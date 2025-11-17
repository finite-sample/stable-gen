"""
Fixed Deterministic Training Demo

Changes from original:
1. More complex task with inherent multimodality
2. Larger model with capacity for multiple modes
3. Explicit temperature sampling to create variance
4. Better margin calculation
5. Separate train/test sets
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm import tqdm
import random
import numpy as np


def create_multimodal_data(n=100):
    """
    Create task with INHERENT multimodality.
    Same input can have multiple valid, different answers.
    """
    data = []
    
    # Templates with genuinely different structures
    templates = [
        "The answer is {result}.",
        "It equals {result}.",  
        "{result} is correct.",
        "You get {result}.",
        "That would be {result}.",
    ]
    
    for _ in range(n):
        a, b = random.randint(10, 50), random.randint(10, 50)
        result = a + b
        
        # For baseline: pick random template (creates multimodality)
        # For canonical: always pick same template (creates unimodality)
        canonical_template = templates[0]  # Always first one
        random_template = random.choice(templates)  # Random for negatives
        
        data.append({
            'input': f"<CANON> What is {a} + {b}?",
            'canonical': canonical_template.format(result=result),
            'alternatives': [t.format(result=result) for t in templates if t != canonical_template],
        })
    
    return data


def margin_loss(logits, targets, gamma=3.0):
    """
    Fixed margin loss - only on OUTPUT tokens, higher gamma.
    """
    # Get logits for target tokens
    target_scores = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Mask out target position to get runner-up
    mask = torch.zeros_like(logits)
    mask.scatter_(-1, targets.unsqueeze(-1), float('-inf'))
    runner_up_scores, _ = (logits + mask).max(dim=-1)
    
    # Only penalize when margin < gamma
    margin = target_scores - runner_up_scores
    loss = F.relu(gamma - margin)
    
    # Only compute on valid positions (not padding)
    valid = targets != -100
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    return loss[valid].mean()


def train_model_fixed(model, tokenizer, data, use_margin=False, epochs=3):
    """Fixed training with proper margin application."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        margin_losses = []
        
        pbar = tqdm(data, desc=f"Epoch {epoch+1}/{epochs}")
        for sample in pbar:
            # Prepare canonical text
            input_text = sample['input']
            canonical_text = sample['canonical']
            full_text = input_text + " " + canonical_text
            
            input_ids = tokenizer.encode(full_text, return_tensors='pt')
            
            # Create target labels (mask input portion)
            target_ids = input_ids.clone()
            input_len = len(tokenizer.encode(input_text))
            target_ids[:, :input_len] = -100
            
            # Forward pass
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss
            
            # Add margin loss ONLY on output tokens
            if use_margin:
                logits = outputs.logits[0]
                
                # Extract only the OUTPUT portion
                output_logits = logits[input_len-1:-1]  # Shift for next-token prediction
                output_targets = target_ids[0, input_len:]
                
                # Filter out -100 (padding)
                valid_mask = output_targets != -100
                if valid_mask.sum() > 0:
                    m_loss = margin_loss(
                        output_logits[valid_mask],
                        output_targets[valid_mask],
                        gamma=3.0  # Higher gamma for more aggressive margins
                    )
                    loss = loss + 0.5 * m_loss  # Higher weight
                    margin_losses.append(m_loss.item())
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if use_margin and len(margin_losses) > 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'margin_loss': f"{np.mean(margin_losses[-10:]):.4f}"
                })
            else:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(data):.4f}")
        if use_margin and len(margin_losses) > 0:
            print(f"  Avg margin loss: {np.mean(margin_losses):.4f}")
    
    return model


def evaluate_with_sampling(model, tokenizer, test_data, num_runs=20):
    """
    Evaluate with small temperature to create realistic variance.
    This simulates real-world numerical jitter.
    """
    model.eval()
    
    sdr_scores = []
    margin_scores = []
    diversity_scores = []
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Evaluating"):
            input_text = sample['input']
            outputs = []
            margins = []
            
            for _ in range(num_runs):
                input_ids = tokenizer.encode(input_text, return_tensors='pt')
                
                # Use small temperature to simulate jitter
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=15,
                    do_sample=True,
                    temperature=0.3,  # Small but nonzero
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                output_text = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                outputs.append(output_text)
                
                # Compute margin on generated output
                full_output = model(output_ids)
                logits = full_output.logits[0]
                
                # Get margins for output portion only
                if len(logits) > len(input_ids[0]):
                    output_portion = logits[len(input_ids[0]):]
                    top2, _ = torch.topk(output_portion, k=2, dim=-1)
                    margin = (top2[:, 0] - top2[:, 1]).mean().item()
                    margins.append(margin)
            
            # Compute metrics
            unique_outputs = list(set(outputs))
            most_common = max(set(outputs), key=outputs.count)
            sdr = outputs.count(most_common) / len(outputs)
            
            diversity = len(unique_outputs)
            
            sdr_scores.append(sdr)
            diversity_scores.append(diversity)
            if len(margins) > 0:
                margin_scores.append(np.mean(margins))
    
    return {
        'mean_sdr': np.mean(sdr_scores),
        'mean_margin': np.mean(margin_scores) if margin_scores else 0,
        'mean_diversity': np.mean(diversity_scores),
        'sdr_scores': sdr_scores,
    }


def main():
    print("=" * 70)
    print("FIXED Deterministic Training Demo")
    print("=" * 70)
    
    # Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CANON>']})
    
    # LARGER model (more capacity for multimodality)
    config = GPT2Config(
        n_layer=6,      # Was 3-4, now 6
        n_head=6,       # Was 3-4, now 6  
        n_embd=384,     # Was 192-256, now 384
        vocab_size=len(tokenizer)
    )
    
    # Create SEPARATE train and test sets
    print("\n1. Creating datasets...")
    train_data = create_multimodal_data(150)
    test_data = create_multimodal_data(30)  # Completely different examples
    
    print(f"   Train: {len(train_data)} examples")
    print(f"   Test: {len(test_data)} examples")
    
    # Show example
    print(f"\n   Example input: {train_data[0]['input']}")
    print(f"   Canonical: {train_data[0]['canonical']}")
    print(f"   Alternatives: {train_data[0]['alternatives'][:2]}")
    
    # Train baseline (NO margin loss)
    print("\n2. Training BASELINE model (CE only)...")
    baseline_model = GPT2LMHeadModel(config)
    baseline_model.resize_token_embeddings(len(tokenizer))
    baseline_model = train_model_fixed(
        baseline_model, tokenizer, train_data,
        use_margin=False,
        epochs=3
    )
    
    # Train enhanced (WITH margin loss)
    print("\n3. Training ENHANCED model (CE + margin)...")
    enhanced_model = GPT2LMHeadModel(config)
    enhanced_model.resize_token_embeddings(len(tokenizer))
    enhanced_model = train_model_fixed(
        enhanced_model, tokenizer, train_data,
        use_margin=True,
        epochs=3
    )
    
    # Evaluate with sampling
    print("\n4. Evaluating baseline...")
    baseline_results = evaluate_with_sampling(
        baseline_model, tokenizer, test_data, num_runs=20
    )
    
    print("\n5. Evaluating enhanced...")
    enhanced_results = evaluate_with_sampling(
        enhanced_model, tokenizer, test_data, num_runs=20
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Enhanced':<15} {'Change':<15}")
    print("-" * 70)
    
    for metric in ['mean_sdr', 'mean_margin', 'mean_diversity']:
        base_val = baseline_results[metric]
        enh_val = enhanced_results[metric]
        
        if metric == 'mean_diversity':
            if base_val > 0:
                change = (base_val - enh_val) / base_val * 100
                change_str = f"{change:+.1f}% ↓"
            else:
                change_str = "N/A"
        else:
            if base_val > 0:
                change = (enh_val - base_val) / base_val * 100
                change_str = f"{change:+.1f}% ↑"
            else:
                change_str = "N/A"
        
        print(f"{metric:<20} {base_val:<15.4f} {enh_val:<15.4f} {change_str:<15}")
    
    # Show example outputs
    print("\n" + "=" * 70)
    print("EXAMPLE OUTPUTS")
    print("=" * 70)
    
    test_input = test_data[0]['input']
    print(f"\nTest input: {test_input}\n")
    
    # Generate 5 samples from each
    baseline_model.eval()
    enhanced_model.eval()
    
    print("BASELINE (5 samples):")
    with torch.no_grad():
        for i in range(5):
            input_ids = tokenizer.encode(test_input, return_tensors='pt')
            output_ids = baseline_model.generate(
                input_ids, max_new_tokens=15,
                do_sample=True, temperature=0.3, top_p=0.95
            )
            text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"  {i+1}. {text}")
    
    print("\nENHANCED (5 samples):")
    with torch.no_grad():
        for i in range(5):
            input_ids = tokenizer.encode(test_input, return_tensors='pt')
            output_ids = enhanced_model.generate(
                input_ids, max_new_tokens=15,
                do_sample=True, temperature=0.3, top_p=0.95
            )
            text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"  {i+1}. {text}")
    
    print("\n" + "=" * 70)
    
    # Interpretation
    if enhanced_results['mean_sdr'] > baseline_results['mean_sdr']:
        improvement = (enhanced_results['mean_sdr'] - baseline_results['mean_sdr']) / baseline_results['mean_sdr'] * 100
        print(f"✅ SUCCESS: Enhanced model is {improvement:.1f}% more deterministic")
    else:
        print("⚠️  Results unclear - may need more training or larger model")
    
    if enhanced_results['mean_margin'] > baseline_results['mean_margin']:
        print(f"✅ Margins increased as expected")
    else:
        print(f"⚠️  Margins did not increase - check margin loss implementation")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
