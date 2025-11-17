"""
Quick Start: Minimal Deterministic Training Demo

Run this for a fast proof-of-concept (takes ~5 minutes).
Shows key improvements without full experimental harness.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm import tqdm
import random


def create_simple_data(n=50):
    """Generate arithmetic task."""
    data = []
    for _ in range(n):
        a, b = random.randint(1, 30), random.randint(1, 30)
        data.append({
            'input': f"<CANON> What is {a} + {b}?",
            'canonical': f"The answer is {a + b}.",
            'alternative': f"That would be {a + b}.",
        })
    return data


def margin_loss(logits, targets, gamma=2.0):
    """Ensure winner beats runner-up by γ."""
    target_scores = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = torch.ones_like(logits).scatter_(-1, targets.unsqueeze(-1), float('-inf'))
    runner_up_scores = (logits + mask).max(dim=-1)[0]
    return F.relu(gamma - (target_scores - runner_up_scores)).mean()


def train_quick(model, tokenizer, data, use_margin=False):
    """Quick training loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    
    for epoch in range(2):
        for sample in tqdm(data, desc=f"Epoch {epoch+1}"):
            text = sample['input'] + " " + sample['canonical']
            input_ids = tokenizer.encode(text, return_tensors='pt')
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            if use_margin:
                # Add margin loss on output tokens only
                input_len = len(tokenizer.encode(sample['input']))
                output_logits = outputs.logits[0, input_len-1:-1]
                output_targets = input_ids[0, input_len:]
                loss = loss + 0.3 * margin_loss(output_logits, output_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def test_consistency(model, tokenizer, prompt, runs=10):
    """Test output consistency."""
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for _ in range(runs):
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            output_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            outputs.append(text)
    
    unique = len(set(outputs))
    consistency = (runs - unique + 1) / runs  # Higher = more consistent
    
    return outputs, consistency, unique


def main():
    print("🚀 Quick Deterministic Training Demo\n")
    
    # Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CANON>']})
    
    config = GPT2Config(n_layer=3, n_head=3, n_embd=192, vocab_size=len(tokenizer))
    
    data = create_simple_data(50)
    test_prompt = "<CANON> What is 15 + 27?"
    
    # Train baseline
    print("\n📊 Training BASELINE (CE only)...")
    baseline_model = GPT2LMHeadModel(config)
    baseline_model = train_quick(baseline_model, tokenizer, data, use_margin=False)
    
    # Train enhanced
    print("\n🎯 Training ENHANCED (CE + margin)...")
    enhanced_model = GPT2LMHeadModel(config)
    enhanced_model = train_quick(enhanced_model, tokenizer, data, use_margin=True)
    
    # Test both
    print(f"\n\n{'='*60}")
    print("TESTING CONSISTENCY")
    print(f"{'='*60}\n")
    print(f"Test input: {test_prompt}\n")
    
    print("BASELINE MODEL (10 runs):")
    print("-" * 40)
    base_outs, base_cons, base_unique = test_consistency(baseline_model, tokenizer, test_prompt)
    for i, out in enumerate(base_outs[:5], 1):
        print(f"  {i}. {out}")
    print(f"  ... ({base_unique} unique outputs)")
    
    print("\nENHANCED MODEL (10 runs):")
    print("-" * 40)
    enh_outs, enh_cons, enh_unique = test_consistency(enhanced_model, tokenizer, test_prompt)
    for i, out in enumerate(enh_outs[:5], 1):
        print(f"  {i}. {out}")
    print(f"  ... ({enh_unique} unique outputs)")
    
    # Summary
    improvement = (enh_cons - base_cons) / base_cons * 100
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline consistency: {base_cons:.1%} ({base_unique} unique)")
    print(f"Enhanced consistency: {enh_cons:.1%} ({enh_unique} unique)")
    print(f"Improvement: {improvement:+.1f}%")
    
    if enh_unique < base_unique:
        print("\n✅ Success! Margin loss reduced output diversity")
    else:
        print("\n⚠️  Try more training epochs or higher margin weight")


if __name__ == '__main__':
    main()
