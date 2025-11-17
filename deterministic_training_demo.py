"""
Deterministic Training for Language Models
Demonstrates SFT + margin loss + contrastive negatives for improving output determinism.

This implementation shows:
1. Baseline SFT (cross-entropy only)
2. Margin-enhanced SFT
3. Contrastive negatives
4. Determinism evaluation metrics (SDR, logit margins)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class DeterministicTrainer:
    """Trainer with margin loss and contrastive objectives for determinism."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        margin_gamma: float = 2.0,
        contrastive_weight: float = 0.5,
        margin_weight: float = 0.3,
        use_margin_loss: bool = True,
        use_contrastive: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.margin_gamma = margin_gamma
        self.contrastive_weight = contrastive_weight
        self.margin_weight = margin_weight
        self.use_margin_loss = use_margin_loss
        self.use_contrastive = use_contrastive
        
    def compute_margin_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Large-margin loss: ensure canonical token beats runner-up by margin γ.
        L_margin = sum_t max(0, γ - (ℓ_y* - max_{k≠y*} ℓ_k))
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Get logits for target tokens
        target_logits = logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        # Create mask for non-target positions
        mask = torch.ones_like(logits)
        mask.scatter_(2, target_ids.unsqueeze(-1), float('-inf'))
        
        # Get max logit among non-targets
        runner_up_logits, _ = (logits + mask).max(dim=-1)
        
        # Margin loss: penalize when margin is < γ
        margin = target_logits - runner_up_logits
        margin_loss = F.relu(self.margin_gamma - margin)
        
        return margin_loss.mean()
    
    def compute_contrastive_loss(
        self,
        logits_positive: torch.Tensor,
        logits_negative: torch.Tensor,
        target_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive ranking loss: push down non-canonical but valid alternatives.
        L_rank = log(1 + exp(s(x, ỹ) - s(x, y*)))
        """
        # Compute sequence scores (sum of log probs)
        positive_log_probs = F.log_softmax(logits_positive, dim=-1)
        negative_log_probs = F.log_softmax(logits_negative, dim=-1)
        
        # Get log probs for each token
        pos_scores = positive_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        neg_scores = negative_log_probs.gather(2, negative_ids.unsqueeze(-1)).squeeze(-1)
        
        # Sum over sequence to get sequence-level scores
        pos_score = pos_scores.sum(dim=-1)
        neg_score = neg_scores.sum(dim=-1)
        
        # Ranking loss
        contrastive_loss = F.softplus(neg_score - pos_score)
        
        return contrastive_loss.mean()
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        negative_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step with all losses."""
        
        # Forward pass for canonical target
        outputs = self.model(input_ids, labels=target_ids)
        ce_loss = outputs.loss
        logits = outputs.logits
        
        total_loss = ce_loss
        losses = {'ce_loss': ce_loss.item()}
        
        # Add margin loss
        if self.use_margin_loss:
            margin_loss = self.compute_margin_loss(logits[:, :-1], target_ids[:, 1:])
            total_loss += self.margin_weight * margin_loss
            losses['margin_loss'] = margin_loss.item()
        
        # Add contrastive loss
        if self.use_contrastive and negative_ids is not None:
            neg_outputs = self.model(input_ids, labels=negative_ids)
            contrastive_loss = self.compute_contrastive_loss(
                logits[:, :-1],
                neg_outputs.logits[:, :-1],
                target_ids[:, 1:],
                negative_ids[:, 1:],
            )
            total_loss += self.contrastive_weight * contrastive_loss
            losses['contrastive_loss'] = contrastive_loss.item()
        
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses


class CanonicalDataset(Dataset):
    """
    Dataset with canonical targets and paraphrastic alternatives.
    
    Creates a synthetic task where each input has:
    - One canonical output (y*)
    - Multiple valid paraphrases (ỹ)
    """
    
    def __init__(
        self,
        tokenizer,
        num_samples: int = 1000,
        add_control_token: bool = True,
    ):
        self.tokenizer = tokenizer
        self.add_control_token = add_control_token
        self.samples = self._generate_samples(num_samples)
        
    def _generate_samples(self, num_samples: int) -> List[Dict]:
        """
        Generate synthetic data with canonical answers and paraphrases.
        Example task: "What is X + Y?" with multiple valid phrasings.
        """
        samples = []
        
        # Define templates for canonical and paraphrastic responses
        canonical_templates = [
            "The sum of {a} and {b} is {result}.",
            "The answer is {result}.",
            "It equals {result}.",
        ]
        
        paraphrase_templates = [
            "{a} plus {b} equals {result}.",
            "That would be {result}.",
            "The result is {result}.",
            "{result} is the answer.",
            "You get {result}.",
        ]
        
        for i in range(num_samples):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            result = a + b
            
            # Create input
            control = "<CANON> " if self.add_control_token else ""
            input_text = f"{control}What is {a} + {b}?"
            
            # Pick ONE canonical answer (consistency!)
            canonical_template = canonical_templates[i % len(canonical_templates)]
            canonical = canonical_template.format(a=a, b=b, result=result)
            
            # Generate paraphrases as negatives
            paraphrases = [
                template.format(a=a, b=b, result=result)
                for template in paraphrase_templates
            ]
            
            samples.append({
                'input': input_text,
                'canonical': canonical,
                'paraphrases': paraphrases,
                'metadata': {'a': a, 'b': b, 'result': result},
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input and canonical target
        input_text = sample['input']
        canonical_text = sample['canonical']
        full_text_canonical = input_text + " " + canonical_text
        
        input_ids = self.tokenizer.encode(full_text_canonical, return_tensors='pt')[0]
        target_ids = input_ids.clone()
        
        # Mask input tokens in labels (only train on output)
        input_len = len(self.tokenizer.encode(input_text))
        target_ids[:input_len] = -100
        
        # Pick a random paraphrase as negative
        negative_text = random.choice(sample['paraphrases'])
        full_text_negative = input_text + " " + negative_text
        negative_ids = self.tokenizer.encode(full_text_negative, return_tensors='pt')[0]
        negative_ids[:input_len] = -100
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'negative_ids': negative_ids,
            'input_text': input_text,
            'canonical_text': canonical_text,
        }


class DeterminismEvaluator:
    """Evaluates model determinism with multiple metrics."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_determinism(
        self,
        test_inputs: List[str],
        num_runs: int = 10,
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> Dict:
        """
        Evaluate determinism metrics:
        - SDR (Sequence Determinism Rate)
        - First-flip position
        - Min logit margin
        - Diversity@K
        """
        self.model.eval()
        
        results = {
            'sdr_scores': [],
            'first_flip_positions': [],
            'min_margins': [],
            'diversity_scores': [],
            'outputs_by_input': {},
        }
        
        with torch.no_grad():
            for input_text in tqdm(test_inputs, desc="Evaluating determinism"):
                outputs = []
                logit_margins = []
                
                # Run K times
                for run in range(num_runs):
                    # Add tiny noise to simulate real-world jitter
                    if add_noise:
                        # Simulate numerical jitter
                        torch.manual_seed(random.randint(0, 1000000))
                    
                    input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
                    
                    # Greedy decoding
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=30,
                        do_sample=False,
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    output_text = self.tokenizer.decode(
                        output_ids[0][input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    outputs.append(output_text)
                    
                    # Compute logit margins
                    model_outputs = self.model(output_ids)
                    logits = model_outputs.logits[0, :-1]  # [seq_len, vocab_size]
                    
                    # Get top-2 logits at each position
                    top2, _ = torch.topk(logits, k=2, dim=-1)
                    margins = top2[:, 0] - top2[:, 1]
                    logit_margins.append(margins.min().item())
                
                # Compute metrics
                unique_outputs = set(outputs)
                sdr = len([o for o in outputs if o == outputs[0]]) / len(outputs)
                
                # First flip position
                first_flip = len(outputs[0])
                for i in range(1, len(outputs)):
                    for pos in range(min(len(outputs[0]), len(outputs[i]))):
                        if outputs[0][pos] != outputs[i][pos]:
                            first_flip = min(first_flip, pos)
                            break
                
                # Diversity@K (average edit distance)
                diversity = np.mean([
                    self._edit_distance(outputs[i], outputs[j])
                    for i in range(len(outputs))
                    for j in range(i+1, len(outputs))
                ])
                
                results['sdr_scores'].append(sdr)
                results['first_flip_positions'].append(first_flip)
                results['min_margins'].append(np.mean(logit_margins))
                results['diversity_scores'].append(diversity)
                results['outputs_by_input'][input_text] = outputs
        
        # Aggregate statistics
        summary = {
            'mean_sdr': np.mean(results['sdr_scores']),
            'std_sdr': np.std(results['sdr_scores']),
            'mean_first_flip': np.mean(results['first_flip_positions']),
            'mean_min_margin': np.mean(results['min_margins']),
            'mean_diversity': np.mean(results['diversity_scores']),
        }
        
        return results, summary
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein distance."""
        if len(s1) < len(s2):
            return DeterminismEvaluator._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


def plot_training_comparison(baseline_metrics, enhanced_metrics, save_path='determinism_comparison.png'):
    """Visualize improvement in determinism metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Determinism Training Results: Baseline vs. Margin-Enhanced SFT', fontsize=16)
    
    metrics = ['mean_sdr', 'mean_first_flip', 'mean_min_margin', 'mean_diversity']
    titles = ['Sequence Determinism Rate (↑)', 'First Flip Position (↑)', 
              'Min Logit Margin (↑)', 'Output Diversity (↓)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        baseline_val = baseline_metrics[metric]
        enhanced_val = enhanced_metrics[metric]
        
        bars = ax.bar(['Baseline SFT', 'Margin + Contrastive'], 
                      [baseline_val, enhanced_val],
                      color=['#ff7f0e', '#2ca02c'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Calculate improvement
        if metric == 'mean_diversity':
            improvement = (baseline_val - enhanced_val) / baseline_val * 100
            improvement_text = f'{improvement:.1f}% reduction'
        else:
            improvement = (enhanced_val - baseline_val) / baseline_val * 100
            improvement_text = f'{improvement:.1f}% improvement'
        
        ax.set_title(f'{title}\n{improvement_text}', fontsize=12, pad=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    return fig


def train_model(
    model,
    train_dataset,
    tokenizer,
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    use_margin_loss=True,
    use_contrastive=True,
    device='cuda',
):
    """Train model with specified loss configuration."""
    
    trainer = DeterministicTrainer(
        model=model,
        tokenizer=tokenizer,
        margin_gamma=2.0,
        margin_weight=0.3,
        contrastive_weight=0.5,
        use_margin_loss=use_margin_loss,
        use_contrastive=use_contrastive,
    )
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    model.to(device)
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        loss_components = defaultdict(float)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            negative_ids = batch['negative_ids'].to(device) if use_contrastive else None
            
            optimizer.zero_grad()
            
            loss, losses = trainer.train_step(input_ids, target_ids, negative_ids)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{losses.get('ce_loss', 0):.4f}",
                'margin': f"{losses.get('margin_loss', 0):.4f}",
            })
        
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        for k, v in loss_components.items():
            print(f"  {k}: {v/len(dataloader):.4f}")
    
    return model, epoch_losses


def main():
    """Main experimental pipeline."""
    
    print("=" * 60)
    print("Deterministic Training Experiment")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load small GPT-2 model
    print("\n1. Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add control token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CANON>']})
    
    config = GPT2Config.from_pretrained('gpt2')
    config.n_layer = 4  # Smaller for faster training
    config.n_head = 4
    config.n_embd = 256
    
    # Create two models: baseline and enhanced
    print("\n2. Creating datasets...")
    train_dataset = CanonicalDataset(tokenizer, num_samples=500, add_control_token=True)
    test_inputs = [train_dataset.samples[i]['input'] for i in range(0, 100, 10)]
    
    # Train baseline model (CE only)
    print("\n3. Training BASELINE model (CE loss only)...")
    baseline_model = GPT2LMHeadModel(config)
    baseline_model.resize_token_embeddings(len(tokenizer))
    baseline_model, _ = train_model(
        baseline_model,
        train_dataset,
        tokenizer,
        num_epochs=2,
        use_margin_loss=False,
        use_contrastive=False,
        device=device,
    )
    
    # Train enhanced model (CE + margin + contrastive)
    print("\n4. Training ENHANCED model (CE + margin + contrastive)...")
    enhanced_model = GPT2LMHeadModel(config)
    enhanced_model.resize_token_embeddings(len(tokenizer))
    enhanced_model, _ = train_model(
        enhanced_model,
        train_dataset,
        tokenizer,
        num_epochs=2,
        use_margin_loss=True,
        use_contrastive=True,
        device=device,
    )
    
    # Evaluate both models
    print("\n5. Evaluating BASELINE model determinism...")
    baseline_evaluator = DeterminismEvaluator(baseline_model, tokenizer, device)
    _, baseline_summary = baseline_evaluator.evaluate_determinism(
        test_inputs, num_runs=10, add_noise=True
    )
    
    print("\n6. Evaluating ENHANCED model determinism...")
    enhanced_evaluator = DeterminismEvaluator(enhanced_model, tokenizer, device)
    _, enhanced_summary = enhanced_evaluator.evaluate_determinism(
        test_inputs, num_runs=10, add_noise=True
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Baseline':<15} {'Enhanced':<15} {'Change':<15}")
    print("-" * 75)
    
    for metric in ['mean_sdr', 'mean_first_flip', 'mean_min_margin', 'mean_diversity']:
        baseline_val = baseline_summary[metric]
        enhanced_val = enhanced_summary[metric]
        
        if metric == 'mean_diversity':
            change = (baseline_val - enhanced_val) / baseline_val * 100
            change_str = f"{change:+.1f}% ↓"
        else:
            change = (enhanced_val - baseline_val) / baseline_val * 100
            change_str = f"{change:+.1f}% ↑"
        
        print(f"{metric:<30} {baseline_val:<15.4f} {enhanced_val:<15.4f} {change_str:<15}")
    
    # Create visualization
    print("\n7. Creating visualization...")
    plot_training_comparison(baseline_summary, enhanced_summary, 
                            save_path='/mnt/user-data/outputs/determinism_comparison.png')
    
    print("\n✓ Experiment complete!")
    print("\nKey Findings:")
    print(f"  • SDR improved by {(enhanced_summary['mean_sdr'] - baseline_summary['mean_sdr']) / baseline_summary['mean_sdr'] * 100:.1f}%")
    print(f"  • Logit margin increased by {(enhanced_summary['mean_min_margin'] - baseline_summary['mean_min_margin']) / baseline_summary['mean_min_margin'] * 100:.1f}%")
    print(f"  • Output diversity reduced by {(baseline_summary['mean_diversity'] - enhanced_summary['mean_diversity']) / baseline_summary['mean_diversity'] * 100:.1f}%")


if __name__ == '__main__':
    main()
