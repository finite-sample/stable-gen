"""
Advanced Extensions: Production-Ready Deterministic Training

This shows how to extend the basic approach to production scenarios:
1. Paraphrase consistency (input-side invariance)
2. DPO-style preference optimization
3. Stability rewards
4. Multi-metric tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np


class AdvancedDeterministicTrainer:
    """
    Production-ready trainer with all techniques from the paper.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,  # Reference model for KL
        tokenizer,
        config: Dict,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        # Hyperparameters
        self.margin_gamma = config.get('margin_gamma', 2.0)
        self.margin_weight = config.get('margin_weight', 0.3)
        self.contrastive_weight = config.get('contrastive_weight', 0.5)
        self.kl_weight = config.get('kl_weight', 0.1)
        self.paraphrase_weight = config.get('paraphrase_weight', 0.2)
        self.stability_weight = config.get('stability_weight', 0.1)
        
    def margin_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Large-margin loss: L_margin = Σ_t max(0, γ - (ℓ_y* - max_{k≠y*} ℓ_k))
        """
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        mask = torch.ones_like(logits)
        mask.scatter_(-1, targets.unsqueeze(-1), float('-inf'))
        runner_up_logits = (logits + mask).max(dim=-1)[0]
        
        margin = target_logits - runner_up_logits
        return F.relu(self.margin_gamma - margin).mean()
    
    def contrastive_loss(
        self,
        logits_pos: torch.Tensor,
        logits_neg: torch.Tensor,
        targets_pos: torch.Tensor,
        targets_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive ranking: L_rank = log(1 + exp(s(x,ỹ) - s(x,y*)))
        """
        pos_log_probs = F.log_softmax(logits_pos, dim=-1)
        neg_log_probs = F.log_softmax(logits_neg, dim=-1)
        
        pos_scores = pos_log_probs.gather(-1, targets_pos.unsqueeze(-1)).squeeze(-1).sum()
        neg_scores = neg_log_probs.gather(-1, targets_neg.unsqueeze(-1)).squeeze(-1).sum()
        
        return F.softplus(neg_scores - pos_scores)
    
    def kl_divergence(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(model || ref) to prevent collapse away from base model.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        kl = (ref_probs * (ref_probs.log() - log_probs)).sum(dim=-1)
        return kl.mean()
    
    def paraphrase_consistency_loss(
        self,
        hidden_states_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Consistency loss across paraphrases: force similar representations.
        Uses contrastive learning on hidden states.
        """
        if len(hidden_states_list) < 2:
            return torch.tensor(0.0, device=hidden_states_list[0].device)
        
        # Pool hidden states to get sequence-level representations
        representations = [h.mean(dim=1) for h in hidden_states_list]  # [batch, hidden]
        
        # Normalize
        representations = [F.normalize(r, dim=-1) for r in representations]
        
        # Contrastive loss: maximize similarity between paraphrases
        consistency_loss = 0
        count = 0
        for i in range(len(representations)):
            for j in range(i + 1, len(representations)):
                # Cosine similarity (want this to be 1)
                similarity = (representations[i] * representations[j]).sum(dim=-1)
                consistency_loss += (1 - similarity).mean()
                count += 1
        
        return consistency_loss / max(count, 1)
    
    def stability_reward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        canonical_output: str,
        num_samples: int = 5,
    ) -> torch.Tensor:
        """
        Reward model for producing consistent outputs across re-decodes.
        Sample M times with tiny noise and reward if all match canonical.
        """
        model.eval()
        outputs = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Add tiny dropout for realistic jitter
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.01,  # Tiny temperature
                    top_p=0.99,
                )
                output_text = self.tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                outputs.append(output_text)
        
        # Reward: +1 if all match canonical, else penalize by diversity
        if all(o == canonical_output for o in outputs):
            reward = 1.0
        else:
            # Penalize based on unique outputs
            unique_count = len(set(outputs))
            reward = -0.1 * unique_count
        
        model.train()
        return torch.tensor(reward, device=input_ids.device)
    
    def dpo_loss(
        self,
        logits_chosen: torch.Tensor,
        logits_rejected: torch.Tensor,
        targets_chosen: torch.Tensor,
        targets_rejected: torch.Tensor,
        ref_logits_chosen: torch.Tensor,
        ref_logits_rejected: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """
        Direct Preference Optimization loss.
        Increases P(y* | x) relative to P(ỹ | x) with KL penalty.
        """
        # Compute log ratios
        chosen_logprobs = F.log_softmax(logits_chosen, dim=-1)
        rejected_logprobs = F.log_softmax(logits_rejected, dim=-1)
        ref_chosen_logprobs = F.log_softmax(ref_logits_chosen, dim=-1)
        ref_rejected_logprobs = F.log_softmax(ref_logits_rejected, dim=-1)
        
        # Get scores
        chosen_scores = chosen_logprobs.gather(-1, targets_chosen.unsqueeze(-1)).squeeze(-1).sum()
        rejected_scores = rejected_logprobs.gather(-1, targets_rejected.unsqueeze(-1)).squeeze(-1).sum()
        
        ref_chosen_scores = ref_chosen_logprobs.gather(-1, targets_chosen.unsqueeze(-1)).squeeze(-1).sum()
        ref_rejected_scores = ref_rejected_logprobs.gather(-1, targets_rejected.unsqueeze(-1)).squeeze(-1).sum()
        
        # DPO objective
        logits_diff = (chosen_scores - rejected_scores) - (ref_chosen_scores - ref_rejected_scores)
        loss = -F.logsigmoid(beta * logits_diff)
        
        return loss
    
    def compute_full_loss(
        self,
        batch: Dict,
        use_dpo: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with all components.
        """
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        # Standard CE loss
        outputs = self.model(input_ids, labels=target_ids, output_hidden_states=True)
        ce_loss = outputs.loss
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        total_loss = ce_loss
        loss_dict = {'ce': ce_loss.item()}
        
        # Margin loss
        valid_mask = target_ids != -100
        valid_logits = logits[valid_mask]
        valid_targets = target_ids[valid_mask]
        
        margin_loss = self.margin_loss(valid_logits, valid_targets)
        total_loss += self.margin_weight * margin_loss
        loss_dict['margin'] = margin_loss.item()
        
        # Contrastive loss (if negatives provided)
        if 'negative_ids' in batch:
            neg_outputs = self.model(input_ids, labels=batch['negative_ids'])
            neg_logits = neg_outputs.logits
            neg_valid_logits = neg_logits[valid_mask]
            neg_valid_targets = batch['negative_ids'][valid_mask]
            
            contrastive = self.contrastive_loss(
                valid_logits, neg_valid_logits,
                valid_targets, neg_valid_targets
            )
            total_loss += self.contrastive_weight * contrastive
            loss_dict['contrastive'] = contrastive.item()
        
        # KL to reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids)
            ref_logits = ref_outputs.logits[valid_mask]
        
        kl_loss = self.kl_divergence(valid_logits, ref_logits)
        total_loss += self.kl_weight * kl_loss
        loss_dict['kl'] = kl_loss.item()
        
        # Paraphrase consistency (if multiple inputs provided)
        if 'paraphrase_hidden_states' in batch:
            paraphrase_hidden_list = [hidden_states] + batch['paraphrase_hidden_states']
            consistency = self.paraphrase_consistency_loss(paraphrase_hidden_list)
            total_loss += self.paraphrase_weight * consistency
            loss_dict['paraphrase'] = consistency.item()
        
        # DPO (alternative to contrastive)
        if use_dpo and 'negative_ids' in batch:
            with torch.no_grad():
                ref_chosen = self.ref_model(input_ids).logits[valid_mask]
                ref_rejected = self.ref_model(input_ids).logits[valid_mask]  # Simplified
            
            dpo = self.dpo_loss(
                valid_logits, neg_valid_logits,
                valid_targets, neg_valid_targets,
                ref_chosen, ref_rejected,
            )
            total_loss += 0.3 * dpo
            loss_dict['dpo'] = dpo.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class ProductionMetricsTracker:
    """
    Comprehensive metrics tracking for production deployment.
    """
    
    def __init__(self):
        self.metrics_history = {
            'sdr': [],
            'min_margin': [],
            'first_flip_pos': [],
            'diversity': [],
            'latency_p50': [],
            'latency_p99': [],
        }
    
    def track_determinism(
        self,
        model,
        tokenizer,
        test_inputs: List[str],
        num_runs: int = 20,
    ) -> Dict:
        """Track all determinism metrics over time."""
        import time
        
        model.eval()
        
        sdr_scores = []
        margin_scores = []
        first_flip_positions = []
        diversity_scores = []
        latencies = []
        
        with torch.no_grad():
            for input_text in test_inputs:
                outputs = []
                margins = []
                run_latencies = []
                
                for _ in range(num_runs):
                    start = time.time()
                    
                    input_ids = tokenizer.encode(input_text, return_tensors='pt')
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=20,
                        do_sample=False,
                    )
                    
                    latency = time.time() - start
                    run_latencies.append(latency)
                    
                    output_text = tokenizer.decode(
                        output_ids[0][input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    outputs.append(output_text)
                    
                    # Compute margin
                    model_outputs = model(output_ids)
                    logits = model_outputs.logits[0]
                    top2, _ = torch.topk(logits, k=2, dim=-1)
                    margin = (top2[:, 0] - top2[:, 1]).min().item()
                    margins.append(margin)
                
                # SDR
                sdr = sum(1 for o in outputs if o == outputs[0]) / len(outputs)
                sdr_scores.append(sdr)
                
                # Min margin
                margin_scores.append(np.mean(margins))
                
                # First flip position
                first_flip = float('inf')
                for i in range(1, len(outputs)):
                    for pos in range(min(len(outputs[0]), len(outputs[i]))):
                        if outputs[0][pos] != outputs[i][pos]:
                            first_flip = min(first_flip, pos)
                            break
                first_flip_positions.append(first_flip if first_flip != float('inf') else len(outputs[0]))
                
                # Diversity
                unique_count = len(set(outputs))
                diversity_scores.append(unique_count)
                
                # Latency
                latencies.extend(run_latencies)
        
        metrics = {
            'mean_sdr': np.mean(sdr_scores),
            'std_sdr': np.std(sdr_scores),
            'mean_margin': np.mean(margin_scores),
            'mean_first_flip': np.mean(first_flip_positions),
            'mean_diversity': np.mean(diversity_scores),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p99': np.percentile(latencies, 99),
        }
        
        # Update history
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
        
        return metrics
    
    def should_promote_model(
        self,
        current_metrics: Dict,
        baseline_metrics: Dict,
        thresholds: Dict = None,
    ) -> Tuple[bool, str]:
        """
        Decide if model should be promoted to production.
        """
        if thresholds is None:
            thresholds = {
                'min_sdr': 0.90,  # At least 90% SDR
                'min_margin': 3.0,  # Comfortable margin
                'max_diversity': 2.0,  # Low diversity
                'max_latency_p99': 0.5,  # Fast enough
            }
        
        checks = []
        
        # SDR check
        sdr_ok = current_metrics['mean_sdr'] >= thresholds['min_sdr']
        checks.append(('SDR', sdr_ok, current_metrics['mean_sdr']))
        
        # Margin check
        margin_ok = current_metrics['mean_margin'] >= thresholds['min_margin']
        checks.append(('Margin', margin_ok, current_metrics['mean_margin']))
        
        # Diversity check
        diversity_ok = current_metrics['mean_diversity'] <= thresholds['max_diversity']
        checks.append(('Diversity', diversity_ok, current_metrics['mean_diversity']))
        
        # Latency check
        latency_ok = current_metrics['latency_p99'] <= thresholds['max_latency_p99']
        checks.append(('Latency P99', latency_ok, current_metrics['latency_p99']))
        
        all_pass = all(check[1] for check in checks)
        
        report = "Model Promotion Report:\n"
        report += "=" * 50 + "\n"
        for name, passed, value in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            report += f"{status} {name}: {value:.4f}\n"
        report += "=" * 50 + "\n"
        report += f"Decision: {'PROMOTE' if all_pass else 'DO NOT PROMOTE'}\n"
        
        return all_pass, report


# Example usage
def production_training_example():
    """
    Example of production training pipeline.
    """
    
    # Setup
    tokenizer = ...  # Your tokenizer
    model = ...  # Your model
    ref_model = ...  # Frozen reference model
    
    config = {
        'margin_gamma': 3.0,  # Aggressive margin
        'margin_weight': 0.4,
        'contrastive_weight': 0.5,
        'kl_weight': 0.1,
        'paraphrase_weight': 0.2,
        'stability_weight': 0.1,
    }
    
    trainer = AdvancedDeterministicTrainer(model, ref_model, tokenizer, config)
    metrics_tracker = ProductionMetricsTracker()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Prepare batch with paraphrases
            batch_data = {
                'input_ids': batch['input_ids'],
                'target_ids': batch['target_ids'],
                'negative_ids': batch['negative_ids'],
                'paraphrase_hidden_states': batch.get('paraphrase_hidden_states', []),
            }
            
            loss, loss_dict = trainer.compute_full_loss(batch_data, use_dpo=True)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate after each epoch
        current_metrics = metrics_tracker.track_determinism(
            model, tokenizer, test_inputs
        )
        
        # Check if ready for production
        should_promote, report = metrics_tracker.should_promote_model(
            current_metrics, baseline_metrics
        )
        
        print(report)
        
        if should_promote:
            print("✅ Model ready for production!")
            break
    
    return model, metrics_tracker


if __name__ == '__main__':
    print("Advanced Deterministic Training Extensions")
    print("=" * 60)
    print("\nThis module provides:")
    print("  • Paraphrase consistency loss")
    print("  • DPO-style preference optimization")
    print("  • Stability rewards")
    print("  • Production metrics tracking")
    print("  • Model promotion gating")
    print("\nSee production_training_example() for usage.")
