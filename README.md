# Deterministic Training Experiment

This experiment demonstrates how SFT with margin loss and contrastive negatives improves output determinism in language models.

## What This Shows

We train two small GPT-2 models on a synthetic task:
- **Baseline**: Standard cross-entropy (CE) loss only
- **Enhanced**: CE + large-margin loss + contrastive negatives

Then we evaluate both models on determinism metrics:
- **SDR (Sequence Determinism Rate)**: % of re-runs producing identical outputs
- **First-flip position**: Where sequences first diverge across runs
- **Min logit margin**: Safety buffer between chosen token and runner-up
- **Diversity@K**: Average edit distance across K re-decodes

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment (takes ~10-15 minutes on GPU, longer on CPU)
python deterministic_training_demo.py
```

## What Happens

### 1. Data Generation
Creates a synthetic arithmetic task:
- Input: "What is X + Y?"
- Canonical output: "The answer is Z." (consistent phrasing)
- Paraphrases: "That would be Z.", "You get Z.", etc. (used as contrastive negatives)

This mimics real-world scenarios where multiple phrasings are valid but you want consistent output.

### 2. Training
Trains two models:

**Baseline**: 
```
L = L_CE(y* | x)
```

**Enhanced**:
```
L = L_CE(y* | x) + λ₁·L_margin + λ₂·L_contrastive
```

Where:
- **L_margin**: Ensures canonical token beats runner-up by margin γ
- **L_contrastive**: Pushes down valid but non-canonical paraphrases

### 3. Evaluation
Runs each test input 10 times with simulated jitter to measure:
- How often the model produces identical outputs (SDR)
- How wide the logit margins are (stability under perturbation)
- How much diversity exists across re-runs (should decrease)

### 4. Visualization
Generates comparison plots showing improvement in all metrics.

## Expected Results

You should see improvements like:
- **SDR**: 40-60% → 85-95% (more consistent outputs)
- **Logit margins**: 2-4 → 6-10 (wider safety buffer)
- **Diversity**: 5-10 → 0-2 (less variation)

## Architecture Details

### DeterministicTrainer
Implements three loss components:

1. **Cross-entropy**: Standard language modeling
2. **Margin loss**: 
   ```
   L_margin = Σ_t max(0, γ - (ℓ_y* - max_{k≠y*} ℓ_k))
   ```
   Penalizes small gaps between canonical and runner-up tokens

3. **Contrastive loss**:
   ```
   L_rank = log(1 + exp(s(x, ỹ) - s(x, y*)))
   ```
   Explicitly reduces probability of valid paraphrases

### DeterminismEvaluator
Measures:
- **SDR**: Fraction of byte-identical outputs across K runs
- **First-flip**: Earliest divergence point in character sequences
- **Logit margin**: min(top_logit - second_logit) across positions
- **Diversity**: Average Levenshtein distance between output pairs

## Customization

### Adjust hyperparameters:
```python
trainer = DeterministicTrainer(
    model=model,
    tokenizer=tokenizer,
    margin_gamma=2.0,        # Margin requirement (higher = more aggressive)
    margin_weight=0.3,       # Weight on margin loss
    contrastive_weight=0.5,  # Weight on contrastive loss
)
```

### Use your own data:
Replace `CanonicalDataset` with your task:
```python
class MyDataset(Dataset):
    def __init__(self, tokenizer):
        self.examples = [
            {
                'input': 'Your prompt',
                'canonical': 'Canonical response',
                'paraphrases': ['Alt 1', 'Alt 2', ...],
            },
            ...
        ]
```

### Test different model sizes:
```python
config = GPT2Config.from_pretrained('gpt2')
config.n_layer = 8  # More layers
config.n_embd = 512  # Larger hidden size
```

## Key Insights

1. **Margin loss is crucial**: Without it, models maintain multimodal distributions even with canonical targets
2. **Contrastive negatives help**: Explicitly pushing down alternatives works better than hoping CE alone will do it
3. **Control tokens work**: The `<CANON>` token allows you to toggle determinism at inference time
4. **Evaluation matters**: Single-run accuracy doesn't capture determinism—you need multi-run metrics

## Limitations

- **Synthetic task**: Real production tasks have more complex multimodality
- **Small model**: Results will be more pronounced on larger models
- **Limited training**: Full results need more epochs and larger datasets
- **Statistical not mathematical**: You get 95%+ determinism, not 100%

## Next Steps

To apply this to production:

1. **Collect canonical data**: Either human-curated or teacher-selected
2. **Identify high-variance prompts**: Find inputs with high output diversity
3. **Scale up training**: Use larger models and datasets
4. **Add paraphrase consistency**: Train on input paraphrase clusters
5. **Deploy with control tokens**: Use `<CANON>` to toggle determinism per request

## References

- Large Margin Deep Networks: https://arxiv.org/abs/1803.05598
- DPO: https://arxiv.org/abs/2305.18290
- Contrastive Learning: https://arxiv.org/abs/2002.05709

## Troubleshooting

**OOM errors**: Reduce `batch_size` or model size
**Slow training**: Use smaller `num_samples` or fewer `num_epochs`
**No GPU**: Works on CPU but slower (expect 30+ minutes)
**Low improvements**: Try increasing `margin_gamma` or `contrastive_weight`
