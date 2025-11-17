# Training for Deterministic Outputs: SFT and RL Approaches

**Short version:** Yes—SFT and RL can help make "same input ⇒ same output" far more likely by collapsing multimodality. You're not fighting randomness anymore; you're fighting the model's natural tendency to keep multiple paraphrastic modes alive for a given prompt. Training can push probability mass onto a single, canonical answer and widen the margin to everything else.

Below is a way to structure the problem cleanly, plus concrete training knobs that target determinism (not format adherence).

---

## 0. Frame it Precisely

You have a conditional distribution $P_\theta(y \mid x)$. Even with greedy decoding, if the top few continuations are close, tiny numeric or scheduling jitters can flip the argmax. The cure is to:

- **(a)** Pick a canonical target $y^*$ per input $x$
- **(b)** Increase the margin $\log P_\theta(y^* \mid x) - \log P_\theta(y \neq y^* \mid x)$ to be comfortably positive across the whole sequence

**Determinism then reduces to:** make $P_\theta(\cdot \mid x)$ behave like a point mass at $y^*$.

---

## 1. Data & Targets: Define a Canonicalization Policy

You need a mapping $c: x \mapsto y^*$. Three pragmatic ways to build it:

### Human-curated canonicalization
For key intents, write crisp targets (one answer, consistent wording).

### Teacher selection
Sample multiple high-quality candidates per $x$ (different prompts/temperatures/models), rank with a deterministic scorer/reranker, and freeze the top candidate as $y^*$ ("silver" labels).

### Style token + policy
Decide on a house style (e.g., terse, declarative) and mark training prompts with a control token like `<CANON>`. At serve time, always include `<CANON>`.

**Whatever you choose, be consistent.** Your training signal can't collapse ambiguity if your labels are ambiguous.

---

## 2. Metrics: Measure Determinism, Not Just Accuracy

You'll want a harness that runs each prompt $K$ times under realistic load and reports:

- **Sequence Determinism Rate (SDR):** fraction of runs that are byte-identical
- **First-flip position:** where sequences first diverge
- **Min logit margin:** per-token safety margin between the chosen token and runner-up
- **Diversity@K:** average pairwise edit distance across $K$ re-decodes (should trend to zero)

These tell you if you're actually narrowing the distribution, not just getting lucky once.

---

## 3. SFT: Push Mass onto the Canonical

Plain SFT already helps—cross-entropy with hard targets concentrates probability on $y^*$. To make it determinism-oriented, add two tweaks:

### (a) Large-margin SFT (sequence or token level)

Augment CE with a margin term so the chosen token is not merely highest, but higher by at least $\gamma$:

$$L_{\text{margin}} = \sum_t \max\left(0, \gamma - \left(\ell_{y_t^*} - \max_{k \neq y_t^*} \ell_k\right)\right)$$

where $\ell_k$ are logits at step $t$. This trains the model to keep a cushion between the canonical token and the runner-up—exactly what fights flips.

### (b) Contrastive negatives

For each $x$, include non-canonical but semantically-valid outputs $\tilde{y}$ as hard negatives. Add a ranking loss:

$$L_{\text{rank}} = \log\left(1 + \exp\left(s(x, \tilde{y}) - s(x, y^*)\right)\right)$$

where $s$ is a sequence score (e.g., sum of token logits or a learned reranker). This explicitly pushes alternatives down.

### Implementation notes

- **Avoid label smoothing** (it softens the very peaks you're trying to sharpen)
- **Up-weight examples** that historically show high variance (many non-identical re-decodes)
- **Use a style/control token** (e.g., `<CANON>`) during SFT so you keep the option to turn determinism on or off at inference

---

## 4. RL: Widen the Margin at the Sequence Level

SFT optimizes token-wise; RL lets you set sequence-level goals. Useful, focused choices:

### (a) Preference optimization (DPO/IPO-style)

For each $x$, treat $(y^*, \tilde{y})$ as preferred/dispreferred pairs. Optimize a pairwise objective that increases $P_\theta(y^* \mid x)$ relative to $P_\theta(\tilde{y} \mid x)$, with a KL term to a reference model to prevent collapse. This is "SFT with teeth."

### (b) Stability reward

Define a per-prompt reward that penalizes diversity across re-decodes:

- Sample $M$ outputs $\{y_i\}$ with small, fixed noise (e.g., top-p > 0 but tiny, or dropout on)
- Reward $+1$ if all $y_i$ are the same and equal $y^*$; otherwise subtract a penalty proportional to the average pairwise distance
- Add a small negative entropy term on the token distribution at decode time to encourage peaky behavior, keeping a KL anchor so the model doesn't become dull globally

### (c) Canonical similarity reward

Even when not identical, grant partial reward based on normalized edit distance, BERTScore, or task-specific similarity to $y^*$. This smooths the landscape and avoids brittle learning.

### Guardrails

- **Keep a KL to the base model.** Determinism without quality is a Pyrrhic victory.
- **Train with the same decode settings you will serve** (greedy, same stop rules), or the policy won't line up with deployment.

---

## 5. Paraphrase-Consistency (Input-side Invariance)

You asked about "same input ⇒ same output," but it's cheap and powerful to also train paraphrase invariance so "similar input ⇒ same output" holds:

- Build small clusters $\{x_j\}$ of paraphrases for the same intent; share the same $y^*$
- Add a consistency loss that forces the model's hidden states or output distributions to match across paraphrases (Siamese/contrastive penalty), while still maximizing $P(y^* \mid x_j)$

This reduces accidental branching on superficial wording.

---

## 6. Practical Recipe You Can Ship

### Create canonical pairs
For top traffic intents, freeze a canonical $y^*$ (human or silver teacher).

### SFT + margin + negatives
Train with CE + margin + contrastive ranking; include `<CANON>` control token.

### Preference optimization
Run DPO/IPO with $(y^*, \tilde{y})$ pairs; add a small stability reward and a KL to your base.

### Paraphrase consistency
For a subset, add paraphrase clusters and a consistency penalty.

### Evaluate with SDR/first-flip/margin
Gate model promotion on SDR under concurrency, not just single-tenant accuracy.

### Serve with `<CANON>` and greedy
Keep numerics pinned; you've now collapsed most intrinsic multimodality.

---

## Limits and Expectations

Training can't give you **mathematical determinism**; it gives you **statistical determinism**—i.e., the model so strongly prefers a single continuation that tiny nudges don't flip it. That's already good enough for production-grade "same input ⇒ same output" once the infra is stable.

If the task is inherently ambiguous (trivia with multiple names, free-form summarization), you'll need a policy decision for what "canonical" means (e.g., always the shortest synonym; always active voice). SFT/RL then enforce that policy.

---

## TL;DR

**Treat determinism as canonicalization + margin maximization.**

- SFT concentrates mass on a single target
- Margin and contrastive losses keep look-alikes below the decision boundary
- RL (preference + stability rewards) widens the gap at sequence level
- Evaluate with multi-run determinism metrics
- Ship with a `<CANON>` control so you can flip the switch across your stack
