# MITR Logical Reasoning Results — BERT & RoBERTa

## Why BERT and RoBERTa?

Our DistilBERT experiments showed that MITR-InfoNCE offers the best accuracy-consistency trade-off. But DistilBERT is a small (66M), 6-layer model distilled *from* BERT. Two questions remained:

1. **Does MITR work on BERT itself?** BERT is the *teacher* model — 12 layers, 110M params. If MITR helps here, the layer redundancy problem exists at the source, not just in distilled models.

2. **Does MITR generalize to RoBERTa?** RoBERTa uses a fundamentally different pretraining recipe — no Next Sentence Prediction, dynamic masking, BPE tokenizer, 10x more data. If MITR works here, the finding is **training-paradigm agnostic**.

| Model | Layers | Params | Pretraining | MI Pairs |
|-------|--------|--------|-------------|----------|
| DistilBERT | 6 | 66M | Distillation | 4 |
| **BERT-base** | **12** | **110M** | MLM + NSP | **10** |
| **RoBERTa-base** | **12** | **125M** | MLM only | **10** |

With 12 layers, there are **10 MI penalty terms** (vs 4 for DistilBERT) — more potential redundancy to catch.

## Results

![MITR BERT & RoBERTa Results](roberta_results.png)

We ran 3 configurations per backbone: Baseline, MITR-Cosine, and MITR-CKA. We skipped CLUB and InfoNCE since CLUB diverges completely on DistilBERT — no point scaling to 12 layers.

### BERT-base-uncased

**Validation Accuracy (top left):**
- All three methods show noisy convergence with a sharp peak around epoch 2–3 before settling
- Baseline and Cosine are volatile mid-training, while CKA converges more smoothly in later epochs
- CKA ends highest; Cosine recovers after a mid-training dip

**Final Accuracy:**
- Baseline: 0.6940
- MITR-Cosine: 0.6987 — +0.47%
- MITR-CKA: **0.7027** — **+0.87%, best for BERT**

**Contradiction Rate (bottom left) — lower is better:**
- Baseline: 0.4160
- MITR-Cosine: 0.4480 — **+3.20% worse than Baseline**
- MITR-CKA: 0.4560 — **+4.00% worse than Baseline, worst of all**

Both strategies make BERT's logical consistency significantly worse. MITR regularization increases BERT's contradiction rate for both methods tested.

### RoBERTa-base

**Validation Accuracy (top middle):**
- Cosine jumps quickly and pulls ahead of Baseline from epoch 2 onward, continuing to rise through epoch 5
- CKA starts very slowly (~0.640 at epoch 1), rises steadily but never catches Baseline or Cosine by epoch 5
- Baseline shows steady improvement then plateaus around epoch 4

**Final Accuracy:**
- RoBERTa Baseline: 0.7027
- MITR-Cosine: **0.7220** — **+1.93%, best for RoBERTa**
- MITR-CKA: 0.6953 — −0.73%, the only result below Baseline

**Contradiction Rate (bottom middle) — lower is better:**
- RoBERTa Baseline: 0.5400
- MITR-Cosine: **0.5080** — **+3.20% reduction, best**
- MITR-CKA: 0.5140 — +2.60% reduction

Both strategies improve RoBERTa's logical consistency, the opposite of what they do to BERT.

## Key findings

### 1. Results are strongly backbone-dependent — there is no universal winner

The effect of MITR on logical consistency reverses between BERT and RoBERTa:

| Model | Strategy | Acc Delta | Contra Delta |
|-------|----------|-----------|-------------|
| BERT | MITR-Cosine | +0.47% | −3.20% (worse) |
| BERT | MITR-CKA | +0.87% | −4.00% (worse) |
| RoBERTa | MITR-Cosine | +1.93% | +3.20% (better) |
| RoBERTa | MITR-CKA | −0.73% | +2.60% (better) |

No single strategy dominates across both backbones on both metrics. Any claim that MITR is universally beneficial or detrimental would be premature.

### 2. MITR hurts BERT's logical consistency

For BERT, both Cosine and CKA significantly increase the contradiction rate (by 3.20% and 4.00% respectively). The MI penalty may be interfering with structures that BERT's MLM+NSP pretraining already established for cross-sentence consistency. CKA is the worst offender here — it improves accuracy most (+0.87%) but at the steepest consistency cost.

### 3. MITR-Cosine is the strongest overall result on RoBERTa

Cosine achieves the best outcome of any configuration tested: highest accuracy across both backbones (0.7220, +1.93% vs RoBERTa Baseline) AND the largest contradiction reduction (+3.20%). RoBERTa, trained without NSP and on 10× more data, appears to benefit more from explicit inter-layer diversity pressure.

### 4. MITR-CKA gives mixed results

CKA's behaviour is the most inconsistent:
- BERT: best accuracy (+0.87%), worst consistency (−4.00%)
- RoBERTa: worst accuracy (−0.73%), second-best consistency (+2.60%)

CKA cannot be recommended as a reliable strategy based on these results.

### 5. Cosine is more consistent across backbones than CKA

Cosine improves accuracy on both BERT (+0.47%) and RoBERTa (+1.93%), and provides meaningful contradiction reduction on RoBERTa (+3.20%). On BERT it worsens consistency, but less severely than CKA. If a single strategy must be chosen for a 12-layer model, Cosine is the safer bet.

## Summary table

| Model | Accuracy | Acc Delta | Contradiction Rate | Contra Delta |
|-------|----------|-----------|--------------------|-------------|
| **BERT Baseline** | 0.6940 | — | 0.4160 | — |
| BERT MITR-Cosine | 0.6987 | +0.47% | 0.4480 | −3.20% (worse) |
| **BERT MITR-CKA** | **0.7027** | **+0.87%** | 0.4560 | −4.00% (worse) |
| | | | | |
| **RoBERTa Baseline** | 0.7027 | — | 0.5400 | — |
| **RoBERTa MITR-Cosine** | **0.7220** | **+1.93%** | **0.5080** | **+3.20% (better)** |
| RoBERTa MITR-CKA | 0.6953 | −0.73% | 0.5140 | +2.60% (better) |

## For the paper

The honest framing of this experiment:

> *"MITR's effect on logical consistency is backbone-dependent. On RoBERTa-base, MITR-Cosine improves both accuracy (+1.93%) and contradiction rate (+3.20% reduction), making it the strongest result in our study. On BERT-base, however, both Cosine and CKA increase the contradiction rate (by 3.20% and 4.00% respectively), despite improving accuracy. This reversal suggests the benefit of explicit inter-layer diversity pressure may depend on what consistency structure the backbone's pretraining has already established — a direction that warrants further investigation."*

## Setup

- Models: `bert-base-uncased`, `roberta-base`
- Dataset: BoolQ (8,000 train / 1,500 val)
- GPU: A100
- Epochs: 5
- Batch size: 32 (grad accumulation 2 = effective 64)
- MI lambda: 0.01 (200-step warmup)
- Precision: BF16
- MI strategies tested: Cosine, CKA (CLUB excluded — diverges; InfoNCE excluded — skipped for scope)
