# MITR Logical Reasoning Results — All MI Strategies

## What is this experiment?

We trained a DistilBERT model on **BoolQ** (yes/no questions about Wikipedia passages) and tested whether adding a **mutual information (MI) penalty** between transformer layers improves logical reasoning.

The idea is simple: each transformer layer should learn something *different*. If two layers learn the same thing, that's wasted capacity. By penalizing this redundancy, we force the model to use each layer for a distinct "reasoning step," which should help with logical consistency.

We tested **4 different ways** to measure layer similarity (and thus penalize redundancy):

| Strategy | What it does | Has learnable parameters? |
|----------|-------------|--------------------------|
| **CLUB** | Trains a small network to estimate mutual information between layers | Yes |
| **InfoNCE** | Uses contrastive learning — treats matching layer pairs as positives, random pairs as negatives | Yes |
| **Cosine** | Measures the angle between layer outputs (cosine similarity) | No |
| **CKA** | Compares the *shape* of representations across the whole batch, not just individual samples | No |

## The two experiments

**Experiment 1 — Accuracy:** Can MITR maintain or improve accuracy on yes/no questions?

**Experiment 2 — Contradiction Rate:** If you ask "Is water wet?" and "Is water not wet?", a logical model should give opposite answers. The contradiction rate measures how often the model gives the *same* answer to both (which is logically impossible).

## Results

![MITR All Strategies Comparison](cka_results.png)

### What the charts show

**Training Loss (top left):**
- Baseline, Cosine, and CKA all decrease smoothly and converge near ~0.38 by epoch 5
- InfoNCE decreases faster than the others, reaching ~0.18 by epoch 5 — the contrastive objective accelerates task-loss minimization
- CLUB diverges: its loss goes negative immediately and hits the −50 clamp by epoch 2, staying clamped throughout. Despite this, the task loss continues to train because the MI loss is separate

**Validation Accuracy (top middle):**
- **Baseline** peaks around epoch 3–4 (~0.703) then settles at 0.6980
- **InfoNCE** starts slow (~0.600 at epoch 1) because the contrastive loss competes with the task loss early on, then catches up rapidly, surpassing all other strategies by epoch 4–5
- **CLUB** tracks below all others throughout (~0.687 final)
- **Cosine** and **CKA** are stable but converge slightly below Baseline (~0.688 and ~0.691)

**MI Regularisation Loss (top right):**
- **Cosine** and **CKA** are flat near zero throughout — the most stable estimators
- **InfoNCE** starts around −5 and gradually decreases to ~−20, stable but non-trivial
- **CLUB** hits the −50 clamp by epoch 2 and stays there — completely diverged. The CLUB internal network fails to keep up with the representation changes, saturating at the loss boundary

**Final Accuracy (bottom left):**
- InfoNCE: **0.7000** (best, +0.20% vs Baseline)
- Baseline: 0.6980
- CKA: 0.6913 (−0.67%)
- Cosine: 0.6880 (−1.00%)
- CLUB: 0.6873 (−1.07%, worst)

**Contradiction Rate (bottom middle) — lower is better:**
- **InfoNCE: 0.4120 (best, −1.80% reduction vs Baseline)**
- **Cosine: 0.4140 (second best, −1.60% reduction)**
- Baseline: 0.4300
- CLUB: 0.4300 (no change — ties Baseline exactly)
- **CKA: 0.4240 (smallest reduction, only −0.60%)**

## Key takeaways

### InfoNCE offers the best accuracy-consistency trade-off

InfoNCE achieves both the highest final accuracy (0.7000, +0.20% vs Baseline) and the largest contradiction rate reduction (0.4120, −1.80%). Despite a slow start at epoch 1, the contrastive estimator converges powerfully and outperforms all other strategies on both metrics simultaneously.

### CLUB diverges — but the task still trains

CLUB's MI loss hits the −50 clamp at epoch 2 and never recovers. The internal variational network saturates completely. Despite this, the task loss continues to train (they are separate objectives), which is why CLUB still achieves 0.6873 accuracy. However, because the MI penalty is effectively disabled after epoch 2, CLUB provides no contradiction reduction whatsoever (ties Baseline at 0.4300).

### Parameter-free methods are stable but not the strongest

CKA and Cosine produce flat, near-zero MI loss curves throughout training — the most stable behavior. However, stability does not translate into the best outcomes: Cosine achieves −1.60% contradiction reduction and CKA only −0.60%. InfoNCE, despite its non-trivial MI loss trajectory, outperforms both on every metric.

### CKA provides the weakest contradiction improvement

CKA is the least effective strategy for reducing contradictions (+0.60% reduction), despite being the most commonly favored for representational analysis. CKA's batch-level geometry comparison may smooth out the signal too aggressively, producing a penalty that is too weak to force meaningful layer differentiation.

### Summary table

| Model | Accuracy | Acc Delta | Contradiction Rate | Contra Reduction |
|-------|----------|-----------|--------------------|-----------------|
| Baseline | 0.6980 | — | 0.4300 | — |
| **MITR-InfoNCE** | **0.7000** | **+0.20%** | **0.4120** | **+1.80%** |
| MITR-Cosine | 0.6880 | −1.00% | 0.4140 | +1.60% |
| MITR-CKA | 0.6913 | −0.67% | 0.4240 | +0.60% |
| MITR-CLUB | 0.6873 | −1.07% | 0.4300 | +0.00% |

### Bottom line

If you care about **accuracy only** → InfoNCE (0.7000, only strategy to beat Baseline)

If you care about **logical consistency** → InfoNCE again (0.4120, largest contradiction reduction)

If you care about **training stability at all costs** → Cosine or CKA (flat MI loss curves), with Cosine giving better consistency than CKA

**CLUB should not be used** — it diverges completely and provides no contradiction benefit over Baseline

## Setup

- Model: `distilbert-base-uncased`
- Dataset: BoolQ (8,000 train / 1,500 val)
- GPU: A100
- Epochs: 5
- Batch size: 64
- MI lambda: 0.01 (with 200-step linear warmup)
- Precision: BF16
