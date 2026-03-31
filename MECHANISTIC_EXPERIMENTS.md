# Mechanistic MITR Experiments (VS Code)

This script addresses the two gaps you identified:

1. **Mechanistic validation** via layer-wise probes (answer and negation signals).
2. **Non-adjacent redundancy** via full layer CKA maps and an `all-pairs` MITR variant.

## File

- `mechanistic_mitr_experiment.py`

## Install dependencies

```bash
pip install --upgrade torch transformers datasets matplotlib
```

## Quick smoke run (fast)

```bash
python mechanistic_mitr_experiment.py \
  --model bert-base-uncased \
  --epochs 1 \
  --train-samples 1200 \
  --val-samples 400 \
  --probe-train-samples 600 \
  --probe-val-samples 300 \
  --variants baseline,mitr_adj \
  --output-dir mechanistic_results_smoke
```

## Full run (stronger evidence)

```bash
python mechanistic_mitr_experiment.py \
  --model bert-base-uncased \
  --epochs 3 \
  --train-samples 8000 \
  --val-samples 1500 \
  --probe-train-samples 2500 \
  --probe-val-samples 1000 \
  --variants baseline,mitr_adj,mitr_all \
  --mi-strategy cosine \
  --output-dir mechanistic_results_bert
```

You can repeat with RoBERTa:

```bash
python mechanistic_mitr_experiment.py \
  --model roberta-base \
  --epochs 3 \
  --train-samples 8000 \
  --val-samples 1500 \
  --probe-train-samples 2500 \
  --probe-val-samples 1000 \
  --variants baseline,mitr_adj,mitr_all \
  --mi-strategy cosine \
  --output-dir mechanistic_results_roberta
```

## Outputs

Each run writes:

- `training_curves.png`
- `probe_answer_profile.png`
- `probe_negation_profile.png`
- `cka_heatmaps.png`
- `summary.json`

## How to interpret

- If `mitr_adj` or `mitr_all` only changes final metrics but probe profiles stay nearly identical to baseline, that weakens the mechanism claim.
- If probe profiles shift by depth and contradiction drops, that supports layer specialization.
- If `mitr_all` reduces non-adjacent CKA means in `summary.json` versus `mitr_adj`, that supports your non-adjacent redundancy hypothesis.

