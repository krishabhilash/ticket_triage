# Ticket triage baseline

A classical classifier for routing support messages to `account-access`,
`transaction-dispute`, `fraud-report`, or `general`.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Train

```bash
.venv/bin/python -m ticket_triage.train \
  --data train.csv \
  --model-output artifacts/ticket_classifier.joblib \
  --class-weight balanced
```

The CLI evaluates a word unigram/bigram TF-IDF plus logistic-regression
pipeline on a reproducible stratified 80/20 split. It then fits a fresh pipeline
on all validated rows and saves it with joblib. Keeping TF-IDF inside the
pipeline ensures it is fitted only on training rows during validation.

`--class-weight` accepts `none` or `balanced`.

## Evaluation caveat

Macro F1 is the overall metric, with fraud recall reported separately because
missing fraud has higher operational cost. The supplied data contains many
synthetic-looking variants of the same messages. A random split therefore puts
closely related templates in both partitions and can produce a suspiciously
high score. It is the requested baseline, but it does not prove generalization
to independently written tickets or the hidden holdout.

## Leakage-aware robustness evaluation

```bash
.venv/bin/python -m ticket_triage.robustness --data train.csv
```

This fixed comparison uses out-of-fold predictions from five-fold
`StratifiedKFold` and five-fold `StratifiedGroupKFold`. The grouping heuristic
normalizes only superficial template variation: case, anchored greetings and
closings, known cryptocurrency/asset names, numeric amounts, punctuation, and
whitespace. It is conservative rather than semantic, and it is used only to
assign evaluation groups. The classifier always receives the untouched message.

Template normalization is therefore a robustness check, not an alternative
preprocessing pipeline. Its results should be read alongside the ordinary CV
result and not repeatedly tuned against.

The fixed-seed evaluation produced 92 heuristic groups from 400 rows:

| Experiment | Accuracy | Macro F1 | Fraud precision | Fraud recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|
| Random CV, unweighted | 0.9925 | 0.9890 | 1.0000 | 0.9400 | 0.9691 |
| Random CV, balanced | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Grouped CV, unweighted | 0.8750 | 0.8518 | 1.0000 | 0.6000 | 0.7500 |
| Grouped CV, balanced | 0.8175 | 0.8043 | 0.6333 | 0.7600 | 0.6909 |

The grouped result confirms that ordinary random CV is overly optimistic.
Balancing sacrifices grouped accuracy, macro F1, and fraud precision, but raises
fraud recall from 0.60 to 0.76. Given the stated higher cost of missing fraud,
the recommended final training configuration remains `balanced`; in production,
the additional false-positive fraud escalations would need operational review.

## Test

```bash
.venv/bin/python -m pytest
```

This stage intentionally excludes APIs, Docker, LLM integration, and deep
learning.
