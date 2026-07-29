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

## Test

```bash
.venv/bin/python -m pytest
```

This stage intentionally excludes APIs, Docker, LLM integration, and deep
learning.
