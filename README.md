# Ticket triage baseline

A classical classifier for routing support messages to `account-access`,
`transaction-dispute`, `fraud-report`, or `general`.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test,api]'
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

## Predict and score a holdout

Load a previously fitted model for one-message inference:

```python
from ticket_triage import load_model

predictor = load_model("artifacts/ticket_classifier.joblib")
label = predictor.predict("I cannot access my account")
result = predictor.predict_with_confidence("I did not authorize this transfer")
```

Score an unseen CSV without retraining:

```bash
.venv/bin/python -m ticket_triage.score \
  --model artifacts/ticket_classifier.joblib \
  --input holdout.csv \
  --output predictions/holdout.csv
```

Use `--text-column message` when the input column is not named `text`. The
output preserves every input row and column in its original order, then adds
`prediction` and `confidence`. Invalid rows fail the complete scoring operation;
they are never silently dropped. Inference is entirely local and never fits the
pipeline on holdout messages.

Confidence is the maximum value returned by logistic regression's
`predict_proba`. It is an uncalibrated model score, not a calibrated probability
that the prediction is correct.

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

## Optional HTTP API

Install the API extra and train the classical model first:

```bash
.venv/bin/pip install -e '.[api]'
.venv/bin/python -m ticket_triage.train \
  --data train.csv \
  --model-output artifacts/model.joblib \
  --class-weight balanced
```

`MODEL_PATH` selects the fitted pipeline and defaults to
`artifacts/model.joblib`.

```bash
MODEL_PATH=artifacts/model.joblib \
  .venv/bin/uvicorn ticket_triage.api:app --host 0.0.0.0 --port 8000
```

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail --request POST http://127.0.0.1:8000/predict \
  --header 'Content-Type: application/json' \
  --data '{"text":"Someone withdrew BTC without my permission."}'
curl --fail --request POST http://127.0.0.1:8000/predict/batch \
  --header 'Content-Type: application/json' \
  --data '{"texts":["I cannot log in.","I never authorized this withdrawal."]}'
```

The process loads the model once through FastAPI lifespan handling and never
trains or changes it. Messages are limited to 4,000 characters and batches to
100 messages. A lock protects concurrent read-only access to the shared fitted
pipeline. Endpoints remain synchronous and thin; for materially larger CPU
workloads, use additional worker processes or move inference to a controlled
thread pool rather than adding ad-hoc async wrappers.

Returned confidence is rounded to six decimal places only at serialization.
The underlying logistic-regression probability is not calibrated and should
not be interpreted as the probability that a prediction is correct.

The production classifier remains usable without FastAPI, Docker, deep learning,
or external services. The provider comparison below remains isolated and
optional.

## Optional Gemini comparison

The classical classifier, scoring command, and default tests do not import the
Gemini SDK and work without credentials or network access. To install the
optional comparison dependency:

```bash
.venv/bin/pip install -e '.[llm]'
```

Configure `GEMINI_API_KEY` and `GEMINI_MODEL` in the process environment (for
example, export them from a local ignored `.env` before running the command).
Pricing is not hard-coded because it varies by model and date; set
`GEMINI_INPUT_COST_PER_MILLION` and `GEMINI_OUTPUT_COST_PER_MILLION` to include
an estimated request cost.

The live command is deliberately gated by explicit acknowledgement:

```bash
.venv/bin/python -m ticket_triage.llm_evaluate \
  --data train.csv \
  --cache artifacts/llm_cache.json \
  --confirm-live
```

It compares Gemini with the classical model on the same deterministic first
fold from the existing template-grouped evaluation. Tickets are not used as
few-shot examples. The prompt contains only the ticket, concise route
definitions, and the fraud/dispute distinction. Responses are constrained and
validated against the four approved labels, with bounded retries and no label
fallback. The local cache uses hashed keys and stores labels, latency, and token
counts—not ticket text or credentials.

This comparison is optional, can incur API cost, and may not be reproducible
across Gemini model versions. Reported latency includes provider latency, and
token-derived cost is only an estimate based on the configured rates.
