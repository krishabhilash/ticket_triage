# Ticket Triage: classical support-ticket classifier

## 1. Project overview

This repository implements the required classical ML solution for routing short
support tickets into four queues:

- `account-access`: sign-in, password, verification-code, or locked-account issues
- `transaction-dispute`: a recognized or user-initiated transaction is delayed,
  incorrect, duplicated, unexpectedly priced, or missing
- `fraud-report`: unauthorized activity, account compromise, phishing, scams,
  theft, or impersonation
- `general`: informational requests outside the other routes

The key boundary is authorization: a recognized transaction with a processing
problem is a `transaction-dispute`; deceptive, unauthorized, or compromised-account
activity is a `fraud-report`.

## 2. Design summary

The primary model is word unigram/bigram TF-IDF followed by multinomial logistic
regression. For approximately 400 short labelled messages, this is fast, auditable,
reproducible, and less prone to overfitting than training a deep model. It also
provides a strong sparse-text baseline without an external service. The classical
pipeline is the default training, prediction, scoring, and production path.

```text
validated CSV -> stratified split -> TF-IDF -> logistic regression -> metrics
                                      |                |
                                      +-- saved sklearn Pipeline --+-> local inference
```

TF-IDF is inside the scikit-learn `Pipeline`, so each validation fold learns its
vocabulary only from that fold's training rows. Optional FastAPI, container, CI,
production safeguards, and Gemini comparison are extensions—not requirements for
running the core solution.

## 3. Repository structure

```text
.
├── src/ticket_triage/
│   ├── constants.py        # labels, columns, and random seed
│   ├── data.py             # CSV validation and stratified split
│   ├── model.py            # TF-IDF/logistic-regression pipeline
│   ├── evaluation.py       # classification metrics
│   ├── train.py            # training CLI
│   ├── predictor.py        # saved-model prediction interface
│   ├── score.py            # holdout CSV scoring CLI
│   ├── template_groups.py  # evaluation-only grouping heuristic
│   ├── robustness.py       # random and grouped cross-validation
│   ├── metadata.py         # model metadata sidecar
│   ├── api.py              # optional FastAPI transport
│   ├── observability.py    # optional logging/metric hooks
│   ├── llm.py              # optional Gemini adapter
│   └── llm_evaluate.py     # optional live comparison
├── tests/                  # deterministic unit and integration tests
├── requirements/           # runtime and development dependency sets
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml
└── train.csv
```

Generated artifacts, caches, `.env`, virtual environments, and datasets are
excluded from version control where appropriate.

## 4. Installation

Python 3.11 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install development/test tooling separately:

```bash
python -m pip install -e '.[test]'
```

The optional HTTP service requires `python -m pip install -e '.[api]'`. The
classical commands do not require FastAPI, Docker, Gemini, an API key, or network
access.

## 5. Training

Input must be a CSV with `text` and `label` columns. Messages must be non-empty
strings and labels must be one of the four routes.

```bash
.venv/bin/python -m ticket_triage.train \
  --data train.csv \
  --model-output artifacts/model.joblib \
  --class-weight balanced
```

The CLI uses a fixed random seed of `42` and a stratified 80/20 validation split,
prints metrics, then refits on all validated rows. `--class-weight` accepts `none`
or `balanced`. The recommended submitted model uses `balanced` because grouped
evaluation measured higher fraud recall, despite lower precision and macro F1.

The `.joblib` artifact is one fitted scikit-learn pipeline containing both TF-IDF
and logistic regression. Training also writes
`artifacts/model.joblib.metadata.json` with aggregate training/configuration data,
not ticket messages.

## 6. Evaluation

### Fixed 80/20 validation split

The balanced command above reproduced these results on 80 validation rows:

| Metric | Value |
|---|---:|
| Accuracy | 1.0000 |
| Macro F1 | 1.0000 |
| Fraud recall | 1.0000 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| account-access | 1.0000 | 1.0000 | 1.0000 | 20 |
| transaction-dispute | 1.0000 | 1.0000 | 1.0000 | 18 |
| fraud-report | 1.0000 | 1.0000 | 1.0000 | 10 |
| general | 1.0000 | 1.0000 | 1.0000 | 32 |

Confusion-matrix label order is account-access, transaction-dispute,
fraud-report, general:

```text
[[20,  0,  0,  0],
 [ 0, 18,  0,  0],
 [ 0,  0, 10,  0],
 [ 0,  0,  0, 32]]
```

This perfect random-split result is suspiciously high because related templates
cross the partition; it does not demonstrate generalization.

### Five-fold random and template-grouped evaluation

```bash
.venv/bin/python -m ticket_triage.robustness --data train.csv
```

The command generates out-of-fold predictions with fixed-seed five-fold
`StratifiedKFold` and `StratifiedGroupKFold`:

| Experiment | Accuracy | Macro F1 | Fraud precision | Fraud recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|
| Random CV, unweighted | 0.9925 | 0.9890 | 1.0000 | 0.9400 | 0.9691 |
| Random CV, balanced | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Grouped CV, unweighted | 0.8750 | 0.8518 | 1.0000 | 0.6000 | 0.7500 |
| Grouped CV, balanced | 0.8175 | 0.8043 | 0.6333 | 0.7600 | 0.6909 |

Template normalization folds case, anchored greetings/closings, known asset names,
amounts, punctuation, and whitespace only to construct groups. The classifier
always receives original text. This conservative robustness check reduces direct
template leakage; it is neither classifier preprocessing nor proof of performance
on an independently collected hidden holdout.

## 7. Metric reasoning

Accuracy can hide weak minority-class behavior, particularly for the 50 fraud
examples. Macro F1 weights each route equally, so it is the overall selection
metric. Fraud recall is reported separately because a false negative may leave
unauthorized activity in a routine queue, with greater customer and financial
cost than an extra fraud review.

The grouped comparison exposes the measured trade-off. Balancing raises fraud
recall from `0.60` to `0.76`, while fraud precision falls from `1.00` to `0.6333`
and macro F1 from `0.8518` to `0.8043`. Given the exercise's stated cost of missed
fraud, `balanced` is the explicit risk-weighted choice. If macro F1 alone were the
only objective, the grouped evidence favors unweighted training.

## 8. Data quality and leakage

`train.csv` contains 400 rows and two required columns:

| Label | Rows | Share |
|---|---:|---:|
| general | 160 | 40.0% |
| account-access | 100 | 25.0% |
| transaction-dispute | 90 | 22.5% |
| fraud-report | 50 | 12.5% |

The audit found zero missing messages, missing labels, empty messages, unknown
labels, exact duplicate rows, or exact duplicate message strings. It found only
92 normalized template groups; 380 of 400 rows belong to groups containing more
than one superficial variant. The wording therefore appears substantially
templated or synthetic even without exact duplicates.

Random validation can place such variants in training and validation, explaining
the near-perfect scores. Grouped validation keeps each heuristic group in one fold
and is more conservative, but the heuristic cannot identify every semantic
duplicate or simulate future production language. Hidden-holdout performance
remains uncertain.

## 9. Single-message prediction

`load_model` returns an interface with `predict(text) -> label`:

```python
from ticket_triage import load_model

predictor = load_model("artifacts/model.joblib")
label = predictor.predict("I cannot access my account")
print(label)  # account-access
```

The example was verified against the balanced artifact generated by the training
command.

## 10. Holdout scoring

The holdout requires a `text` column by default; use `--text-column` to select a
different name. No retraining or TF-IDF fitting occurs.

```bash
.venv/bin/python -m ticket_triage.score \
  --model artifacts/model.joblib \
  --input holdout.csv \
  --output predictions/holdout.csv
```

Input:

```csv
ticket_id,text
T-2,I cannot log in
T-1,How does staking work
```

Verified output:

```csv
ticket_id,text,prediction,confidence
T-2,I cannot log in,account-access,0.4161020631167093
T-1,How does staking work,general,0.6488530746471604
```

Every input row and column is preserved in order; `prediction` and `confidence`
are appended. Invalid rows fail clearly rather than being dropped. Confidence is
the maximum logistic-regression `predict_proba` value and is not necessarily
calibrated.

## 11. Testing

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
```

The current suite has 44 passing tests. It covers dataset and message validation,
pipeline fitting and save/reload, single prediction, row-preserving CSV scoring and
CLI execution, model metadata, template grouping, FastAPI endpoints/lifecycle and
review policy, logging privacy, and mocked Gemini response/retry/cache behavior.
Tests use temporary small datasets and make no live external calls.

## 12. Optional extension: FastAPI service

Install the API extra and train the model before starting the service:

```bash
.venv/bin/pip install -e '.[api]'
MODEL_PATH=artifacts/model.joblib \
LOW_CONFIDENCE_THRESHOLD=0.60 \
  .venv/bin/uvicorn ticket_triage.api:app --host 0.0.0.0 --port 8000
```

FastAPI lifespan handling loads exactly one model for the process. It never trains
or changes the model. `MODEL_PATH` defaults to `artifacts/model.joblib`.

- `GET /health`: liveness; HTTP 200 even if the model failed to load
- `GET /ready`: HTTP 200 only with a loaded model, otherwise HTTP 503
- `POST /predict`: one ticket, maximum 4,000 characters
- `POST /predict/batch`: 1–100 tickets, preserving order

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Someone withdrew BTC without my permission."}'
curl --fail -X POST http://127.0.0.1:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["I cannot log in.","I never authorized this withdrawal."]}'
```

A verified single response from the current artifact is:

```json
{"label":"fraud-report","confidence":0.650233,"requires_review":false}
```

`requires_review` marks scores below `LOW_CONFIDENCE_THRESHOLD` without changing
the label. The default `0.60` is an operational placeholder, not an automatically
validated threshold or calibrated correctness probability; deployment must select
it from validation, calibration, capacity, and business-cost analysis.

## 13. Optional extension: container

Training and serving are separate lifecycle stages. Train on the host, build the
image, and mount the artifact read-only:

```bash
.venv/bin/python -m ticket_triage.train \
  --data train.csv --model-output artifacts/model.joblib --class-weight balanced
docker build --tag ticket-triage-api .
docker run --rm --detach \
  --name ticket-triage-api \
  --publish 8000:8000 \
  --env MODEL_PATH=/app/artifacts/model.joblib \
  --volume "$(pwd)/artifacts:/app/artifacts:ro" \
  ticket-triage-api
```

```bash
until curl --fail --silent http://127.0.0.1:8000/ready; do sleep 1; done
curl --fail -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Someone withdrew BTC without my permission."}'
test "$(docker exec ticket-triage-api id -u)" -ne 0
docker stop ticket-triage-api
```

The verified container returned the same valid fraud label/confidence/review
schema shown above. The Python 3.12 slim image exposes port 8000, health-checks
`/ready`, installs runtime dependencies only, and runs as non-root UID 10001. It
never trains a classical or LLM model at startup. Dependencies use bounded direct
versions rather than a lockfile, so builds are not claimed to be byte-identical.

## 14. Optional extension: continuous integration

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`, with
least-privilege `contents: read`. It tests Python 3.11 and 3.12 using:

```bash
ruff check .
ruff format --check .
pytest -q
```

After both Python jobs pass, CI builds—but does not publish—the Docker image. It
requires no repository secrets and excludes live Gemini evaluation, so CI neither
incurs provider cost nor transmits ticket data to an external LLM.

## 15. Optional experiment: Gemini comparison

Gemini is an isolated experiment for unfamiliar or semantically ambiguous tickets,
not the mandatory classifier or default production path. Install it explicitly:

```bash
.venv/bin/pip install -e '.[llm]'
export GEMINI_API_KEY='set-outside-source-control'
export GEMINI_MODEL='your-explicit-model-version'
.venv/bin/python -m ticket_triage.llm_evaluate \
  --data train.csv --cache artifacts/llm_cache.json --confirm-live
```

The prompt defines the same four labels and explicitly distinguishes recognized
processing disputes from unauthorized/deceptive fraud. The adapter requests a
structured JSON label constrained to the approved set, validates every response,
uses bounded retries, and never silently substitutes a label. Configuration comes
from environment variables; keys are not logged or stored in the cache.

The comparison uses the same deterministic first template-grouped fold for the
balanced classical model and Gemini, without few-shot evaluation examples. The
command reports macro F1, fraud precision/recall/F1, classical and Gemini latency,
failures, invalid/retry outputs, and token usage. Cost is reported only when the
operator supplies current input/output price rates.

No complete, reproducible live Gemini result is claimed in this README. Therefore
no LLM macro F1, fraud recall, latency, invalid-output rate, token total, or cost is
reported. Results can vary with provider/model versions; tickets sent for live
evaluation leave the local environment, so privacy approval and data minimization
are required.

## 16. Production architecture

An evidence-based next architecture would keep the classical classifier on the
ordinary path and send high-risk or genuinely uncertain tickets to human review.
After calibration and cost analysis, an optional LLM escalation could be tested for
semantically ambiguous cases; it should not silently replace the classical result.
No arbitrary confidence threshold should be promoted automatically. Thresholds
must be selected using validation data, calibrated scores where feasible, fraud
false-negative cost, review capacity, and measured operational outcomes.

## 17. Scaling to 10,000 requests per minute

The sparse classical model is the default because inference is CPU-friendly and
local, but this repository contains no throughput benchmark. Expected scaling is:

- stateless API replicas behind a gateway/load balancer;
- one immutable model loaded once per worker, with replica memory sized accordingly;
- horizontal CPU scaling and bounded batch inference where latency requirements
  permit it;
- gateway rate limiting, authentication, TLS termination, and request-size limits;
- dashboards and alerts for latency, errors, readiness, label mix, review rate,
  drift, and fraud false negatives.

Calling a generative LLM for every request would be expected to add network latency,
provider dependency, privacy exposure, and variable cost. It is justified only if
measured semantic gains on representative ambiguous tickets outweigh those costs,
for example as a bounded escalation path rather than the default.

## 18. Production considerations

The optional API emits structured JSON completion logs with request ID, route,
status, latency, and model version; raw ticket text and secrets are omitted.
Thread-safe instrumentation hooks count requests and latency, validation and model
load failures, predicted-label distribution, and low-confidence predictions. A
production adapter could export these to Prometheus.

Operations should monitor data drift, label-definition drift, predicted-label and
low-confidence rates, calibration, and especially fraud false negatives. Model
metadata records aggregate dataset/configuration/validation information to support
versioning, retraining decisions, and rollback to an immutable validated artifact.
Human review outcomes should feed controlled relabelling and retraining rather than
automatic online learning.

Ticket data is sensitive: minimize retention, redact infrastructure logs, restrict
access, and define deletion policies. Authentication, TLS termination, rate
limiting, WAF/request limits, and abuse controls belong at a managed API gateway;
this exercise intentionally implements no custom authentication system.

## 19. Scope and trade-offs

The work prioritized validated local data handling, a leakage-safe classical
pipeline, reproducible evaluation, robust holdout scoring, and focused tests.
Deep-learning training, hyperparameter sweeps, semantic clustering, probability
calibration, a hybrid router, custom authentication, a frontend, and a model
registry were deliberately left out. They are unnecessary for the small exercise
without stronger evidence or deployment requirements.

FastAPI, container, CI, observability hooks, and Gemini isolation were included as
small optional extensions demonstrating deployability; none is required to train
or use the core classifier. With more time, the priority would be independently
written production-like evaluation data, label-guideline adjudication, calibration,
threshold/cost analysis, drift baselines, load testing, and monitored shadow trials.

**Candidate time spent:** `[complete before submission: ___ hours]`

## 20. Limitations

- The training set has only 400 messages, including just 50 fraud examples.
- Wording is heavily templated or synthetic-looking despite no exact duplicates.
- Fraud and transaction-dispute boundaries can remain ambiguous in real language.
- Random validation is optimistic; grouped validation is heuristic and uncertain.
- Logistic-regression confidence is uncalibrated.
- Real production-ticket evaluation and representative error review are required.
- Language, fraud patterns, customer behavior, and label policy can drift.

## 21. Submission checklist

- GitHub repository: <https://github.com/krishabhilash/ticket_triage>
- Real, unsquashed commit history with focused implementation stages
- README with reproducible commands, measured results, and explicit limitations
- Local checks: Ruff lint/format and **44 passing tests**
