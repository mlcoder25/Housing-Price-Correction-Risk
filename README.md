# Housing Price Correction Risk Classifier — Production MLOps Pipeline

Predicts whether a European country will report a **year-over-year house
price decline** in a given quarter, using only information knowable at the
end of the *prior* quarter. Built on real Eurostat-style House Price Index
data (30 countries, 2022-Q4 to 2025-Q3/Q4).

This started as a prototype on synthetic data to demonstrate MLOps practices
(experiment tracking, CI/CD, drift monitoring); it now runs on the real
dataset end-to-end, including an honest check against a naive baseline —
see [Model performance, honestly](#model-performance-honestly) below.

## Why this project

Training a good model is one skill. Shipping and operating it safely — and
knowing when it *isn't* actually adding value — is another. This repo
demonstrates the second skill:

| Concern | Without MLOps | This repo |
|---|---|---|
| Comparing runs | Re-run notebook, eyeball printed metrics | Every run logged to **MLflow** — params, metrics, and the model artifact |
| Promoting a model | Copy a `.pkl` file and hope | Best run is **registered** in the MLflow Model Registry as a versioned artifact |
| Catching a bad model | Found in production | **CI gate**: build fails if ROC-AUC/F1 drop below threshold |
| Catching bad data | Silent failure downstream | **Data validation tests** (schema, duplicates, row count, class balance) run on every push |
| Knowing if the model is even useful | Assume the metric = the value | **Naive persistence baseline** computed and compared on every run, not just the fancy model's own score |
| Serving | Ad-hoc script | **FastAPI** `/predict` endpoint, containerized |
| Knowing when to retrain | Nobody notices | **Evidently** drift report compares live vs. training distribution |
| Scaling / self-healing | Single process | **Kubernetes** manifests with readiness/liveness probes |

## Model performance, honestly

The trained model (logistic regression won the search) scores **ROC-AUC
0.99 / F1 0.80** on the held-out test set. Those numbers alone would look
great in a slide — but `yearly_change_pct` is a rolling 4-quarter measure,
so it's strongly autocorrelated with itself one quarter later almost by
construction. A one-line naive baseline —

> "predict a decline this quarter if last quarter was already declining"

— scores **F1 0.82** on the *same* test set, matching or beating the
trained model. `src/train.py` computes this baseline on every run, logs it
to MLflow (`f1_vs_baseline`), and saves it into `model/metrics.json`
alongside the champion model's metrics, and prints an explicit warning when
the model doesn't clearly beat it (as it currently doesn't).

This isn't a bug to hide — with ~300 rows across 30 countries and a target
this autocorrelated, it's the expected, honest result. The useful next
steps are noted in [What's next](#whats-next), not a bigger grid search on
the same 300 rows.

## Architecture

```
            ┌───────────────┐
  git push  │ GitHub Actions│
  ───────▶  │  CI/CD        │
            └───────┬───────┘
                    │ 1. train.py  → MLflow tracking + registry + baseline check
                    │ 2. pytest    → data + performance gate
                    │ 3. drift     → Evidently report (informational)
                    │ 4. docker build
                    ▼
            ┌───────────────┐        ┌──────────────┐
            │ Docker image  │ ─────▶ │  Kubernetes  │
            └───────────────┘        │  Deployment  │
                                      │  + Service   │
                                      └──────┬───────┘
                                             │
                                      FastAPI /predict
```

## Repo layout

```
data/european_housing_prices_clean.csv  # real Eurostat-style HPI data
src/data.py               # loads real data, builds leakage-safe lag features
src/train.py              # SMOTE + GridSearchCV, MLflow tracking + registry + baseline
tests/test_pipeline.py    # data validation + model performance gate
api/main.py                # FastAPI serving layer
frontend/                  # Next.js 14 UI — calls the API via a server-side proxy route
monitoring/check_drift.py  # Evidently drift report (reference vs. current data)
Dockerfile
.github/workflows/ci.yml   # train → test → gate → build, on every push
k8s/deployment.yaml        # Deployment with readiness/liveness probes
k8s/service.yaml
```

## The data and the task

`data/european_housing_prices_clean.csv` is quarterly House Price Index
data for 30 individual European countries (5 EU/Euro-area aggregate rows
are excluded — they're weighted combinations of the individual countries,
not independent observations).

**Target**: `yearly_change_pct < 0` this quarter (a YoY price decline).

**Features** — deliberately built only from what was knowable at the end
of the *prior* quarter, so nothing about the current quarter's own price
figures leaks into its own prediction:

| Feature | What it is |
|---|---|
| `lag1_quarterly_change_pct` | last quarter's QoQ price change |
| `lag1_yearly_change_pct` | last quarter's YoY price change |
| `lag2_quarterly_change_pct` | QoQ change two quarters ago |
| `momentum_change` | `lag1_qoq - lag2_qoq` — is momentum accelerating or fading? |
| `eu_member` / `eurozone_member` | static country attributes (1/0) |
| `quarter_num` | 1-4, captures seasonality |

Raw `price_index` is intentionally **not** used as a feature — each
country's index is rebased to its own reference year, so raw levels
aren't comparable across countries (Türkiye's index, inflated by real
hyperinflation, would otherwise dominate any model that used it directly).

**Known limitation**: the train/test split is a stratified *random* split,
not a walk-forward time split. With only ~300 rows spread across 30
countries × ~11 usable quarters each, a strict time-based split would
leave too little data per fold to train or evaluate reliably. A production
version should move to walk-forward validation (train on quarters 1..k,
test on k+1) once more history accumulates — this is flagged here rather
than hidden.

## Running it locally

```bash
pip install -r requirements.txt

# 1. Train — logs to MLflow (sqlite backend), registers the best model,
#    and prints the baseline comparison
python src/train.py

# 2. Inspect experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → open http://localhost:5000

# 3. Run the test gate
pytest tests/ -v

# 4. Check for drift
python monitoring/check_drift.py
# → open monitoring/drift_report.html

# 5. Serve the model
uvicorn api.main:app --reload --port 8000
# → open http://localhost:8000/docs for an interactive Swagger UI
```

## Frontend (Next.js 14)

A small UI in `frontend/` — a form for the model's 7 features plus a
result panel with a momentum sparkline (the two real QoQ inputs you
entered, not decoration). It talks to the API through a Next.js API
route (`frontend/app/api/predict/route.ts`) that proxies server-side to
FastAPI, so the browser never needs CORS configured — the same pattern
you'd use in front of any internal API in production.

```bash
# Terminal 1 — backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
# → open http://localhost:3000
```

Override which API the frontend calls by copying
`frontend/.env.local.example` to `frontend/.env.local` and setting `API_URL`
(e.g. to a Kubernetes service's cluster URL once deployed).

## Running on Kubernetes (local minikube)

```bash
docker build -t housing-price-correction-risk:latest .
minikube image load housing-price-correction-risk:latest
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl port-forward svc/housing-price-correction-risk-svc 8000:80
```

(Update the image name in `k8s/deployment.yaml` if you rename the built image.)

## What's next

- Move to walk-forward (time-based) validation once more quarters of data
  are available — the current random split is a documented compromise for
  small-N, not the end state
- Try a harder, less autocorrelated target: predicting a **regime change**
  (a country flips from growth into decline) rather than "is this quarter
  declining" — this removes most of the persistence signal and would be a
  more meaningful test of whether the model adds real value
- Swap the local MLflow/sqlite backend for a hosted tracking server
- Make the drift check block deployment (not just informational) once a
  real quarterly data feed exists
