# CLAUDE.md

## Project Overview
MLOps learning/portfolio project: demand forecasting for product orders using real QuickBooks data. Predicts weekly order volume per SKU. Built for MLA-C01 exam prep and interview portfolio — not production.

## Key Decisions
- **Keep it simple.** No superpowers skills, no formal planning/TDD/code-review workflows. This is for learning.
- **Parquet + PostgreSQL dual storage.** Parquet files for fast local dev, Postgres for learning the database-backed feature store pattern.
- **GPU training.** RTX 4070 available via WSL2 — PyTorch installed with CUDA (cu126). Always use GPU when training.
- **Top SKUs have sparse demand.** Even the most active SKU only has orders ~50% of weeks. Zero-fill is critical for the time series.
- **QuickBooks CSV encoding.** Exports use `cp1252` encoding, not UTF-8. All CSV readers must specify `encoding="cp1252"`.
- **QuickBooks report format.** Sales.CSV is a grouped report with section headers, item group headers, and "Total" summary rows mixed in. The first column being blank identifies actual transaction rows. Extra columns (Memo, U/M, etc.) are ignored automatically.
- **Multiple CSV exports supported.** Ingest accepts numbered files (Sales.CSV, Sales2.CSV, Sales3.CSV, etc.) for all three report types. Duplicates are removed automatically. This works around QuickBooks CSV export dropping rows on large reports.
- **MAPE is inflated.** Most SKU-weeks have zero demand, so MAPE is misleadingly high across all models. MAE is the primary comparison metric.
- **MLflow version pinning.** Server runs MLflow 2.19 (in `Dockerfiles/mlflow.Dockerfile`). Client uses `mlflow-skinny` 2.22. Do NOT upgrade to MLflow 3.x — the `logged-model` API breaks with OSS server. Full `mlflow` package won't install on Python 3.14 due to pyarrow<20 constraint.
- **Artifact logging uses pickle/torch.save.** Since `mlflow-skinny` doesn't include flavor-specific `log_model` (sklearn, pytorch, xgboost), we save models as `.pkl`/`.pt` files and use `mlflow.log_artifact()`. Artifact upload goes through the server proxy (`--serve-artifacts`).
- **Private container registry** at `192.168.2.203:5000`. All custom images are pushed here.

## Data
- Raw QuickBooks exports go in `data/raw/` (Sales.CSV, Sales2.CSV, ..., Items.CSV, Customers.CSV)
- Sales.CSV columns: Type, Date, Num (invoice number), Name (customer), Item, Qty, Sales Price, Amount, Balance
- Processed parquet files written to `data/processed/` (gitignored)
- Date range extends back to 2009 with multiple exports

## Architecture
```
data/raw/*.CSV → ingest.py → data/processed/*.parquet → features.py → features.parquet
                                                                    → db.py → PostgreSQL (mlops-db)
features.parquet → train.py (baseline + LSTM) → MLflow (http://192.168.2.202)
                → evaluate.py → MLflow Model Registry
                                    ↓
                          app.py (FastAPI) ← PostgreSQL (features)
                          /predict, /predict/batch, /predict/forecast
```

## Infrastructure

### Local (docker-compose)
- PostgreSQL 16, container `mlops-db`
- Connection: `postgresql://mlops:mlops@localhost:5432/mlops`
- Schema defined in `db/init.sql`
- `docker-compose down -v` required when changing schema (init.sql only runs on first volume creation)

### Kubernetes (Talos cluster)
- **Namespace:** `mlops`
- **MLflow Tracking Server:** `http://192.168.2.202` (LoadBalancer via MetalLB)
  - Image: `192.168.2.203:5000/mlflow:latest` (custom build with psycopg2)
  - Backend: PostgreSQL in-cluster (`mlflow-postgres` service)
  - Artifacts: Ceph PVC (`mlflow-artifacts-pvc`, 10Gi)
  - Flags: `--serve-artifacts`, `--artifacts-destination=/mlflow/artifacts`, `--default-artifact-root=mlflow-artifacts:/`
- **MLflow PostgreSQL:** ClusterIP service, Ceph PVC (`mlflow-postgres-pvc`, 5Gi)
- **Storage:** `ceph-block` StorageClass (default), Rook-Ceph
- **Registry:** `192.168.2.203:5000` (in-cluster, `registry` namespace)
- **Manifests are NOT in this repo.** They live in the Homelab-Configuration repo (`/home/vba2/Documents/git/Homelab-Configuration/Kubernetes/mlflow/`: namespace.yaml, postgres.yaml, mlflow.yaml) so they can be shared across projects. Edit and commit them there.
- Makefile reads them via `HOMELAB_K8S` (default `../Homelab-Configuration/Kubernetes`); override with `HOMELAB_K8S=/path make deploy-k8s`

## Makefile Commands

### Environment
- `make venv` — create virtual environment and install deps
- `make db` / `make db-down` — start/stop local PostgreSQL

### Data Pipeline
- `make pipeline` — run full pipeline (ingest → features → validate → load-db)
- `make ingest` / `make features` / `make validate` / `make load-db` — individual steps

### Model Training
- `make train` — train all models (linear, XGBoost, LSTM)
- `make evaluate` — compare runs and promote best to MLflow registry

### Model Serving
- `make serve` — start FastAPI serving app (pulls staging model from MLflow, queries PostgreSQL)
- `PORT=8001 make serve` — use alternate port
- `MODEL_ALIAS=production make serve` — serve a different model alias
- `make forecast ARGS="SKU --weeks N"` — plot demand forecast in terminal (requires serving app running)

### Docker
- `make build-mlflow` / `make push-mlflow` — MLflow server image
- `make build-train` / `make push-train` — training image
- `make build-serve` / `make push-serve` — serving image
- `make build-pipeline` / `make push-pipeline` — pipeline step image
- `make build-all` / `make push-all` — everything

### Kubernetes
- `make deploy-mlflow` — full cycle: build, push, deploy, restart MLflow
- `make deploy-k8s` — apply all k8s manifests (from Homelab-Configuration repo)
- `make teardown-k8s` — tear down everything
- `make status` — check pods and services in mlops namespace

## Project Phases (from project-outline.md)
- Phase 1: Data Pipeline & Feature Engineering ✅
- Phase 2: Model Development with PyTorch ✅
- Phase 3: MLflow Deployment on Kubernetes ✅
- Phase 4: Kubeflow Pipelines for Automated Retraining
- Phase 5: Model Serving with KServe — **FastAPI serving app complete** (local, not yet on k8s/KServe)
- Phase 6: Monitoring & Drift Detection

## Features (in features.parquet)
- **Target:** qty, amount, order_count (weekly per SKU)
- **Lags:** qty_lag_1 through qty_lag_4
- **Rolling:** qty_rolling_mean_4w, _std_4w, _mean_12w, _std_12w (shift(1) to avoid leakage)
- **Calendar:** week_of_year, month, quarter, cyclical sin/cos encodings
- **SKU profile:** sku_lifetime_mean, _std, _max, sku_active_rate
- SKUs with <10 total orders are filtered out

## Model Results (Phase 2)
| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| XGBoost | 0.59 | 10.42 | 380.7% |
| Linear Regression | 1.04 | 5.32 | 325.2% |
| LSTM | 1.21 | 5.90 | 110.0% |

XGBoost promoted to MLflow registry as `demand-forecaster` v1 (alias: staging).
Experiment name: `demand-forecast` (on cluster MLflow at http://192.168.2.202).
