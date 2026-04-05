VENV := .venv
PYTHON := python3
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python3

REGISTRY := 192.168.2.203:5000

.PHONY: venv activate deactivate \
	db db-down \
	ingest features validate load-db pipeline \
	train evaluate \
	build-mlflow push-mlflow deploy-mlflow \
	deploy-k8s teardown-k8s

# ── Environment ──────────────────────────────────────────
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install pandas pyarrow psycopg2-binary sqlalchemy
	@echo ""
	@echo "Venv created. Run: source $(VENV)/bin/activate"

activate:
	@echo "Run this in your shell:"
	@echo "  source $(VENV)/bin/activate"

deactivate:
	@echo "Run this in your shell:"
	@echo "  deactivate"

# ── Local Database ───────────────────────────────────────
db:
	docker-compose up -d

db-down:
	docker-compose down

# ── Data Pipeline ────────────────────────────────────────
ingest:
	$(PY) src/data_pipeline/ingest.py

features:
	$(PY) src/data_pipeline/features.py

validate:
	$(PY) src/data_pipeline/validate.py

load-db:
	$(PY) src/data_pipeline/db.py

pipeline: ingest features validate load-db

# ── Model Serving ────────────────────────────────────────
serve:
	$(PY) -m src.serving.app

forecast:
	$(PY) -m src.serving.forecast_cli $(ARGS)

# ── Model Training ───────────────────────────────────────
train:
	$(PY) -m src.models.train

evaluate:
	$(PY) -m src.models.evaluate

# ── Docker Builds ────────────────────────────────────────
build-mlflow:
	docker build -f Dockerfiles/mlflow.Dockerfile -t $(REGISTRY)/mlflow:latest .

push-mlflow: build-mlflow
	docker push $(REGISTRY)/mlflow:latest

build-train:
	docker build -f Dockerfiles/train.Dockerfile -t $(REGISTRY)/train:latest .

push-train: build-train
	docker push $(REGISTRY)/train:latest

build-serve:
	docker build -f Dockerfiles/serve.Dockerfile -t $(REGISTRY)/serve:latest .

push-serve: build-serve
	docker push $(REGISTRY)/serve:latest

build-pipeline:
	docker build -f Dockerfiles/pipeline.Dockerfile -t $(REGISTRY)/pipeline:latest .

push-pipeline: build-pipeline
	docker push $(REGISTRY)/pipeline:latest

build-all: build-mlflow build-train build-serve build-pipeline
push-all: push-mlflow push-train push-serve push-pipeline

# ── Kubernetes Deployments ───────────────────────────────
deploy-mlflow: push-mlflow
	kubectl apply -f k8s/mlflow/namespace.yaml
	kubectl apply -f k8s/mlflow/postgres.yaml
	kubectl apply -f k8s/mlflow/mlflow.yaml
	kubectl rollout restart deployment/mlflow -n mlops
	kubectl rollout status deployment/mlflow -n mlops --timeout=90s

deploy-k8s:
	kubectl apply -f k8s/mlflow/namespace.yaml
	kubectl apply -f k8s/mlflow/postgres.yaml
	kubectl apply -f k8s/mlflow/mlflow.yaml

teardown-k8s:
	kubectl delete -f k8s/mlflow/mlflow.yaml --ignore-not-found
	kubectl delete -f k8s/mlflow/postgres.yaml --ignore-not-found
	kubectl delete -f k8s/mlflow/namespace.yaml --ignore-not-found

status:
	@echo "=== Pods ==="
	@kubectl get pods -n mlops
	@echo ""
	@echo "=== Services ==="
	@kubectl get svc -n mlops
