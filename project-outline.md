PROJECT OUTLINE
Product Demand Forecasting
End-to-End MLOps with PyTorch, MLflow & Kubeflow
Target Infrastructure: 4-Node Talos Kubernetes Cluster + RTX PRO 6000 Inference Server
Timeline: 6 Weeks | Deadline: Before September 10, 2026

1. Project Overview
Build a demand forecasting model for product orders using real business data. The model predicts weekly order volume per SKU, enabling proactive inventory management and supplier communication. This project serves as a portfolio-grade MLOps demonstration covering the full lifecycle from data ingestion to automated retraining.
Why This Project
This project is deliberately chosen to maximize both learning and career value. It uses real production data from an active B2B SaaS business rather than a toy dataset, which means dealing with real-world data quality issues, seasonal patterns, and business constraints. It touches every component of the MLOps stack that interviewers care about: data pipelines, experiment tracking, model registry, automated training, serving, and monitoring. It also maps directly to MLA-C01 exam domains, reinforcing cert study with hands-on practice.

2. Architecture Overview
LayerToolAWS EquivalentMLA-C01 DomainData StorageTrueNAS NFS + PostgreSQLS3 + RDSDomain 1Experiment TrackingMLflow Tracking ServerSageMaker ExperimentsDomain 2Model RegistryMLflow Model RegistrySageMaker Model RegistryDomain 2Pipeline OrchestrationKubeflow PipelinesSageMaker PipelinesDomain 3Model TrainingPyTorch (GPU node)SageMaker TrainingDomain 2Model ServingKServe + FastAPISageMaker EndpointsDomain 3MonitoringEvidently AI + GrafanaSageMaker Model MonitorDomain 4CI/CDGitHub Actions + Argo CDCodePipeline + CodeBuildDomain 4

3. Project Phases
Phase 1: Data Pipeline & Feature Engineering (Week 1)
Extract order history from QuickBooks exports and Google Places lead data. Build a reproducible data pipeline that transforms raw exports into ML-ready features.
Deliverables

Data ingestion script: Python script that reads QuickBooks CSV/IIF exports, cleans and normalizes order records (handle missing values, duplicate orders, date parsing).
Feature engineering module: Generate features including rolling averages of order volume (7-day, 30-day), day-of-week and month seasonality encodings, SKU-level aggregations, lead-to-order conversion rates from Google Places data, and lag features for autoregressive input.
PostgreSQL schema: Store processed features in a PostgreSQL database on the Talos cluster with versioned feature tables.
Data validation: Basic data quality checks using Great Expectations or custom assertions (null checks, range validation, schema conformance).

Key Learning
This phase maps to MLA-C01 Domain 1 (Data Preparation). Understand the parallels with AWS Glue for ETL, SageMaker Data Wrangler for feature engineering, and SageMaker Feature Store for versioned features.

Phase 2: Model Development with PyTorch (Week 2)
Build a time-series forecasting model using PyTorch. Start simple, iterate, and track everything with MLflow.
Model Architecture

Baseline: Simple linear regression and XGBoost for comparison benchmarks.
Primary model: LSTM-based sequence model that takes the engineered features as input and predicts the next 4 weeks of order volume per SKU.
Stretch goal: Transformer-based temporal fusion transformer (TFT) if LSTM performance plateaus.

Deliverables

PyTorch training script: Modular training code with configurable hyperparameters (learning rate, hidden size, number of layers, sequence length, dropout rate).
MLflow integration: Log every training run with parameters, metrics (MAE, RMSE, MAPE), loss curves, and model artifacts. Tag runs by model type and data version.
Experiment comparison: Use MLflow UI to compare baseline vs LSTM runs, visualize metric trends across hyperparameter sweeps.
Best model registered: Promote the best-performing model to the MLflow Model Registry with a 'Staging' tag.

Key Learning
This phase maps to MLA-C01 Domain 2 (Model Development). Understand the parallels with SageMaker Training Jobs, SageMaker Experiments for tracking, hyperparameter tuning with SageMaker Automatic Model Tuning, and the SageMaker Model Registry lifecycle (staging, production, archived).

Phase 3: MLflow Deployment on Kubernetes (Week 3)
Deploy MLflow as a persistent service on the Talos cluster so it becomes the central metadata store for all ML work.
Deliverables

MLflow Helm deployment: Deploy MLflow Tracking Server as a Kubernetes deployment with a PostgreSQL backend for metadata and TrueNAS NFS for artifact storage.
Persistent storage configuration: Configure Democratic-CSI PVCs for both the PostgreSQL database and the MLflow artifact store, ensuring experiments survive pod restarts.
Ingress and access: Set up an ingress route so the MLflow UI is accessible from your local network for experiment review.
Model serving endpoint: Use MLflow's built-in model serving to deploy the staged model as a REST endpoint for initial testing.

Key Learning
This phase reinforces container orchestration and persistent storage concepts. The SageMaker parallel is that SageMaker manages all of this infrastructure for you (Experiments, Model Registry, Endpoints are all managed services), which is the tradeoff between managed vs self-hosted MLOps you should be able to articulate in interviews.

Phase 4: Kubeflow Pipelines for Automated Retraining (Week 4)
Build an automated ML pipeline using Kubeflow Pipelines that handles the full retraining workflow. This is the core MLOps automation piece.
Pipeline Steps

Step 1 - Data Pull: Container that extracts the latest order data from PostgreSQL and validates data quality.
Step 2 - Feature Engineering: Container that runs the feature engineering module from Phase 1, outputs versioned feature sets.
Step 3 - Model Training: Container that runs PyTorch training on the GPU node, logs to MLflow, registers the new model.
Step 4 - Evaluation: Container that compares the new model against the current production model using a holdout test set. Calculates MAE, RMSE, and MAPE.
Step 5 - Conditional Promotion: If the new model improves MAPE by more than 5% over the production model, automatically promote it to 'Production' in the MLflow registry. Otherwise, archive the run.
Step 6 - Deployment: If promoted, trigger a rolling update of the model serving endpoint.

Deliverables

Kubeflow Pipeline definition: Python SDK pipeline compiled to an Argo workflow YAML.
Container images: One image per pipeline step, pushed to private registry at 192.168.2.203:5000.
Scheduled trigger: CronWorkflow that runs the pipeline weekly (simulating real-world retraining cadence).
Pipeline UI: Kubeflow dashboard showing run history, step logs, and artifact lineage.

Key Learning
This is the most exam-relevant phase. It maps directly to MLA-C01 Domain 3 (Deployment and Orchestration) and Domain 4 (Monitoring and Operations). Understand how SageMaker Pipelines accomplishes the same thing with Step decorators, how SageMaker Model Monitor triggers retraining, and how CodePipeline/CodeBuild handles the CI/CD integration.

Phase 5: Model Serving with KServe (Week 5)
Replace the basic MLflow serving endpoint with production-grade model serving using KServe on Kubernetes.
Deliverables

KServe InferenceService: Deploy the PyTorch model as a KServe InferenceService with autoscaling based on request volume.
Canary deployment: Configure KServe to route 10% of traffic to the new model version and 90% to the current production model. Gradually increase if metrics are healthy.
FastAPI wrapper: Build a lightweight FastAPI service that accepts domain-specific requests (SKU, date range) and translates them into model input tensors.
Load testing: Use Locust or k6 to simulate realistic request patterns and validate autoscaling behavior.

Key Learning
This maps to SageMaker Endpoints with production variants (canary/A-B testing), auto-scaling policies, and the multi-model endpoint pattern. Understanding traffic shifting and rollback strategies is a common exam topic.

Phase 6: Monitoring & Drift Detection (Week 6)
Implement production monitoring to detect when the model degrades and trigger retraining automatically.
Deliverables

Evidently AI integration: Generate data drift and prediction drift reports by comparing incoming request distributions against the training data baseline.
Grafana dashboards: Visualize model latency (p50, p95, p99), prediction distribution, feature drift scores, and error rates. Connect to your existing kube-prometheus-stack.
Alerting rules: Configure Prometheus alerts that fire when prediction drift exceeds a threshold or when error rate spikes, which trigger the Kubeflow retraining pipeline.
Feedback loop: Log predictions alongside actual outcomes (when orders arrive) to calculate ground-truth accuracy over time.

Key Learning
This maps to MLA-C01 Domain 4 (Monitoring and Maintenance). Understand SageMaker Model Monitor (data quality, model quality, bias drift, feature attribution drift monitors), CloudWatch integration for operational metrics, and how SageMaker Clarify detects bias drift over time.

4. Technology Stack
CategoryTechnologyPurposeLanguagePython 3.11+All ML code, pipeline stepsML FrameworkPyTorch 2.xLSTM/Transformer model trainingExperiment TrackingMLflow 2.xParams, metrics, artifacts, registryPipeline OrchestrationKubeflow Pipelines 2.xDAG-based ML workflow automationModel ServingKServe + FastAPIProduction inference endpointMonitoringEvidently AI + GrafanaDrift detection, dashboardsData StorePostgreSQL + NFSFeatures, artifacts, raw dataContainer RegistryPrivate (192.168.2.203:5000)Pipeline step imagesInfrastructureTalos Linux / Proxmox / K8s4-node cluster + GPU nodeCI/CDGitHub Actions + Argo CDPipeline triggers, GitOps deploys

5. Repository Structure
demand-forecast-mlops/
├── data/
│   ├── raw/                    # QuickBooks exports, Google Places data
│   └── processed/              # Cleaned, feature-engineered datasets
├── src/
│   ├── data_pipeline/
│   │   ├── ingest.py             # QuickBooks CSV/IIF parser
│   │   ├── features.py           # Feature engineering module
│   │   └── validate.py           # Data quality checks
│   ├── models/
│   │   ├── baseline.py           # Linear regression + XGBoost
│   │   ├── lstm.py               # LSTM model definition
│   │   ├── train.py              # Training loop with MLflow logging
│   │   └── evaluate.py           # Model comparison logic
│   ├── serving/
│   │   ├── app.py                # FastAPI wrapper
│   │   └── kserve_config.yaml    # InferenceService manifest
│   └── monitoring/
│       ├── drift_report.py       # Evidently AI drift detection
│       └── dashboards/           # Grafana JSON exports
├── pipelines/
│   ├── pipeline.py             # Kubeflow Pipeline definition
│   ├── components/             # Containerized pipeline step definitions
│   └── cron_trigger.yaml       # CronWorkflow for scheduled runs
├── Dockerfiles/
│   ├── train.Dockerfile        # Training step container
│   ├── serve.Dockerfile        # Serving container
│   └── pipeline.Dockerfile     # Base image for pipeline steps
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_pipeline.py
├── README.md
└── pyproject.toml

6. MLA-C01 Exam Domain Mapping
DomainExam TopicProject PhaseAWS Service to StudyDomain 1Data PreparationPhase 1: Data PipelineGlue, Data Wrangler, Feature StoreDomain 2Model DevelopmentPhase 2: PyTorch TrainingSageMaker Training, ExperimentsDomain 2Model RegistryPhase 3: MLflow RegistrySageMaker Model RegistryDomain 3Deployment & OrchestrationPhase 4: Kubeflow PipelinesSageMaker Pipelines, EndpointsDomain 3Inference InfrastructurePhase 5: KServe ServingSageMaker Endpoints, Auto ScalingDomain 4Monitoring & MaintenancePhase 6: Drift DetectionModel Monitor, CloudWatch, Clarify

7. Interview Talking Points
This project gives you concrete answers to the most common MLOps interview questions. Practice articulating these:

"Walk me through an ML pipeline you built." Describe the 6-step Kubeflow pipeline end-to-end, emphasizing the conditional promotion logic and automated retraining trigger.
"How do you handle model versioning?" Explain the MLflow Model Registry lifecycle: every training run is logged, best models are staged, promoted to production only after evaluation, and previous versions are archived with full lineage.
"How do you detect model degradation?" Describe the Evidently AI drift detection feeding Grafana dashboards with Prometheus alerts, and how drift alerts trigger the Kubeflow retraining pipeline automatically.
"How do you deploy model updates safely?" Explain the KServe canary deployment pattern: 10/90 traffic split, monitoring during the canary window, automatic rollback if error rates spike.
"Managed vs self-hosted MLOps: when would you choose each?" Use this project as the self-hosted reference point. Articulate the tradeoffs: SageMaker reduces operational burden but increases vendor lock-in and cost; self-hosted gives full control and portability but requires more infrastructure expertise. Your DevOps background makes you uniquely qualified to evaluate this tradeoff.


8. Timeline Summary
WeekPhaseKey OutputHours (Est.)Week 1Data Pipeline & FeaturesClean dataset in PostgreSQL8-10 hoursWeek 2PyTorch Model + MLflowTrained model in registry10-12 hoursWeek 3MLflow on KubernetesPersistent tracking server6-8 hoursWeek 4Kubeflow PipelineAutomated retraining DAG12-15 hoursWeek 5KServe Model ServingProduction endpoint + canary8-10 hoursWeek 6Monitoring & DriftGrafana dashboards + alerts8-10 hours
Total estimated effort: 52-65 hours over 6 weeks (roughly 9-11 hours per week).
