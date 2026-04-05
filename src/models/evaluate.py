"""
Model comparison and promotion logic.

Compares all runs in the demand-forecasting experiment and promotes
the best model to the MLflow Model Registry with a 'Staging' alias.
"""

import os

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.2.202")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "demand-forecast"
REGISTERED_MODEL_NAME = "demand-forecaster"


def compare_runs():
    """Print a comparison table of all runs in the experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print(f"No experiment named '{EXPERIMENT_NAME}' found.")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
    )

    print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("-" * 53)

    for run in runs:
        name = run.info.run_name or run.info.run_id[:8]
        mae = run.data.metrics.get("mae", float("inf"))
        rmse = run.data.metrics.get("rmse", float("inf"))
        mape = run.data.metrics.get("mape", float("inf"))
        print(f"{name:<25} {mae:>8.2f} {rmse:>8.2f} {mape:>7.1f}%")

    return runs


def promote_best(metric: str = "mae"):
    """Register the best model and assign it the 'Staging' alias."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("No experiment found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )

    if not runs:
        print("No runs found.")
        return

    best_run = runs[0]
    best_name = best_run.info.run_name or best_run.info.run_id[:8]
    best_metric = best_run.data.metrics.get(metric, float("inf"))
    model_uri = f"runs:/{best_run.info.run_id}/model"

    print(f"\nBest model: {best_name} ({metric}={best_metric:.4f})")
    print(f"Registering to '{REGISTERED_MODEL_NAME}'...")

    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "staging", mv.version)
    print(f"  Version {mv.version} registered with alias 'staging'")


if __name__ == "__main__":
    compare_runs()
    promote_best()
