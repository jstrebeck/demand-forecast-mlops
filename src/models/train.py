"""
Training script with MLflow logging.

Trains baseline models (Linear Regression, XGBoost) and the LSTM model,
logging all parameters, metrics, and artifacts to MLflow.
"""

import os
import pickle
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import torch

from src.models.baseline import FEATURE_COLS, TARGET_COL, train_linear, train_xgboost
from src.models.lstm import train_lstm

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# MLflow config — points to the cluster MLflow server
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.2.202")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "demand-forecast"


def log_model_artifact(model, name: str):
    """Save a model to a temp file and log it as an MLflow artifact."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if isinstance(model, torch.nn.Module):
                path = os.path.join(tmpdir, f"{name}.pt")
                torch.save(model.state_dict(), path)
            else:
                path = os.path.join(tmpdir, f"{name}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(model, f)
            mlflow.log_artifact(path, artifact_path="model")
    except Exception as e:
        print(f"  Warning: could not log model artifact: {e}")


def time_split(df: pd.DataFrame, test_weeks: int = 12):
    """Split features by time: last N weeks for test, rest for train."""
    cutoff = df["date"].max() - pd.Timedelta(weeks=test_weeks)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


def run_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train and log baseline models."""
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    # Linear Regression
    with mlflow.start_run(run_name="linear-regression"):
        mlflow.set_tag("model_type", "linear_regression")
        model, preds, metrics = train_linear(X_train, y_train, X_test, y_test)
        mlflow.log_metrics(metrics)
        log_model_artifact(model, "linear_regression")
        print(f"  Linear Regression — MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.1f}%")

    # XGBoost
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    with mlflow.start_run(run_name="xgboost"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.log_params(xgb_params)
        model, preds, metrics = train_xgboost(X_train, y_train, X_test, y_test, xgb_params)
        mlflow.log_metrics(metrics)
        log_model_artifact(model, "xgboost")
        print(f"  XGBoost — MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.1f}%")


def run_lstm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int = 8,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
):
    """Train and log the LSTM model."""
    params = {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    with mlflow.start_run(run_name="lstm"):
        mlflow.set_tag("model_type", "lstm")
        mlflow.log_params(params)

        print("  Training LSTM...")
        model, preds, metrics, loss_history = train_lstm(
            train_df, test_df, **params
        )

        mlflow.log_metrics(metrics)

        # Log loss curve
        for i, loss in enumerate(loss_history):
            mlflow.log_metric("train_loss", loss, step=i)

        log_model_artifact(model, "lstm")
        print(f"  LSTM — MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.1f}%")

    return metrics


if __name__ == "__main__":
    print("Loading features...")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nSplitting data (last 12 weeks for test)...")
    train_df, test_df = time_split(features, test_weeks=12)
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    print("\nTraining baselines...")
    run_baselines(train_df, test_df)

    print("\nTraining LSTM...")
    run_lstm(train_df, test_df)

    print(f"\nDone. View results at: {MLFLOW_TRACKING_URI}")
