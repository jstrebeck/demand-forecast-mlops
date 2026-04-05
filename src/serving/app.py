"""
FastAPI serving app for demand forecasting.

Pulls the best model from MLflow registry at startup and queries
PostgreSQL for feature vectors at inference time.
"""

import os
import pickle
import tempfile
from collections import deque
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from src.models.baseline import FEATURE_COLS

# ── Config ──────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.2.202")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlops:mlops@localhost:5432/mlops")
REGISTERED_MODEL_NAME = os.environ.get("MODEL_NAME", "demand-forecaster")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "staging")

# ── Global state ────────────────────────────────────────────
model = None
engine = None


def load_model():
    """Download the model artifact from MLflow and load it."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get the model version by alias
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    run_id = mv.run_id
    version = mv.version
    print(f"Loading model: {REGISTERED_MODEL_NAME} v{version} (alias={MODEL_ALIAS}, run={run_id})")

    # Download the artifact directory
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=tmpdir,
        )
        # Find the .pkl file (XGBoost/Linear) or .pt file (LSTM)
        import glob
        pkl_files = glob.glob(f"{artifact_path}/*.pkl")
        if pkl_files:
            with open(pkl_files[0], "rb") as f:
                loaded = pickle.load(f)
            print(f"  Loaded pickle model from {os.path.basename(pkl_files[0])}")
            return loaded

        pt_files = glob.glob(f"{artifact_path}/*.pt")
        if pt_files:
            raise RuntimeError(
                "LSTM .pt serving not yet implemented — promote an XGBoost or Linear model"
            )

    raise RuntimeError(f"No model artifact found for run {run_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and DB connection on startup."""
    global model, engine
    model = load_model()
    engine = create_engine(DATABASE_URL)
    # Verify DB connection
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM features")).scalar()
        print(f"  Connected to PostgreSQL — features table has {count} rows")
    yield
    engine.dispose()


app = FastAPI(
    title="Demand Forecaster",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ──────────────────────────────
class PredictRequest(BaseModel):
    sku: str


class PredictResponse(BaseModel):
    sku: str
    predicted_qty: float
    as_of_date: str
    model_name: str
    model_alias: str


class BatchPredictRequest(BaseModel):
    skus: list[str]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class ForecastRequest(BaseModel):
    sku: str
    weeks: int = 12


class WeekForecast(BaseModel):
    date: str
    predicted_qty: float


class ForecastResponse(BaseModel):
    sku: str
    weeks: int
    forecast: list[WeekForecast]
    model_name: str
    model_alias: str


# ── Recursive forecasting helpers ───────────────────────────
def build_feature_vector(
    qty_history: deque,
    target_date: pd.Timestamp,
    sku_stats: dict,
    last_order_count: float,
) -> pd.DataFrame:
    """Build a single feature vector from rolling qty history and a target date.

    qty_history should contain at least 12 recent qty values, most recent last.
    This mirrors the logic in features.py:
      - lags use shift(N) → history[-N]
      - rolling stats use shift(1).rolling(W) → window over history[-W-1:-1]
    """
    hist = list(qty_history)

    # Lags: lag_1 = most recent value, lag_2 = one before that, etc.
    lag_1 = hist[-1] if len(hist) >= 1 else 0.0
    lag_2 = hist[-2] if len(hist) >= 2 else 0.0
    lag_3 = hist[-3] if len(hist) >= 3 else 0.0
    lag_4 = hist[-4] if len(hist) >= 4 else 0.0

    # Rolling stats: shift(1) means exclude the most recent, then take window
    # So for a row at date T: rolling_mean_4w = mean(qty at T-1, T-2, T-3, T-4)
    # But here we're building features for the *next* week, so the "current" qty
    # is hist[-1], meaning:
    #   shift(1) window = hist[-1] back through hist[-W]
    window_4 = hist[-4:] if len(hist) >= 4 else hist
    window_12 = hist[-12:] if len(hist) >= 12 else hist

    rolling_mean_4w = np.mean(window_4) if window_4 else 0.0
    rolling_std_4w = np.std(window_4, ddof=1) if len(window_4) > 1 else 0.0
    rolling_mean_12w = np.mean(window_12) if window_12 else 0.0
    rolling_std_12w = np.std(window_12, ddof=1) if len(window_12) > 1 else 0.0

    # Calendar features from target date
    week_of_year = target_date.isocalendar().week
    month = target_date.month
    quarter = target_date.quarter
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    week_sin = np.sin(2 * np.pi * week_of_year / 52)
    week_cos = np.cos(2 * np.pi * week_of_year / 52)

    row = {
        "qty_lag_1": lag_1,
        "qty_lag_2": lag_2,
        "qty_lag_3": lag_3,
        "qty_lag_4": lag_4,
        "qty_rolling_mean_4w": rolling_mean_4w,
        "qty_rolling_std_4w": rolling_std_4w,
        "qty_rolling_mean_12w": rolling_mean_12w,
        "qty_rolling_std_12w": rolling_std_12w,
        "week_of_year": week_of_year,
        "month": month,
        "quarter": quarter,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
        "sku_lifetime_mean": sku_stats["sku_lifetime_mean"],
        "sku_lifetime_std": sku_stats["sku_lifetime_std"],
        "sku_lifetime_max": sku_stats["sku_lifetime_max"],
        "sku_active_rate": sku_stats["sku_active_rate"],
        "order_count": last_order_count,
    }
    return pd.DataFrame([{col: row[col] for col in FEATURE_COLS}])


# ── Endpoints ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": REGISTERED_MODEL_NAME, "alias": MODEL_ALIAS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict next week's demand for a single SKU."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT * FROM features
                WHERE sku = :sku
                ORDER BY date DESC
                LIMIT 1
            """),
            {"sku": req.sku},
        ).mappings().fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"No features found for SKU '{req.sku}'")

    features = pd.DataFrame([{col: row[col] for col in FEATURE_COLS}])
    prediction = float(model.predict(features.values)[0])

    return PredictResponse(
        sku=req.sku,
        predicted_qty=round(prediction, 2),
        as_of_date=str(row["date"]),
        model_name=REGISTERED_MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    """Predict demand for multiple SKUs at once."""
    if not req.skus:
        raise HTTPException(status_code=400, detail="skus list cannot be empty")

    results = []
    with engine.connect() as conn:
        for sku in req.skus:
            row = conn.execute(
                text("""
                    SELECT * FROM features
                    WHERE sku = :sku
                    ORDER BY date DESC
                    LIMIT 1
                """),
                {"sku": sku},
            ).mappings().fetchone()

            if row is None:
                raise HTTPException(status_code=404, detail=f"No features found for SKU '{sku}'")

            features = pd.DataFrame([{col: row[col] for col in FEATURE_COLS}])
            prediction = float(model.predict(features.values)[0])

            results.append(PredictResponse(
                sku=sku,
                predicted_qty=round(prediction, 2),
                as_of_date=str(row["date"]),
                model_name=REGISTERED_MODEL_NAME,
                model_alias=MODEL_ALIAS,
            ))

    return BatchPredictResponse(predictions=results)


@app.post("/predict/forecast", response_model=ForecastResponse)
def predict_forecast(req: ForecastRequest):
    """Recursively forecast demand for a SKU over multiple weeks.

    Fetches the last 12 weeks of actuals from the DB, then iteratively
    predicts each future week — feeding each prediction back in as the
    new lag/rolling values for the next step.
    """
    if req.weeks < 1 or req.weeks > 52:
        raise HTTPException(status_code=400, detail="weeks must be between 1 and 52")

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT date, qty, order_count,
                       sku_lifetime_mean, sku_lifetime_std,
                       sku_lifetime_max, sku_active_rate
                FROM features
                WHERE sku = :sku
                ORDER BY date DESC
                LIMIT 12
            """),
            {"sku": req.sku},
        ).mappings().fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No features found for SKU '{req.sku}'")

    # Rows are newest-first from the query; reverse to chronological order
    rows = list(reversed(rows))

    # Seed the qty history buffer (chronological, most recent last)
    qty_history = deque([float(r["qty"]) for r in rows], maxlen=12)

    # Static SKU profile stats (same for all weeks)
    latest = rows[-1]
    sku_stats = {
        "sku_lifetime_mean": float(latest["sku_lifetime_mean"]),
        "sku_lifetime_std": float(latest["sku_lifetime_std"]),
        "sku_lifetime_max": float(latest["sku_lifetime_max"]),
        "sku_active_rate": float(latest["sku_active_rate"]),
    }
    last_order_count = float(latest["order_count"])

    # Step forward from the most recent date
    current_date = pd.Timestamp(latest["date"])

    forecast = []
    for _ in range(req.weeks):
        target_date = current_date + pd.Timedelta(weeks=1)

        features = build_feature_vector(
            qty_history, target_date, sku_stats, last_order_count,
        )
        pred = max(float(model.predict(features.values)[0]), 0.0)

        forecast.append(WeekForecast(
            date=str(target_date.date()),
            predicted_qty=round(pred, 2),
        ))

        # Roll forward: prediction becomes the newest history value
        qty_history.append(pred)
        current_date = target_date

    return ForecastResponse(
        sku=req.sku,
        weeks=req.weeks,
        forecast=forecast,
        model_name=REGISTERED_MODEL_NAME,
        model_alias=MODEL_ALIAS,
    )


@app.get("/skus")
def list_skus():
    """List all SKUs that have features available."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT sku FROM features ORDER BY sku")
        ).fetchall()
    return {"skus": [r[0] for r in rows], "count": len(rows)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=port, reload=True)
