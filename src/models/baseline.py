"""
Baseline models for demand forecasting: Linear Regression and XGBoost.

These provide benchmark metrics to compare against the LSTM model.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


FEATURE_COLS = [
    "qty_lag_1",
    "qty_lag_2",
    "qty_lag_3",
    "qty_lag_4",
    "qty_rolling_mean_4w",
    "qty_rolling_std_4w",
    "qty_rolling_mean_12w",
    "qty_rolling_std_12w",
    "week_of_year",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "sku_lifetime_mean",
    "sku_lifetime_std",
    "sku_lifetime_max",
    "sku_active_rate",
    "order_count",
]

TARGET_COL = "qty"


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, ignoring zeros in y_true."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true, y_pred):
    """Return a dict of MAE, RMSE, and MAPE."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


def train_linear(X_train, y_train, X_test, y_test):
    """Train a linear regression baseline and return model + metrics."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    return model, preds, metrics


def train_xgboost(X_train, y_train, X_test, y_test, params=None):
    """Train an XGBoost regressor and return model + metrics."""
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    return model, preds, metrics
