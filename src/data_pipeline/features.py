"""
Feature engineering for demand forecasting.

Takes the cleaned sales data from ingest.py and produces weekly,
SKU-level feature sets ready for model training. The target variable
is weekly order quantity per SKU.
"""

from pathlib import Path

import pandas as pd
import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_weekly_demand(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily transactions into weekly demand per SKU.

    - Filters out zero-qty rows (unfulfilled SO placeholders)
    - Nets negative qty rows (returns/credits) into the totals
    - Fills missing weeks with zero so every SKU has a continuous time series
    """
    df = sales[sales["qty"] != 0].copy()

    weekly = (
        df.groupby([pd.Grouper(key="date", freq="W-MON"), "sku"])
        .agg(qty=("qty", "sum"), amount=("amount", "sum"), order_count=("qty", "size"))
        .reset_index()
    )

    # Build a full week x SKU grid so gaps become explicit zeros
    all_weeks = pd.date_range(
        weekly["date"].min(), weekly["date"].max(), freq="W-MON", name="date"
    )
    all_skus = weekly["sku"].unique()
    full_index = pd.MultiIndex.from_product([all_weeks, all_skus], names=["date", "sku"])

    weekly = weekly.set_index(["date", "sku"]).reindex(full_index, fill_value=0).reset_index()

    return weekly


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged qty values as autoregressive inputs.

    Each lag is the qty from N weeks prior for the same SKU.
    """
    if lags is None:
        lags = [1, 2, 3, 4]

    df = df.sort_values(["sku", "date"])
    for lag in lags:
        df[f"qty_lag_{lag}"] = df.groupby("sku")["qty"].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Add rolling mean and std of qty over past N weeks per SKU."""
    if windows is None:
        windows = [4, 12]  # ~1 month and ~3 months

    df = df.sort_values(["sku", "date"])
    for w in windows:
        grp = df.groupby("sku")["qty"]
        df[f"qty_rolling_mean_{w}w"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"qty_rolling_std_{w}w"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonality encodings from the date."""
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Cyclical encodings so week 52 is close to week 1
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


def add_sku_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add static SKU-level aggregations as features.

    These capture the overall demand profile of each SKU (high-volume vs
    low-volume, steady vs bursty).
    """
    stats = df.groupby("sku")["qty"].agg(
        sku_lifetime_mean="mean",
        sku_lifetime_std="std",
        sku_lifetime_max="max",
    ).fillna(0)

    # Fraction of weeks with any orders
    active = df[df["qty"] > 0].groupby("sku").size() / df.groupby("sku").size()
    stats["sku_active_rate"] = active.fillna(0)

    df = df.merge(stats, on="sku", how="left")
    return df


def build_features(
    sales: pd.DataFrame,
    min_total_orders: int = 10,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Args:
        sales: Cleaned sales DataFrame from ingest.load_sales().
        min_total_orders: Drop SKUs with fewer total orders than this
            (too sparse to forecast).
    """
    # Filter to SKUs with enough history
    sku_totals = sales[sales["qty"] > 0].groupby("sku")["qty"].sum()
    valid_skus = sku_totals[sku_totals >= min_total_orders].index
    sales = sales[sales["sku"].isin(valid_skus)]

    df = build_weekly_demand(sales)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_sku_stats(df)

    # Drop rows where lag features are NaN (first few weeks per SKU)
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Loading sales data...")
    sales = pd.read_parquet(PROCESSED_DIR / "sales.parquet")

    print("Building features...")
    features = build_features(sales)

    print(f"  Shape: {features.shape}")
    print(f"  SKUs: {features['sku'].nunique()}")
    print(f"  Date range: {features['date'].min()} to {features['date'].max()}")
    print(f"  Columns: {features.columns.tolist()}")

    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print(f"\nSaved to {PROCESSED_DIR / 'features.parquet'}")
