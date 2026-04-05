"""
Database helpers for loading DataFrames into PostgreSQL.
"""

import os

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://mlops:mlops@localhost:5432/mlops",
)


def get_engine():
    return create_engine(DB_URL)


def load_to_db(df: pd.DataFrame, table: str, if_exists: str = "replace"):
    """Write a DataFrame to a PostgreSQL table.

    Args:
        df: DataFrame to load.
        table: Target table name.
        if_exists: 'replace' drops and recreates, 'append' adds rows.
    """
    engine = get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False, method="multi", chunksize=1000)
    row_count = pd.read_sql(text(f"SELECT count(*) FROM {table}"), engine).iloc[0, 0]
    print(f"  {table}: {row_count} rows loaded")


if __name__ == "__main__":
    from pathlib import Path

    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

    print("Loading parquet files to PostgreSQL...")

    sales = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    load_to_db(sales, "sales")

    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    load_to_db(items, "items")

    customers = pd.read_parquet(PROCESSED_DIR / "customers.parquet")
    load_to_db(customers, "customers")

    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    load_to_db(features, "features")

    print("\nDone.")
