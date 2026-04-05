"""
Data validation checks for the demand forecasting pipeline.

Runs assertions against the sales, items, and features DataFrames
to catch data quality issues early. Each check prints PASS/FAIL
and the script exits non-zero if any check fails.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = ""):
    """Record a check result."""
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def validate_sales(sales: pd.DataFrame):
    print("Sales checks:")

    check("no null dates", sales["date"].notna().all())

    check("no null SKUs", (sales["sku"] != "").all() and sales["sku"].notna().all())

    check(
        "dates in valid range",
        sales["date"].min() >= pd.Timestamp("2005-01-01")
        and sales["date"].max() <= pd.Timestamp("2027-01-01"),
        f"{sales['date'].min().date()} to {sales['date'].max().date()}",
    )

    neg_count = (sales["qty"] < 0).sum()
    neg_pct = neg_count / len(sales) * 100
    check(
        "negative qty under 5%",
        neg_pct < 5,
        f"{neg_count} rows ({neg_pct:.1f}%)",
    )

    zero_count = (sales["qty"] == 0).sum()
    zero_pct = zero_count / len(sales) * 100
    check(
        "zero qty under 15%",
        zero_pct < 15,
        f"{zero_count} rows ({zero_pct:.1f}%)",
    )

    # Fully identical rows can occur when the same SKU appears as multiple
    # line items on one SO (e.g., different install locations). Warn if high.
    dupe_count = sales.duplicated(keep="first").sum()
    dupe_pct = dupe_count / len(sales) * 100
    check(
        "identical rows under 3%",
        dupe_pct < 3,
        f"{dupe_count} rows ({dupe_pct:.1f}%)",
    )


def validate_items(items: pd.DataFrame):
    print("\nItems checks:")

    check("no null item names", (items["item"] != "").all() and items["item"].notna().all())

    check(
        "no negative prices",
        (items["price"] >= 0).all(),
        f"min price: {items['price'].min()}",
    )

    check("has inventory parts", (items["type"] == "Inventory Part").any())


def validate_features(features: pd.DataFrame):
    print("\nFeatures checks:")

    check("no NaN values", features.notna().all().all())

    # Lag/rolling values can be negative due to returns (negative qty weeks).
    # Just check they aren't wildly out of range.
    lag_cols = [c for c in features.columns if "lag" in c]
    lag_min = features[lag_cols].min().min()
    lag_max = features[lag_cols].max().max()
    check(
        "lag values in reasonable range",
        lag_min > -10000 and lag_max < 100000,
        f"range [{lag_min}, {lag_max}]",
    )

    rolling_cols = [c for c in features.columns if "rolling_mean" in c]
    rolling_min = features[rolling_cols].min().min()
    check(
        "rolling means not deeply negative",
        rolling_min > -1000,
        f"min rolling mean: {rolling_min:.2f}",
    )

    # Every SKU should have a continuous weekly series (no gaps)
    sku_week_counts = features.groupby("sku")["date"].count()
    expected = sku_week_counts.iloc[0]
    check(
        "all SKUs have same number of weeks",
        (sku_week_counts == expected).all(),
        f"expected {expected} weeks each, got range [{sku_week_counts.min()}, {sku_week_counts.max()}]",
    )

    check(
        "sku_active_rate in [0, 1]",
        features["sku_active_rate"].between(0, 1).all(),
    )

    min_skus = 100
    n_skus = features["sku"].nunique()
    check(
        f"at least {min_skus} SKUs after filtering",
        n_skus >= min_skus,
        f"{n_skus} SKUs",
    )


if __name__ == "__main__":
    sales = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    validate_sales(sales)
    validate_items(items)
    validate_features(features)

    failures = [r for r in results if not r[1]]
    print(f"\n{'=' * 40}")
    print(f"{len(results)} checks, {len(results) - len(failures)} passed, {len(failures)} failed")

    if failures:
        print("\nFailures:")
        for name, _, detail in failures:
            print(f"  - {name}: {detail}")
        exit(1)
    else:
        print("All checks passed.")
