"""
Ingest QuickBooks CSV exports into clean, flat DataFrames.

Sales.CSV is a grouped QuickBooks report — not a flat table. It contains:
  - A header row
  - Section type rows ("Inventory")
  - Item group headers (SKU name in first column, rest blank)
  - Transaction rows (first column blank, then Date/Name/etc.)
  - "Total ..." summary rows per item group

This module extracts only the actual transaction rows and parses them
into a clean DataFrame suitable for downstream feature engineering.
"""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _find_files(base_name: str, ext: str = ".CSV") -> list[Path]:
    """Find all files matching a base name with optional numeric suffix.

    E.g. base_name="Sales", ext=".CSV" matches:
      Sales.CSV, Sales2.CSV, Sales123.CSV
    """
    pattern = re.compile(
        rf"^{re.escape(base_name)}\d*{re.escape(ext)}$", re.IGNORECASE
    )
    files = sorted(f for f in RAW_DIR.iterdir() if pattern.match(f.name))
    if not files:
        raise FileNotFoundError(f"No {base_name}*{ext} files found in {RAW_DIR}")
    return files


def load_sales(path: Path | None = None) -> pd.DataFrame:
    """Parse the QuickBooks Sales by Item Detail report into a flat DataFrame.

    Filters out section headers, item group headers, and total rows.
    Keeps only actual transaction lines. Supports multiple files
    (Sales.CSV, Sales2.CSV, etc.) which are concatenated together.
    """
    if path:
        paths = [path]
    else:
        paths = _find_files("Sales")

    frames = []
    for p in paths:
        print(f"  Reading {p.name}...")
        frames.append(_parse_sales_file(p))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "invoice_number", "customer", "item", "qty", "amount"],
    )
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def _parse_sales_file(path: Path) -> pd.DataFrame:
    """Parse a single Sales CSV file."""
    raw = pd.read_csv(
        path,
        header=0,
        dtype=str,
        keep_default_na=False,
        encoding="cp1252",
    )

    # The first (unnamed) column holds group headers / "Total" rows.
    # Actual transaction rows have this column blank.
    first_col = raw.columns[0]
    transactions = raw[raw[first_col] == ""].copy()
    transactions = transactions.drop(columns=[first_col])

    # Map QuickBooks header names to our clean names.
    # Only keep the columns we need — ignore extras like Memo, U/M.
    col_map = {
        "Type": "type",
        "Date": "date",
        "Num": "invoice_number",
        "Name": "customer",
        "Item": "item",
        "Qty": "qty",
        "Sales Price": "unit_price",
        "Amount": "amount",
        "Balance": "balance",
    }
    # Keep only columns we recognize
    transactions = transactions[[c for c in transactions.columns if c in col_map]]
    transactions = transactions.rename(columns=col_map)

    # Parse types
    transactions["date"] = pd.to_datetime(transactions["date"], format="%m/%d/%Y")
    transactions["qty"] = pd.to_numeric(transactions["qty"], errors="coerce").fillna(0).astype(int)
    transactions["unit_price"] = pd.to_numeric(
        transactions["unit_price"].str.replace(",", ""), errors="coerce"
    ).fillna(0.0)
    transactions["amount"] = pd.to_numeric(
        transactions["amount"].str.replace(",", ""), errors="coerce"
    ).fillna(0.0)
    transactions["balance"] = pd.to_numeric(
        transactions["balance"].str.replace(",", ""), errors="coerce"
    ).fillna(0.0)

    # Extract the SKU code from the item field: "05-2419 (05-2419: description...)" -> "05-2419"
    transactions["sku"] = transactions["item"].apply(_extract_sku)

    # Strip whitespace from string columns
    for col in ["type", "customer", "item", "invoice_number", "sku"]:
        transactions[col] = transactions[col].str.strip()

    return transactions


def _extract_sku(item_str: str) -> str:
    """Extract the leading SKU code from a QuickBooks item string.

    Examples:
        "05-2419 (05-2419: description...)" -> "05-2419"
        "PRP2020-20 (PRP2020-20: desc...)"  -> "PRP2020-20"
    """
    item_str = item_str.strip()
    match = re.match(r"^([\w./-]+)", item_str)
    return match.group(1) if match else item_str


def load_items(path: Path | None = None) -> pd.DataFrame:
    """Load the QuickBooks Item Listing export.

    Supports multiple files (Items.CSV, Items2.CSV, etc.).
    """
    if path:
        paths = [path]
    else:
        paths = _find_files("Items")

    frames = []
    for p in paths:
        print(f"  Reading {p.name}...")
        raw = pd.read_csv(p, header=0, dtype=str, keep_default_na=False, encoding="cp1252")
        first_col = raw.columns[0]
        df = raw.drop(columns=[first_col])

        df.columns = [
            "item",
            "description",
            "type",
            "cost",
            "price",
            "tax_code",
            "qty_on_hand",
            "qty_on_so",
            "reorder_point",
            "qty_on_po",
            "uom",
            "preferred_vendor",
        ]
        frames.append(df)

    items = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["item"])

    for col in ["cost", "price"]:
        items[col] = pd.to_numeric(items[col].str.replace(",", ""), errors="coerce").fillna(0.0)

    for col in ["qty_on_hand", "qty_on_so", "reorder_point", "qty_on_po"]:
        items[col] = pd.to_numeric(items[col], errors="coerce").fillna(0).astype(int)

    return items


def load_customers(path: Path | None = None) -> pd.DataFrame:
    """Load the QuickBooks Customer Contact List export.

    Supports multiple files (Customers.CSV, Customers2.CSV, etc.).
    """
    if path:
        paths = [path]
    else:
        paths = _find_files("Customers")

    frames = []
    for p in paths:
        print(f"  Reading {p.name}...")
        raw = pd.read_csv(p, header=0, dtype=str, keep_default_na=False, encoding="cp1252")
        first_col = raw.columns[0]
        df = raw.drop(columns=[first_col])

        df.columns = [
            "customer",
            "bill_to",
            "primary_contact",
            "main_phone",
            "fax",
            "balance_total",
        ]
        frames.append(df)

    customers = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["customer"])

    customers["balance_total"] = pd.to_numeric(
        customers["balance_total"].str.replace(",", ""), errors="coerce"
    ).fillna(0.0)

    return customers


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sales...")
    sales = load_sales()
    print(f"  {len(sales)} transactions, {sales['sku'].nunique()} unique SKUs")
    print(f"  Date range: {sales['date'].min()} to {sales['date'].max()}")
    print(f"  Zero-qty rows: {(sales['qty'] == 0).sum()}")
    sales.to_parquet(PROCESSED_DIR / "sales.parquet", index=False)

    print("\nLoading items...")
    items = load_items()
    print(f"  {len(items)} items")
    print(f"  Types: {items['type'].value_counts().to_dict()}")
    items.to_parquet(PROCESSED_DIR / "items.parquet", index=False)

    print("\nLoading customers...")
    customers = load_customers()
    print(f"  {len(customers)} customers")
    customers.to_parquet(PROCESSED_DIR / "customers.parquet", index=False)

    print(f"\nProcessed files written to {PROCESSED_DIR}")
