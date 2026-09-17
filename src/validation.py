from __future__ import annotations

import pandas as pd

MIN_COLUMNS = {"product_name", "price_current"}


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in MIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_business_rules(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty after preprocessing.")
    if (df["price_local"] <= 0).any():
        raise ValueError("price_local must be > 0")
    if (df["price_usd"] <= 0).any():
        raise ValueError("price_usd must be > 0")
    if not pd.api.types.is_datetime64_any_dtype(df["snapshot_date"]):
        raise TypeError("snapshot_date must be datetime")
