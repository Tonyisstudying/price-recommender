from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "sku_name", "brand", "category", "price_usd",
    "month", "country", "marketplace"
]

def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_business_rules(df: pd.DataFrame) -> None:
    if (df["price_usd"] <= 0).any():
        raise ValueError("price_usd must be > 0")
    if not pd.api.types.is_datetime64_any_dtype(df["month"]):
        raise TypeError("month must be datetime")
