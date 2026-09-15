from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import clean_text
from .validation import validate_schema, validate_business_rules

NUMERIC_COLUMNS = [
    "price_usd",
    "discount", 
    "sold", 
    "gmv_usd", 
    "rating", 
    "reviews",
]

def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    validate_schema(df)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sku_name", "brand", "category", "country", "marketplace"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["sku_name", "price_usd", "month"])
    df = df[df["price_usd"] > 0].copy()
    if "discount" in df:
        df["discount"] = df["discount"].fillna(0).clip(0, 100)
    if "sold" in df:
        df["sold"] = df["sold"].fillna(0).clip(lower=0)
    if "rating" in df:
        df["rating"] = df["rating"].fillna(df["rating"].median()).clip(0, 5)
    if "reviews" in df:
        df["reviews"] = df["reviews"].fillna(0).clip(lower=0)
    df["sku_name_clean"] = df["sku_name"].apply(clean_text)
    df["brand_clean"] = df["brand"].apply(clean_text)
    dedupe = ["sku_name", "merchant", "month", "marketplace", "country"]
    df = df.drop_duplicates(subset=[c for c in dedupe if c in df], keep="last")
    upper = df["price_usd"].quantile(0.999)
    df = df[df["price_usd"] <= upper].copy()
    validate_business_rules(df)
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except ImportError:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
