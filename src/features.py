"""
Feature engineering for the market-price and demand models (Phase 3).

Built now (Phase 1) because EDA and the competitor engine both want
log-scaled sold/review counts and a clean `month` ordinal, and defining
them once here avoids three copies of the same log1p() scattered around
notebooks later.

IMPORTANT - leakage rule (see brief section 31):
Every function in this file is a pure row-wise / within-row transform.
None of them look at any other row's price or any future date. Anything
that aggregates across rows (competitor medians, etc.) lives in
competitor.py and MUST be computed only from data available at or before
the row's own snapshot_date - that join happens in competitor.py, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """log1p-transform heavily right-skewed count features. Using log1p
    (not log) so sold_count=0 / review_count=0 don't produce -inf."""
    df = df.copy()
    df["log_sold"] = np.log1p(df["sold_count"].astype(float))
    df["log_reviews"] = np.log1p(df["review_count"].astype(float))
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal month index (0,1,2,...) in addition to the string `month`
    column, so models can use month as an ordered numeric feature without
    treating March/April/May as unrelated categories. Rows with unknown
    snapshot_date (the Lazada scrape - see preprocessing.py) get NaN here,
    which correctly excludes them from anything using this feature."""
    df = df.copy()
    month_order = sorted(df["month"].dropna().unique())
    month_to_idx = {m: i for i, m in enumerate(month_order)}
    df["month_index"] = df["month"].map(month_to_idx)
    return df


def add_discount_features(df: pd.DataFrame) -> pd.DataFrame:
    """discount_pct already exists from preprocessing; add a clipped,
    non-negative version for modeling (a negative "discount" is a data
    artifact we chose to keep visible upstream, but should not feed a
    model as if raising price were a discount)."""
    df = df.copy()
    df["discount_pct_clipped"] = df["discount_pct"].clip(lower=0, upper=0.95).fillna(0.0)
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every row-wise feature transform. Call this once, right after
    src.preprocessing.run(), before competitor-stat joins."""
    df = add_log_transforms(df)
    df = add_temporal_features(df)
    df = add_discount_features(df)
    return df


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    cleaned = pd.read_parquet(cfg.processed_dir / "cleaned_products.parquet")
    featured = build_feature_table(cleaned)
    out_path = cfg.processed_dir / "featured_products.parquet"
    featured.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(featured)} rows, {featured.shape[1]} cols)")
