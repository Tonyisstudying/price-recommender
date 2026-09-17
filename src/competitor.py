from __future__ import annotations

import numpy as np
import pandas as pd


def seller_deduplicated_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    key = [c for c in ["seller_id", "seller_name"] if c in df.columns]
    if not key:
        return df.copy()
    work = df.copy()
    work["seller_key"] = work["seller_id"].where(work["seller_id"].ne(""), work["seller_name"])
    return work.groupby("seller_key", as_index=False).agg(
        price_local=("price_local", "median"),
        sold_count=("sold_count", "max"),
        review_count=("review_count", "max"),
        rating_score=("rating_score", "max"),
        product_name=("product_name", "first"),
        brand_name=("brand_name", "first"),
        seller_name=("seller_name", "first"),
    )


def price_distribution(competitor_df: pd.DataFrame) -> dict:
    if competitor_df.empty:
        raise ValueError("No competitor products found.")
    work = seller_deduplicated_prices(competitor_df)
    price_col = "price_local" if "price_local" in work.columns else "price_usd"
    values = work[price_col].dropna().to_numpy()
    if len(values) == 0:
        raise ValueError("No valid competitor prices found.")
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(len(values)),
        "seller_count": int(len(work)),
    }


def weighted_price_mean(competitor_df: pd.DataFrame) -> float:
    work = seller_deduplicated_prices(competitor_df)
    if work.empty:
        raise ValueError("No competitors.")
    price_col = "price_local" if "price_local" in work.columns else "price_usd"
    weights = np.clip(np.log1p(work.get("sold_count", pd.Series(0, index=work.index)).fillna(0).to_numpy()), 0.1, None)
    return float(np.average(work[price_col].to_numpy(), weights=weights))


def build_competitor_features(competitor_df: pd.DataFrame) -> dict:
    stats = price_distribution(competitor_df)
    stats["weighted_mean"] = weighted_price_mean(competitor_df)
    return stats
