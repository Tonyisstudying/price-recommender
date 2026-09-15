from __future__ import annotations

import numpy as np
import pandas as pd


def price_distribution(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("No competitor products found")
    prices = pd.to_numeric(df["price_usd"], errors="coerce").dropna().to_numpy()
    if len(prices) == 0:
        raise ValueError("No valid competitor prices")
    return {
        "min": float(np.min(prices)),
        "p10": float(np.percentile(prices, 10)),
        "p25": float(np.percentile(prices, 25)),
        "median": float(np.percentile(prices, 50)),
        "p75": float(np.percentile(prices, 75)),
        "p90": float(np.percentile(prices, 90)),
        "max": float(np.max(prices)),
        "mean": float(np.mean(prices)),
        "std": float(np.std(prices)),
        "count": int(len(prices)),
    }


def weighted_price_mean(df: pd.DataFrame) -> float:
    if df.empty:
        raise ValueError("No competitors")
    prices = df["price_usd"].to_numpy()
    sold = df.get("sold", pd.Series(0, index=df.index)).fillna(0).to_numpy()
    weights = np.clip(np.log1p(sold), 0.1, None)
    return float(np.average(prices, weights=weights))


def build_competitor_features(df: pd.DataFrame) -> dict:
    stats = price_distribution(df)
    stats["weighted_mean"] = weighted_price_mean(df)
    return stats
