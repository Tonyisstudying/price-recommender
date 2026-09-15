"""Feature engineering for market-price and demand models.

Leakage rules (mandatory):
    * Competitor statistics for an observation at month t use other listings
      whose month is <= t, never future months.
    * A listing does not enter its own competitor distribution.
    * Same-month competitor *listings* are treated as observable market context
      (you can see the live catalog). That is not the same as using the label
      price of the row being predicted.
    * Demand features may include listed price because that is the policy
      variable; they must not include future sold/gmv.

Skewed counts are log1p-transformed. Categorical fields stay as strings for
CatBoost later; one-hot encoding is applied only for linear baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.competitor import compute_price_distribution
from src.utils import get_logger

logger = get_logger(__name__)

LOG_COLUMNS = ("reviews", "sold")


def add_log_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in LOG_COLUMNS:
        if column in out.columns:
            numeric = pd.to_numeric(out[column], errors="coerce").fillna(0).clip(lower=0)
            out[f"log_{column}"] = np.log1p(numeric)
    return out


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    periods = pd.PeriodIndex(out["month"].astype(str), freq="M")
    out["year"] = periods.year
    out["month_number"] = periods.month
    origin = periods.min()
    out["months_since_start"] = (periods.year - origin.year) * 12 + (periods.month - origin.month)
    return out


def _group_competitor_features(group: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out competitor stats inside one (country, marketplace, category, month)."""
    n = len(group)
    prices = pd.to_numeric(group["price_usd"], errors="coerce").to_numpy(dtype=float)
    sold = pd.to_numeric(group.get("sold", pd.Series(np.zeros(n))), errors="coerce").fillna(0).to_numpy(dtype=float)

    p25 = np.full(n, np.nan)
    p50 = np.full(n, np.nan)
    p75 = np.full(n, np.nan)
    mean = np.full(n, np.nan)
    count = np.zeros(n, dtype=int)
    weighted_median = np.full(n, np.nan)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        others = prices[mask]
        others = others[~np.isnan(others)]
        count[i] = int(others.size)
        if others.size == 0:
            continue
        p25[i] = float(np.percentile(others, 25))
        p50[i] = float(np.percentile(others, 50))
        p75[i] = float(np.percentile(others, 75))
        mean[i] = float(others.mean())
        dist = compute_price_distribution(
            pd.Series(prices[mask]),
            pd.Series(sold[mask]),
        )
        weighted_median[i] = dist.weighted_median if dist.weighted_median is not None else np.nan

    result = group.copy()
    result["competitor_count"] = count
    result["competitor_p25"] = p25
    result["competitor_median"] = p50
    result["competitor_p75"] = p75
    result["competitor_mean"] = mean
    result["competitor_weighted_median"] = weighted_median
    return result


def add_same_month_competitor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Category-level competitor features at month t, excluding the current row.

    This is a coarse feature (all SKUs in the category cell), not TF-IDF
    similarity. Similarity-based features are computed at inference time by
    CompetitorEngine. Using only month == t (not future) avoids test-period
    leakage when the frame is already sliced to train or test.
    """
    if frame.empty:
        return frame
    keys = ["country", "marketplace", "category", "month"]
    parts = [_group_competitor_features(part) for _, part in frame.groupby(keys, dropna=False)]
    out = pd.concat(parts, axis=0).sort_index()
    logger.info("Added leave-one-out competitor features for %s rows", len(out))
    return out


def build_model_frame(frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Full feature table for later market-price / demand training."""
    config = config or load_config()
    featured = add_log_features(frame)
    featured = add_temporal_features(featured)
    featured = add_same_month_competitor_features(featured)
    return featured


def temporal_split(frame: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = Mar–May 2026, test = Jun 2026. Never a random row split."""
    config = config or load_config()
    train_months = set(config["temporal"]["train_months"])
    test_months = set(config["temporal"]["test_months"])
    months = frame["month"].astype(str)
    train = frame[months.isin(train_months)].copy()
    test = frame[months.isin(test_months)].copy()
    if train.empty or test.empty:
        logger.warning(
            "Temporal split produced empty fold (train=%s, test=%s). Check month coverage.",
            len(train),
            len(test),
        )
    logger.info("Temporal split: train=%s rows (%s), test=%s rows (%s)", len(train), sorted(train_months), len(test), sorted(test_months))
    return train, test


def save_competitor_features(frame: pd.DataFrame, config: dict | None = None) -> None:
    config = config or load_config()
    path = get_path(config, "competitor_features_parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("Wrote competitor features to %s", path)


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline

    cleaned = run_preprocessing_pipeline()
    featured = build_model_frame(cleaned)
    save_competitor_features(featured)
    temporal_split(featured)
