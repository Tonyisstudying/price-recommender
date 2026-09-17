from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .config import CONFIG, ROOT
from .competitor import price_distribution
from .data_manager import dataset_fingerprint
from .demand_model import train_model as train_demand_model, evaluate as evaluate_demand_model, save_model as save_demand_model
from .market_price_model import train_model as train_market_model, evaluate as evaluate_market_model, save_model as save_market_model
from .model_registry import ModelRegistry
from .similarity import SimilarityEngine
from .utils import save_json


def add_competitor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build time-local market features without looking into future months."""
    group_cols = ["country", "marketplace", "category", "month"]
    grouped = (
        df.groupby(group_cols)["price_usd"]
        .agg(
            competitor_p25=lambda s: s.quantile(0.25),
            competitor_median="median",
            competitor_p75=lambda s: s.quantile(0.75),
            competitor_count="count",
        )
        .reset_index()
    )
    return df.merge(grouped, on=group_cols, how="left")


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(df["month"].dropna().dt.to_period("M").unique())
    if len(months) < 2:
        raise ValueError("At least two distinct months are required for temporal validation.")

    test_period = months[-1]
    test_mask = df["month"].dt.to_period("M") == test_period
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train/test set.")

    return train_df, test_df


def train_challenger(df: pd.DataFrame, registry: ModelRegistry) -> dict:
    version = registry.new_version_id()
    version_dir = registry.version_path(version)

    enriched = add_competitor_features(df)
    train_df, test_df = temporal_split(enriched)

    market_model, market_model_type = train_market_model(
        train_df,
        prefer_catboost=True,
    )
    market_metrics = evaluate_market_model(
        market_model,
        test_df,
    )
    save_market_model(
        market_model,
        version_dir / "market_price_model.joblib",
    )

    demand_model = train_demand_model(train_df)
    demand_metrics = evaluate_demand_model(
        demand_model,
        test_df,
    )
    save_demand_model(
        demand_model,
        version_dir / "demand_model.joblib",
    )

    similarity = SimilarityEngine.fit(enriched)
    joblib.dump(
        similarity.vectorizer,
        version_dir / "tfidf_vectorizer.joblib",
    )

    # Persist the processed corpus alongside the model. This makes the model
    # version self-contained and prevents an old model from accidentally using
    # a future vocabulary/data snapshot.
    corpus_path = version_dir / "corpus.csv"
    enriched.to_csv(corpus_path, index=False)

    metadata = {
        "version": version,
        "dataset_fingerprint": dataset_fingerprint(df),
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df["month"].min().date()),
        "train_end": str(train_df["month"].max().date()),
        "test_start": str(test_df["month"].min().date()),
        "test_end": str(test_df["month"].max().date()),
        "market_price_model": {
            "type": market_model_type,
            **market_metrics,
        },
        "demand_model": {
            "type": "HistGradientBoostingRegressor",
            **demand_metrics,
        },
    }
    save_json(metadata, version_dir / "metadata.json")
    return metadata


def challenger_is_better(
    challenger: dict,
    champion: dict | None,
) -> bool:
    if champion is None:
        return True

    c_mae = challenger["market_price_model"]["mae"]
    p_mae = champion.get("market_price_model", {}).get("mae", float("inf"))

    c_rmse = challenger["demand_model"]["rmse"]
    p_rmse = champion.get("demand_model", {}).get("rmse", float("inf"))

    # Primary decision: market-price MAE. Demand RMSE is a tie-breaker.
    if c_mae < p_mae * 0.995:
        return True
    if c_mae > p_mae * 1.005:
        return False
    return c_rmse <= p_rmse

