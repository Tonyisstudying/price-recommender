from __future__ import annotations

from pathlib import Path
import joblib

import pandas as pd

from src.config import CONFIG, ROOT
from src.preprocessing import load_raw_data, preprocess, save_processed
from src.market_price_model import train_model as train_market_model, evaluate as evaluate_market_model, save_model as save_market_model
from src.demand_model import train_model as train_demand_model, evaluate as evaluate_demand_model, save_model as save_demand_model
from src.similarity import SimilarityEngine
from src.utils import save_json


def add_basic_competitor_features(df: pd.DataFrame) -> pd.DataFrame:
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


def main() -> None:
    raw_path = ROOT / CONFIG["data"]["raw_path"]
    processed_path = ROOT / CONFIG["data"]["processed_path"]
    model_dir = ROOT / CONFIG["model"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(raw_path)
    df = preprocess(raw)
    df = add_basic_competitor_features(df)
    save_processed(df, processed_path)

    cutoff = pd.Timestamp("2026-06-01")
    train_df = df[df["month"] < cutoff].copy()
    test_df = df[df["month"] >= cutoff].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Temporal split failed; dataset must contain data before and during June 2026.")

    market_model, market_model_type = train_market_model(train_df, prefer_catboost=True)
    market_metrics = evaluate_market_model(market_model, test_df)
    save_market_model(market_model, model_dir / CONFIG["model"]["market_price_model"])

    demand_model = train_demand_model(train_df)
    demand_metrics = evaluate_demand_model(demand_model, test_df)
    save_demand_model(demand_model, model_dir / CONFIG["model"]["demand_model"])

    similarity = SimilarityEngine.fit(df)
    joblib.dump(similarity.vectorizer, model_dir / CONFIG["model"]["tfidf_vectorizer"])

    metrics = {
        "market_price_model": {"type": market_model_type, "metrics": market_metrics},
        "demand_model": {"type": "HistGradientBoostingRegressor", "metrics": demand_metrics},
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": str(train_df["month"].min().date()),
        "train_end": str(train_df["month"].max().date()),
        "test_start": str(test_df["month"].min().date()),
        "test_end": str(test_df["month"].max().date()),
    }
    save_json(metrics, model_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
