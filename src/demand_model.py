from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CATEGORICAL = ["country", "marketplace", "category", "brand_name", "currency"]
NUMERIC = [
    "price_usd", "discount_rate", "rating_score", "log_reviews",
    "competitor_median_usd", "competitor_p25_usd", "competitor_p75_usd",
    "competitor_count", "price_vs_median",
]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_reviews"] = np.log1p(out.get("review_count", 0).fillna(0))
    if "price_vs_median" not in out:
        out["price_vs_median"] = out["price_usd"] / out["competitor_median_usd"].replace(0, np.nan)
        out["price_vs_median"] = out["price_vs_median"].fillna(1.0)
    for col in NUMERIC:
        if col not in out:
            out[col] = 0.0
    for col in CATEGORICAL:
        if col not in out:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def train_model(train_df: pd.DataFrame, target_col: str = "demand_target"):
    frame = prepare_frame(train_df)
    x = frame[CATEGORICAL + NUMERIC]
    y = np.log1p(frame[target_col].clip(lower=0))
    model = CatBoostRegressor(
        iterations=600,
        depth=7,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y, cat_features=CATEGORICAL)
    return model, target_col


def predict_demand(model, df: pd.DataFrame) -> np.ndarray:
    frame = prepare_frame(df)
    pred = np.expm1(model.predict(frame[CATEGORICAL + NUMERIC]))
    return np.maximum(pred, 0)


def evaluate(model, test_df: pd.DataFrame, target_col: str = "demand_target") -> dict:
    y_true = test_df[target_col].clip(lower=0).to_numpy()
    y_pred = predict_demand(model, test_df)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_model(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)
