from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor


CATEGORICAL = [
    "country", "marketplace", "category", "brand_name", "currency"
]
NUMERIC = [
    "rating_score", "log_reviews", "log_sold", "discount_rate",
    "competitor_p25_usd", "competitor_median_usd",
    "competitor_p75_usd", "competitor_count",
]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_reviews"] = np.log1p(out.get("review_count", 0).fillna(0))
    out["log_sold"] = np.log1p(out.get("sold_count", 0).fillna(0))
    for col in NUMERIC:
        if col not in out:
            out[col] = 0.0
    for col in CATEGORICAL:
        if col not in out:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def train_model(train_df: pd.DataFrame):
    frame = prepare_frame(train_df)
    x = frame[CATEGORICAL + NUMERIC].copy()
    y = np.log1p(frame["price_usd"])  # stabilize across categories/countries
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
    return model, "CatBoostRegressor(log-price)"


def predict_usd(model, df: pd.DataFrame) -> np.ndarray:
    frame = prepare_frame(df)
    x = frame[CATEGORICAL + NUMERIC]
    return np.maximum(np.expm1(model.predict(x)), 0)


def evaluate(model, test_df: pd.DataFrame) -> dict:
    y_true = test_df["price_usd"].to_numpy()
    y_pred = predict_usd(model, test_df)
    nonzero = y_true != 0
    return {
        "mae_usd": float(mean_absolute_error(y_true, y_pred)),
        "rmse_usd": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100),
    }


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)
