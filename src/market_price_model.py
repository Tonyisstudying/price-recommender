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
    def bounded(values: np.ndarray) -> np.ndarray:
        median = pd.to_numeric(df.get("competitor_median_usd", 0), errors="coerce").to_numpy()
        upper = np.where(median > 0, median * 3.0, np.inf)
        return np.minimum(np.maximum(values, 0), upper)

    # Support artifacts produced by the pre-canonical feature contract.
    if getattr(model, "feature_names_", None) and "brand" in model.feature_names_:
        legacy = pd.DataFrame(index=df.index)
        legacy["country"] = df.get("country", "Unknown").astype(str)
        legacy["marketplace"] = df.get("marketplace", "Unknown").astype(str)
        legacy["category"] = df.get("category", "Unknown").astype(str)
        legacy["brand"] = df.get("brand_name", df.get("brand", "Unknown")).astype(str)
        legacy["rating"] = pd.to_numeric(df.get("rating_score", 0), errors="coerce").fillna(0)
        legacy["log_reviews"] = np.log1p(pd.to_numeric(df.get("review_count", 0), errors="coerce").fillna(0))
        legacy["log_sold"] = np.log1p(pd.to_numeric(df.get("sold_count", 0), errors="coerce").fillna(0))
        legacy["discount"] = pd.to_numeric(df.get("discount_rate", 0), errors="coerce").fillna(0)
        dates = df["snapshot_date"] if "snapshot_date" in df else pd.Series(pd.NaT, index=df.index)
        legacy["month_num"] = pd.to_datetime(dates, errors="coerce").dt.month.fillna(0)
        legacy["competitor_p25"] = df.get("competitor_p25_usd", 0)
        legacy["competitor_median"] = df.get("competitor_median_usd", 0)
        legacy["competitor_p75"] = df.get("competitor_p75_usd", 0)
        legacy["competitor_count"] = df.get("competitor_count", 0)
        return bounded(np.expm1(model.predict(legacy[model.feature_names_])))
    frame = prepare_frame(df)
    x = frame[CATEGORICAL + NUMERIC]
    return bounded(np.expm1(model.predict(x)))


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
