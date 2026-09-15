from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

CATEGORICAL = ["country", "marketplace", "category", "brand"]
NUMERIC = ["price_usd", "discount", "rating", "log_reviews", "competitor_median", "competitor_p25", "competitor_p75"]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["log_reviews"] = np.log1p(frame.get("reviews", 0).fillna(0))
    for col in ["competitor_median", "competitor_p25", "competitor_p75"]:
        if col not in frame:
            frame[col] = frame["price_usd"]
    return frame


def build_model() -> Pipeline:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer([
        ("cat", categorical, CATEGORICAL),
        ("num", numeric, NUMERIC),
    ])
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_model(train_df: pd.DataFrame):
    frame = prepare_frame(train_df)
    x = frame[CATEGORICAL + NUMERIC]
    y = np.log1p(frame["sold"].clip(lower=0))
    model = build_model()
    model.fit(x, y)
    return model


def predict_demand(model, df: pd.DataFrame) -> np.ndarray:
    frame = prepare_frame(df)
    values = model.predict(frame[CATEGORICAL + NUMERIC])
    return np.maximum(np.expm1(values), 0)


def evaluate(model, test_df: pd.DataFrame) -> dict:
    y_true = test_df["sold"].clip(lower=0).to_numpy()
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
