from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

CATEGORICAL = ["country", "marketplace", "category", "brand"]
NUMERIC = [
    "rating", "log_reviews", "log_sold", "discount", "month_num",
    "competitor_p25", "competitor_median", "competitor_p75", "competitor_count",
]


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_reviews"] = np.log1p(out.get("reviews", 0).fillna(0))
    out["log_sold"] = np.log1p(out.get("sold", 0).fillna(0))
    out["month_num"] = out["month"].dt.month
    for col in NUMERIC:
        if col not in out:
            out[col] = 0.0
    return out


def build_sklearn_model() -> Pipeline:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer([
        ("cat", categorical, CATEGORICAL),
        ("num", numeric, NUMERIC),
    ])
    model = RandomForestRegressor(
        n_estimators=350, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_model(train_df: pd.DataFrame, prefer_catboost: bool = True):
    frame = prepare_training_frame(train_df)
    x = frame[CATEGORICAL + NUMERIC]
    y = frame["price_usd"]
    if prefer_catboost and CatBoostRegressor is not None:
        model = CatBoostRegressor(
            iterations=500, depth=7, learning_rate=0.05,
            loss_function="RMSE", verbose=False, random_seed=42
        )
        model.fit(x, y, cat_features=CATEGORICAL)
        return model, "CatBoostRegressor"
    model = build_sklearn_model()
    model.fit(x, y)
    return model, "RandomForestRegressor"


def predict(model, df: pd.DataFrame) -> np.ndarray:
    frame = prepare_training_frame(df)
    return model.predict(frame[CATEGORICAL + NUMERIC])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    y_true = test_df["price_usd"].to_numpy()
    y_pred = predict(model, test_df)
    nonzero = y_true != 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100),
    }


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)
