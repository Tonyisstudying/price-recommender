from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_reviews"] = np.log1p(df.get("reviews", pd.Series(0, index=df.index)).fillna(0))
    df["log_sold"] = np.log1p(df.get("sold", pd.Series(0, index=df.index)).fillna(0))
    df["month_num"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    return df
