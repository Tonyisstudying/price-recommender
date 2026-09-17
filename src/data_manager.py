from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .adapter.registry import get_adapter
from .preprocessing import preprocess
from .utils import ensure_dir

SUPPORTED = {".csv", ".parquet", ".pq", ".json", ".jsonl"}


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def read_external_file(
    path: str | Path,
    *,
    source: str = "generic",
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    path = Path(path)
    raw = _read(path)
    adapted = get_adapter(source).normalize(raw)
    return preprocess(adapted, snapshot_date=snapshot_date)


def read_external_files(input_dir: str | Path, source: str = "generic") -> list[tuple[Path, pd.DataFrame]]:
    input_dir = Path(input_dir)
    results = []
    if not input_dir.exists():
        return results
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            results.append((path, read_external_file(path, source=source)))
    return results


def load_master_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = _read(path)
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def save_master_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def dataset_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    stable = df.sort_index(axis=1).copy()
    sort_cols = [
        c for c in [
            "snapshot_date", "country", "marketplace",
            "category", "product_id", "seller_id", "product_name"
        ] if c in stable.columns
    ]
    if sort_cols:
        stable = stable.sort_values(sort_cols).reset_index(drop=True)
    payload = pd.util.hash_pandas_object(stable, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def merge_datasets(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = [x for x in [existing, incoming] if not x.empty]
    if not frames:
        raise ValueError("No data to merge.")
    merged = pd.concat(frames, ignore_index=True)

    key = ["product_id", "seller_id", "snapshot_date", "marketplace", "country"]
    fallback_key = ["product_name_clean", "seller_name", "snapshot_date", "marketplace", "country"]
    dedupe_key = key if merged["product_id"].ne("").any() else fallback_key

    before = len(merged)
    merged = merged.drop_duplicates(subset=dedupe_key, keep="last")
    duplicates_removed = before - len(merged)
    merged = merged.sort_values(
        ["snapshot_date", "country", "marketplace", "category", "product_name"]
    ).reset_index(drop=True)

    merged["sold_delta"] = (
        merged.sort_values(["product_id", "seller_id", "snapshot_date"])
        .groupby(["product_id", "seller_id"])['sold_count']
        .diff().fillna(0).clip(lower=0)
    )
    merged["review_delta"] = (
        merged.sort_values(["product_id", "seller_id", "snapshot_date"])
        .groupby(["product_id", "seller_id"])['review_count']
        .diff().fillna(0).clip(lower=0)
    )

    report = {
        "existing_rows": int(len(existing)),
        "incoming_rows": int(len(incoming)),
        "merged_rows": int(len(merged)),
        "duplicates_removed": int(duplicates_removed),
        "min_snapshot": str(merged["snapshot_date"].min().date()),
        "max_snapshot": str(merged["snapshot_date"].max().date()),
        "marketplaces": sorted(merged["marketplace"].dropna().astype(str).unique().tolist()),
        "countries": sorted(merged["country"].dropna().astype(str).unique().tolist()),
        "fingerprint": dataset_fingerprint(merged),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return merged, report


def update_master_dataset(*, master_path: str | Path, incoming: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    existing = load_master_dataset(master_path)
    merged, report = merge_datasets(existing, incoming)
    save_master_dataset(merged, master_path)
    return merged, report


def register_data_event(report: dict, log_path: str | Path) -> None:
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
