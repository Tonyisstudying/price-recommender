from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.config import CONFIG, ROOT
from src.data_manager import read_external_file, update_master_dataset, register_data_event
from src.model_registry import ModelRegistry
from src.retraining import train_challenger, challenger_is_better


def read_input(path: Path, source: str, snapshot_date: str | None) -> pd.DataFrame:
    if path.is_file():
        return read_external_file(path, source=source, snapshot_date=snapshot_date)
    frames = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix.lower() in {".csv", ".parquet", ".pq", ".json", ".jsonl"}:
            frames.append(read_external_file(file, source=source, snapshot_date=snapshot_date))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source", default="generic", choices=["generic", "lazada", "shopee", "tiktok", "tiktok_shop", "tokopedia"])
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args()

    incoming = read_input(Path(args.input), args.source, args.snapshot_date)
    if incoming.empty:
        raise SystemExit("No valid records found.")

    master_path = ROOT / CONFIG["data"]["master_path"]
    event_log = ROOT / CONFIG["data"]["update_log"]
    registry = ModelRegistry(ROOT / CONFIG["model"]["registry_dir"])

    merged, report = update_master_dataset(
        master_path=master_path,
        incoming=incoming,
    )

    if incoming.empty:
        raise SystemExit(
            f"No CSV/Parquet records found in {input_path}. "
            "Add a new dataset and run the command again."
        )
    
    register_data_event(report, event_log)
    print("DATA UPDATE")
    print(report)

    if args.no_retrain:
        return

    challenger = train_challenger(
        merged, 
        registry,
    )
    champion = registry.get_current()
    champion_meta = None
    if champion:
        meta = registry.versions / champion["version"] / "metadata.json"
        if meta.exists():
            import json
            champion_meta = json.loads(meta.read_text(encoding="utf-8"))

    if challenger_is_better(challenger, champion_meta):
        registry.promote(
            challenger["version"],
            {
                "dataset_fingerprint": challenger["dataset_fingerprint"],
                "rows": challenger["rows"],
                "market_price_model": challenger["market_price_model"],
                "demand_model": challenger["demand_model"],
            },
        )
        registry.cleanup_old_versions(keep=5)
        print(f"PROMOTED: {challenger['version']}")
    else:
        print("REJECTED: current champion remains; new data is still in the master catalog.")


if __name__ == "__main__":
    main()
