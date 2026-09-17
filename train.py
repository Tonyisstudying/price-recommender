from __future__ import annotations

import json

from src.config import CONFIG, ROOT
from src.data_manager import load_master_dataset
from src.model_registry import ModelRegistry
from src.retraining import train_challenger


def main():
    master = load_master_dataset(ROOT / CONFIG["data"]["master_path"])
    if master.empty:
        raise SystemExit("No master data. Run update_data.py first.")

    registry = ModelRegistry(ROOT / CONFIG["model"]["registry_dir"])
    metadata = train_challenger(master, registry)
    registry.promote(
        metadata["version"],
        {
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "rows": metadata["rows"],
            "market_price_model": metadata["market_price_model"],
            "demand_model": metadata["demand_model"],
        },
    )
    registry.cleanup_old_versions(keep=5)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
