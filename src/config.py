from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

@dataclass(frozen=True)
class Config:
    """Typed, read-only wrapper around the parsed config.yaml."""

    raw: dict[str, Any]

    # -- paths ---------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["processed_dir"]

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["models_dir"]

    @property
    def raw_sources(self) -> list[dict[str, str]]:
        return self.raw["paths"].get("raw_sources", [])

    # -- domain ----------------------------------------------------------
    @property
    def valid_countries(self) -> list[str]:
        return self.raw.get("domain", self.raw.get("markets", {}))["countries"]

    @property
    def valid_marketplaces(self) -> list[str]:
        return self.raw.get("domain", self.raw.get("markets", {}))["marketplaces"]

    @property
    def valid_categories(self) -> list[str]:
        return self.raw.get("domain", self.raw.get("markets", {}))["categories"]

    @property
    def valid_strategies(self) -> list[str]:
        domain = self.raw.get("domain", {})
        return domain.get("pricing_strategies", self.raw["pricing"]["strategies"])

    @property
    def country_currency(self) -> dict[str, str]:
        return self.raw["country_currency"]

    @property
    def fx_to_usd(self) -> dict[str, float]:
        return self.raw["fx_to_usd"]

    @property
    def vietnam_location_aliases(self) -> set[str]:
        return set(self.raw.get("vietnam_location_aliases", []))

    @property
    def category_keywords(self) -> dict[str, list[str]]:
        return self.raw.get("category_keywords", {})

    @property
    def quality(self) -> dict[str, Any]:
        return self.raw["quality"]

    @property
    def similarity(self) -> dict[str, Any]:
        return self.raw["similarity"]

    @property
    def pricing(self) -> dict[str, Any]:
        pricing = self.raw["pricing"]
        return {
            **pricing,
            "market_weight": pricing.get(
                "market_weight", pricing.get("market_anchor_competitor_weight", 0.60)
            ),
            "ml_weight": pricing.get(
                "ml_weight", pricing.get("market_anchor_model_weight", 0.40)
            ),
        }


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and parse configs/config.yaml into a Config object."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)


PROJECT_ROOT = PROJECT_ROOT
CONFIG = load_config().raw
ROOT = PROJECT_ROOT
