from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class SourceAdapter:
    name: str

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class LazadaAdapter(SourceAdapter):
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["marketplace"] = "Lazada"
        if "currency" not in out.columns:
            out["currency"] = "VND"
        return out


class ShopeeAdapter(SourceAdapter):
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["marketplace"] = "Shopee"
        if "currency" not in out.columns:
            out["currency"] = "VND"
        return out

class GenericAdapter(SourceAdapter):
    pass

ADAPTERS = {
    "lazada": LazadaAdapter("Lazada"),
    "shopee": ShopeeAdapter("Shopee"),
    "generic": GenericAdapter("Generic"),
}

def get_adapter(source: str | None) -> SourceAdapter:
    key = (source or "generic").strip().lower().replace(" ", "_")
    if key not in ADAPTERS:
        raise ValueError(
            f"Unknown source '{source}'. Supported: {', '.join(sorted(ADAPTERS))}"
        )
    return ADAPTERS[key]