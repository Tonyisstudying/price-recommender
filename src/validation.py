"""Schema and business-rule validation for marketplace listings.

Validation is a gate before modeling. It records *why* a row is unusable
instead of silently dropping it, which matters when Magpie extracts contain
missing sales, promotional prices, or malformed dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

CANONICAL_REQUIRED = (
    "sku_name",
    "brand",
    "category",
    "price_usd",
    "month",
    "country",
    "marketplace",
)

CANONICAL_OPTIONAL = (
    "merchant",
    "discount",
    "sold",
    "gmv_usd",
    "rating",
    "reviews",
)


@dataclass
class ValidationReport:
    n_rows: int
    missing_required: dict[str, int] = field(default_factory=dict)
    invalid_price_rows: int = 0
    invalid_date_rows: int = 0
    invalid_country_rows: int = 0
    invalid_marketplace_rows: int = 0
    duplicate_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "missing_required": self.missing_required,
            "invalid_price_rows": self.invalid_price_rows,
            "invalid_date_rows": self.invalid_date_rows,
            "invalid_country_rows": self.invalid_country_rows,
            "invalid_marketplace_rows": self.invalid_marketplace_rows,
            "duplicate_rows": self.duplicate_rows,
            "notes": self.notes,
        }


def _series_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def validate_canonical_frame(frame: pd.DataFrame, config: dict) -> ValidationReport:
    """Inspect a canonical-schema DataFrame. Does not mutate rows."""
    report = ValidationReport(n_rows=len(frame))
    allowed_countries = {c.lower() for c in config["markets"]["countries"]}
    allowed_marketplaces = {m.lower() for m in config["markets"]["marketplaces"]}
    rules = config["validation"]

    for column in CANONICAL_REQUIRED:
        if column not in frame.columns:
            report.missing_required[column] = report.n_rows
            report.notes.append(f"Required column missing entirely: {column}")
            continue
        missing = frame[column].isna() | (frame[column].astype(str).str.strip() == "")
        report.missing_required[column] = int(missing.sum())

    prices = pd.to_numeric(_series_or_empty(frame, "price_usd"), errors="coerce")
    invalid_price = prices.isna() | (prices < rules["min_price_usd"]) | (
        prices > rules["max_price_usd"]
    )
    report.invalid_price_rows = int(invalid_price.sum())

    months = _series_or_empty(frame, "month")
    report.invalid_date_rows = int(months.isna().sum())

    countries = _series_or_empty(frame, "country").astype(str).str.strip().str.lower()
    report.invalid_country_rows = int(
        (~countries.isin(allowed_countries) | countries.eq("nan") | countries.eq("")).sum()
    )

    marketplaces = _series_or_empty(frame, "marketplace").astype(str).str.strip().str.lower()
    report.invalid_marketplace_rows = int(
        (
            ~marketplaces.isin(allowed_marketplaces)
            | marketplaces.eq("nan")
            | marketplaces.eq("")
        ).sum()
    )

    identity = [
        col
        for col in ("sku_name", "merchant", "country", "marketplace", "month", "price_usd")
        if col in frame.columns
    ]
    if identity:
        report.duplicate_rows = int(frame.duplicated(subset=identity, keep="first").sum())

    logger.info("Validation report: %s", report.as_dict())
    return report


def valid_price_mask(frame: pd.DataFrame, config: dict) -> pd.Series:
    rules = config["validation"]
    prices = pd.to_numeric(frame["price_usd"], errors="coerce")
    return prices.notna() & (prices >= rules["min_price_usd"]) & (
        prices <= rules["max_price_usd"]
    )
