"""Ingest heterogeneous marketplace extracts into a canonical product table.

Responsibilities:
    1. Load CSV or Parquet.
    2. Map source columns via config aliases (never assume Magpie names).
    3. Normalize country / marketplace / category / text.
    4. Convert local currency to USD when `price_usd` is absent.
    5. Parse months, discounts, and numeric fields.
    6. Drop only *invalid* prices/dates — flag statistical outliers instead of
       deleting premium, bundle, or flash-sale listings.
    7. Persist cleaned parquet for later stages.

This module does not train models and does not look at future periods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_path, load_config, resolve_path
from src.utils import (
    get_logger,
    infer_marketplace_from_url,
    map_alias,
    month_to_str,
    normalize_text,
    parse_month,
    safe_float,
    title_case_label,
)
from src.validation import validate_canonical_frame, valid_price_mask

logger = get_logger(__name__)

CANONICAL_COLUMNS = [
    "sku_name",
    "brand",
    "category",
    "merchant",
    "price_usd",
    "discount",
    "sold",
    "gmv_usd",
    "rating",
    "reviews",
    "month",
    "country",
    "marketplace",
    "currency_source",
    "price_local",
    "is_price_outlier",
    "is_mock",
    "source_file",
    "sku_name_norm",
    "search_text",
]

_CATEGORY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("mouse", "Accessories"),
    ("keyboard", "Accessories"),
    ("cable", "Accessories"),
    ("charger", "Accessories"),
    ("holder", "Accessories"),
    ("case", "Accessories"),
    ("rice", "Groceries"),
    ("noodle", "Groceries"),
    ("snack", "Groceries"),
    ("coffee", "Groceries"),
    ("tea", "Groceries"),
    ("oil", "Groceries"),
    ("cleanser", "Beauty"),
    ("serum", "Beauty"),
    ("moisturizer", "Beauty"),
    ("sunscreen", "Beauty"),
    ("toner", "Beauty"),
    ("balm", "Beauty"),
    ("cream", "Beauty"),
    ("shampoo", "Beauty"),
    ("headphone", "Electronics"),
    ("earphone", "Electronics"),
    ("speaker", "Electronics"),
    ("phone", "Electronics"),
    ("laptop", "Electronics"),
    ("vacuum", "Home"),
    ("kettle", "Home"),
    ("cookware", "Home"),
    ("blender", "Home"),
)


def load_table(path: str | Path) -> pd.DataFrame:
    """Load CSV or Parquet from a filesystem path."""
    file_path = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    suffix = file_path.suffix.lower()
    logger.info("Loading %s", file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Parquet.")


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in frame.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _pick_series(frame: pd.DataFrame, aliases: list[str]) -> pd.Series:
    column = _first_present(frame, aliases)
    if column is None:
        return pd.Series([pd.NA] * len(frame), index=frame.index)
    return frame[column]


def infer_category(sku_name: str, existing: object, allowed: list[str]) -> str:
    existing_text = ""
    if existing is not None and not pd.isna(existing):
        existing_text = str(existing).strip()
    if existing_text and existing_text in allowed:
        return existing_text
    text = (sku_name or "").lower()
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in text:
            return category
    if existing_text:
        return existing_text
    return "Accessories"


def convert_to_usd(
    price_local: float | None,
    country: str,
    currency: str | None,
    fx_to_usd: dict[str, float],
    country_currency: dict[str, str],
) -> float | None:
    if price_local is None or pd.isna(price_local):
        return None
    ccy = (currency or country_currency.get(country, "USD")).upper()
    rate = fx_to_usd.get(ccy)
    if rate is None:
        logger.warning("No FX rate for currency %s; leaving price unconverted", ccy)
        return float(price_local)
    return float(price_local) * float(rate)


def _compute_discount(price_local: float | None, original: float | None, given: float | None) -> float | None:
    if given is not None and not pd.isna(given):
        value = float(given)
        # Magpie-style extracts sometimes store 25 instead of 0.25.
        if value > 1.0:
            value = value / 100.0
        return float(np.clip(value, 0.0, 0.95))
    if price_local is None or original is None:
        return None
    if original <= 0 or pd.isna(original) or pd.isna(price_local):
        return None
    if price_local >= original:
        return 0.0
    return float(np.clip(1.0 - (price_local / original), 0.0, 0.95))


def map_to_canonical(frame: pd.DataFrame, config: dict, *, source_file: str = "") -> pd.DataFrame:
    """Apply the alias mapping layer onto a raw extract."""
    aliases = config["column_aliases"]
    countries_allowed = config["markets"]["countries"]
    marketplaces_allowed = config["markets"]["marketplaces"]
    categories_allowed = config["markets"]["categories"]
    country_aliases = {str(k).lower(): v for k, v in config["country_aliases"].items()}
    marketplace_aliases = {str(k).lower(): v for k, v in config["marketplace_aliases"].items()}

    sku = _pick_series(frame, aliases["sku_name"]).map(title_case_label)
    brand = _pick_series(frame, aliases["brand"]).map(title_case_label)
    category_raw = _pick_series(frame, aliases["category"]).map(title_case_label)
    merchant = _pick_series(frame, aliases["merchant"]).map(title_case_label)
    price_usd_raw = _pick_series(frame, aliases["price_usd"]).map(safe_float)
    price_local = _pick_series(frame, aliases["price_local"]).map(safe_float)
    price_original = _pick_series(frame, aliases["price_original_local"]).map(safe_float)
    discount_raw = _pick_series(frame, aliases["discount"]).map(safe_float)
    sold = _pick_series(frame, aliases["sold"]).map(safe_float)
    gmv = _pick_series(frame, aliases["gmv_usd"]).map(safe_float)
    rating = _pick_series(frame, aliases["rating"]).map(safe_float)
    reviews = _pick_series(frame, aliases["reviews"]).map(safe_float)
    month_raw = _pick_series(frame, aliases["month"])
    country_raw = _pick_series(frame, aliases["country"])
    marketplace_raw = _pick_series(frame, aliases["marketplace"])
    url_raw = _pick_series(frame, aliases["product_url"])
    currency_raw = _pick_series(frame, aliases["currency"]).map(
        lambda v: str(v).upper() if pd.notna(v) and str(v).strip() else None
    )

    if "is_mock" in frame.columns:
        is_mock = frame["is_mock"].astype(bool)
    else:
        is_mock = pd.Series([False] * len(frame), index=frame.index)

    rows: list[dict] = []
    for idx in frame.index:
        country = map_alias(country_raw.loc[idx], country_aliases)
        marketplace = map_alias(marketplace_raw.loc[idx], marketplace_aliases)
        if not marketplace:
            marketplace = infer_marketplace_from_url(url_raw.loc[idx])
        if marketplace and marketplace not in marketplaces_allowed:
            marketplace = map_alias(marketplace, marketplace_aliases, default=marketplace)
        if country and country not in countries_allowed:
            country = map_alias(country, country_aliases, default=country)

        listed_usd = price_usd_raw.loc[idx]
        local = price_local.loc[idx]
        currency = currency_raw.loc[idx]
        if listed_usd is None:
            listed_usd = convert_to_usd(
                local,
                country,
                currency,
                config["fx_to_usd"],
                config["country_currency"],
            )
            currency_source = currency or config["country_currency"].get(country, "UNKNOWN")
        else:
            currency_source = "USD"

        sku_name = sku.loc[idx]
        category = infer_category(sku_name, category_raw.loc[idx], categories_allowed)
        month_value = month_to_str(parse_month(month_raw.loc[idx]))
        discount = _compute_discount(local, price_original.loc[idx], discount_raw.loc[idx])
        sold_value = sold.loc[idx]
        gmv_value = gmv.loc[idx]
        if gmv_value is None and listed_usd is not None and sold_value is not None:
            gmv_value = listed_usd * sold_value

        search_text = normalize_text(" ".join(
            part for part in (sku_name, brand.loc[idx], category) if part
        ))

        rows.append(
            {
                "sku_name": sku_name,
                "brand": brand.loc[idx] or "Unknown",
                "category": category,
                "merchant": merchant.loc[idx] or "Unknown",
                "price_usd": listed_usd,
                "discount": discount,
                "sold": sold_value,
                "gmv_usd": gmv_value,
                "rating": rating.loc[idx],
                "reviews": reviews.loc[idx],
                "month": month_value if month_value else pd.NA,
                "country": country,
                "marketplace": marketplace,
                "currency_source": currency_source,
                "price_local": local,
                "is_price_outlier": False,
                "is_mock": bool(is_mock.loc[idx]),
                "source_file": source_file,
                "sku_name_norm": normalize_text(sku_name),
                "search_text": search_text,
            }
        )

    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def flag_price_outliers(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag log-price IQR extremes within category x country. Do not drop them.

    Marketplace catalogs contain luxury SKUs, bundles, and flash sales. Those
    are business cases, not automatic data errors. We keep the rows and mark
    them so EDA and modeling can treat them explicitly.
    """
    out = frame.copy()
    out["is_price_outlier"] = False
    multiplier = float(config["validation"]["outlier_log_iqr_multiplier"])
    if out.empty:
        return out

    prices = pd.to_numeric(out["price_usd"], errors="coerce")
    log_price = np.log(prices.clip(lower=1e-6))
    grouped = out.groupby(["category", "country"], dropna=False)

    flags = pd.Series(False, index=out.index)
    for _, idx in grouped.groups.items():
        subset = log_price.loc[idx].dropna()
        if len(subset) < 12:
            continue
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low = q1 - multiplier * iqr
        high = q3 + multiplier * iqr
        member = log_price.loc[idx]
        flags.loc[idx] = (member < low) | (member > high)

    out["is_price_outlier"] = flags.fillna(False).astype(bool)
    logger.info("Flagged %s statistical price outliers (kept in dataset)", int(out["is_price_outlier"].sum()))
    return out


def clean_canonical(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove invalid rows, optional duplicates; keep flagged outliers."""
    report = validate_canonical_frame(frame, config)
    logger.info("Pre-clean validation: %s", report.as_dict())

    cleaned = frame.copy()
    cleaned["sku_name"] = cleaned["sku_name"].fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned["sku_name"] != ""]

    if config["validation"]["drop_invalid_prices"]:
        cleaned = cleaned[valid_price_mask(cleaned, config)]

    cleaned = cleaned[cleaned["month"].notna() & (cleaned["month"].astype(str).str.strip() != "")]
    cleaned = cleaned[cleaned["country"].isin(config["markets"]["countries"])]
    cleaned = cleaned[cleaned["marketplace"].isin(config["markets"]["marketplaces"])]

    cleaned["sold"] = pd.to_numeric(cleaned["sold"], errors="coerce").clip(lower=0)
    cleaned["reviews"] = pd.to_numeric(cleaned["reviews"], errors="coerce").clip(lower=0)
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned["discount"] = pd.to_numeric(cleaned["discount"], errors="coerce").clip(0, 0.95)
    cleaned["price_usd"] = pd.to_numeric(cleaned["price_usd"], errors="coerce")

    rating_cap = config["validation"]["max_rating"]
    cleaned.loc[cleaned["rating"] > rating_cap, "rating"] = rating_cap
    cleaned.loc[cleaned["rating"] < 0, "rating"] = pd.NA

    identity = ["sku_name", "merchant", "country", "marketplace", "month", "price_usd"]
    if config["validation"]["drop_duplicates"]:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=identity, keep="first")
        logger.info("Dropped %s duplicate listing rows", before - len(cleaned))

    if config["validation"]["flag_statistical_outliers"]:
        cleaned = flag_price_outliers(cleaned, config)

    cleaned = cleaned.reset_index(drop=True)
    post = validate_canonical_frame(cleaned, config)
    logger.info("Post-clean validation: %s", post.as_dict())
    return cleaned


def preprocess_file(path: str | Path, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    file_path = Path(path)
    raw = load_table(file_path)
    mapped = map_to_canonical(raw, config, source_file=file_path.name)
    return clean_canonical(mapped, config)


def save_processed(frame: pd.DataFrame, config: dict | None = None) -> Path:
    config = config or load_config()
    out_path = get_path(config, "cleaned_parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    logger.info("Wrote %s rows to %s", len(frame), out_path)
    return out_path


def run_preprocessing_pipeline(input_path: str | Path | None = None) -> pd.DataFrame:
    """Default Phase-1 entry: mock CSV if present, else local pages-new extract."""
    config = load_config()
    if input_path is not None:
        source = resolve_path(input_path) if not Path(input_path).is_absolute() else Path(input_path)
    else:
        mock_path = get_path(config, "mock_csv")
        magpie_path = get_path(config, "magpie_csv")
        local_path = get_path(config, "local_sample_csv")
        if mock_path.exists():
            source = mock_path
        elif magpie_path.exists():
            source = magpie_path
        elif local_path.exists():
            logger.warning(
                "%s has no month field; rows without parseable months will be dropped. "
                "Prefer the mock dataset or a Magpie extract for temporal modeling.",
                local_path.name,
            )
            source = local_path
        else:
            raise FileNotFoundError(
                "No input found. Generate mock data with `python -m src.generate_mock_data` "
                "or place a Magpie extract at data/raw/marketplace_data.csv."
            )

    cleaned = preprocess_file(source, config)
    save_processed(cleaned, config)
    return cleaned


if __name__ == "__main__":
    run_preprocessing_pipeline()
