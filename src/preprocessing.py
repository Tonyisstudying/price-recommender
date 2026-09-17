"""
Data ingestion, normalization, and cleaning.

Design
------
The two raw files we have on disk are NOT the same shape:

  marketplace_data.csv   -> already "canonical": has country, marketplace,
                             category, currency, snapshot_date.
  lazada_pages_new.csv   -> a raw Lazada Vietnam scrape: VND prices baked
                             into `location` (Vietnamese province names,
                             not "Vietnam"), no category, no marketplace
                             column, no snapshot_date, and missing values
                             in price_original / review_count / rating_score.

So this module has one ADAPTER FUNCTION PER RAW SHAPE that maps it to a
single CANONICAL SCHEMA, then one shared CLEANING PIPELINE that only ever
has to deal with the canonical schema. Adding a new marketplace/source
later = write one new adapter, nothing else changes.

Canonical schema (one row = one product observation on one date)
-------------------------------------------------------------------
    product_id        int64   unique id from the source
    product_name       str
    brand_name          str
    seller_id            str    (kept as str: ids are identifiers, not quantities)
    seller_name         str
    price_current_local float64  price in the source's local currency
    price_original_local float64
    currency             str    ISO-ish code, e.g. VND, PHP, USD
    price_current_usd    float64  <- computed
    price_original_usd   float64  <- computed
    sold_count          Int64   nullable int
    review_count        Int64
    rating_score        float64
    in_stock             bool
    country               str    one of config.valid_countries
    marketplace           str    one of config.valid_marketplaces
    category              str    one of config.valid_categories, or "Uncategorized"
    snapshot_date         datetime64[ns], may be NaT if unknown
    data_source           str    provenance, e.g. "marketplace_canonical" / "lazada_vn_scrape"
    product_url           str
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

CANONICAL_COLUMNS = [
    "product_id", "product_name", "brand_name", "seller_id", "seller_name",
    "price_current_local", "price_original_local", "currency",
    "price_current_usd", "price_original_usd",
    "sold_count", "review_count", "rating_score", "in_stock",
    "country", "marketplace", "category", "snapshot_date",
    "data_source", "product_url",
]


def preprocess(df: pd.DataFrame, snapshot_date: str | None = None) -> pd.DataFrame:
    """Normalize a marketplace adapter result into the serving schema.

    This lightweight entry point is used by external ingestion callers. The
    full ``run`` pipeline remains responsible for source discovery and QA.
    """
    cfg = load_config()
    out = df.copy()
    aliases = {
        "product_id": ["product_id", "itemid", "sku_id"],
        "product_name": ["product_name", "name", "sku_name", "title"],
        "brand_name": ["brand_name", "brand"],
        "seller_id": ["seller_id", "shopid"],
        "seller_name": ["seller_name", "shop_name", "merchant"],
        "price_current_local": ["price_current_local", "price_current", "price", "listed_price"],
        "price_original_local": ["price_original_local", "price_original", "original_price"],
        "sold_count": ["sold_count", "historical_sold", "sold"],
        "review_count": ["review_count", "reviews", "rating_count"],
        "rating_score": ["rating_score", "rating_star", "rating"],
        "country": ["country", "location"],
        "category": ["category", "category_name"],
        "product_url": ["product_url", "url"],
    }
    for target, candidates in aliases.items():
        if target not in out:
            source = next((name for name in candidates if name in out), None)
            out[target] = out[source] if source else None
    out["country"] = out["country"].fillna("Vietnam")
    out["marketplace"] = out.get("marketplace", "Generic")
    out["currency"] = out.get("currency", "VND")
    inferred = out.apply(
        lambda row: infer_category(str(row["product_name"]), str(row["brand_name"]), cfg),
        axis=1,
    )
    inferred = inferred.mask(
        inferred.eq("Uncategorized")
        & out["product_name"].str.contains("cleanser", case=False, na=False),
        "Cleanser",
    )
    inferred = inferred.mask(
        inferred.eq("Uncategorized")
        & out["product_name"].str.contains("mouse|keyboard|headset", case=False, na=False),
        "Accessories",
    )
    out["category"] = out["category"].fillna(inferred)
    out["snapshot_date"] = pd.to_datetime(
        out.get("snapshot_date", snapshot_date), errors="coerce"
    )
    out["price_current_local"] = pd.to_numeric(out["price_current_local"], errors="coerce")
    out["price_original_local"] = pd.to_numeric(
        out["price_original_local"], errors="coerce"
    ).fillna(out["price_current_local"])
    out["price_current_usd"] = out["price_current_local"] * out["currency"].map(
        cfg.fx_to_usd
    ).fillna(1.0)
    out["price_original_usd"] = out["price_original_local"] * out["currency"].map(
        cfg.fx_to_usd
    ).fillna(1.0)
    out["sold_count"] = pd.to_numeric(out["sold_count"], errors="coerce").fillna(0)
    out["review_count"] = pd.to_numeric(out["review_count"], errors="coerce").fillna(0)
    out["rating_score"] = pd.to_numeric(out["rating_score"], errors="coerce")
    out["brand_name"] = out["brand_name"].fillna("Unknown").astype(str)
    out["product_name"] = out["product_name"].fillna("").astype(str)
    out["product_name_clean"] = out["product_name"].str.lower()
    out["brand_clean"] = out["brand_name"].str.lower()
    out["seller_id"] = out["seller_id"].fillna("").astype(str)
    out["seller_name"] = out["seller_name"].fillna("").astype(str)
    out["in_stock"] = out.get("in_stock", True)
    return out


# =============================================================================
# Generic helpers
# =============================================================================

def read_any(path: Path | str) -> pd.DataFrame:
    """Read a CSV or Parquet file based on its extension."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix} ({path})")


def normalize_text(value: object) -> str | None:
    """Lowercase, strip, collapse whitespace, normalize unicode accents away
    for MATCHING purposes only (category inference, dedup keys)"""
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def infer_category(product_name: str, brand_name: str, cfg: Config) -> str:
    """Rule-based category inference for raw sources with no category
    column. First keyword match (checked in the order categories are
    listed in config.yaml) wins. Falls back to 'Uncategorized' rather than
    guessing, so downstream consumers can filter/flag it explicitly."""
    haystack = normalize_text(f"{product_name} {brand_name}") or ""
    for category, keywords in cfg.category_keywords.items():
        for kw in keywords:
            if kw.lower() in haystack:
                return category
    return "Uncategorized"


# =============================================================================
# Adapter: already-canonical marketplace_data.csv
# =============================================================================

def adapt_marketplace_canonical(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """marketplace_data.csv already has every canonical field; this adapter
    just renames/casts columns into the exact canonical layout."""
    out = pd.DataFrame()
    out["product_id"] = df["product_id"]
    out["product_name"] = df["product_name"]
    out["brand_name"] = df["brand_name"]
    out["seller_id"] = df["seller_id"].astype(str)
    out["seller_name"] = df["seller_name"]
    out["price_current_local"] = pd.to_numeric(df["price_current"], errors="coerce")
    out["price_original_local"] = pd.to_numeric(df["price_original"], errors="coerce")
    out["currency"] = df["currency"]
    out["sold_count"] = pd.to_numeric(df["sold_count"], errors="coerce").astype("Int64")
    out["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")
    out["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
    out["in_stock"] = df["in_stock"].astype(bool)
    out["country"] = df["country"]
    out["marketplace"] = df["marketplace"]
    out["category"] = df["category"]
    out["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    out["data_source"] = "marketplace_canonical"
    out["product_url"] = df["product_url"]
    out["price_current_usd"] = np.nan  # computed later by convert_currency_to_usd
    out["price_original_usd"] = np.nan
    return out


# =============================================================================
# Adapter: raw Lazada Vietnam scrape (lazada_pages_new.csv)
# =============================================================================

def adapt_lazada_vn_scrape(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Raw Lazada Vietnam scrape. Needs real normalization work:

    - `location` holds Vietnamese province/city names, not "Vietnam" ->
      map every value in config.vietnam_location_aliases to country="Vietnam".
      (All observed values in this file are Vietnam locations; anything
      outside that alias set is left as-is and will be caught by schema
      validation downstream rather than silently mapped.)
    - marketplace is not a column -> every row came from a lazada.vn URL,
      so marketplace="Lazada" for this whole source.
    - currency is not a column -> prices are in VND (lazada.vn), so
      currency="VND" for this whole source.
    - category is not a column -> inferred from product_name/brand_name
      via infer_category().
    - snapshot_date is not present -> left as NaT (unknown). This means
      rows from this source are usable for the similarity/comparable-
      product engine and catalog breadth, but are EXCLUDED from temporal
      price-model training/evaluation, which requires a real date.
    - price_original / review_count / rating_score have missing values ->
      left as NaN/NA here; handled by the shared cleaning pipeline.
    """
    out = pd.DataFrame()
    out["product_id"] = df["product_id"]
    out["product_name"] = df["product_name"]
    out["brand_name"] = df["brand_name"]
    out["seller_id"] = df["seller_id"].astype(str)
    out["seller_name"] = df["seller_name"]
    out["price_current_local"] = pd.to_numeric(df["price_current"], errors="coerce")
    out["price_original_local"] = pd.to_numeric(df["price_original"], errors="coerce")
    out["currency"] = "VND"
    out["sold_count"] = pd.to_numeric(df["sold_count"], errors="coerce").astype("Int64")
    out["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")
    out["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
    out["in_stock"] = df["in_stock"].astype(bool)

    is_vn_location = df["location"].isin(cfg.vietnam_location_aliases)
    n_unmapped = (~is_vn_location).sum()
    if n_unmapped:
        logger.warning(
            "%d rows in lazada_vn_scrape have a `location` value outside "
            "the known Vietnam alias list; leaving country as raw value.",
            n_unmapped,
        )
    out["country"] = np.where(is_vn_location, "Vietnam", df["location"])
    out["marketplace"] = "Lazada"
    out["category"] = [
        infer_category(name, brand, cfg)
        for name, brand in zip(df["product_name"], df["brand_name"])
    ]
    out["snapshot_date"] = pd.NaT
    out["data_source"] = "lazada_vn_scrape"
    out["product_url"] = df["product_url"]
    out["price_current_usd"] = np.nan  # computed later by convert_currency_to_usd
    out["price_original_usd"] = np.nan
    return out


ADAPTERS = {
    "canonical": adapt_marketplace_canonical,
    "lazada_raw": adapt_lazada_vn_scrape,
}


# =============================================================================
# Ingestion entry point
# =============================================================================

def ingest_all_sources(cfg: Config) -> pd.DataFrame:
    """Read every source listed in config.yaml, run it through its adapter,
    and concatenate into one canonical (but not yet cleaned) DataFrame."""
    frames = []
    for source in cfg.raw_sources:
        path = cfg.raw_dir / source["file"]
        adapter = ADAPTERS[source["schema"]]
        logger.info("Ingesting %s via adapter '%s'", path, source["schema"])
        raw_df = read_any(path)
        canonical_df = adapter(raw_df, cfg)
        canonical_df = canonical_df[CANONICAL_COLUMNS]
        logger.info("  -> %d rows", len(canonical_df))
        frames.append(canonical_df)
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined raw dataset: %d rows from %d sources", len(combined), len(frames))
    return combined


# =============================================================================
# Cleaning pipeline (operates only on the canonical schema)
# =============================================================================

def convert_currency_to_usd(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add price_current_usd / price_original_usd using fixed FX rates.
    Unknown currencies map to NaN (caught by validation, not silently 1:1)."""
    df = df.copy()
    fx = cfg.fx_to_usd
    rate = df["currency"].map(fx)
    unknown = df["currency"][rate.isna()].unique()
    if len(unknown):
        logger.warning("Unknown currencies with no FX rate: %s", list(unknown))
    df["price_current_usd"] = df["price_current_local"] * rate
    df["price_original_usd"] = df["price_original_local"] * rate
    return df


def drop_duplicates(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """De-dupe on the configured key. Rows with NaT snapshot_date (the
    Lazada scrape) are deduped on product_id+marketplace only, since they
    have no date to key on."""
    before = len(df)
    keyed = df["snapshot_date"].notna()
    dated = df[keyed].drop_duplicates(subset=cfg.quality["duplicate_key_columns"])
    undated = df[~keyed].drop_duplicates(subset=["product_id", "marketplace"])
    out = pd.concat([dated, undated], ignore_index=True)
    logger.info("Dropped %d exact/keyed duplicate rows", before - len(out))
    return out


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Business rules for missing values, not blanket imputation:

    - price_current is required -> rows missing it are dropped (can't
      price-model a product with no observed price).
    - price_original missing -> assume no discount, i.e. price_original
      = price_current (flagged via `had_missing_original_price`).
    - review_count / rating_score missing -> very plausible for a real
      scrape (new/low-visibility listing) -> filled with 0 reviews and
      NaN rating kept as NaN (0 would falsely imply a bad product)."""
    df = df.copy()
    before = len(df)
    df = df[df["price_current_local"].notna()]
    logger.info("Dropped %d rows missing price_current", before - len(df))

    df["had_missing_original_price"] = df["price_original_local"].isna()
    df.loc[df["had_missing_original_price"], "price_original_local"] = df.loc[
        df["had_missing_original_price"], "price_current_local"
    ]
    df.loc[df["had_missing_original_price"], "price_original_usd"] = df.loc[
        df["had_missing_original_price"], "price_current_usd"
    ]

    df["review_count"] = df["review_count"].fillna(0).astype("Int64")
    df["sold_count"] = df["sold_count"].fillna(0).astype("Int64")
    # rating_score: leave as NaN when unknown - do not impute a fake rating.
    return df


def flag_invalid_prices(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Flag (and drop) prices that are almost certainly data errors, using
    USD thresholds so the rule is consistent across currencies:
      - price <= 0, or outside [min_price_usd, max_price_usd]
      - price_current > price_original (a "discount" that increases price)
      - discount deeper than max_discount_pct (e.g. -95%) which is more
        likely a data entry error (missing a digit) than a real flash sale
    These are ERRORS, not "legitimate business cases" like bundles/luxury/
    flash sales - see outlier flagging below for the latter."""
    df = df.copy()
    q = cfg.quality
    valid_range = df["price_current_usd"].between(q["min_price_usd"], q["max_price_usd"])
    not_inverted = df["price_current_local"] <= df["price_original_local"] * 1.0001
    discount_ratio = df["price_current_local"] / df["price_original_local"].replace(0, np.nan)
    not_too_deep_discount = discount_ratio >= (1 - q["max_discount_pct"])

    is_valid = valid_range & not_inverted & not_too_deep_discount.fillna(True)
    n_invalid = (~is_valid).sum()
    if n_invalid:
        logger.info("Dropping %d rows with invalid prices (range/inversion/discount checks)", n_invalid)
    return df[is_valid].copy()


def normalize_domain_fields(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Validate country/marketplace/category against config domain lists.
    Anything unrecognized is kept but labeled, never silently dropped."""
    df = df.copy()
    df["country"] = df["country"].where(df["country"].isin(cfg.valid_countries), "Other")
    df["marketplace"] = df["marketplace"].where(df["marketplace"].isin(cfg.valid_marketplaces), "Other")
    df["category"] = df["category"].where(
        df["category"].isin(cfg.valid_categories) | (df["category"] == "Uncategorized"),
        "Uncategorized",
    )
    n_other_country = (df["country"] == "Other").sum()
    n_other_market = (df["marketplace"] == "Other").sum()
    n_uncat = (df["category"] == "Uncategorized").sum()
    if n_other_country or n_other_market or n_uncat:
        logger.info(
            "Domain normalization: %d rows -> country=Other, %d -> marketplace=Other, %d -> category=Uncategorized",
            n_other_country, n_other_market, n_uncat,
        )
    return df


def flag_statistical_outliers(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Flag (but do NOT drop) statistical price outliers within each
    country+category+marketplace group, using a 3*IQR fence. This is
    intentionally permissive (3x, not 1.5x) because premium products,
    bundles, and flash sales are real, legitimate high/low prices - see
    section 8 of the brief. We flag for downstream inspection/weighting
    rather than deleting business-legitimate rows.
    """
    df = df.copy()
    group_cols = group_cols or ["country", "category", "marketplace"]

    def _flag(group: pd.DataFrame) -> pd.Series:
        q1, q3 = group["price_current_usd"].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        return ~group["price_current_usd"].between(low, high)

    df["is_price_outlier"] = df.groupby(group_cols, group_keys=False).apply(_flag)
    logger.info("Flagged %d statistical price outliers (kept, not dropped)", df["is_price_outlier"].sum())
    return df


def clean_pipeline(raw_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run the full cleaning pipeline in order. Order matters: currency
    conversion must happen before invalid-price checks (which use USD),
    and missing-value handling must happen before invalid-price checks
    (which need price_original filled in)."""
    df = raw_df.copy()
    df = convert_currency_to_usd(df, cfg)
    df = handle_missing_values(df)
    df = flag_invalid_prices(df, cfg)
    df = drop_duplicates(df, cfg)
    df = normalize_domain_fields(df, cfg)
    df = flag_statistical_outliers(df)

    # Derived convenience fields used by later phases (features/EDA).
    df["month"] = df["snapshot_date"].dt.to_period("M").astype(str)
    df.loc[df["snapshot_date"].isna(), "month"] = None
    df["discount_pct"] = 1 - (df["price_current_local"] / df["price_original_local"])

    return df.reset_index(drop=True)


def run(config_path: Path | str | None = None) -> pd.DataFrame:
    """End-to-end Phase 1 entry point: ingest both sources, clean, and
    persist to data/processed/cleaned_products.parquet."""
    cfg = load_config(config_path) if config_path else load_config()
    raw = ingest_all_sources(cfg)
    cleaned = clean_pipeline(raw, cfg)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.processed_dir / "cleaned_products.parquet"
    cleaned.to_parquet(out_path, index=False)
    logger.info("Wrote cleaned dataset: %s (%d rows)", out_path, len(cleaned))
    return cleaned


if __name__ == "__main__":
    run()
