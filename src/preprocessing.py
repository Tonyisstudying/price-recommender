from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .config import CONFIG
from .utils import clean_text
from .validation import validate_schema, validate_business_rules


COLUMN_ALIASES = {
    "product_id": "product_id",
    "id": "product_id",
    "itemid": "product_id",
    "product_name": "product_name",
    "sku_name": "product_name",
    "title": "product_name",
    "product_title": "product_name",
    "name": "product_name",
    "brand_id": "brand_id",
    "brand_name": "brand_name",
    "brand": "brand_name",
    "seller_id": "seller_id",
    "shopid": "seller_id",
    "seller_name": "seller_name",
    "seller": "seller_name",
    "store": "seller_name",
    "store_name": "seller_name",
    "shop_name": "seller_name",
    "shop": "seller_name",
    "price_current": "price_current",
    "price": "price_current",
    "current_price": "price_current",
    "sale_price": "price_current",
    "selling_price": "price_current",
    "price_original": "price_original",
    "original_price": "price_original",
    "list_price": "price_original",
    "sold_count": "sold_count",
    "units_sold": "sold_count",
    "sales": "sold_count",
    "historical_sold": "sold_count",
    "review_count": "review_count",
    "reviews": "review_count",
    "rating_score": "rating_score",
    "rating": "rating_score",
    "rating_star": "rating_score",
    "in_stock": "in_stock",
    "location": "location",
    "country": "country",
    "country_name": "country",
    "marketplace": "marketplace",
    "platform_name": "marketplace",
    "platform": "marketplace",
    "site": "marketplace",
    "category": "category",
    "subcategory": "subcategory",
    "snapshot_date": "snapshot_date",
    "date": "snapshot_date",
    "scrape_date": "snapshot_date",
    "product_url": "product_url",
    "url": "product_url",
    "currency": "currency",
    "currency_code": "currency",
}

NUMERIC_COLUMNS = [
    "price_current",
    "price_original",
    "sold_count",
    "review_count",
    "rating_score",
]

CATEGORY_RULES = [
    ("Cleanser", ["cleanser", "cleansing", "face wash", "wash" ]),
    ("Serum", ["serum", "ampoule"]),
    ("Moisturizer", ["moisturizer", "moisturiser", "cream", "gel cream", "hydrating cream"]),
    ("Sunscreen", ["sunscreen", "sun screen", "spf"]),
    ("Lip Care", ["lip balm", "lip care", "lip treatment"]),
    ("Haircare", ["shampoo", "conditioner", "hair mask", "hair serum"]),
    ("Makeup", ["mascara", "foundation", "lipstick", "lip tint", "blush", "concealer"]),
    ("Accessories", ["mouse", "keyboard", "cable", "charger", "holder", "stand", "power bank"]),
    ("Groceries", ["coffee", "tea", "snack", "cereal", "chocolate", "sauce", "milk"]),
    ("Electronics", ["tablet", "monitor", "router", "speaker", "earbuds", "headphones", "ssd"]),
    ("Home & Living", ["lamp", "storage", "container", "bottle", "air fryer", "fan"]),
    ("Cleanser", ["cleanser", "face wash", "facial wash"]),
    ("Serum", ["serum", "ampoule"]),
    ("Toner", ["toner", "essence"]),
    ("Sunscreen", ["sunscreen", "spf"]),
    ("Mouse", ["wireless mouse", "mouse"]),
    ("Keyboard", ["keyboard"]),
    ("Headphones", ["headphones", "headset", "earbuds"]),
    ("Speaker", ["bluetooth speaker", "speaker"]),
    ("Coffee", ["coffee", "espresso"]),
    ("Tea", ["tea", "green tea"]),
    ("Snacks", ["snack", "chips", "cracker"]),
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lookup = {str(c).strip().lower(): c for c in out.columns}
    rename: dict[str, str] = {}
    for alias, canonical in COLUMN_ALIASES.items():
        actual = lookup.get(alias)
        if actual is not None and canonical not in out.columns:
            rename[actual] = canonical
    return out.rename(columns=rename)


def infer_country(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "country" not in out:
        out["country"] = ""
    out["country"] = out["country"].fillna("").astype(str).str.strip()
    if "location" in out.columns:
        location = out["location"].fillna("").astype(str).str.lower()
        location_map = {
            "vietnam": "Vietnam", "viet nam": "Vietnam", "hanoi": "Vietnam", "ho chi minh": "Vietnam",
            "thailand": "Thailand", "bangkok": "Thailand",
            "malaysia": "Malaysia", "kuala lumpur": "Malaysia",
            "singapore": "Singapore",
            "indonesia": "Indonesia", "jakarta": "Indonesia",
            "philippines": "Philippines", "manila": "Philippines",
        }
        inferred = location.map(location_map)
        out.loc[out["country"].eq(""), "country"] = inferred[out["country"].eq("")]
    out["country"] = out["country"].replace("", pd.NA).fillna(CONFIG["data"]["default_country"])
    out["country"] = out["country"].astype(str).str.strip()
    return out


def infer_marketplace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "marketplace" not in out:
        out["marketplace"] = ""
    out["marketplace"] = out["marketplace"].fillna("").astype(str).str.strip()
    out["marketplace"] = out["marketplace"].str.lower().map({"lazada": "Lazada", "shopee": "Shopee", "tiktok": "TikTok Shop", "tokopedia": "Tokopedia"}).fillna(out["marketplace"])
    url = out.get("product_url", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    out.loc[out["marketplace"].eq(""), "marketplace"] = url[out["marketplace"].eq("")].str.extract(r"(lazada|shopee|tiktok|tokopedia)", expand=False)
    out["marketplace"] = out["marketplace"].fillna("").str.lower().map({"lazada": "Lazada", "shopee": "Shopee", "tiktok": "TikTok Shop", "tokopedia": "Tokopedia"}).fillna(out["marketplace"])
    out["marketplace"] = out["marketplace"].replace("", CONFIG["data"]["default_marketplace"])
    return out


def infer_currency(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = CONFIG["supported"]["currency_by_country"]
    if "currency" not in out:
        out["currency"] = ""
    out["currency"] = out["currency"].fillna("").astype(str).str.upper().str.strip()
    for country, currency in mapping.items():
        mask = out["country"].eq(country) & out["currency"].eq("")
        out.loc[mask, "currency"] = currency
    out.loc[out["currency"].eq(""), "currency"] = "USD"
    return out


def infer_category(name: str) -> str:
    text = str(name).lower()
    for category, keywords in CATEGORY_RULES:
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords):
            return category
    return "Other"


def infer_category_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "category" not in out:
        out["category"] = ""
    out["category"] = out["category"].fillna("").astype(str).str.strip()
    mask = out["category"].eq("")
    out.loc[mask, "category"] = out.loc[mask, "product_name"].map(infer_category)
    return out


def infer_snapshot_date(df: pd.DataFrame, snapshot_date: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if "snapshot_date" not in out:
        out["snapshot_date"] = pd.NaT
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    if snapshot_date is not None:
        supplied = pd.Timestamp(snapshot_date)
        out["snapshot_date"] = out["snapshot_date"].fillna(supplied)
    return out


def apply_currency_conversion(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fx = CONFIG["supported"]["fx_to_usd"]
    out["fx_to_usd"] = out["currency"].map(fx)
    out["fx_to_usd"] = out["fx_to_usd"].fillna(1.0)
    out["price_usd"] = out["price_local"] * out["fx_to_usd"]
    out["price_original_usd"] = out["price_original_local"] * out["fx_to_usd"]
    return out


def preprocess(df: pd.DataFrame, snapshot_date: str | None = None) -> pd.DataFrame:
    df = normalize_column_names(df)
    validate_schema(df)

    df = infer_country(df)
    df = infer_marketplace(df)

    for col in ["product_id", "brand_id", "seller_id", "product_name", "brand_name", "seller_name", "location", "product_url"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in NUMERIC_COLUMNS:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["price_original"] = df["price_original"].fillna(df["price_current"])
    df["sold_count"] = df["sold_count"].fillna(0).clip(lower=0)
    df["review_count"] = df["review_count"].fillna(0).clip(lower=0)
    df["rating_score"] = df["rating_score"].fillna(df["rating_score"].median()).fillna(0).clip(0, 5)
    if "in_stock" not in df.columns:
        df["in_stock"] = True
    df["in_stock"] = (
        df["in_stock"].astype(str).str.lower().map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(True)
    )

    df["price_local"] = df["price_current"]
    df["price_original_local"] = df["price_original"]
    df["discount_rate"] = ((df["price_original_local"] - df["price_local"]) / df["price_original_local"].replace(0, pd.NA)).fillna(0).clip(0, 1)

    df = infer_category_column(df)
    df = infer_currency(df)
    df = infer_snapshot_date(df, snapshot_date=snapshot_date)
    df = apply_currency_conversion(df)

    df = df[df["product_name"].ne("") & df["price_local"].notna() & (df["price_local"] > 0)].copy()
    df = df.dropna(subset=["snapshot_date"])

    df["product_name_clean"] = df["product_name"].apply(clean_text)
    df["brand_clean"] = df["brand_name"].apply(clean_text)
    df["category_clean"] = df["category"].apply(clean_text)

    # One product/seller/listing per snapshot.
    key = ["product_id", "seller_id", "snapshot_date", "marketplace", "country"]
    fallback_key = ["product_name_clean", "seller_name", "snapshot_date", "marketplace", "country"]
    dedupe = key if df["product_id"].ne("").any() else fallback_key
    df = df.drop_duplicates(subset=dedupe, keep="last")

    validate_business_rules(df)
    return df.reset_index(drop=True)


def load_raw_data(path: str | Path, snapshot_date: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    return preprocess(df, snapshot_date=snapshot_date)


def save_processed(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path
