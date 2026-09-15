"""Generate a clearly labeled MOCK SEA marketplace catalog.

This is not Magpie data and must not be treated as real transactions.
It exists so ingestion, EDA, competitor stats, and later model code can be
tested before a licensed extract is available.

Coverage:
    * 6 SEA countries
    * Shopee and Lazada
    * 5 categories
    * multiple brands and merchants
    * March 2026 – June 2026
    * listed prices, discounts, sold, reviews, ratings
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.utils import get_logger

logger = get_logger(__name__)

# Country listed-price multipliers vs a USD-like base (Singapore ~ 1.00).
COUNTRY_PRICE_FACTOR = {
    "Singapore": 1.00,
    "Malaysia": 0.72,
    "Thailand": 0.70,
    "Vietnam": 0.62,
    "Indonesia": 0.60,
    "Philippines": 0.64,
}

MARKETPLACE_PRICE_FACTOR = {
    "Shopee": 0.98,
    "Lazada": 1.03,
}

CATALOG: list[dict[str, object]] = [
    # Accessories
    {"sku_name": "Logitech M331 Wireless Mouse", "brand": "Logitech", "category": "Accessories", "base_price": 12.5},
    {"sku_name": "Logitech M331 Silent Mouse", "brand": "Logitech", "category": "Accessories", "base_price": 13.0},
    {"sku_name": "Logitech Wireless M331 Mouse", "brand": "Logitech", "category": "Accessories", "base_price": 12.4},
    {"sku_name": "Logitech M331 Bluetooth Mouse", "brand": "Logitech", "category": "Accessories", "base_price": 13.2},
    {"sku_name": "Logitech M170 Wireless Mouse", "brand": "Logitech", "category": "Accessories", "base_price": 10.5},
    {"sku_name": "Rapoo M10 Wireless Mouse", "brand": "Rapoo", "category": "Accessories", "base_price": 8.0},
    {"sku_name": "Xiaomi Wireless Mouse Lite", "brand": "Xiaomi", "category": "Accessories", "base_price": 9.5},
    {"sku_name": "Anker USB-C to USB-C Cable 1m", "brand": "Anker", "category": "Accessories", "base_price": 8.9},
    {"sku_name": "Baseus USB-C Fast Charging Cable 2m", "brand": "Baseus", "category": "Accessories", "base_price": 7.5},
    {"sku_name": "Baseus Magnetic Car Phone Holder", "brand": "Baseus", "category": "Accessories", "base_price": 11.0},
    {"sku_name": "Ugreen USB-C Hub 6-in-1", "brand": "Ugreen", "category": "Accessories", "base_price": 24.0},
    {"sku_name": "Spigen Tough Armor Phone Case", "brand": "Spigen", "category": "Accessories", "base_price": 16.0},
    {"sku_name": "JSAUX USB-C Cable 1m", "brand": "JSAUX", "category": "Accessories", "base_price": 6.8},
    {"sku_name": "Aukey 30W USB-C Charger", "brand": "Aukey", "category": "Accessories", "base_price": 14.5},
    {"sku_name": "Ringke Fusion Phone Case", "brand": "Ringke", "category": "Accessories", "base_price": 12.0},
    # Groceries
    {"sku_name": "Jasmine Rice 5kg Premium", "brand": "Dragon", "category": "Groceries", "base_price": 9.8},
    {"sku_name": "Thai Hom Mali Rice 5kg", "brand": "Royal Umbrella", "category": "Groceries", "base_price": 11.5},
    {"sku_name": "Indomie Mi Goreng 5 Pack", "brand": "Indomie", "category": "Groceries", "base_price": 3.2},
    {"sku_name": "Nissin Cup Noodles Seafood", "brand": "Nissin", "category": "Groceries", "base_price": 1.4},
    {"sku_name": "Oreo Chocolate Sandwich 137g", "brand": "Oreo", "category": "Groceries", "base_price": 1.9},
    {"sku_name": "Kopiko Instant Coffee 3in1", "brand": "Kopiko", "category": "Groceries", "base_price": 4.5},
    {"sku_name": "Nescafe Classic Instant Coffee 100g", "brand": "Nescafe", "category": "Groceries", "base_price": 5.8},
    {"sku_name": "Dutch Lady Full Cream Milk 1L", "brand": "Dutch Lady", "category": "Groceries", "base_price": 1.8},
    {"sku_name": "Knife Cooking Oil 2L", "brand": "Knife", "category": "Groceries", "base_price": 4.2},
    {"sku_name": "Maggi Chili Sauce 500g", "brand": "Maggi", "category": "Groceries", "base_price": 2.4},
    {"sku_name": "Knorr Chicken Powder 1kg", "brand": "Knorr", "category": "Groceries", "base_price": 6.1},
    {"sku_name": "Yeo's Chrysanthemum Tea 6 Pack", "brand": "Yeo's", "category": "Groceries", "base_price": 3.6},
    {"sku_name": "Pocari Sweat 500ml 12 Pack", "brand": "Pocari", "category": "Groceries", "base_price": 8.4},
    {"sku_name": "Milo Activ-Go 1kg", "brand": "Milo", "category": "Groceries", "base_price": 9.2},
    {"sku_name": "Gardenia White Bread 400g", "brand": "Gardenia", "category": "Groceries", "base_price": 1.5},
    # Beauty
    {"sku_name": "Paula's Choice Skin Perfecting 2% BHA Liquid", "brand": "Paula's Choice", "category": "Beauty", "base_price": 32.0},
    {"sku_name": "Paula's Choice C15 Super Booster", "brand": "Paula's Choice", "category": "Beauty", "base_price": 52.0},
    {"sku_name": "CeraVe Hydrating Facial Cleanser 236ml", "brand": "CeraVe", "category": "Beauty", "base_price": 14.0},
    {"sku_name": "CeraVe PM Facial Moisturizing Lotion", "brand": "CeraVe", "category": "Beauty", "base_price": 16.5},
    {"sku_name": "La Roche-Posay Anthelios SPF50", "brand": "La Roche-Posay", "category": "Beauty", "base_price": 22.0},
    {"sku_name": "The Ordinary Niacinamide 10% + Zinc 1%", "brand": "The Ordinary", "category": "Beauty", "base_price": 8.5},
    {"sku_name": "Innisfree Green Tea Seed Serum", "brand": "Innisfree", "category": "Beauty", "base_price": 19.0},
    {"sku_name": "COSRX Advanced Snail 96 Mucin Essence", "brand": "COSRX", "category": "Beauty", "base_price": 18.5},
    {"sku_name": "Hada Labo Gokujyun Lotion", "brand": "Hada Labo", "category": "Beauty", "base_price": 12.8},
    {"sku_name": "SK-II Facial Treatment Essence 75ml", "brand": "SK-II", "category": "Beauty", "base_price": 99.0},
    {"sku_name": "Nivea Soft Moisturizing Cream 200ml", "brand": "Nivea", "category": "Beauty", "base_price": 5.5},
    {"sku_name": "Garnier Micellar Water 400ml", "brand": "Garnier", "category": "Beauty", "base_price": 7.2},
    {"sku_name": "L'Oreal Elseve Total Repair Shampoo", "brand": "L'Oreal", "category": "Beauty", "base_price": 6.8},
    {"sku_name": "Maybelline Fit Me Foundation", "brand": "Maybelline", "category": "Beauty", "base_price": 11.4},
    {"sku_name": "Vaseline Repairing Jelly 50ml", "brand": "Vaseline", "category": "Beauty", "base_price": 3.1},
    # Electronics
    {"sku_name": "Xiaomi Redmi Buds 6", "brand": "Xiaomi", "category": "Electronics", "base_price": 28.0},
    {"sku_name": "Anker Soundcore Life P2i", "brand": "Anker", "category": "Electronics", "base_price": 25.0},
    {"sku_name": "Sony WH-CH520 Headphones", "brand": "Sony", "category": "Electronics", "base_price": 48.0},
    {"sku_name": "JBL Tune 510BT Headphones", "brand": "JBL", "category": "Electronics", "base_price": 39.0},
    {"sku_name": "Samsung USB-C Earphones", "brand": "Samsung", "category": "Electronics", "base_price": 15.0},
    {"sku_name": "TP-Link Archer AX10 Router", "brand": "TP-Link", "category": "Electronics", "base_price": 55.0},
    {"sku_name": "SanDisk Ultra 128GB Flash Drive", "brand": "SanDisk", "category": "Electronics", "base_price": 14.0},
    {"sku_name": "Logitech H111 Headset", "brand": "Logitech", "category": "Electronics", "base_price": 12.0},
    {"sku_name": "Baseus 20W Power Bank 10000mAh", "brand": "Baseus", "category": "Electronics", "base_price": 22.0},
    {"sku_name": "Anker 737 Power Bank", "brand": "Anker", "category": "Electronics", "base_price": 89.0},
    {"sku_name": "Xiaomi Smart Band 9", "brand": "Xiaomi", "category": "Electronics", "base_price": 35.0},
    {"sku_name": "Apple Lightning Cable 1m", "brand": "Apple", "category": "Electronics", "base_price": 19.0},
    {"sku_name": "Realme Buds T300", "brand": "Realme", "category": "Electronics", "base_price": 27.0},
    {"sku_name": "Oppo Enco Air3", "brand": "Oppo", "category": "Electronics", "base_price": 29.0},
    {"sku_name": "Huawei FreeBuds SE", "brand": "Huawei", "category": "Electronics", "base_price": 31.0},
    # Home
    {"sku_name": "LocknLock Vacuum Bottle 500ml", "brand": "LocknLock", "category": "Home", "base_price": 12.0},
    {"sku_name": "Tefal Non-Stick Frying Pan 24cm", "brand": "Tefal", "category": "Home", "base_price": 28.0},
    {"sku_name": "Philips Daily Collection Kettle", "brand": "Philips", "category": "Home", "base_price": 32.0},
    {"sku_name": "Xiaomi Mi Electric Kettle 2", "brand": "Xiaomi", "category": "Home", "base_price": 29.0},
    {"sku_name": "Midea 2L Rice Cooker", "brand": "Midea", "category": "Home", "base_price": 34.0},
    {"sku_name": "Sharp Air Purifier Compact", "brand": "Sharp", "category": "Home", "base_price": 95.0},
    {"sku_name": "Dyson V8 Absolute Vacuum", "brand": "Dyson", "category": "Home", "base_price": 320.0},
    {"sku_name": "Panasonic Nanoe Hair Dryer", "brand": "Panasonic", "category": "Home", "base_price": 42.0},
    {"sku_name": "LocknLock Food Container Set", "brand": "LocknLock", "category": "Home", "base_price": 16.0},
    {"sku_name": "3M Scotch-Brite Sponge 8 Pack", "brand": "3M", "category": "Home", "base_price": 4.5},
    {"sku_name": "Camelbak Water Bottle 750ml", "brand": "Camelbak", "category": "Home", "base_price": 18.0},
    {"sku_name": "Zojirushi Rice Cooker 1.0L", "brand": "Zojirushi", "category": "Home", "base_price": 140.0},
    {"sku_name": "Ikea LED Desk Lamp", "brand": "Ikea", "category": "Home", "base_price": 15.0},
    {"sku_name": "Muji Aroma Diffuser", "brand": "Muji", "category": "Home", "base_price": 38.0},
    {"sku_name": "Unicharm Wet Wipes 80s", "brand": "Unicharm", "category": "Home", "base_price": 3.8},
]


MERCHANT_POOL = [
    "Official Store",
    "SEA Digital Mart",
    "GadgetHub SG",
    "Indo SuperMart",
    "VN Value Shop",
    "Thai Daily Goods",
    "KL Best Deals",
    "Manila Express",
    "Lazada Mall Partner",
    "Shopee Preferred",
    "Family Mart Online",
    "TechCorner",
    "BeautyLane",
    "GroceryGo",
    "HomeNest",
    "Mouse World",
    "Cable King",
    "Daily Essentials",
]


def _month_drift(month: str) -> float:
    # Mild inflation / promo seasonality across Mar–Jun 2026.
    return {
        "2026-03": 1.00,
        "2026-04": 1.01,
        "2026-05": 0.97,  # mid-year promo dip
        "2026-06": 1.02,
    }[month]


def generate_mock_catalog(config: dict | None = None, rng: np.random.Generator | None = None) -> pd.DataFrame:
    config = config or load_config()
    mock_cfg = config["mock"]
    rng = rng or np.random.default_rng(int(mock_cfg["seed"]))

    countries = list(config["markets"]["countries"])
    marketplaces = list(config["markets"]["marketplaces"])
    months = list(mock_cfg["months"])
    merchants = MERCHANT_POOL[: int(mock_cfg["n_merchants"])]

    catalog = CATALOG
    if mock_cfg.get("n_base_skus"):
        catalog = catalog[: int(mock_cfg["n_base_skus"])]

    records: list[dict[str, object]] = []
    for sku in catalog:
        # Each SKU appears in a subset of country-marketplace cells to mimic coverage gaps.
        n_cells = int(rng.integers(4, 8))
        cells = [
            (str(countries[i]), str(marketplaces[j]))
            for i, j in zip(
                rng.integers(0, len(countries), size=n_cells),
                rng.integers(0, len(marketplaces), size=n_cells),
            )
        ]
        cells = list(dict.fromkeys(cells))
        for country, marketplace in cells:
            sellers = list(rng.choice(merchants, size=int(rng.integers(2, 5)), replace=False))
            for merchant in sellers:
                for month in months:
                    if rng.random() < 0.08:
                        continue  # some listings vanish in a given month
                    base = float(sku["base_price"])
                    factor = (
                        COUNTRY_PRICE_FACTOR[country]
                        * MARKETPLACE_PRICE_FACTOR[marketplace]
                        * _month_drift(month)
                    )
                    noise = float(rng.normal(1.0, 0.07))
                    price = max(0.4, base * factor * noise)
                    discount = float(np.clip(rng.beta(1.2, 8.0) * 0.45, 0.0, 0.55))
                    if rng.random() < 0.12:
                        discount = float(rng.uniform(0.25, 0.50))  # flash sale
                    rating = float(np.clip(rng.normal(4.55, 0.28), 3.2, 5.0))
                    reviews = int(max(0, rng.lognormal(mean=3.4, sigma=1.1)))
                    # Lower price relative to base tends to sell more — observational, not causal.
                    relative = price / (base * factor)
                    sold = int(max(0, rng.lognormal(mean=3.8 - 1.6 * (relative - 1.0), sigma=1.0)))
                    records.append(
                        {
                            "sku_name": sku["sku_name"],
                            "brand": sku["brand"],
                            "category": sku["category"],
                            "merchant": merchant,
                            "price_usd": round(price, 2),
                            "discount": round(discount, 4),
                            "sold": sold,
                            "gmv_usd": round(price * sold, 2),
                            "rating": round(rating, 2),
                            "reviews": reviews,
                            "month": month,
                            "country": country,
                            "marketplace": marketplace,
                            "currency": "USD",
                            "is_mock": True,
                            "data_disclaimer": "MOCK DATA — not real marketplace observations",
                        }
                    )

    frame = pd.DataFrame.from_records(records)
    logger.info(
        "Generated MOCK dataset: %s rows, %s skus, countries=%s, marketplaces=%s, months=%s",
        len(frame),
        frame["sku_name"].nunique(),
        sorted(frame["country"].unique()),
        sorted(frame["marketplace"].unique()),
        sorted(frame["month"].unique()),
    )
    return frame


def write_mock_dataset(output_path: str | Path | None = None) -> Path:
    config = load_config()
    frame = generate_mock_catalog(config)
    path = Path(output_path) if output_path is not None else get_path(config, "mock_csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote MOCK CSV to %s (%s rows). Do not treat as real Magpie data.", path, len(frame))
    return path


if __name__ == "__main__":
    write_mock_dataset()
