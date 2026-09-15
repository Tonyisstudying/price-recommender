from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

COUNTRIES = [
    "Vietnam", "Thailand", "Malaysia", "Singapore", "Indonesia", "Philippines"
]
MARKETPLACES = ["Shopee", "Lazada"]

CATALOG = {
    "Accessories": [
        ("Logitech", ["Wireless Mouse", "Mechanical Keyboard", "Webcam"]),
        ("Anker", ["USB-C Hub", "Charging Cable", "Power Bank"]),
        ("Baseus", ["USB-C Cable", "Laptop Stand", "Phone Holder"]),
    ],
    "Groceries": [
        ("Oishi", ["Green Tea", "Snack Mix", "Crackers"]),
        ("Nestle", ["Coffee", "Cereal", "Chocolate Drink"]),
        ("Kikkoman", ["Soy Sauce", "Teriyaki Sauce"]),
    ],
    "Beauty": [
        ("L'Oreal", ["Shampoo", "Hair Serum", "Face Wash"]),
        ("Innisfree", ["Clay Mask", "Moisturizer", "Cleanser"]),
        ("Maybelline", ["Mascara", "Foundation", "Lip Tint"]),
    ],
    "Electronics": [
        ("Xiaomi", ["Smart Band", "Bluetooth Speaker", "Router"]),
        ("JBL", ["Bluetooth Speaker", "Earbuds", "Headphones"]),
        ("Samsung", ["SSD", "Monitor", "Tablet"]),
    ],
    "Home & Living": [
        ("IKEA", ["Desk Lamp", "Storage Box", "Kitchen Organizer"]),
        ("LocknLock", ["Water Bottle", "Food Container", "Lunch Box"]),
        ("Philips", ["LED Bulb", "Air Fryer", "Desk Fan"]),
    ],
}

COUNTRY_MULTIPLIER = {
    "Vietnam": 1.00,
    "Thailand": 1.04,
    "Malaysia": 1.03,
    "Singapore": 1.23,
    "Indonesia": 0.96,
    "Philippines": 1.01,
}

CATEGORY_PRICE = {
    "Accessories": (6, 80),
    "Groceries": (2, 35),
    "Beauty": (4, 55),
    "Electronics": (25, 450),
    "Home & Living": (8, 160),
}


def generate(rows_per_month: int = 800) -> pd.DataFrame:
    rows: list[dict] = []
    months = pd.date_range("2026-03-01", "2026-06-01", freq="MS")

    for month in months:
        for _ in range(rows_per_month):
            country = random.choice(COUNTRIES)
            marketplace = random.choice(MARKETPLACES)
            category = random.choice(list(CATALOG))
            brand, products = random.choice(CATALOG[category])
            product_type = random.choice(products)
            variant = random.choice(["Standard", "2026", "Pro", "Plus", "Mini", "Max", ""])
            product_name = f"{brand} {product_type} {variant}".strip()

            low, high = CATEGORY_PRICE[category]
            base_price = np.random.uniform(low, high)
            price = base_price * COUNTRY_MULTIPLIER[country] * np.random.lognormal(0, 0.08)
            discount = float(np.clip(np.random.normal(12, 8), 0, 50))
            rating = float(np.clip(np.random.normal(4.55, 0.35), 2.5, 5.0))
            reviews = int(np.random.lognormal(5.0, 1.2))

            base_demand = 1200 / (1 + price**0.60)
            marketplace_effect = 1.15 if marketplace == "Shopee" else 0.95
            category_effect = {
                "Accessories": 1.0,
                "Groceries": 1.35,
                "Beauty": 1.15,
                "Electronics": 0.55,
                "Home & Living": 0.75,
            }[category]
            promo_effect = 1 + discount / 100
            sold = int(np.random.poisson(max(
                base_demand * marketplace_effect * category_effect * promo_effect, 1
            )))

            rows.append({
                "sku_name": product_name,
                "brand": brand,
                "category": category,
                "merchant": f"{brand.lower().replace(' ', '-')}-seller-{random.randint(1,60):03d}",
                "price_usd": round(float(price), 2),
                "discount": round(discount, 2),
                "sold": sold,
                "gmv_usd": round(float(price * sold), 2),
                "rating": round(rating, 2),
                "reviews": reviews,
                "month": month.strftime("%Y-%m-%d"),
                "country": country,
                "marketplace": marketplace,
            })

    return pd.DataFrame(rows)


def main() -> None:
    df = generate()
    output = Path("data/raw/marketplace_data.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df):,} synthetic rows -> {output}")


if __name__ == "__main__":
    main()
