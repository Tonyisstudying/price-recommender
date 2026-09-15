import pandas as pd
from src.preprocessing import preprocess


def test_preprocess():
    df = pd.DataFrame([{
        "sku_name": "Test Product", "brand": "Test", "category": "Accessories",
        "merchant": "seller", "price_usd": "10.5", "discount": None,
        "sold": None, "gmv_usd": 100, "rating": None, "reviews": None,
        "month": "2026-03-01", "country": "Vietnam", "marketplace": "Shopee",
    }])
    out = preprocess(df)
    assert out.iloc[0]["price_usd"] == 10.5
    assert out.iloc[0]["month"].year == 2026
    assert out.iloc[0]["discount"] == 0
    assert out.iloc[0]["sold"] == 0
