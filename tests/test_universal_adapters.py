import pandas as pd
from src.adapters import get_adapter
from src.preprocessing import preprocess


def test_lazada_adapter():
    df = pd.DataFrame([{
        "product_id": "1",
        "product_name": "Paula's Choice Cleanser",
        "brand_name": "Paula's Choice",
        "seller_id": "2",
        "seller_name": "Store",
        "price_current": 650000,
        "price_original": 700000,
        "sold_count": 100,
        "review_count": 20,
        "rating_score": 4.8,
        "in_stock": True,
        "location": "Vietnam",
    }])
    out = preprocess(
        get_adapter("lazada").normalize(df),
        snapshot_date="2026-09-16",
    )
    assert out.iloc[0]["marketplace"] == "Lazada"
    assert out.iloc[0]["currency"] == "VND"
    assert out.iloc[0]["category"] == "Cleanser"


def test_shopee_adapter():
    df = pd.DataFrame([{
        "itemid": "2",
        "name": "Logitech Wireless Mouse",
        "brand": "Logitech",
        "shopid": "3",
        "shop_name": "Store",
        "price": 500000,
        "historical_sold": 100,
        "rating_star": 4.7,
        "country": "Vietnam",
    }])
    out = preprocess(
        get_adapter("shopee").normalize(df),
        snapshot_date="2026-09-16",
    )
    assert out.iloc[0]["marketplace"] == "Shopee"
    assert out.iloc[0]["product_id"] == "2"
    assert out.iloc[0]["category"] == "Accessories"
