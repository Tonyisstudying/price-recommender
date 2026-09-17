import pandas as pd

from src.data_manager import merge_datasets
from src.preprocessing import preprocess


def make_row(date):
    return preprocess(pd.DataFrame([{
        "product_id": "1",
        "product_name": "Test Cleanser",
        "brand_name": "Test",
        "seller_id": "2",
        "seller_name": "Seller",
        "price_current": 100000,
        "price_original": 120000,
        "sold_count": 10,
        "review_count": 3,
        "in_stock": True,
        "rating_score": 5,
        "location": "Vietnam",
        "product_url": "https://lazada.vn/x",
        "snapshot_date": date,
        "marketplace": "Lazada",
        "country": "Vietnam",
    }]))


def test_same_listing_same_snapshot_is_deduplicated():
    a = make_row("2026-09-01")
    b = make_row("2026-09-01")
    merged, report = merge_datasets(a, b)
    assert len(merged) == 1
    assert report["duplicates_removed"] == 1


def test_new_snapshot_is_kept():
    a = make_row("2026-09-01")
    b = make_row("2026-10-01")
    merged, _ = merge_datasets(a, b)
    assert len(merged) == 2
