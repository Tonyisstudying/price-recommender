import pandas as pd
from src.competitor import price_distribution


def test_price_distribution():
    stats = price_distribution(pd.DataFrame({"price_usd": [10, 11, 12, 13, 14]}))
    assert stats["median"] == 12
    assert stats["p25"] == 11
    assert stats["p75"] == 13
    assert stats["count"] == 5
