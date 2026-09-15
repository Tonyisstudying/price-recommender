import pytest
from src.pricing import minimum_profitable_price, baseline_recommendation


def test_minimum_profitable_price():
    assert round(minimum_profitable_price(8, 0.30), 2) == 11.43


def test_invalid_margin():
    with pytest.raises(ValueError):
        minimum_profitable_price(8, 1.0)


def test_margin_floor_is_respected():
    stats = {"p25": 10.8, "median": 11.2, "p75": 12.1}
    result = baseline_recommendation(8, 0.30, stats, 11.5, "balanced")
    assert result.recommended_price >= 11.43


def test_margin_conflict_warning():
    stats = {"p25": 18, "median": 20, "p75": 21}
    result = baseline_recommendation(15, 0.40, stats, 20, "balanced")
    assert result.warning is not None
