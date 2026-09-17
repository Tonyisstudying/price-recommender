from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import minimum_profitable_price


def optimize_price(
    *,
    cost: float,
    target_margin: float,
    upper_price: float,
    demand_model,
    base_row: pd.DataFrame,
    strategy_price: float,
    grid_points: int = 80,
    competitor_p90: float | None = None,
    max_competitor_multiple: float = 1.20,
) -> dict:
    # All optimization prices must use the same currency as `price_usd` in base_row.
    min_price = minimum_profitable_price(cost, target_margin)
    effective_upper = max(upper_price, min_price, strategy_price)
    if competitor_p90 is not None:
        effective_upper = max(
            min_price,
            min(effective_upper, competitor_p90 * max_competitor_multiple),
        )

    prices = np.linspace(min_price, effective_upper, grid_points)
    profits, demands = [], []

    for price in prices:
        row = base_row.copy()
        row["price_usd"] = price
        predicted_demand = float(demand_model.predict(row)[0])
        predicted_demand = max(predicted_demand, 0.0)
        profit = (price - cost) * predicted_demand
        demands.append(predicted_demand)
        profits.append(profit)

    best = int(np.argmax(profits))
    return {
        "optimized_price": float(prices[best]),
        "expected_demand": float(demands[best]),
        "expected_profit": float(profits[best]),
        "grid": [
            {
                "price": float(p),
                "expected_demand": float(d),
                "expected_profit": float(x),
            }
            for p, d, x in zip(prices, demands, profits)
        ],
    }
