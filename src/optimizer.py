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
        model_features = getattr(demand_model, "feature_names_in_", None)
        if model_features is not None:
            compatible = row.copy()
            aliases = {
                "brand": "brand_name",
                "rating": "rating_score",
                "discount": "discount_rate",
                "competitor_p25": "competitor_p25_usd",
                "competitor_median": "competitor_median_usd",
                "competitor_p75": "competitor_p75_usd",
            }
            for target, source in aliases.items():
                if target in model_features and target not in compatible:
                    compatible[target] = compatible[source]
            row = compatible[list(model_features)]
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
