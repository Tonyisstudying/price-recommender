from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PricingResult:
    recommended_price: float
    minimum_profitable_price: float
    predicted_market_price: float
    competitor_p25: float
    competitor_median: float
    competitor_p75: float
    target_margin: float
    achieved_margin: float
    strategy: str
    competitiveness: str
    warning: str | None
    explanation: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def minimum_profitable_price(cost: float, target_margin: float) -> float:
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not 0 < target_margin < 1:
        raise ValueError("target_margin must be between 0 and 1")
    return cost / (1 - target_margin)


def market_anchor(stats: dict, strategy: str) -> float:
    mapping = {
        "competitive": stats["p25"],
        "balanced": stats["median"],
        "premium": stats["p75"],
    }
    if strategy not in mapping:
        raise ValueError(f"Unknown strategy: {strategy}")
    return float(mapping[strategy])


def competitiveness(price: float, p25: float, median: float, p75: float) -> str:
    if price <= p25:
        return "very_competitive"
    if price <= median:
        return "competitive"
    if price <= p75:
        return "balanced"
    return "premium"


def baseline_recommendation(
    cost: float,
    target_margin: float,
    stats: dict,
    predicted_market_price: float,
    strategy: str,
    market_weight: float = 0.60,
    ml_weight: float = 0.40,
) -> PricingResult:
    if abs(market_weight + ml_weight - 1.0) > 1e-9:
        raise ValueError("market_weight + ml_weight must equal 1")

    minimum_price = minimum_profitable_price(cost, target_margin)
    anchor = market_anchor(stats, strategy)
    blended = market_weight * anchor + ml_weight * predicted_market_price
    recommended = max(minimum_price, blended)
    achieved_margin = (recommended - cost) / recommended
    comp = competitiveness(recommended, stats["p25"], stats["median"], stats["p75"])

    warning = None
    if minimum_price > stats["p75"]:
        warning = "Required gross margin places the price above competitor P75."

    explanation = [
        f"Minimum profitable price: {minimum_price:.2f}.",
        f"{strategy.title()} market anchor: {anchor:.2f}.",
        f"ML market-price estimate: {predicted_market_price:.2f}.",
        f"Final recommendation: {recommended:.2f}.",
    ]
    if warning:
        explanation.append("The requested margin conflicts with the observed competitive range.")
    else:
        if recommended < stats["p25"]:
            explanation.append("The recommendation is below competitor P25 while still satisfying the margin floor.")
        elif recommended <= stats["p75"]:
            explanation.append("The recommendation is inside the normal P25-P75 competitive range.")
        else:
            explanation.append("The recommendation is above competitor P75 because of the selected pricing strategy or ML estimate.")

    return PricingResult(
        recommended_price=round(recommended, 2),
        minimum_profitable_price=round(minimum_price, 2),
        predicted_market_price=round(predicted_market_price, 2),
        competitor_p25=round(stats["p25"], 2),
        competitor_median=round(stats["median"], 2),
        competitor_p75=round(stats["p75"], 2),
        target_margin=target_margin,
        achieved_margin=round(achieved_margin, 4),
        strategy=strategy,
        competitiveness=comp,
        warning=warning,
        explanation=explanation,
    )
