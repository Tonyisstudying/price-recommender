from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field
#BaseModel is used to create a data model
Strategy = Literal["competitive", "balanced", "premium"]

class PricingRequest(BaseModel):
    country: str = Field(min_length=2)
    marketplace: str = Field(min_length=2)
    category: str = Field(min_length=2)
    product_name: str = Field(min_length=2)
    brand: str = "
    cost: float = Field(gt=0)
    target_margin: float = Field(gt=0, lt=1)
    strategy: Strategy = "balanced"

class PricingResponse(BaseModel):
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
    competitor_count: int
    comparable_products: list[dict]
    optional_optimized_price: float | None = None
    expected_demand: float | None = None
    expected_profit: float | None = None
