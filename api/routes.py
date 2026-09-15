from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config import CONFIG, ROOT
from src.competitor import build_competitor_features
from src.demand_model import load_model as load_demand_model, prepare_frame as prepare_demand_frame
from src.market_price_model import load_model as load_market_model, predict as predict_market
from src.optimizer import optimize_price
from src.pricing import baseline_recommendation
from src.similarity import SimilarityEngine

from .schemas import PricingRequest, PricingResponse

router = APIRouter(prefix="/api/v1", tags=["pricing"])

DATA_PATH = ROOT / CONFIG["data"]["processed_path"]
MODEL_DIR = ROOT / CONFIG["model"]["model_dir"]
MARKET_MODEL_PATH = MODEL_DIR / CONFIG["model"]["market_price_model"]
DEMAND_MODEL_PATH = MODEL_DIR / CONFIG["model"]["demand_model"]
VECTORIZER_PATH = MODEL_DIR / CONFIG["model"]["tfidf_vectorizer"]

DATA = None
SIMILARITY = None
MARKET_MODEL = None
DEMAND_MODEL = None


def ensure_loaded() -> None:
    global DATA, SIMILARITY, MARKET_MODEL, DEMAND_MODEL
    if not DATA_PATH.exists():
        raise RuntimeError("Processed dataset not found. Run generate_mock_data.py then train.py.")
    if DATA is None:
        DATA = pd.read_parquet(DATA_PATH) if str(DATA_PATH).lower().endswith(".parquet") else pd.read_csv(DATA_PATH)
        DATA["month"] = pd.to_datetime(DATA["month"], errors="coerce")
    if SIMILARITY is None:
        SIMILARITY = SimilarityEngine.load(str(VECTORIZER_PATH), str(DATA_PATH))
    if MARKET_MODEL is None:
        MARKET_MODEL = load_market_model(MARKET_MODEL_PATH)
    if DEMAND_MODEL is None:
        DEMAND_MODEL = load_demand_model(DEMAND_MODEL_PATH)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/model-info")
def model_info() -> dict:
    ensure_loaded()
    return {
        "market_model": MARKET_MODEL.__class__.__name__,
        "demand_model": DEMAND_MODEL.__class__.__name__,
        "data_rows": int(len(DATA)),
        "date_min": str(DATA["month"].min().date()),
        "date_max": str(DATA["month"].max().date()),
    }


@router.post("/pricing/recommend", response_model=PricingResponse)
def recommend(request: PricingRequest) -> PricingResponse:
    try:
        ensure_loaded()
        competitors = SIMILARITY.find_candidates(
            request.product_name,
            request.brand,
            request.category,
            request.country,
            request.marketplace,
            top_k=30,
            min_similarity=0.20,
        )
        if competitors.empty:
            raise HTTPException(status_code=404, detail="No comparable products found.")

        stats = build_competitor_features(competitors)

        scope = DATA[
            (DATA["country"] == request.country) &
            (DATA["marketplace"] == request.marketplace) &
            (DATA["category"] == request.category)
        ]
        latest_month = scope["month"].max()
        context = scope[scope["month"] == latest_month].copy()
        if context.empty:
            context = competitors.copy()

        row = pd.DataFrame([{
            "country": request.country,
            "marketplace": request.marketplace,
            "category": request.category,
            "brand": request.brand or "Unknown",
            "rating": float(context["rating"].median()),
            "reviews": float(context["reviews"].median()),
            "sold": float(context["sold"].median()),
            "discount": float(context["discount"].median()),
            "month": latest_month,
            "competitor_p25": stats["p25"],
            "competitor_median": stats["median"],
            "competitor_p75": stats["p75"],
            "competitor_count": stats["count"],
            "price_usd": stats["median"],
        }])

        predicted_market_price = float(predict_market(MARKET_MODEL, row)[0])

        result = baseline_recommendation(
            cost=request.cost,
            target_margin=request.target_margin,
            stats=stats,
            predicted_market_price=predicted_market_price,
            strategy=request.strategy,
            market_weight=float(CONFIG["pricing"]["market_weight"]),
            ml_weight=float(CONFIG["pricing"]["ml_weight"]),
        )

        demand_row = prepare_demand_frame(row)
        demand_row = demand_row[[
            "country", "marketplace", "category", "brand", "rating",
            "log_reviews", "price_usd", "discount",
            "competitor_median", "competitor_p25", "competitor_p75"
        ]]

        optimized = optimize_price(
            cost=request.cost,
            target_margin=request.target_margin,
            upper_price=max(stats["p90"], result.recommended_price * 1.15),
            demand_model=DEMAND_MODEL,
            base_row=demand_row,
            strategy_price=result.recommended_price,
            grid_points=int(CONFIG["pricing"]["price_grid_points"]),
            competitor_p90=stats["p90"],
            max_competitor_multiple=float(CONFIG["pricing"]["max_competitor_multiple"]),
        )

        comparable_cols = ["sku_name", "brand", "price_usd", "merchant", "similarity"]
        comparable = competitors[[c for c in comparable_cols if c in competitors]].head(10).to_dict("records")

        payload = result.to_dict()
        payload.update({
            "competitor_count": stats["count"],
            "comparable_products": comparable,
            "optional_optimized_price": round(optimized["optimized_price"], 2),
            "expected_demand": round(optimized["expected_demand"], 2),
            "expected_profit": round(optimized["expected_profit"], 2),
        })
        return PricingResponse(**payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
