from __future__ import annotations

from threading import Lock
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config import CONFIG, ROOT
from src.data_manager import load_master_dataset, dataset_fingerprint
from src.competitor import build_competitor_features
from src.demand_model import load_model as load_demand_model, prepare_frame as prepare_demand_frame
from src.market_price_model import load_model as load_market_model, predict_usd
from src.optimizer import optimize_price
from src.pricing import baseline_recommendation
from src.preprocessing import infer_category
from src.similarity import SimilarityEngine
from src.model_registry import ModelRegistry

from .schemas import PricingRequest, PricingResponse

router = APIRouter(prefix="/api/v1", tags=["pricing"])

DATA_PATH = ROOT / CONFIG["data"]["processed_path"]
MODEL_DIR = ROOT / CONFIG["model"]["model_dir"]
MARKET_MODEL_PATH = MODEL_DIR / CONFIG["model"]["market_price_model"]
DEMAND_MODEL_PATH = MODEL_DIR / CONFIG["model"]["demand_model"]
VECTORIZER_PATH = MODEL_DIR / CONFIG["model"]["tfidf_vectorizer"]

REGISTRY = ModelRegistry(ROOT / CONFIG["model"]["registry_dir"])
_CACHE = {"version": None, "data": None, "similarity": None, "market": None, "demand": None, "metadata": None}
_CACHE_LOCK = Lock()

DATA = None
SIMILARITY = None
MARKET_MODEL = None
DEMAND_MODEL = None


def _load_production_bundle() -> None:
    current = REGISTRY.get_current()
    if not current:
        raise RuntimeError("No promoted model exists. Run `python train.py` first.")
    version = current["version"]
    if _CACHE["version"] == version:
        return
    with _CACHE_LOCK:
        current = REGISTRY.get_current()
        if not current:
            raise RuntimeError("No promoted model exists.")
        version = current["version"]
        if _CACHE["version"] == version:
            return
        version_dir = REGISTRY.versions / version
        # Serving catalog is always the latest validated master dataset.
        # The ML artifacts remain versioned and protected by the registry.
        master_path = ROOT / CONFIG["data"]["master_path"]
        data = load_master_dataset(master_path)
        similarity = SimilarityEngine.fit(data)
        market = load_market_model(version_dir / "market_price_model.joblib")
        demand = load_demand_model(version_dir / "demand_model.joblib")
        metadata_path = version_dir / "metadata.json"
        metadata = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else "{}"
        _CACHE.update({
            "version": version,
            "data": data,
            "similarity": similarity,
            "market": market,
            "demand": demand,
            "metadata": metadata,
        })


def ensure_loaded() -> None:
    _load_production_bundle()

@router.get("/health")
def health() -> dict:
    try:
        ensure_loaded()
        return {"status": "ok", "model_version": _CACHE["version"]}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}


@router.get("/model-info")
def model_info() -> dict:
    ensure_loaded()
    data = _CACHE["data"]
    current = REGISTRY.get_current() or {}
    return {
        "model_version": _CACHE["version"],
        "market_model": _CACHE["market"].__class__.__name__,
        "demand_model": _CACHE["demand"].__class__.__name__,
        "data_rows": int(len(data)),
        "date_min": str(data["snapshot_date"].min().date()),
        "date_max": str(data["snapshot_date"].max().date()),
        "model_dataset_fingerprint": current.get("dataset_fingerprint"),
        "serving_dataset_fingerprint": dataset_fingerprint(data),
    }

@router.post("/pricing/recommend", response_model=PricingResponse)
def recommend(request: PricingRequest) -> PricingResponse:
    try:
        ensure_loaded()
        data = _CACHE["data"]
        similarity = _CACHE["similarity"]
        market_model = _CACHE["market"]
        demand_model = _CACHE["demand"]

        category = request.category.strip() or infer_category(request.product_name)
        currency = request.currency.upper().strip()
        if not currency:
            currency = CONFIG["supported"]["currency_by_country"].get(request.country, "USD")
        fx = CONFIG["supported"]["fx_to_usd"].get(currency)
        if fx is None:
            raise HTTPException(status_code=400, detail=f"Unsupported currency: {currency}")

        scope = data[
            (data["country"] == request.country)
            & (data["marketplace"] == request.marketplace)
            & (data["category"] == category)
        ].copy()
        if scope.empty:
            raise HTTPException(status_code=404, detail=f"No market data for {request.country}/{request.marketplace}/{category}.")

        latest = scope["snapshot_date"].max()
        competitors = similarity.find_candidates(
            request.product_name,
            request.brand,
            category,
            request.country,
            request.marketplace,
            snapshot_date=latest,
            top_k=40,
            min_similarity=0.15,
        )
        if competitors.empty:
            raise HTTPException(status_code=404, detail="No comparable products found.")

        stats = build_competitor_features(competitors)
        context = scope[scope["snapshot_date"] == latest].copy()
        if context.empty:
            context = competitors.copy()

        row = pd.DataFrame([{
            "country": request.country,
            "marketplace": request.marketplace,
            "category": category,
            "brand_name": request.brand or "Unknown",
            "currency": currency,
            "rating_score": float(context["rating_score"].median()),
            "review_count": float(context["review_count"].median()),
            "sold_count": float(context["sold_count"].median()),
            "discount_rate": float(context["discount_rate"].median()),
            "competitor_p25_usd": stats["p25"] * fx,
            "competitor_median_usd": stats["median"] * fx,
            "competitor_p75_usd": stats["p75"] * fx,
            "competitor_count": stats["count"],
            "price_usd": stats["median"] * fx,
        }])

        predicted_market_usd = float(predict_usd(market_model, row)[0])
        predicted_market_local = predicted_market_usd / fx

        local_stats = stats
        result = baseline_recommendation(
            cost=request.cost,
            target_margin=request.target_margin,
            stats=local_stats,
            predicted_market_price=predicted_market_local,
            strategy=request.strategy,
            market_weight=float(CONFIG["pricing"]["market_weight"]),
            ml_weight=float(CONFIG["pricing"]["ml_weight"]),
        )

        demand_row = prepare_demand_frame(row)
        demand_base = demand_row[[
            "country", "marketplace", "category", "brand_name", "currency",
            "rating_score", "log_reviews", "price_usd", "discount_rate",
            "competitor_median_usd", "competitor_p25_usd", "competitor_p75_usd",
            "competitor_count", "price_vs_median",
        ]]

        cost_usd = request.cost * fx
        strategy_usd = result.recommended_price * fx
        upper_usd = max(stats["p90"] * fx, strategy_usd * 1.15)
        optimized = optimize_price(
            cost=cost_usd,
            target_margin=request.target_margin,
            upper_price=upper_usd,
            demand_model=demand_model,
            base_row=demand_base,
            strategy_price=strategy_usd,
            grid_points=int(CONFIG["pricing"]["price_grid_points"]),
            competitor_p90=stats["p90"] * fx,
            max_competitor_multiple=float(CONFIG["pricing"]["max_competitor_multiple"]),
        )

        comparable = competitors[[
            c for c in ["product_id", "product_name", "brand_name", "seller_name", "price_local", "rating_score", "sold_count", "product_url", "similarity"]
            if c in competitors.columns
        ]].head(10).to_dict("records")
        for item in comparable:
            if "price_local" in item:
                item["price"] = round(float(item.pop("price_local")), 2)

        payload = result.to_dict()
        payload.update({
            "currency": currency,
            "model_version": _CACHE["version"],
            "data_snapshot": str(latest.date()),
            "competitor_count": stats["count"],
            "comparable_products": comparable,
            "optional_optimized_price": round(optimized["optimized_price"] / fx, 2),
            "expected_demand": round(optimized["expected_demand"], 2),
            "expected_profit": round(optimized["expected_profit"] / fx, 2),
        })
        return PricingResponse(**payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
