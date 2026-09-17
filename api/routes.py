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

DATA_PATH = ROOT / CONFIG["paths"].get("cleaned_parquet", "data/processed/cleaned_products.csv")
MODEL_DIR = ROOT / CONFIG["paths"]["models_dir"]
MARKET_MODEL_PATH = MODEL_DIR / "market_price_model.joblib"
DEMAND_MODEL_PATH = MODEL_DIR / "demand_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

REGISTRY = ModelRegistry(ROOT / CONFIG["paths"].get("registry_dir", "models/registry"))
_CACHE = {"version": None, "data": None, "similarity": None, "market": None, "demand": None, "metadata": None}
_CACHE_LOCK = Lock()

DATA = None
SIMILARITY = None
MARKET_MODEL = None
DEMAND_MODEL = None


def _load_production_bundle() -> None:
    current = REGISTRY.get_current()
    version = current["version"] if current else "local"
    if _CACHE["version"] == version:
        return
    with _CACHE_LOCK:
        current = REGISTRY.get_current()
        version = current["version"] if current else "local"
        if _CACHE["version"] == version:
            return
        version_dir = REGISTRY.versions / version if current else MODEL_DIR
        # Serving catalog is always the latest validated master dataset.
        # The ML artifacts remain versioned and protected by the registry.
        master_path = ROOT / CONFIG["paths"].get("cleaned_parquet", "data/processed/cleaned_products.csv")
        if not master_path.exists():
            master_path = ROOT / "data" / "processed" / "cleaned_products.csv"
        data = load_master_dataset(master_path)
        data = _canonicalize_serving_data(data)
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


def _canonicalize_serving_data(data: pd.DataFrame) -> pd.DataFrame:
    """Accept both the current canonical schema and legacy processed exports."""
    out = data.copy()
    renames = {
        "sku_name": "product_name",
        "brand": "brand_name",
        "merchant": "seller_name",
        "price_usd": "price_current_usd",
        "sold": "sold_count",
        "reviews": "review_count",
        "rating": "rating_score",
    }
    out = out.rename(columns={old: new for old, new in renames.items() if old in out})
    if "snapshot_date" not in out:
        out["snapshot_date"] = pd.to_datetime(out.get("month"), errors="coerce")
    if "price_current_local" not in out:
        out["price_current_local"] = out["price_current_usd"]
    if "discount_rate" not in out:
        out["discount_rate"] = pd.to_numeric(out.get("discount", 0), errors="coerce").fillna(0)
    if "currency" not in out:
        out["currency"] = out["country"].map(CONFIG["country_currency"]).fillna("USD")
    if "product_name_clean" not in out:
        out["product_name_clean"] = out["product_name"].fillna("").astype(str).str.lower()
    if "brand_clean" not in out:
        out["brand_clean"] = out["brand_name"].fillna("").astype(str).str.lower()
    return out


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
        "date_min": str(pd.to_datetime(data["snapshot_date"]).min().date()),
        "date_max": str(pd.to_datetime(data["snapshot_date"]).max().date()),
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

        category = request.category.strip()
        currency = request.currency.upper().strip()
        if not currency:
            # Costs in the public API contract are USD unless a currency is
            # explicitly supplied; this keeps the example request unit-safe.
            currency = "USD"
        fx = CONFIG["fx_to_usd"].get(currency)
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
            market_weight=float(CONFIG["pricing"].get(
                "market_weight", CONFIG["pricing"].get("market_anchor_competitor_weight", 0.60)
            )),
            ml_weight=float(CONFIG["pricing"].get(
                "ml_weight", CONFIG["pricing"].get("market_anchor_model_weight", 0.40)
            )),
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
            grid_points=int(CONFIG["pricing"].get("price_grid_points", 80)),
            competitor_p90=stats["p90"] * fx,
            max_competitor_multiple=float(CONFIG["pricing"].get("max_competitor_multiple", 1.20)),
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
