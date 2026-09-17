"""
Comparable-product engine: Stage 1 candidate filtering + Stage 2 TF-IDF/
cosine similarity retrieval (brief section 10), plus competitor price
distribution statistics, including volume-weighted stats (section 11).

Kept deliberately swappable: `find_similar_products` takes a candidate
DataFrame and returns (index, score) pairs. A future embedding-based
retriever (sentence-transformers + FAISS) only needs to implement the
same signature - nothing else in the project touches TfidfVectorizer
directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config


# =============================================================================
# Stage 1: candidate filtering
# =============================================================================

def filter_candidates(
    df: pd.DataFrame,
    country: str,
    marketplace: str,
    category: str,
    brand_name: str | None = None,
    as_of_date: pd.Timestamp | None = None,
    exclude_product_id: int | None = None,
) -> pd.DataFrame:
    """Stage 1: cheap categorical filter before any text similarity math.

    as_of_date enforces the no-leakage rule from brief section 31: when
    filtering competitors for a prediction at time t, only rows with
    snapshot_date <= t (or unknown/undated catalog rows used purely for
    breadth, which is a caller decision - see `include_undated`) should
    be visible. Pass as_of_date=None to disable this (e.g. Phase 1
    catalog-wide similarity checks where no specific "current time"
    applies yet).
    """
    candidates = df[
        (df["country"] == country)
        & (df["marketplace"] == marketplace)
        & (df["category"] == category)
    ]
    if brand_name is not None:
        brand_matches = candidates[candidates["brand_name"] == brand_name]
        # Only narrow to brand if it doesn't wipe out the candidate pool -
        # a brand-only match with too few rows is worse than category-wide.
        if len(brand_matches) >= 3:
            candidates = brand_matches
    if as_of_date is not None:
        candidates = candidates[
            candidates["snapshot_date"].isna() | (candidates["snapshot_date"] <= as_of_date)
        ]
    if exclude_product_id is not None:
        candidates = candidates[candidates["product_id"] != exclude_product_id]
    return candidates


# =============================================================================
# Stage 2: TF-IDF + cosine similarity
# =============================================================================

def _build_text_field(df: pd.DataFrame, text_fields: list[str]) -> pd.Series:
    return df[text_fields].fillna("").astype(str).agg(" ".join, axis=1)


def find_similar_products(
    query_text: str,
    candidates: pd.DataFrame,
    cfg: Config,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Rank `candidates` by TF-IDF cosine similarity to `query_text`.

    Returns a copy of `candidates` with a `similarity_score` column,
    sorted descending, truncated to top_k. Text corpus = query + every
    candidate's (product_name, brand_name, category) joined, so the
    vectorizer's vocabulary is built fresh each call - fine at this data
    scale (thousands of rows); a production version would persist a
    fitted vectorizer instead of refitting per query.
    """
    top_k = top_k or cfg.similarity["top_k"]
    text_fields = cfg.similarity["text_fields"]
    if candidates.empty:
        return candidates.assign(similarity_score=pd.Series(dtype=float))

    corpus_candidates = _build_text_field(candidates, text_fields)
    corpus = pd.concat([pd.Series([query_text]), corpus_candidates], ignore_index=True)

    vectorizer = TfidfVectorizer(ngram_range=tuple(cfg.similarity["ngram_range"]), lowercase=True)
    tfidf = vectorizer.fit_transform(corpus)

    query_vec = tfidf[0:1]
    candidate_vecs = tfidf[1:]
    scores = cosine_similarity(query_vec, candidate_vecs).ravel()

    out = candidates.copy()
    out["similarity_score"] = scores
    out = out.sort_values("similarity_score", ascending=False)
    return out.head(top_k)


# =============================================================================
# Competitor price distribution (section 11)
# =============================================================================

PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def price_distribution(
    competitors: pd.DataFrame, price_col: str = "price_usd"
) -> dict:
    """Backward-compatible compact distribution for pricing decisions."""
    stats = competitor_price_distribution(competitors, price_col)
    return {
        "count": stats["n_competitors"],
        "min": stats["min"],
        "p10": stats["p10"],
        "p25": stats["p25"],
        "median": stats["median"],
        "p75": stats["p75"],
        "p90": stats["p90"],
        "max": stats["max"],
        "mean": stats["mean"],
        "std": stats["std"],
    }


def build_competitor_features(
    competitors: pd.DataFrame, price_col: str = "price_local"
) -> dict:
    """Return the local-price stats consumed by the pricing engine."""
    if price_col not in competitors.columns:
        if "price_current_local" in competitors.columns:
            price_col = "price_current_local"
        elif "price_current_usd" in competitors.columns:
            price_col = "price_current_usd"
        else:
            price_col = "price_usd"
    return price_distribution(competitors, price_col)


def competitor_price_distribution(competitors: pd.DataFrame, price_col: str = "price_current_usd") -> dict:
    """Unweighted distribution stats. Returns None-valued dict if empty
    rather than raising, so callers can decide how to handle 'no
    competitors found' (e.g. widen the filter, or return low confidence)."""
    if competitors.empty:
        return {
            "n_competitors": 0, "min": None, "p10": None, "p25": None,
            "median": None, "p75": None, "p90": None, "max": None,
            "mean": None, "std": None,
        }
    prices = competitors[price_col].dropna()
    pcts = prices.quantile(PERCENTILES)
    return {
        "n_competitors": int(len(prices)),
        "min": float(prices.min()),
        "p10": float(pcts.loc[0.10]),
        "p25": float(pcts.loc[0.25]),
        "median": float(pcts.loc[0.50]),
        "p75": float(pcts.loc[0.75]),
        "p90": float(pcts.loc[0.90]),
        "max": float(prices.max()),
        "mean": float(prices.mean()),
        "std": float(prices.std()) if len(prices) > 1 else 0.0,
    }


def volume_weighted_price_stats(
    competitors: pd.DataFrame,
    price_col: str = "price_current_usd",
    sold_col: str = "sold_count",
) -> dict:
    """Volume-weighted mean/median using weight = log(1 + sold), per
    section 11: high-volume competitors get more influence on the
    'true' market price than a low-sold listing with an outlier price.

    Weighted median is computed by sorting prices and walking cumulative
    weight to the 50% point (there's no closed-form weighted median in
    numpy/pandas)."""
    if competitors.empty:
        return {"weighted_mean": None, "weighted_median": None, "total_weight": 0.0}

    prices = competitors[price_col].to_numpy(dtype=float)
    sold = competitors[sold_col].fillna(0).to_numpy(dtype=float)
    weights = np.log1p(sold)

    if weights.sum() == 0:
        # No sales signal at all -> fall back to unweighted (equal weight),
        # rather than dividing by zero.
        weights = np.ones_like(prices)

    weighted_mean = float(np.average(prices, weights=weights))

    order = np.argsort(prices)
    sorted_prices, sorted_weights = prices[order], weights[order]
    cum_weight = np.cumsum(sorted_weights)
    half = cum_weight[-1] / 2.0
    median_idx = int(np.searchsorted(cum_weight, half))
    median_idx = min(median_idx, len(sorted_prices) - 1)
    weighted_median = float(sorted_prices[median_idx])

    return {
        "weighted_mean": weighted_mean,
        "weighted_median": weighted_median,
        "total_weight": float(weights.sum()),
    }


def get_competitor_summary(
    df: pd.DataFrame,
    cfg: Config,
    product_name: str,
    country: str,
    marketplace: str,
    category: str,
    brand_name: str | None = None,
    as_of_date: pd.Timestamp | None = None,
    exclude_product_id: int | None = None,
) -> dict:
    """End-to-end: filter -> rank by similarity -> compute both plain and
    volume-weighted distribution stats. This is the function Phase 4's
    pricing engine will call."""
    candidates = filter_candidates(
        df, country, marketplace, category, brand_name, as_of_date, exclude_product_id
    )
    ranked = find_similar_products(product_name, candidates, cfg)
    stats = competitor_price_distribution(ranked)
    weighted = volume_weighted_price_stats(ranked)
    return {**stats, **weighted, "top_competitors": ranked}


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    df = pd.read_parquet(cfg.processed_dir / "cleaned_products.parquet")

    summary = get_competitor_summary(
        df, cfg,
        product_name="Logitech M331 Wireless Mouse",
        country="Vietnam", marketplace="Lazada", category="Accessories",
    )
    top = summary.pop("top_competitors")
    print("Competitor summary:", {k: v for k, v in summary.items()})
    print("\nTop matches:")
    print(top[["product_name", "brand_name", "price_current_usd", "similarity_score"]].head(5).to_string(index=False))
