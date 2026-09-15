"""Comparable-product retrieval and competitor price distributions.

Two-stage design:

    Stage 1 — hard filters (country, marketplace, category; optional brand)
    Stage 2 — TF-IDF + cosine similarity on product name + brand + category

The vectorizer is wrapped behind a small interface so a future
sentence-transformer + FAISS retriever can replace TF-IDF without changing
callers.

Price distributions are the business object: P10/P25/median/P75/P90 matter
more than the mean. Volume-weighted stats use log(1 + sold) so high-velocity
listings pull the center of the market without letting a single viral SKU
dominate linearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config
from src.utils import get_logger, normalize_text

logger = get_logger(__name__)


class SimilarityIndex(Protocol):
    """Swap-point for TF-IDF today and embeddings later."""

    def fit(self, corpus: list[str]) -> None: ...

    def similar(self, query: str, candidate_indices: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...


class TfidfSimilarityIndex:
    """Baseline lexical retriever. ngram (1,2) catches 'M331 Mouse' vs 'Mouse M331'."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2, max_features: int = 50_000) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            lowercase=True,
            strip_accents=None,
        )
        self.matrix = None

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("Cannot fit similarity index on an empty corpus.")
        try:
            self.matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Tiny corpora (unit tests) may not satisfy min_df.
            self.vectorizer.set_params(min_df=1)
            self.matrix = self.vectorizer.fit_transform(corpus)

    def similar(
        self, query: str, candidate_indices: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.matrix is None:
            raise RuntimeError("Similarity index is not fitted.")
        if candidate_indices.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        query_vec = self.vectorizer.transform([query])
        subset = self.matrix[candidate_indices]
        scores = cosine_similarity(query_vec, subset).ravel()
        order = np.argsort(-scores)
        k = min(top_k, order.size)
        chosen = order[:k]
        return candidate_indices[chosen], scores[chosen]


@dataclass
class PriceDistribution:
    count: int
    minimum: float | None
    p10: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p90: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    weighted_median: float | None
    weighted_mean: float | None
    volume_weight: str = "log1p_sold"

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.minimum,
            "p10": self.p10,
            "p25": self.p25,
            "median": self.median,
            "p75": self.p75,
            "p90": self.p90,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "weighted_median": self.weighted_median,
            "weighted_mean": self.weighted_mean,
            "volume_weight": self.volume_weight,
        }


@dataclass
class CompetitorMatch:
    sku_name: str
    brand: str
    merchant: str
    price_usd: float
    sold: float | None
    similarity: float
    month: str
    country: str
    marketplace: str
    category: str


@dataclass
class CompetitorSet:
    query: str
    n_filtered: int
    matches: list[CompetitorMatch] = field(default_factory=list)
    distribution: PriceDistribution | None = None


def _percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return float(values.mean())
    return float(np.dot(values, weights) / weight_sum)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    if float(weights.sum()) <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cutoff = 0.5 * cdf[-1]
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(idx, len(sorted_values) - 1)
    return float(sorted_values[idx])


def compute_price_distribution(
    prices: pd.Series,
    sold: pd.Series | None = None,
    *,
    volume_weight: str = "log1p_sold",
) -> PriceDistribution:
    values = pd.to_numeric(prices, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return PriceDistribution(
            count=0,
            minimum=None,
            p10=None,
            p25=None,
            median=None,
            p75=None,
            p90=None,
            maximum=None,
            mean=None,
            std=None,
            weighted_median=None,
            weighted_mean=None,
            volume_weight=volume_weight,
        )

    if sold is None:
        weights = np.ones_like(values)
    else:
        sold_aligned = pd.to_numeric(sold, errors="coerce").fillna(0).to_numpy(dtype=float)
        if sold_aligned.size != values.size:
            sold_numeric = pd.to_numeric(sold, errors="coerce").fillna(0)
            joined = pd.DataFrame({"price": pd.to_numeric(prices, errors="coerce"), "sold": sold_numeric})
            joined = joined.dropna(subset=["price"])
            values = joined["price"].to_numpy(dtype=float)
            sold_aligned = joined["sold"].to_numpy(dtype=float)
        if volume_weight == "log1p_sold":
            weights = np.log1p(np.clip(sold_aligned, 0, None))
        elif volume_weight == "sold":
            weights = np.clip(sold_aligned, 0, None)
        else:
            raise ValueError(f"Unknown volume_weight: {volume_weight}")

    return PriceDistribution(
        count=int(values.size),
        minimum=float(values.min()),
        p10=_percentile(values, 10),
        p25=_percentile(values, 25),
        median=_percentile(values, 50),
        p75=_percentile(values, 75),
        p90=_percentile(values, 90),
        maximum=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        weighted_median=_weighted_median(values, weights),
        weighted_mean=_weighted_mean(values, weights),
        volume_weight=volume_weight,
    )


def filter_candidates(
    catalog: pd.DataFrame,
    *,
    country: str,
    marketplace: str,
    category: str,
    brand: str | None = None,
    month: str | None = None,
    months_allowed: list[str] | None = None,
) -> pd.DataFrame:
    """Stage-1 hard filters. Month restriction is required for leakage control."""
    mask = (
        (catalog["country"] == country)
        & (catalog["marketplace"] == marketplace)
        & (catalog["category"] == category)
    )
    if brand:
        mask = mask & (catalog["brand"].str.lower() == brand.lower())
    if month:
        mask = mask & (catalog["month"].astype(str) == str(month))
    if months_allowed is not None:
        mask = mask & catalog["month"].astype(str).isin(months_allowed)
    return catalog.loc[mask].copy()


class CompetitorEngine:
    def __init__(self, catalog: pd.DataFrame, config: dict | None = None, index: SimilarityIndex | None = None) -> None:
        self.config = config or load_config()
        self.catalog = catalog.reset_index(drop=True)
        sim_cfg = self.config["similarity"]
        self.top_k = int(sim_cfg["top_k"])
        self.min_cosine = float(sim_cfg["min_cosine"])
        ngram = tuple(sim_cfg["ngram_range"])
        self.index = index or TfidfSimilarityIndex(
            ngram_range=(int(ngram[0]), int(ngram[1])),
            min_df=int(sim_cfg["min_df"]),
            max_features=int(sim_cfg["max_features"]),
        )
        corpus = self.catalog["search_text"].fillna("").tolist()
        if not any(corpus) and "sku_name" in self.catalog.columns:
            corpus = (
                self.catalog["sku_name"].fillna("").astype(str)
                + " "
                + self.catalog.get("brand", pd.Series([""] * len(self.catalog))).fillna("").astype(str)
                + " "
                + self.catalog.get("category", pd.Series([""] * len(self.catalog))).fillna("").astype(str)
            ).map(normalize_text).tolist()
            self.catalog = self.catalog.copy()
            self.catalog["search_text"] = corpus
        self.index.fit(corpus)
        logger.info("Fitted competitor TF-IDF index on %s listings", len(self.catalog))

    def find_competitors(
        self,
        product_name: str,
        *,
        country: str,
        marketplace: str,
        category: str,
        brand: str | None = None,
        month: str | None = None,
        months_allowed: list[str] | None = None,
        top_k: int | None = None,
        exclude_exact_name: bool = False,
    ) -> CompetitorSet:
        """Return similar listings. Restrict `months_allowed` to history at time t."""
        k = top_k or self.top_k
        query_text = normalize_text(" ".join(part for part in (product_name, brand or "", category) if part))
        filtered = filter_candidates(
            self.catalog,
            country=country,
            marketplace=marketplace,
            category=category,
            brand=None,
            month=month,
            months_allowed=months_allowed,
        )
        if exclude_exact_name:
            filtered = filtered[filtered["sku_name"].str.lower() != product_name.lower()]
        if filtered.empty:
            return CompetitorSet(query=product_name, n_filtered=0, matches=[], distribution=compute_price_distribution(pd.Series(dtype=float)))

        # Matrix rows align with the reset catalog index (0..n-1).
        positions = filtered.index.to_numpy(dtype=int)
        retrieved_idx, scores = self.index.similar(query_text, positions, top_k=max(k * 3, k))
        matches: list[CompetitorMatch] = []
        for idx, score in zip(retrieved_idx, scores):
            if score < self.min_cosine:
                continue
            row = self.catalog.iloc[int(idx)]
            matches.append(
                CompetitorMatch(
                    sku_name=str(row["sku_name"]),
                    brand=str(row["brand"]),
                    merchant=str(row.get("merchant", "")),
                    price_usd=float(row["price_usd"]),
                    sold=None if pd.isna(row.get("sold")) else float(row.get("sold")),
                    similarity=float(score),
                    month=str(row["month"]),
                    country=str(row["country"]),
                    marketplace=str(row["marketplace"]),
                    category=str(row["category"]),
                )
            )
            if len(matches) >= k:
                break

        match_frame = pd.DataFrame([m.__dict__ for m in matches]) if matches else pd.DataFrame(columns=["price_usd", "sold"])
        distribution = compute_price_distribution(
            match_frame["price_usd"] if not match_frame.empty else pd.Series(dtype=float),
            match_frame["sold"] if "sold" in match_frame.columns else None,
            volume_weight=self.config["competitor"]["volume_weight"],
        )
        return CompetitorSet(
            query=product_name,
            n_filtered=int(len(filtered)),
            matches=matches,
            distribution=distribution,
        )
