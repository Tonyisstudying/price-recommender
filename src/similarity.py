from __future__ import annotations

from dataclasses import dataclass
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityEngine:
    vectorizer: TfidfVectorizer
    matrix: object
    products: pd.DataFrame

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "SimilarityEngine":
        corpus = (
            df["sku_name_clean"].fillna("") + " " +
            df["brand_clean"].fillna("") + " " +
            df["category"].fillna("").str.lower()
        )
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer, matrix, df.reset_index(drop=True))

    def _query(self, product_name: str, brand: str, category: str):
        return self.vectorizer.transform([f"{product_name} {brand} {category}".lower()])

    def find_candidates(
        self,
        product_name: str,
        brand: str,
        category: str,
        country: str,
        marketplace: str,
        top_k: int = 30,
        min_similarity: float = 0.20,
    ) -> pd.DataFrame:
        pool = self.products[
            (self.products["country"] == country) &
            (self.products["marketplace"] == marketplace) &
            (self.products["category"] == category)
        ].copy()
        if pool.empty:
            return pool
        positions = pool.index.to_numpy()
        sim = cosine_similarity(self._query(product_name, brand, category), self.matrix[positions]).ravel()
        pool["similarity"] = sim
        if brand:
            same = pool["brand"].str.lower().str.strip() == brand.lower().strip()
            pool.loc[same, "similarity"] += 0.10
        return pool[pool["similarity"] >= min_similarity].sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)

    @classmethod
    def load(cls, vectorizer_path: str, products_path: str) -> "SimilarityEngine":
        vectorizer = joblib.load(vectorizer_path)
        products = pd.read_parquet(products_path) if str(products_path).lower().endswith(".parquet") else pd.read_csv(products_path)
        corpus = (
            products["sku_name_clean"].fillna("") + " " +
            products["brand_clean"].fillna("") + " " +
            products["category"].fillna("").str.lower()
        )
        matrix = vectorizer.transform(corpus)
        return cls(vectorizer, matrix, products.reset_index(drop=True))
