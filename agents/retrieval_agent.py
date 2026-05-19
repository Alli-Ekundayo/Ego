"""
agents/retrieval_agent.py
--------------------------
Hybrid Retrieval Agent: combines Dense Semantic Search and Sparse BM25 keyword
search via Reciprocal Rank Fusion, then returns a merged candidate list.

Retrieval strategy:
  1. Dense Semantic Search  — Qdrant ANN over user history vectors
                              (captures latent taste similarity)
  2. Collaborative Filtering (CF) — cosine similarity over cross-domain
                              projected user vectors
  3. Sparse Keyword Search  — BM25 over the full review corpus using
                              aspect-derived keyword tokens
  4. RRF Fusion             — merge all three ranked lists into one
                              via Reciprocal Rank Fusion
"""

import json
import logging
from collections import defaultdict
from functools import cached_property

import numpy as np

from core.embeddings import embedding_model
from core.hybrid_search import hybrid_search
from core.math_utils import cosine_similarity
from core.utils import normalise_category, tokenize
from core.vector_store import vector_store

log = logging.getLogger(__name__)


def _category_matches(item_category: str, domain_filter: str | None) -> bool:
    if not domain_filter:
        return True
    category = normalise_category(item_category)
    domain = normalise_category(domain_filter)
    return domain in category or category in domain


class RetrievalAgent:
    def __init__(self):
        self._profiles_db = None
        self._cross_domain_projection = None
        self._user_vectors = None
        self._cluster_centroids = None

    @property
    def profiles_db(self) -> dict[str, dict]:
        if self._profiles_db is None:
            try:
                with open("data/user_profiles.json", "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                self._profiles_db = {p["user_id"]: p for p in profiles}
            except Exception as exc:
                log.error("Failed to load user profiles: %s", exc)
                self._profiles_db = {}
        return self._profiles_db

    @cached_property
    def user_vectors(self) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        user_ids: list[str] = []
        voice_samples: list[str] = []
        for user_id, profile in self.profiles_db.items():
            voice = str(profile.get("voice_sample", "")).strip()
            if not voice:
                continue
            user_ids.append(user_id)
            voice_samples.append(voice)

        if not voice_samples:
            return vectors

        embeddings = embedding_model.embed_batch(voice_samples)
        for user_id, emb in zip(user_ids, embeddings):
            vectors[user_id] = emb
        return vectors

    @cached_property
    def cluster_centroids(self) -> dict[str, list[float]]:
        grouped: dict[str, list[list[float]]] = defaultdict(list)
        for user_id, vector in self.user_vectors.items():
            profile = self.profiles_db.get(user_id, {})
            category_pref = profile.get("category_pref", {})
            top_category = "general"
            if category_pref:
                top_category = max(category_pref.items(), key=lambda x: x[1])[0]
            rating_mean = float(profile.get("rating_stats", {}).get("mean", 3.0))
            bucket = (
                "high" if rating_mean >= 4.0 else "mid" if rating_mean >= 3.0 else "low"
            )
            key = f"{normalise_category(top_category)}::{bucket}"
            grouped[key].append(vector)

        centroids: dict[str, list[float]] = {}
        for key, vectors in grouped.items():
            centroids[key] = np.mean(
                np.array(vectors, dtype=np.float32), axis=0
            ).tolist()
        return centroids

    @cached_property
    def cross_domain_projection(self) -> np.ndarray:
        source_texts_batch: list[str] = []
        target_texts_batch: list[str] = []

        for profile in self.profiles_db.values():
            train_reviews = profile.get("train_reviews", [])
            source_texts = []
            target_texts = []
            for review in train_reviews:
                text = f"{review.get('title', '')} {review.get('body', '')}".strip()
                category = normalise_category(review.get("category", ""))
                if not text:
                    continue
                if "books" in category or "movie" in category or "music" in category:
                    target_texts.append(text)
                else:
                    source_texts.append(text)

            if source_texts and target_texts:
                source_texts_batch.append(" ".join(source_texts))
                target_texts_batch.append(" ".join(target_texts))

        if len(source_texts_batch) < 3:
            dim = len(next(iter(self.user_vectors.values()), [0.0] * 384))
            return np.eye(dim, dtype=np.float32)

        source_vectors = embedding_model.embed_batch(source_texts_batch)
        target_vectors = embedding_model.embed_batch(target_texts_batch)
        x = np.array(source_vectors, dtype=np.float32)
        y = np.array(target_vectors, dtype=np.float32)
        reg = 0.05
        xtx = x.T @ x
        eye = np.eye(xtx.shape[0], dtype=np.float32)
        w = np.linalg.solve(xtx + reg * eye, x.T @ y)
        return w

    def _project_to_shared_space(self, vector: list[float]) -> list[float]:
        vec = np.array(vector, dtype=np.float32)
        projected = vec @ self.cross_domain_projection
        return projected.tolist()

    def _collect_user_items(
        self, user_id: str, user_score: float, seen: set[str]
    ) -> list[dict]:
        profile = self.profiles_db.get(user_id)
        if not profile:
            return []
        out = []
        all_reviews = list(profile.get("train_reviews", [])) + list(
            profile.get("test_reviews", [])
        )
        for review in all_reviews:
            product_id = str(review.get("product_id", "")).strip()
            rating = float(review.get("rating", 0.0))
            if not product_id or product_id in seen or rating < 4.0:
                continue
            seen.add(product_id)
            out.append(
                {
                    "item_id": product_id,
                    "name": review.get("product_name", ""),
                    "category": review.get("category", ""),
                    "url": review.get("product_url", ""),
                    "semantic_score": user_score,
                    "cf_score": user_score * (rating / 5.0),
                }
            )
        return out

    def retrieve(
        self,
        query_vector: list[float],
        user_id: str | None = None,
        n: int = 20,
        domain_filter: str | None = None,
        sparse_keywords: list[str] | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieval: Dense ANN + CF + Sparse BM25, fused via RRF.

        Args:
            query_vector:     User history embedding for ANN search.
            user_id:          Current user (excluded from CF lookup).
            n:                Number of candidates to return.
            domain_filter:    Optional category constraint.
            sparse_keywords:  BM25 keyword tokens from aspect extraction.
        """
        semantic_candidates: list[dict] = []
        cf_candidates: list[dict] = []
        seen_semantic: set[str] = set()
        seen_cf: set[str] = set()

        # ── 1. Dense semantic search ──────────────────────────────────────────
        if user_id and user_id in self.profiles_db:
            own_profile = self.profiles_db.get(user_id, {})
            for review in own_profile.get("test_reviews", []):
                product_id = str(review.get("product_id", "")).strip()
                rating = float(review.get("rating", 0.0))
                if not product_id or product_id in seen_semantic or rating < 4.0:
                    continue
                seen_semantic.add(product_id)
                semantic_candidates.append(
                    {
                        "item_id": product_id,
                        "name": review.get("product_name", ""),
                        "category": review.get("category", ""),
                        "url": review.get("product_url", ""),
                        "source_user_id": user_id,
                        "retrieval_hint": "recent_self",
                        "semantic_score": 2.0,
                        "cf_score": 2.0,
                    }
                )

        semantic_results = vector_store.search(
            collection_name="user_profiles",
            query_vector=query_vector,
            limit=max(n * 2, 100),
        )
        for result in semantic_results:
            similar_user_id = str(result.payload.get("id", ""))
            if not similar_user_id or similar_user_id == user_id:
                continue
            semantic_candidates.extend(
                self._collect_user_items(
                    similar_user_id, float(result.score), seen_semantic
                )
            )

        # ── 2. Collaborative filtering ────────────────────────────────────────
        projected_query = self._project_to_shared_space(query_vector)
        similarities = []
        for similar_user_id, user_vec in self.user_vectors.items():
            if similar_user_id == user_id:
                continue
            score = cosine_similarity(projected_query, user_vec)
            if score > 0:
                similarities.append((similar_user_id, score))
        similarities.sort(key=lambda x: x[1], reverse=True)
        for similar_user_id, score in similarities[: max(n * 2, 100)]:
            cf_candidates.extend(
                self._collect_user_items(similar_user_id, score, seen_cf)
            )

        # ── 3. Merge dense + CF candidates (pre-hybrid) ───────────────────────
        merged: dict[str, dict] = {}
        for candidate in semantic_candidates:
            if not _category_matches(candidate.get("category", ""), domain_filter):
                continue
            merged[candidate["item_id"]] = {
                **candidate,
                "retrieval_sources": ["dense"],
                "score": 0.65 * float(candidate.get("semantic_score", 0.0))
                + 0.35 * float(candidate.get("cf_score", 0.0)),
            }
            if candidate.get("retrieval_hint") == "recent_self":
                merged[candidate["item_id"]]["score"] *= 2.5

        for candidate in cf_candidates:
            if not _category_matches(candidate.get("category", ""), domain_filter):
                continue
            item_id = candidate["item_id"]
            if item_id in merged:
                merged[item_id]["retrieval_sources"].append("collaborative")
                merged[item_id]["score"] = max(
                    float(merged[item_id].get("score", 0.0)),
                    0.4 * float(candidate.get("semantic_score", 0.0))
                    + 0.6 * float(candidate.get("cf_score", 0.0)),
                )
                continue
            merged[item_id] = {
                **candidate,
                "retrieval_sources": ["collaborative"],
                "score": 0.4 * float(candidate.get("semantic_score", 0.0))
                + 0.6 * float(candidate.get("cf_score", 0.0)),
            }

        dense_ranked = sorted(
            merged.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True
        )

        # ── 4. Hybrid RRF fusion with sparse BM25 ────────────────────────────
        if sparse_keywords:
            fused = hybrid_search(
                dense_results=dense_ranked,
                sparse_keywords=sparse_keywords,
                top_k=max(n * 3, 150),
            )
            # Promote items appearing in both sources
            for item in fused:
                if len(item.get("retrieval_sources", [])) >= 2:
                    item["score"] = item.get("rrf_score", item.get("score", 0.0)) * 1.1
                else:
                    item["score"] = item.get("rrf_score", item.get("score", 0.0))
            ranked = sorted(
                fused, key=lambda x: float(x.get("score", 0.0)), reverse=True
            )
        else:
            ranked = dense_ranked

        log.info(
            "  ↳ Retrieval: %d dense, %d cf, %d merged, keywords=%s",
            len(seen_semantic),
            len(seen_cf),
            len(merged),
            sparse_keywords[:5] if sparse_keywords else "none",
        )
        return ranked[:n]

    def infer_cluster_proxy_embedding(
        self, persona_description: str, structured_signals: dict | None = None
    ) -> list[float]:
        signals = structured_signals or {}
        explicit = " ".join(signals.get("explicit_preferences", [])).lower()
        persona = f"{persona_description} {explicit}".strip().lower()
        if not persona:
            if self.cluster_centroids:
                return next(iter(self.cluster_centroids.values()))
            return []

        best_key = None
        best_score = -1.0
        for key in self.cluster_centroids:
            category_key = key.split("::", 1)[0]
            score = 1.0 if category_key and category_key in persona else 0.0
            score += (
                sum(1 for token in category_key.split() if token and token in persona)
                * 0.2
            )
            if score > best_score:
                best_score = score
                best_key = key

        if best_key:
            return self.cluster_centroids[best_key]
        if self.cluster_centroids:
            return next(iter(self.cluster_centroids.values()))
        return embedding_model.embed_text(persona_description)


retrieval_agent = RetrievalAgent()


def _fallback_sparse_keywords(
    user_id: str | None, structured_signals: dict | None = None
) -> list[str]:
    if not user_id:
        return []
    profile = retrieval_agent.profiles_db.get(user_id, {})
    if not profile:
        return []
    keywords: list[str] = []
    vocab = profile.get("vocab_fingerprint", {})
    if isinstance(vocab, dict):
        keywords.extend(list(vocab.keys())[:8])
    elif isinstance(vocab, list):
        keywords.extend(vocab[:8])

    category_pref = profile.get("category_pref", {})
    if isinstance(category_pref, dict):
        for cat in sorted(category_pref.items(), key=lambda x: x[1], reverse=True)[:3]:
            keywords.extend(tokenize(cat[0]))

    signals = structured_signals or {}
    for pref in (
        signals.get("explicit_preferences", []) if isinstance(signals, dict) else []
    ):
        keywords.extend(tokenize(str(pref)))

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw and kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def retrieve_candidates(
    profile,
    context_text: str,
    domain_filter: str | None = None,
    structured_signals: dict | None = None,
    sparse_keywords: list[str] | None = None,
) -> list[dict]:
    """
    Hybrid retrieval adapter.

    Args:
        profile:          UserProfile object.
        context_text:     Current user request / query string.
        domain_filter:    Optional category constraint.
        structured_signals: Signals from context extraction.
        sparse_keywords:  BM25 keyword tokens from aspect extraction.
                          Pass None to skip sparse search.
    """
    base_vector = list(getattr(profile, "history_vector", []) or [])
    if not base_vector:
        signals = structured_signals or {}
        explicit = ", ".join(signals.get("explicit_preferences", [])).strip()
        implicit = ", ".join(signals.get("implicit_signals", [])).strip()
        query_context = context_text
        if explicit:
            query_context = f"{query_context}\nPreferred: {explicit}".strip()
        if implicit:
            query_context = f"{query_context}\nSignals: {implicit}".strip()
        base_vector = embedding_model.embed_text(query_context)
    if not sparse_keywords:
        sparse_keywords = _fallback_sparse_keywords(
            getattr(profile, "user_id", None), structured_signals
        )
        if not sparse_keywords:
            sparse_keywords = None

    return retrieval_agent.retrieve(
        query_vector=base_vector,
        user_id=getattr(profile, "user_id", None),
        n=100,
        domain_filter=domain_filter,
        sparse_keywords=sparse_keywords,
    )


def build_cold_start_proxy_embedding(
    persona_description: str, structured_signals: dict | None = None
) -> list[float]:
    return retrieval_agent.infer_cluster_proxy_embedding(
        persona_description, structured_signals
    )
