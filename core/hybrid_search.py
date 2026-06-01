"""
core/hybrid_search.py
---------------------
Hybrid Search: Reciprocal Rank Fusion (RRF) of:
  1. Dense Semantic Search  — Turbovec cosine-similarity ANN over history vectors
  2. Sparse Keyword Search  — BM25 over the text corpus of user reviews

The final merged score balances both signals so that candidates that appear
in both ranked lists get a strong boost, implementing the classic RRF formula:

    RRF(d) = Σ_r  1 / (k + rank_r(d))

Where k=60 is the standard RRF constant recommended in the literature.

Design choices:
- The BM25 index is built on-demand from user_profiles.json (train_reviews).
- Dense results come from VectorStore.search() (already cached in Turbovec).
- This module is stateless and callable per-request with minimal overhead.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from core.profiles import profiles_list as _load_profiles
from core.utils import tokenize as _tokenize_util

log = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    log.warning(
        "rank-bm25 is not installed. Sparse keyword search will be skipped. "
        "Install it with: pip install rank-bm25"
    )

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Thin wrapper around core.utils.tokenize (re-exported for local use)."""
    return _tokenize_util(text)


class ReviewCorpus:
    """
    Builds and manages a BM25 index over all historical user reviews.

    Each "document" in the corpus is a single review (title + body),
    and we associate each document with its product_id and user_id.
    """

    def __init__(self):
        self._bm25: Any = None
        self._doc_meta: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, profiles: list[dict]) -> None:
        """
        (Re)build the BM25 index from a list of user profile dicts.
        Each profile must have a 'train_reviews' list.
        """
        self._doc_meta = []
        self._tokenized_corpus = []

        for profile in profiles:
            user_id = profile.get("user_id", "")
            for review in profile.get("train_reviews", []):
                text = (
                    str(review.get("title", "")) + " " + str(review.get("body", ""))
                ).strip()
                if not text:
                    continue
                tokens = _tokenize(text)
                self._tokenized_corpus.append(tokens)
                self._doc_meta.append(
                    {
                        "user_id": user_id,
                        "product_id": str(review.get("product_id", "")),
                        "product_name": review.get("product_name", ""),
                        "category": review.get("category", ""),
                        "rating": float(review.get("rating", 3.0)),
                        "text": text,
                    }
                )

        if _BM25_AVAILABLE and self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            log.info(
                "BM25 index built: %d review documents from %d profiles.",
                len(self._doc_meta),
                len(profiles),
            )
        else:
            self._bm25 = None

    def search(self, keywords: list[str], top_k: int = 100) -> list[tuple[dict, float]]:
        """
        BM25 keyword search.

        Returns list of (doc_meta, bm25_score) tuples, sorted descending.
        """
        if not self._bm25 or not keywords:
            return []

        scores = self._bm25.get_scores(keywords)
        indexed = sorted(
            ((i, float(s)) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(self._doc_meta[i], score) for i, score in indexed[:top_k]]


_corpus: ReviewCorpus | None = None
_corpus_mtime: float = 0.0
_corpus_lock = threading.Lock()


def _get_corpus() -> ReviewCorpus:
    """Return the BM25 corpus, rebuilding only when the profiles file changes."""
    global _corpus, _corpus_mtime

    _profiles_path = Path(__file__).parent.parent / "data" / "user_profiles.json"
    current_mtime = (
        _profiles_path.stat().st_mtime if _profiles_path.exists() else 0.0
    )
    # Fast path: no lock needed if already built and file hasn't changed
    if _corpus is not None and current_mtime == _corpus_mtime:
        return _corpus

    with _corpus_lock:
        # Re-check under the lock (another thread may have just built it)
        if _corpus is not None and current_mtime == _corpus_mtime:
            return _corpus

        profiles = _load_profiles()

        _corpus = ReviewCorpus()
        _corpus.build(profiles)
        _corpus_mtime = current_mtime

    return _corpus


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[tuple[dict, float]],
    k: int = RRF_K,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict]:
    """
    Merge dense (semantic) and sparse (BM25) results via Reciprocal Rank Fusion.

    Dense results are keyed by product_id (from the item payload).
    Sparse results are keyed by product_id in the doc_meta.

    Each unique product_id receives an RRF score contribution from whichever
    lists it appears in. Items in both lists receive the strongest signal.

    Returns merged list of candidate dicts, sorted by fused RRF score descending.
    """
    scores: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        pid = str(item.get("item_id", item.get("product_id", "")))
        if not pid:
            continue
        rrf_score = dense_weight / (k + rank)
        if pid not in scores:
            scores[pid] = {**item, "rrf_score": 0.0, "retrieval_sources": []}
        scores[pid]["rrf_score"] += rrf_score
        if "dense" not in scores[pid]["retrieval_sources"]:
            scores[pid]["retrieval_sources"].append("dense")

    for rank, (meta, _bm25_score) in enumerate(sparse_results, start=1):
        pid = str(meta.get("product_id", ""))
        if not pid:
            continue
        rrf_score = sparse_weight / (k + rank)
        if pid not in scores:
            scores[pid] = {
                "item_id": pid,
                "name": meta.get("product_name", ""),
                "category": meta.get("category", ""),
                "url": "",
                "rrf_score": 0.0,
                "retrieval_sources": [],
                "sparse_rating": meta.get("rating", 3.0),
            }
        scores[pid]["rrf_score"] += rrf_score
        if "sparse" not in scores[pid]["retrieval_sources"]:
            scores[pid]["retrieval_sources"].append("sparse")

    merged = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged


def hybrid_search(
    dense_results: list[dict],
    sparse_keywords: list[str],
    top_k: int = 100,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict]:
    """
    Run hybrid search: fuse pre-computed dense results with BM25 sparse results.

    Args:
        dense_results:    Already-ranked candidates from Turbovec semantic search.
                          Each dict must have 'item_id' (or 'product_id').
        sparse_keywords:  Keyword tokens for BM25 query (from aspect extraction).
        top_k:            Number of final merged candidates to return.
        dense_weight:     RRF weight for dense results (default 0.7).
        sparse_weight:    RRF weight for sparse results (default 0.3).

    Returns:
        Merged candidate list sorted by RRF score, length ≤ top_k.
    """
    corpus = _get_corpus()
    sparse_results = corpus.search(sparse_keywords, top_k=top_k * 2)

    if not sparse_results:
        log.debug("Sparse search returned 0 results; using dense-only ranking.")
        return dense_results[:top_k]

    merged = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    log.info(
        "Hybrid search: %d dense + %d sparse → %d merged (top %d)",
        len(dense_results),
        len(sparse_results),
        len(merged),
        top_k,
    )
    return merged[:top_k]
