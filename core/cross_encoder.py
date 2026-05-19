"""
core/cross_encoder.py
---------------------
Cross-Encoder Reranking with Persona-Conditioned Emotional Intensity Sorting.

Pipeline:
  1. Score each (query, candidate_text) pair with a cross-encoder model,
     producing a fine-grained relevance score.
  2. Apply a persona-conditioned emotional intensity multiplier derived from
     the user's historical rating variance and sentiment signal.
  3. Final score = cross_encoder_score × emotional_intensity_weight
  4. Sort descending and return top-N candidates.

The cross-encoder used here is a lightweight sentence-transformer cross-encoder
(e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) that runs entirely locally,
keeping latency acceptable for top-30 candidate reranking.

Persona-Conditioned Intensity:
    Users with high rating variance (strong opinions) get a multiplier > 1
    for candidates that match their emotional profile (high or low rated).
    Users with low variance (neutral reviewers) get uniform weights.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Lazy-load cross-encoder ───────────────────────────────────────────────────
_cross_encoder_model: Any = None
_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder() -> Any:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            # Check if model is locally cached to prevent hanging downloads
            cache_dir = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
            )
            snapshot_dir = cache_dir / "snapshots"
            has_local = False
            if snapshot_dir.exists():
                for p in snapshot_dir.glob("**/model.safetensors"):
                    if p.exists() and p.stat().st_size > 80_000_000:
                        has_local = True
                        break

            if not has_local:
                log.warning(
                    "Cross-encoder model not found/fully cached locally. "
                    "Using cosine-score/RRF fallback to prevent hanging downloads."
                )
                return None

            from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore

            _cross_encoder_model = CrossEncoder(_CE_MODEL_NAME, local_files_only=True)
            log.info("Cross-encoder loaded: %s", _CE_MODEL_NAME)
        except Exception as exc:
            log.warning(
                "Cross-encoder load failed (%s). Falling back to cosine-score pass-through.",
                exc,
            )
            _cross_encoder_model = None
    return _cross_encoder_model


# ── Emotional intensity scoring ───────────────────────────────────────────────


def compute_emotional_intensity(profile_summary: dict) -> float:
    """
    Derive a persona-conditioned emotional intensity weight [0.8, 1.5].

    High intensity: user has strong opinions (high rating variance,
                    many extreme 1- or 5-star reviews).
    Low intensity:  user is consistently neutral (low variance, all 3-4 stars).

    Args:
        profile_summary: dict with keys like 'rating_mean', 'rating_std',
                         'category_preferences', 'recent_reviews'.

    Returns:
        float multiplier in [0.8, 1.5]
    """
    rating_mean: float = float(profile_summary.get("rating_mean", 3.0))
    rating_std: float = float(profile_summary.get("rating_std", 1.0))

    # If std is missing, estimate from recent_reviews
    if rating_std == 1.0:
        recent: list[dict] = profile_summary.get("recent_reviews", [])
        ratings = []
        for r in recent:
            if isinstance(r, dict):
                ratings.append(float(r.get("rating", 3.0)))
            elif isinstance(r, (int, float)):
                ratings.append(float(r))
        if len(ratings) >= 2:
            mean_r = sum(ratings) / len(ratings)
            variance = sum((x - mean_r) ** 2 for x in ratings) / len(ratings)
            rating_std = math.sqrt(variance)

    # Map std [0, 2.5] → intensity [0.8, 1.5]
    # std=0 (all same ratings)  → 0.8 (uniform, low intensity)
    # std=2.0+ (extreme spread) → 1.5 (high intensity)
    intensity = 0.8 + min(rating_std / 2.0, 1.0) * 0.7
    log.debug(
        "Emotional intensity: mean=%.2f, std=%.2f → weight=%.3f",
        rating_mean,
        rating_std,
        intensity,
    )
    return intensity


def compute_candidate_emotional_match(
    candidate: dict,
    rating_mean: float,
    intensity: float,
) -> float:
    """
    Score how well a candidate's historical rating signal aligns with
    the user's emotional profile.

    - For high-intensity users (strong opinions): boost candidates that
      have extreme ratings (very high or very low), penalize neutral ones.
    - For low-intensity users: keep scores uniform.

    Returns a multiplier in [0.9, 1.1].
    """
    candidate_rating = float(
        candidate.get("sparse_rating", candidate.get("cf_score", rating_mean))
    )
    deviation = abs(candidate_rating - 3.0)  # distance from neutral

    # High intensity users prefer emotionally polarised candidates
    if intensity > 1.2:
        match = 0.9 + (deviation / 2.0) * 0.2  # [0.9, 1.1]
    else:
        match = 1.0  # uniform for neutral users
    return match


# ── Cross-encoder scoring ─────────────────────────────────────────────────────


def _build_candidate_text(candidate: dict) -> str:
    """Build a short passage from candidate metadata for cross-encoder input."""
    parts = [
        candidate.get("name", ""),
        candidate.get("category", ""),
        candidate.get("reasoning", ""),
    ]
    return " | ".join(p for p in parts if p).strip()


def cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    profile_summary: dict,
    top_n: int = 10,
) -> list[dict]:
    """
    Rerank candidates using a cross-encoder + persona-conditioned multiplier.

    Steps:
      1. Build (query, passage) pairs for each candidate.
      2. Score all pairs with the cross-encoder in a single batch call.
      3. Apply emotional intensity multiplier per candidate.
      4. Sort by final score descending, return top_n.

    Args:
        query:          The user's current request / context string.
        candidates:     List of candidate item dicts (with at least 'item_id', 'name').
        profile_summary: Dict with 'rating_mean', 'rating_std', 'recent_reviews'.
        top_n:          Number of candidates to return.

    Returns:
        Reranked list of candidate dicts, each enriched with
        'cross_encoder_score' and 'final_rerank_score'.
    """
    if not candidates:
        return []

    ce_model = _get_cross_encoder()
    intensity = compute_emotional_intensity(profile_summary)
    rating_mean = float(profile_summary.get("rating_mean", 3.0))

    # Build pairs
    pairs = [(query, _build_candidate_text(c)) for c in candidates]

    if ce_model is not None:
        try:
            raw_scores: list[float] = ce_model.predict(pairs).tolist()
        except Exception as exc:
            log.warning("Cross-encoder predict failed: %s. Using fallback scores.", exc)
            raw_scores = [
                float(c.get("rrf_score", c.get("score", 0.5))) for c in candidates
            ]
    else:
        # Fallback: use existing retrieval score (already computed by hybrid_search)
        raw_scores = [
            float(c.get("rrf_score", c.get("score", 0.5))) for c in candidates
        ]

    # Normalise cross-encoder scores to [0, 1] with sigmoid
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    final_candidates = []
    for candidate, ce_score in zip(candidates, raw_scores):
        norm_ce = sigmoid(float(ce_score)) if ce_model is not None else float(ce_score)
        emotional_match = compute_candidate_emotional_match(
            candidate, rating_mean, intensity
        )
        final_score = norm_ce * intensity * emotional_match

        enriched = dict(candidate)
        enriched["cross_encoder_score"] = round(norm_ce, 4)
        enriched["emotional_intensity"] = round(intensity, 4)
        enriched["final_rerank_score"] = round(final_score, 4)
        final_candidates.append(enriched)

    final_candidates.sort(key=lambda x: x["final_rerank_score"], reverse=True)

    log.info(
        "Cross-encoder reranked %d candidates → top %d (intensity=%.2f)",
        len(candidates),
        top_n,
        intensity,
    )
    return final_candidates[:top_n]
