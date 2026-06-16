"""
core/cross_encoder.py
---------------------
Cross-Encoder Reranking with Persona-Conditioned Emotional Intensity Sorting
and Embedding-Based Aspect Alignment.

Pipeline:
  1. Score each (query, candidate_text) pair with a cross-encoder model,
     producing a fine-grained relevance score.
  2. Apply a persona-conditioned emotional intensity multiplier derived from
     the user's historical rating variance and sentiment signal.
  3. Compute an aspect alignment score: cosine similarity between each
     candidate embedding and the user's target aspect query embeddings.
  4. Apply a category preference weight from the user's historical category
     distribution.
  5. Final blended score:
       0.45 × CE score
     + 0.35 × aspect alignment
     + 0.10 × category preference
     + 0.10 × retrieval score
     all multiplied by the emotional intensity weight.
  6. Sort descending and return top-N candidates.

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
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_cross_encoder_model: Any = None
_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder() -> Any:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = (
                    Path(hf_home)
                    / "hub"
                    / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
                )
            else:
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

    if intensity > 1.2:
        match = 0.9 + (deviation / 2.0) * 0.2  # [0.9, 1.1]
    else:
        match = 1.0  # uniform for neutral users
    return match


def _build_candidate_text(candidate: dict) -> str:
    """Build a short passage from candidate metadata for cross-encoder input."""
    parts = [
        candidate.get("name", ""),
        candidate.get("category", ""),
        candidate.get("reasoning", ""),
    ]
    return " | ".join(p for p in parts if p).strip()


def aspect_alignment_score(
    candidate_embedding: list[float],
    aspect_embeddings: list[list[float]],
) -> float:
    """
    Score how well a candidate aligns with the user's target aspects using
    embedding cosine similarity.

    Takes the max cosine similarity across all aspect query embeddings so that
    a candidate matching *any* high-priority aspect scores well.

    Args:
        candidate_embedding: Dense vector for the candidate's name/category text.
        aspect_embeddings:   One embedding per user-specified aspect query string.

    Returns:
        float in [0, 1]. Returns 0.5 (neutral) when no aspect embeddings supplied.
    """
    from core.math_utils import cosine_similarity

    if not aspect_embeddings or not candidate_embedding:
        return 0.5

    scores = [
        # cosine_similarity returns [-1, 1]; shift to [0, 1]
        (cosine_similarity(candidate_embedding, aq) + 1.0) / 2.0
        for aq in aspect_embeddings
    ]
    return float(max(scores))


def _category_preference_weight(candidate: dict, profile_summary: dict) -> float:
    """
    Return a [0.5, 1.0] weight based on how well the candidate's category
    matches the user's historical category preferences.

    A category the user engages with heavily maps to 1.0; an unknown category
    maps to 0.5, giving a mild penalty over a fully known preferred category.
    """
    from core.utils import normalise_category

    cat_prefs: dict = profile_summary.get("category_preferences", {})
    if not cat_prefs:
        return 0.75  # neutral when no preference data

    candidate_cat = normalise_category(candidate.get("category", ""))
    for pref_cat, pref_weight in cat_prefs.items():
        norm_pref = normalise_category(pref_cat)
        if norm_pref in candidate_cat or candidate_cat in norm_pref:
            # Map preference weight [0, 1] → output [0.5, 1.0]
            return 0.5 + float(pref_weight) * 0.5

    return 0.5  # unknown category


# Blend weights for the final scoring step.
# Adjust these constants to tune the retrieval/aspect/preference balance.
_W_CE: float = 0.45        # cross-encoder relevance
_W_ASPECT: float = 0.35    # embedding-based aspect alignment
_W_CAT: float = 0.10       # user category preference
_W_RETRIEVAL: float = 0.10 # upstream RRF / retrieval score


def cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    profile_summary: dict,
    top_n: int = 10,
    aspect_embeddings: list[list[float]] | None = None,
    candidate_embeddings: list[list[float]] | None = None,
) -> list[dict]:
    """
    Rerank candidates using a blended local scorer:
      CE relevance + aspect alignment + category preference + retrieval score.

    Steps:
      1. Build (query, passage) pairs and score with the cross-encoder.
      2. Compute embedding-based aspect alignment per candidate.
      3. Look up the user's category preference weight per candidate.
      4. Blend all signals with persona-conditioned emotional intensity.
      5. Sort descending, return top_n.

    Args:
        query:               The user's current request / context string.
        candidates:          List of candidate item dicts (with 'item_id', 'name').
        profile_summary:     Dict with 'rating_mean', 'rating_std',
                             'category_preferences', 'recent_reviews'.
        top_n:               Number of candidates to return.
        aspect_embeddings:   Pre-computed embeddings for user aspect query strings.
                             When provided, enables the aspect alignment scoring
                             component. Pass None to skip (falls back to 0.5).
        candidate_embeddings: Pre-computed embeddings for each candidate text,
                             aligned by index with `candidates`. When provided,
                             avoids re-embedding inside this function.

    Returns:
        Reranked list of candidate dicts, each enriched with
        'cross_encoder_score', 'aspect_alignment_score', 'category_weight',
        'emotional_intensity', and 'final_rerank_score'.
    """
    if not candidates:
        return []

    ce_model = _get_cross_encoder()
    intensity = compute_emotional_intensity(profile_summary)
    rating_mean = float(profile_summary.get("rating_mean", 3.0))
    aspect_embs: list[list[float]] = aspect_embeddings or []

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
        raw_scores = [
            float(c.get("rrf_score", c.get("score", 0.5))) for c in candidates
        ]

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    final_candidates = []
    for i, (candidate, ce_score) in enumerate(zip(candidates, raw_scores)):
        norm_ce = sigmoid(float(ce_score)) if ce_model is not None else float(ce_score)
        emotional_match = compute_candidate_emotional_match(
            candidate, rating_mean, intensity
        )

        # Aspect alignment: cosine similarity to user's aspect query embeddings.
        cand_emb: list[float] = (
            candidate_embeddings[i]
            if candidate_embeddings and i < len(candidate_embeddings)
            else []
        )
        asp_score = aspect_alignment_score(cand_emb, aspect_embs)

        # Category preference weight from the user's historical distribution.
        cat_weight = _category_preference_weight(candidate, profile_summary)

        # Upstream retrieval score (RRF or merged score), clamped to [0, 1].
        retrieval_score = min(
            float(candidate.get("rrf_score", candidate.get("score", 0.5))), 1.0
        )

        blended = (
            _W_CE * norm_ce
            + _W_ASPECT * asp_score
            + _W_CAT * cat_weight
            + _W_RETRIEVAL * retrieval_score
        )
        final_score = blended * intensity * emotional_match

        enriched = dict(candidate)
        enriched["cross_encoder_score"] = round(norm_ce, 4)
        enriched["aspect_alignment_score"] = round(asp_score, 4)
        enriched["category_weight"] = round(cat_weight, 4)
        enriched["emotional_intensity"] = round(intensity, 4)
        enriched["final_rerank_score"] = round(final_score, 4)
        final_candidates.append(enriched)

    final_candidates.sort(key=lambda x: x["final_rerank_score"], reverse=True)

    log.info(
        "Cross-encoder reranked %d candidates → top %d "
        "(intensity=%.2f, aspect_embs=%d)",
        len(candidates),
        top_n,
        intensity,
        len(aspect_embs),
    )
    return final_candidates[:top_n]
