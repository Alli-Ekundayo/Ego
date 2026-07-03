"""
agents/rerank_agent.py
-----------------------
Reranking Agent: two-stage pipeline.

Stage 1 — Cross-Encoder + Embedding-Based Scoring (local, fast):
    Score each (query, candidate) pair with a cross-encoder model and blend with:
      - Aspect alignment: cosine similarity between candidate and user aspect embeddings.
      - Category preference: weight from the user's historical category distribution.
      - Retrieval score: upstream RRF / merged retrieval signal.
    Apply a persona-conditioned emotional intensity multiplier.
    → Prunes candidates from ~50 down to top-N deterministically.

Stage 2 — LLM Reason Generation (cloud, top-N only):
    The LLM receives ONLY the top-N already-ranked items (not the full candidate
    list) and writes a short, personalized reason string for each.
    → Does NOT re-rank; ranking is fixed after Stage 1.
    → Falls back to templated reasons on any LLM error; ranking is never lost.

This architecture replaces the previous design where the LLM was responsible for
both ranking and explaining, which was expensive (~1500 tokens/request), slow
(~3-5s cloud round-trip), and fragile (bad JSON / hallucinated item IDs).
"""

import json
import logging
import re
from functools import cached_property
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from core.cross_encoder import (
    _build_candidate_text,
    cross_encoder_rerank,
)
from core.embeddings import embedding_model
from core.llm import get_llm

log = logging.getLogger(__name__)


def _extract_json(text: Any) -> dict:
    """
    Robustly extract a JSON object from an LLM response.
    Handles markdown code fences (```json ... ```) that LLMs frequently add.
    """
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    if not isinstance(text, str):
        text = str(text)

    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in LLM response: {text[:200]!r}")


def _templated_reason(candidate: dict, aspects: list[str] | None) -> str:
    """Generate a safe fallback reason without an LLM call."""
    name = candidate.get("name", "this item")
    if aspects:
        return (
            f"{name} aligns well with your interest in "
            f"{' and '.join(aspects[:2])} and matches your preferences."
        )
    return f"{name} is highly relevant based on your profile and purchase history."


class RerankAgent:
    """
    Two-stage reranker:
      1. Local blended scorer: CE + aspect cosine + category pref + retrieval.
      2. LLM reason generation for top-N items only (not re-ranking).

    The LLM is initialised lazily on first use so that a missing or invalid
    DASHSCOPE_API_KEY does not crash FastAPI at startup.
    """

    @cached_property
    def llm(self):
        """Return a JSON-bound LLM instance, constructed on first access."""
        return get_llm(settings.LLM_MODEL)

    def _cross_encoder_prerank(
        self,
        query: str,
        candidates: list[dict],
        profile_summary: dict,
        aspect_embeddings: list[list[float]],
        candidate_embeddings: list[list[float]],
        pre_n: int = 30,
    ) -> list[dict]:
        """
        Prune from ≤50 candidates → top-30 using the blended local scorer:
        CE score × aspect alignment × category preference × retrieval signal.
        """
        try:
            return cross_encoder_rerank(
                query=query,
                candidates=candidates,
                profile_summary=profile_summary,
                top_n=pre_n,
                aspect_embeddings=aspect_embeddings,
                candidate_embeddings=candidate_embeddings,
            )
        except Exception as exc:
            log.warning(
                "Cross-encoder pre-ranking failed: %s. Using original order.", exc, exc_info=True
            )
            return candidates[:pre_n]

    def _generate_reasons(
        self,
        ranked: list[dict],
        profile: dict,
        context: str,
        aspects: list[str] | None,
        session_history: list[dict] | None,
    ) -> dict[str, str]:
        """
        Ask the LLM to write one short personalized reason per already-ranked item.

        The LLM does NOT re-rank — it receives items in their final order and
        produces a `reasons` dict keyed by item_id. If it fails for any reason,
        the caller falls back to templated reasons so the ranking is preserved.

        Prompt is intentionally small: only top-N item names + user context,
        no profile JSON blob, no scoring metadata.
        """
        items_text = "\n".join(
            f"- ID: {c['item_id']} | {c.get('name', '')}"
            for c in ranked
        )
        aspect_hint = (
            f"The user specifically cares about: {', '.join(aspects)}.\n"
            if aspects
            else ""
        )
        history_hint = ""
        if session_history:
            history_lines = [
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                for m in session_history[-3:]
                if m.get("content")
            ]
            if history_lines:
                history_hint = "Recent conversation:\n" + "\n".join(history_lines) + "\n"

        prompt_text = (
            "You are a friendly recommendation assistant for Jumia Nigeria.\n"
            "The items below are already ranked by relevance. "
            "For each item, write a short (1-2 sentence) personalized reason "
            "referencing the user's context and interests.\n"
            "{aspect_hint}"
            "Context: {context}\n"
            "{history_hint}\n"
            "Items (in ranked order):\n{items}\n\n"
            "Return ONLY a JSON object with a 'reasons' key mapping item_id to reason string.\n"
            'Example: {{"reasons": {{"abc123": "Great battery life for gym sessions.", ...}}}}'
        )

        try:
            response = (
                ChatPromptTemplate.from_template(prompt_text) | self.llm
            ).invoke(
                {
                    "context": context,
                    "aspect_hint": aspect_hint,
                    "history_hint": history_hint,
                    "items": items_text,
                }
            )
            output = _extract_json(response.content)
            reasons = output.get("reasons", {}) if isinstance(output, dict) else {}
            if isinstance(reasons, dict):
                return {str(k): str(v) for k, v in reasons.items()}
        except Exception as exc:
            log.warning("LLM reason generation failed: %s. Using templated reasons.", exc)

        return {}

    def rerank(
        self,
        profile: dict,
        context: str,
        candidates: list[dict],
        top_n: int = 10,
        aspects: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> list[dict]:
        if not candidates:
            return []

        item_map = {str(c.get("item_id")): c for c in candidates}

        # ------------------------------------------------------------------
        # Stage 1a: Embed aspects once — one batch call, all cached.
        # Aspect queries are constructed as "<aspect>: <context>" so the
        # embedding captures both the concept and the user's intent.
        # ------------------------------------------------------------------
        aspect_embeddings: list[list[float]] = []
        if aspects:
            aspect_queries = [f"{asp}: {context}" for asp in aspects]
            aspect_embeddings = embedding_model.embed_batch(aspect_queries)
            log.info("  ↳ Embedded %d aspect queries", len(aspect_embeddings))

        # ------------------------------------------------------------------
        # Stage 1b: Embed all candidates in one batch (disk-cached, fast).
        # ------------------------------------------------------------------
        candidate_texts = [_build_candidate_text(c) for c in candidates]
        candidate_embeddings = embedding_model.embed_batch(candidate_texts)
        log.info("  ↳ Embedded %d candidate texts", len(candidate_embeddings))

        # ------------------------------------------------------------------
        # Stage 1c: Blended local scoring → deterministic top-30 ranking.
        # ------------------------------------------------------------------
        pre_ranked = self._cross_encoder_prerank(
            query=context,
            candidates=candidates,
            profile_summary=profile,
            aspect_embeddings=aspect_embeddings,
            candidate_embeddings=candidate_embeddings,
            pre_n=30,
        )
        log.info(
            "  ↳ Local rerank: %d → %d candidates (aspects=%s)",
            len(candidates),
            len(pre_ranked),
            aspects or [],
        )

        top_candidates = pre_ranked[:top_n]

        # ------------------------------------------------------------------
        # Stage 2: LLM writes reasons for top-N only — does NOT re-rank.
        # Prompt is ~300 tokens (5 items) vs ~1500 tokens (30 items + profile).
        # ------------------------------------------------------------------
        reasons = self._generate_reasons(
            ranked=top_candidates,
            profile=profile,
            context=context,
            aspects=aspects,
            session_history=session_history,
        )

        final_recs = []
        for candidate in top_candidates:
            iid = str(candidate.get("item_id", "")).strip()
            if iid not in item_map:
                continue
            enriched = dict(item_map[iid])
            # Carry over scoring metadata from the reranker.
            for score_key in (
                "cross_encoder_score",
                "aspect_alignment_score",
                "category_weight",
                "emotional_intensity",
                "final_rerank_score",
            ):
                if score_key in candidate:
                    enriched[score_key] = candidate[score_key]
            # Use LLM reason if available; fall back to template.
            enriched["reasoning"] = reasons.get(
                iid, _templated_reason(candidate, aspects)
            )
            final_recs.append(enriched)

        return final_recs


def _profile_summary(profile, structured_signals: dict | None = None) -> dict:
    return {
        "user_id": getattr(profile, "user_id", ""),
        "rating_mean": getattr(profile, "rating_stats", {}).get("mean", 3.0),
        "rating_std": getattr(profile, "rating_stats", {}).get("std", 1.0),
        "category_preferences": getattr(profile, "category_pref", {}),
        "recent_reviews": getattr(profile, "sample_reviews", []),
        "signals": structured_signals or {},
    }


_rerank_agent: RerankAgent = RerankAgent()


def _get_rerank_agent() -> RerankAgent:
    """Return the module-level RerankAgent singleton."""
    return _rerank_agent


def rerank_candidates(
    profile,
    candidates: list[dict],
    context_text: str,
    session_history: list[dict] | None = None,
    structured_signals: dict | None = None,
    top_n: int = 10,
    aspects: list[str] | None = None,
) -> list[dict]:
    """
    Adapter used by Task A and Task B graphs.

    Args:
        profile:            UserProfile object.
        candidates:         Candidate item dicts from retrieval.
        context_text:       User's current request string.
        session_history:    Prior conversation turns — passed to LLM for reason writing.
        structured_signals: Signals from context extraction node.
        top_n:              Number of final recommendations.
        aspects:            Extracted product aspects for embedding alignment.
    """
    summary = _profile_summary(profile, structured_signals)
    return _get_rerank_agent().rerank(
        profile=summary,
        context=context_text,
        candidates=candidates,
        top_n=top_n,
        aspects=aspects,
        session_history=session_history,
    )
