"""
agents/rerank_agent.py
-----------------------
Reranking Agent: two-stage pipeline.

Stage 1 — Cross-Encoder Reranking (fast, local):
    Score each (query, candidate) pair with a cross-encoder model.
    Apply a persona-conditioned emotional intensity multiplier so that
    historically opinionated users get aspect-aligned candidates boosted.
    → Prunes candidates from ~50 down to top-30.

Stage 2 — LLM Personalisation Reranking:
    The LLM reads the user profile, context, and top-30 cross-encoder
    candidates, then returns a personalised ranking with natural-language
    reasons for each recommendation.
    → Final top-N for the user.
"""

import json
import logging
import re
from functools import cached_property
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from core.config import settings
from core.cross_encoder import cross_encoder_rerank
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


class RerankAgent:
    """
    Two-stage reranker:
      1. Cross-encoder + persona emotional intensity (local, fast).
      2. LLM personalisation pass (cloud, with reasoning).

    The LLM is initialised lazily on first use so that a missing or invalid
    GOOGLE_API_KEY does not crash FastAPI at startup.
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
        pre_n: int = 30,
    ) -> list[dict]:
        """
        Prune from ≤50 candidates → top-30 using cross-encoder scores
        weighted by persona emotional intensity.
        """
        try:
            return cross_encoder_rerank(
                query=query,
                candidates=candidates,
                profile_summary=profile_summary,
                top_n=pre_n,
            )
        except Exception as exc:
            log.warning(
                "Cross-encoder pre-ranking failed: %s. Using original order.", exc, exc_info=True
            )
            return candidates[:pre_n]

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

        pre_ranked = self._cross_encoder_prerank(
            query=context,
            candidates=candidates,
            profile_summary=profile,
            pre_n=30,
        )
        log.info(
            "  ↳ Cross-encoder pre-rank: %d → %d candidates (aspects=%s)",
            len(candidates),
            len(pre_ranked),
            aspects or [],
        )

        formatted = "\n".join(
            f"ID: {c.get('item_id')} | Name: {c.get('name')} | "
            f"Category: {c.get('category')} | "
            f"CE-Score: {c.get('cross_encoder_score', 'N/A')} | "
            f"Intensity: {c.get('emotional_intensity', 'N/A')}"
            for c in pre_ranked
        )

        aspect_hint = ""
        if aspects:
            aspect_hint = (
                f"\nThe user is specifically interested in these product aspects: "
                f"{', '.join(aspects)}. Prioritise candidates that match these aspects."
            )

        history_hint = ""
        if session_history:
            history_lines = [
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                for m in session_history[-5:]
                if m.get("content")
            ]
            if history_lines:
                history_hint = "\nRecent conversation:\n" + "\n".join(history_lines)

        prompt_text = (
            "You are an expert recommendation engine for Jumia Nigeria.\n"
            "Given the user's profile and current context, rank the top {top_n} items "
            "from the provided list.\n"
            "{aspect_hint}\n\n"
            "User Profile Summary: {profile}\n"
            "Context: {context}\n"
            "{history_hint}\n\n"
            "Candidates List (pre-ranked by cross-encoder + emotional intensity):\n"
            "{candidates}\n\n"
            "CRITICAL: Return a JSON object with a 'ranked_items' key. "
            "Each item MUST include the exact 'item_id' from the candidates list. "
            "For each item, provide a 'reason' in a helpful, personalized tone "
            "(referencing the user's interests and the matched aspects).\n"
            'Example format: {{"ranked_items": [{{"item_id": "...", "reason": "..."}}]}}'
        )

        try:
            response = (
                ChatPromptTemplate.from_template(prompt_text) | self.llm
            ).invoke(
                {
                    "profile": json.dumps(profile, ensure_ascii=False),
                    "context": context,
                    "candidates": formatted,
                    "top_n": top_n,
                    "aspect_hint": aspect_hint,
                    "history_hint": history_hint,
                }
            )
            output = _extract_json(response.content)
            raw_ranked = (
                output.get("ranked_items", []) if isinstance(output, dict) else output
            )

            final_recs = []
            for item in raw_ranked:
                iid = str(item.get("item_id", "")).strip()
                if iid in item_map:
                    enriched = dict(item_map[iid])
                    enriched["reasoning"] = item.get(
                        "reason", "Highly relevant based on your preferences."
                    )
                    final_recs.append(enriched)

                if len(final_recs) >= top_n:
                    break

            return final_recs

        except Exception as exc:
            log.warning(
                "  ↳ LLM reranking failed: %s. Falling back to cross-encoder order.",
                exc,
            )
            return [
                {
                    **item_map[str(c.get("item_id"))],
                    "reasoning": "Recommended based on your profile.",
                }
                for c in pre_ranked[:top_n]
                if str(c.get("item_id")) in item_map
            ]


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
        session_history:    Prior conversation turns — passed to LLM for contextual reranking.
        structured_signals: Signals from context extraction node.
        top_n:              Number of final recommendations.
        aspects:            Extracted product aspects for LLM hint injection.
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
