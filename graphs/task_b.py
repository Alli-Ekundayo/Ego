"""
Task B — Recommendation Graph
==============================

Nodes:
  load_profile_node         → fetch user profile and cold-start flag
  context_extraction_node   → parse persona/history/context into structured signals
  aspect_extraction_node    → extract target product aspects + sparse BM25 keywords
  cold_start_node           → build proxy embedding from nearest user cluster
  hybrid_retrieval_node     → Dense semantic + Sparse BM25 + CF (via RRF fusion)
  cross_encoder_rerank_node → Cross-encoder scoring + persona emotional intensity sort
  llm_reranking_node        → LLM personalisation pass (top-N with reasons)
  multiturn_node            → conversational refinement + re-retrieval
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from agents.rerank_agent import _extract_json, rerank_candidates
from agents.retrieval_agent import (
    build_cold_start_proxy_embedding,
    retrieve_candidates,
)
from core.aspect_extractor import (
    aspects_to_query_strings,
    extract_aspects_rule_based,
    extract_sparse_keywords,
)
from core.config import settings
from core.llm import get_llm
from core.user_profile import UserProfile, build_profile, profile_from_payload
from core.vector_store import get_user_profile

# ── Local profile disambiguation (handles short-ID collisions) ────────────────


@lru_cache(maxsize=1)
def _profiles_by_user_id() -> dict[str, list[dict]]:
    path = Path(__file__).resolve().parent.parent / "data" / "user_profiles.json"
    try:
        with path.open(encoding="utf-8") as f:
            profiles = json.load(f)
    except Exception:
        return {}
    grouped: dict[str, list[dict]] = {}
    for p in profiles:
        uid = str(p.get("user_id", "")).strip()
        if not uid:
            continue
        grouped.setdefault(uid, []).append(p)
    return grouped


def _build_profile_from_record(user_id: str, record: dict) -> UserProfile:
    reviews = []
    for r in record.get("train_reviews", []):
        text = f"{r.get('title', '')} {r.get('body', '')}".strip()
        if text:
            reviews.append({"text": text, "rating": float(r.get("rating", 3.0))})
    profile = build_profile(user_id, reviews)
    profile.rating_stats = dict(record.get("rating_stats", profile.rating_stats))
    profile.category_pref = dict(record.get("category_pref", {}))
    profile.sample_reviews = [
        str(r.get("body", "")).strip() for r in record.get("train_reviews", [])[:5]
    ]
    return profile


# ── State ─────────────────────────────────────────────────────────────────────


class TaskBState(TypedDict):
    # Inputs
    user_id: str
    persona_description: str  # free-text persona (for cold-start + context enrichment)
    context_text: str  # the user's current request / conversational turn
    session_history: list[dict]  # prior turns [{role, content}, ...]
    n: int  # number of recommendations requested
    domain_filter: str | None  # optional domain constraint

    # New product feature context (drives aspect extraction)
    new_product_features: dict  # optional metadata for the new/target product

    # Populated by nodes
    profile: UserProfile | None
    is_cold_start: bool
    structured_signals: dict
    extracted_domain: str | None

    # Aspect extraction outputs
    extracted_aspects: list[str]  # e.g. ["Battery Life", "Price / Value"]
    aspect_queries: list[str]  # natural-language semantic query per aspect
    sparse_keywords: list[str]  # BM25 keyword tokens for sparse search

    proxy_embedding: list[float]
    candidates: list[dict]
    ranked_recommendations: list[dict]
    refined_context_text: str
    error: str | None


# ── Nodes ─────────────────────────────────────────────────────────────────────


def load_profile_node(state: TaskBState) -> dict:
    """Try to load an existing user profile. Flag cold-start if absent."""
    user_id = state["user_id"]
    payload = get_user_profile(user_id)

    # Disambiguate shortened hashed IDs using persona name when possible.
    persona_name = str(state.get("persona_description", "")).strip().lower()
    candidates = _profiles_by_user_id().get(user_id, [])
    if persona_name and candidates:
        for record in candidates:
            if str(record.get("name", "")).strip().lower() == persona_name:
                profile = _build_profile_from_record(user_id, record)
                return {"profile": profile, "is_cold_start": False, "error": None}

    if payload:
        profile = profile_from_payload(payload)
        return {"profile": profile, "is_cold_start": False, "error": None}

    # No profile found — flag for cold-start handling
    return {"profile": None, "is_cold_start": True, "error": None}


def context_extraction_node(state: TaskBState) -> dict:
    """
    Parse the user persona into structured signals:
    - explicit preferences
    - implicit signals from conversation history
    - current context (mood, occasion, location)
    """
    llm = get_llm(settings.LLM_MODEL).bind(response_format={"type": "json_object"})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract recommendation signals and return a JSON object with keys: "
                "explicit_preferences (array of strings), "
                "implicit_signals (array of strings), "
                "current_context (object with mood, occasion, location), "
                "inferred_domain (string). "
                "Use empty strings/lists when unknown.",
            ),
            (
                "user",
                "Persona:\n{persona}\n\nConversation history:\n{history}\n\nCurrent request:\n{context}",
            ),
        ]
    )

    history = state.get("session_history", [])
    history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history
    )
    context_text = state.get("context_text", "")
    persona_text = state.get("persona_description", "")

    raw = (prompt | llm | StrOutputParser()).invoke(
        {
            "persona": persona_text,
            "history": history_text,
            "context": context_text,
        }
    )
    structured = _extract_json(raw)
    if not isinstance(structured, dict):
        structured = {}

    extracted_domain = (
        str(structured.get("inferred_domain", "")).strip().lower() or None
    )
    valid_domains = {
        "food",
        "music",
        "movies",
        "books",
        "electronics",
        "fashion",
        "general",
        "computing",
        "video games",
    }
    domain_aliases = {
        "gaming": "video games",
        "gaming audio": None,
        "audio": "electronics",
        "headphones": "electronics",
        "earbuds": "electronics",
        "gadgets": "electronics",
    }
    if extracted_domain in domain_aliases:
        extracted_domain = domain_aliases[extracted_domain]
    elif extracted_domain not in valid_domains:
        extracted_domain = None
    if state.get("domain_filter"):
        extracted_domain = str(state["domain_filter"]).strip().lower()

    import logging

    logger = logging.getLogger(__name__)
    logger.info("  ↳ Inferred domain: %s", extracted_domain)
    return {"structured_signals": structured, "extracted_domain": extracted_domain}


def aspect_extraction_node(state: TaskBState) -> dict:
    """
    Aspect Extraction Node
    ----------------------
    Extract product aspects (e.g. "Battery Life", "Price / Value") from:
      a. new_product_features (if provided — target product for recommendation)
      b. context_text (fallback — treat user request as product description)

    Produces:
      - extracted_aspects: list of aspect labels
      - aspect_queries:    dense semantic search queries (one per aspect)
      - sparse_keywords:   BM25 keyword tokens for sparse search
    """
    import logging

    logger = logging.getLogger(__name__)

    # Build item metadata from available signals
    new_product = state.get("new_product_features") or {}
    if not new_product:
        # Synthesise from context + persona when no explicit product is given
        structured = state.get("structured_signals", {})
        prefs = structured.get("explicit_preferences", [])
        new_product = {
            "name": " ".join(prefs),
            "category": state.get("extracted_domain") or "general",
            "description": " ".join(structured.get("implicit_signals", [])),
        }

    extracted_aspects = extract_aspects_rule_based(new_product)
    aspect_queries = aspects_to_query_strings(extracted_aspects, new_product)
    if not new_product.get("name") and not new_product.get("description"):
        sparse_keywords = []
    else:
        sparse_keywords = extract_sparse_keywords(new_product, extracted_aspects)

    logger.info(
        "  ↳ Aspect extraction: %s | keywords: %s",
        extracted_aspects,
        sparse_keywords[:8] if sparse_keywords else "none",
    )
    return {
        "extracted_aspects": extracted_aspects,
        "aspect_queries": aspect_queries,
        "sparse_keywords": sparse_keywords,
    }


def cold_start_node(state: TaskBState) -> dict:
    """
    Cold-start strategy:
    - infer demographic/preference signals from persona
    - map to nearest user cluster
    - use cluster centroid as proxy embedding
    """
    persona = state.get("persona_description", "")
    if not persona:
        return {"error": "No persona description provided for cold-start user."}

    proxy_embedding = build_cold_start_proxy_embedding(
        persona_description=persona,
        structured_signals=state.get("structured_signals", {}),
    )
    synthetic_review = {
        "text": persona,
        "rating": 3.0,
        "category": state.get("extracted_domain") or "general",
    }
    profile = build_profile(state["user_id"], [synthetic_review])
    profile.history_vector = proxy_embedding
    return {
        "profile": profile,
        "proxy_embedding": proxy_embedding,
        "is_cold_start": True,
        "error": None,
    }


def hybrid_retrieval_node(state: TaskBState) -> dict:
    """
    Hybrid Retrieval Node
    ---------------------
    Dense semantic + CF + Sparse BM25 keyword search, fused via RRF.

    Uses:
      - profile.history_vector   → ANN dense search
      - sparse_keywords           → BM25 search over review corpus
      - extracted_domain          → category filter
    """
    if state.get("error"):
        return {}

    profile = state["profile"]
    domain = state.get("extracted_domain")
    context = state.get("context_text", "") or state.get("persona_description", "")
    sparse_keywords = state.get("sparse_keywords") or []

    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "  ↳ Hybrid retrieval: domain=%s, keywords=%s",
        domain,
        sparse_keywords[:6],
    )

    candidates = retrieve_candidates(
        profile=profile,
        context_text=context,
        domain_filter=domain,
        structured_signals=state.get("structured_signals", {}),
        sparse_keywords=sparse_keywords if sparse_keywords else None,
    )
    return {"candidates": candidates[:100]}  # Pass up to 100 to cross-encoder


def reranking_node(state: TaskBState) -> dict:
    """
    Reranking Node
    --------------
    Two-stage reranker:
    Stage 1 — Cross-encoder + persona emotional intensity (local)
    Stage 2 — LLM personalisation with aspect hints (cloud)
    """
    if state.get("error"):
        return {}

    structured = state.get("structured_signals", {})
    prefs = structured.get("explicit_preferences", [])

    enriched_context = state.get("context_text", "")
    if state.get("persona_description"):
        enriched_context += f" | Persona: {state['persona_description']}"
    if prefs:
        enriched_context += f" | Interests: {', '.join(prefs)}"

    ranked = rerank_candidates(
        profile=state["profile"],
        candidates=state["candidates"],
        context_text=enriched_context,
        session_history=state.get("session_history", []),
        structured_signals=state.get("structured_signals", {}),
        top_n=min(state.get("n", 10), 10),
        aspects=state.get("extracted_aspects"),
    )
    return {"ranked_recommendations": ranked}


def multiturn_node(state: TaskBState) -> dict:
    """
    Conversational refinement:
    if latest turn signals a refinement ("actually", "instead", "more ..."),
    re-retrieve and rerank with updated context.
    """
    history = state.get("session_history", [])
    if not history:
        return {"refined_context_text": state.get("context_text", "")}

    latest_user_turn = ""
    for message in reversed(history):
        if str(message.get("role", "")).lower() == "user":
            latest_user_turn = str(message.get("content", "")).strip()
            break

    refinement_markers = ("actually", "instead", "more ", "less ", "prefer ", "focus ")
    if not latest_user_turn.lower().startswith(refinement_markers):
        return {"refined_context_text": state.get("context_text", "")}

    refined_context = (
        f"{state.get('context_text', '')}\nUser refinement: {latest_user_turn}".strip()
    )
    refined_candidates = retrieve_candidates(
        profile=state["profile"],
        context_text=refined_context,
        domain_filter=state.get("extracted_domain"),
        structured_signals=state.get("structured_signals", {}),
        sparse_keywords=state.get("sparse_keywords") or None,
    )[:100]
    refined_ranked = rerank_candidates(
        profile=state["profile"],
        candidates=refined_candidates,
        context_text=refined_context,
        session_history=history,
        structured_signals=state.get("structured_signals", {}),
        top_n=min(state.get("n", 10), 10),
        aspects=state.get("extracted_aspects"),
    )
    return {
        "refined_context_text": refined_context,
        "candidates": refined_candidates,
        "ranked_recommendations": refined_ranked,
    }


# ── Routing ───────────────────────────────────────────────────────────────────


def route_after_context(state: TaskBState) -> str:
    if state.get("error"):
        return "abort"
    return "cold_start" if state.get("is_cold_start", False) else "hybrid_retrieval"


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_task_b_graph() -> StateGraph:
    g = StateGraph(TaskBState)

    g.add_node("load_profile_node", load_profile_node)
    g.add_node("context_extraction_node", context_extraction_node)
    g.add_node("aspect_extraction_node", aspect_extraction_node)
    g.add_node("cold_start_node", cold_start_node)
    g.add_node("hybrid_retrieval_node", hybrid_retrieval_node)
    g.add_node("reranking_node", reranking_node)
    g.add_node("multiturn_node", multiturn_node)

    g.set_entry_point("load_profile_node")
    g.add_edge("load_profile_node", "context_extraction_node")
    # Aspect extraction always runs after context (gives keywords to retrieval)
    g.add_edge("context_extraction_node", "aspect_extraction_node")

    # After aspect extraction: cold users route through cold_start_node
    g.add_conditional_edges(
        "aspect_extraction_node",
        route_after_context,
        {
            "cold_start": "cold_start_node",
            "hybrid_retrieval": "hybrid_retrieval_node",
            "abort": END,
        },
    )

    g.add_edge("cold_start_node", "hybrid_retrieval_node")
    g.add_edge("hybrid_retrieval_node", "reranking_node")
    g.add_edge("reranking_node", "multiturn_node")
    g.add_edge("multiturn_node", END)

    return g.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

_graph = None


def run_task_b(
    user_id: str,
    context_text: str,
    persona_description: str = "",
    session_history: list[dict] | None = None,
    n: int | None = None,
    domain_filter: str | None = None,
    new_product_features: dict | None = None,
) -> dict:
    """
    Entry point for the API layer.

    Args:
        user_id:              Stable user ID.
        context_text:         User's current request / query.
        persona_description:  Free-text persona (used for cold-start).
        session_history:      Prior conversation turns.
        n:                    Number of recommendations (max 10).
        domain_filter:        Optional category constraint.
        new_product_features: Optional metadata dict for a new/target product
                              to drive aspect extraction.

    Returns:
        {
          "recommendations": [ {rank, item_id, name, category, reasoning}, ... ],
          "is_cold_start": bool,
          "domain": str | None,
          "aspects": list[str],
          "signals": dict,
          "error": str | None,
        }
    """
    global _graph
    if _graph is None:
        _graph = build_task_b_graph()

    initial_state: TaskBState = {
        "user_id": user_id,
        "persona_description": persona_description,
        "context_text": context_text,
        "session_history": session_history or [],
        "n": n or 10,
        "domain_filter": domain_filter,
        "new_product_features": new_product_features or {},
        "profile": None,
        "is_cold_start": False,
        "structured_signals": {},
        "extracted_domain": None,
        "extracted_aspects": [],
        "aspect_queries": [],
        "sparse_keywords": [],
        "proxy_embedding": [],
        "candidates": [],
        "ranked_recommendations": [],
        "refined_context_text": context_text,
        "error": None,
    }

    final = _graph.invoke(initial_state)

    return {
        "recommendations": final.get("ranked_recommendations", []),
        "is_cold_start": final.get("is_cold_start", False),
        "domain": final.get("extracted_domain"),
        "aspects": final.get("extracted_aspects", []),
        "signals": final.get("structured_signals", {}),
        "error": final.get("error"),
    }


task_b_graph = build_task_b_graph()
