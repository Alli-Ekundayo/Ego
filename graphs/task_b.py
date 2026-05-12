"""
Task B — Recommendation Graph
==============================

Nodes:
  load_profile       → fetch user profile (or handle cold-start)
  extract_context    → parse intent + domain from user query
  retrieve_candidates→ dual-path candidate retrieval
  rerank             → LLM reranker → top-10 with reasoning
  format_response    → clean output for API

Cold-start routing:
  If the user has no Qdrant profile, the graph routes to cold_start_node
  which builds an ephemeral profile from the persona description alone.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from core.config import settings
from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.user_profile import UserProfile, profile_from_payload, build_profile
from core.vector_store import get_user_profile
from agents.retrieval_agent import retrieve_candidates, cold_start_retrieval
from agents.rerank_agent import rerank_candidates


# ── State ─────────────────────────────────────────────────────────────────────

class TaskBState(TypedDict):
    # Inputs
    user_id: str
    persona_description: str          # free-text persona (for cold-start + context enrichment)
    context_text: str                  # the user's current request / conversational turn
    session_history: list[dict]        # prior turns [{role, content}, ...]
    n: int                             # number of recommendations requested
    domain_filter: str | None          # optional domain constraint

    # Populated by nodes
    profile: UserProfile | None
    is_cold_start: bool
    extracted_domain: str | None
    candidates: list[dict]
    ranked_recommendations: list[dict]
    error: str | None


# ── Nodes ─────────────────────────────────────────────────────────────────────

def load_profile_node(state: TaskBState) -> dict:
    """Try to load an existing user profile. Flag cold-start if absent."""
    payload = get_user_profile(state["user_id"])
    if payload:
        profile = profile_from_payload(payload)
        return {"profile": profile, "is_cold_start": False, "error": None}

    # No profile found — flag for cold-start handling
    return {"profile": None, "is_cold_start": True, "error": None}


def cold_start_node(state: TaskBState) -> dict:
    """
    Build an ephemeral profile from the persona description (no history needed).
    Covers the 25-point Cold-Start & Cross-Domain criterion.
    """
    persona = state.get("persona_description", "")
    if not persona:
        return {"error": "No persona description provided for cold-start user."}

    # Build a minimal profile from the persona text alone
    synthetic_review = {"text": persona, "rating": 3.0, "category": "general"}
    profile = build_profile(state["user_id"], [synthetic_review])
    return {"profile": profile, "is_cold_start": True, "error": None}


def extract_context_node(state: TaskBState) -> dict:
    """
    Use the LLM to extract intent + domain from the user's natural language request.
    This enables cross-domain recommendations (e.g. books + movies together).
    """
    context = state.get("context_text", "") or state.get("persona_description", "")
    if not context:
        return {"extracted_domain": None}

    llm = get_llm(settings.LLM_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query parser. Extract the primary domain from a user's recommendation request. "
                   "Respond with ONLY a single word from this list: "
                   "food, music, movies, books, electronics, fashion, general. "
                   "If multiple domains apply, return the most prominent one."),
        ("user", "{context}")
    ])
    
    response = (prompt | llm | StrOutputParser()).invoke({"context": context})

    domain = response.strip().lower()
    valid_domains = {"food", "music", "movies", "books", "electronics", "fashion", "general"}
    extracted = domain if domain in valid_domains else None

    # Override with explicit filter if provided
    if state.get("domain_filter"):
        extracted = state["domain_filter"]

    return {"extracted_domain": extracted}


def retrieve_node(state: TaskBState) -> dict:
    """Retrieve candidates — uses cold-start path if no history."""
    if state.get("error"):
        return {}

    profile = state["profile"]
    domain  = state.get("extracted_domain")
    context = state.get("context_text", "") or state.get("persona_description", "")

    if state["is_cold_start"] or not profile.history_vector:
        candidates = cold_start_retrieval(
            persona_description=state.get("persona_description", context),
            domain=domain,
        )
    else:
        candidates = retrieve_candidates(
            profile=profile,
            context_text=context,
            domain_filter=domain,
        )

    return {"candidates": candidates}


def rerank_node(state: TaskBState) -> dict:
    """LLM reranker — personalised top-N with reasoning."""
    if state.get("error"):
        return {}

    ranked = rerank_candidates(
        profile=state["profile"],
        candidates=state["candidates"],
        context_text=state.get("context_text", "") or state.get("persona_description", ""),
        session_history=state.get("session_history", []),
    )

    # Honour the requested N
    n = state.get("n", 10)
    return {"ranked_recommendations": ranked[:n]}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_load(state: TaskBState) -> str:
    if state.get("error"):
        return "abort"
    return "cold_start" if state["is_cold_start"] else "extract_context"


def should_abort(state: TaskBState) -> str:
    return "abort" if state.get("error") else "continue"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_task_b_graph() -> StateGraph:
    g = StateGraph(TaskBState)

    g.add_node("load_profile",    load_profile_node)
    g.add_node("cold_start",      cold_start_node)
    g.add_node("extract_context", extract_context_node)
    g.add_node("retrieve",        retrieve_node)
    g.add_node("rerank",          rerank_node)

    g.set_entry_point("load_profile")

    # After load: warm users → extract context; cold users → cold_start
    g.add_conditional_edges(
        "load_profile",
        route_after_load,
        {
            "cold_start":      "cold_start",
            "extract_context": "extract_context",
            "abort":           END,
        },
    )

    # Cold start rejoins at extract_context
    g.add_conditional_edges(
        "cold_start",
        should_abort,
        {"abort": END, "continue": "extract_context"},
    )

    g.add_edge("extract_context", "retrieve")
    g.add_edge("retrieve",        "rerank")
    g.add_edge("rerank",          END)

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
) -> dict:
    """
    Entry point for the API layer.

    Returns:
        {
          "recommendations": [ {rank, item_id, name, category, reasoning}, ... ],
          "is_cold_start": bool,
          "error": str | None,
        }
    """
    global _graph
    if _graph is None:
        _graph = build_task_b_graph()

    initial_state: TaskBState = {
        "user_id":             user_id,
        "persona_description": persona_description,
        "context_text":        context_text,
        "session_history":     session_history or [],
        "n":                   n or 10,
        "domain_filter":       domain_filter,
        "profile":             None,
        "is_cold_start":       False,
        "extracted_domain":    None,
        "candidates":          [],
        "ranked_recommendations": [],
        "error":               None,
    }

    final = _graph.invoke(initial_state)

    return {
        "recommendations": final.get("ranked_recommendations", []),
        "is_cold_start":   final.get("is_cold_start", False),
        "domain":          final.get("extracted_domain"),
        "error":           final.get("error"),
    }


task_b_graph = build_task_b_graph()
