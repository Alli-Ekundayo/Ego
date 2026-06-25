"""agents/memory_agent.py
-------------------------
MemoryAgent — LangGraph-orchestrated memory consolidation agent.

Responsibilities:
  1. Ingest new interaction events and persist them to the MemoryStore.
  2. Run a Qwen-powered consolidation pass: merge redundant memories, promote
     high-signal items to "preference" type, and generate a rolling long-term
     summary that fits in any downstream context window.
  3. Prune decayed memories and enforce the per-user hard cap.

This agent is called:
  - After every /recommend response (post-hook, async)
  - After every /simulate-review response (post-hook, async)
  - Directly via POST /memory/consolidate

Qwen Cloud (DashScope) is used for the consolidation LLM pass because its
context window and instruction-following are well-suited to structured
memory management tasks.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from core.config import settings
from core.memory import MemoryStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class MemoryAgentState(TypedDict):
    user_id: str
    session_id: str
    # New events to ingest
    events: list[dict]           # [{"type": "interaction"|"feedback"|"preference", "content": str, "importance": float, ...}]
    # Computed
    memories_before: int
    memories_after: int
    pruned: int
    evicted: int
    summary: str
    preferences: dict
    error: str | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def ingest_node(state: MemoryAgentState) -> dict:
    """Persist all new interaction events to the MemoryStore."""
    store = MemoryStore(state["user_id"])
    events = state.get("events") or []
    ingested = 0
    for ev in events:
        try:
            store.add_memory(
                content=str(ev.get("content", "")).strip(),
                memory_type=str(ev.get("type", "interaction")),
                importance=float(ev.get("importance", 0.5)),
                session_id=state.get("session_id", ""),
                metadata=ev.get("metadata"),
            )
            ingested += 1
        except Exception as exc:
            log.warning("MemoryAgent.ingest: failed to store event: %s", exc)

    snapshot = store.snapshot()
    log.info(
        "MemoryAgent.ingest: ingested %d events for user %s (total: %d)",
        ingested, state["user_id"], snapshot["memory_count"],
    )
    return {
        "memories_before": snapshot["memory_count"],
        "error": None,
    }


def consolidate_node(state: MemoryAgentState) -> dict:
    """
    Qwen-powered consolidation pass.

    Reads all memories for the user, asks Qwen to:
      - Identify the most important preferences and facts
      - Write a concise long-term summary (≤300 words)
      - Flag any explicit preference keys to upsert

    The summary is stored in memory_summaries and returned in the state.
    Preferences are upserted into user_preferences.
    """
    if state.get("error"):
        return {}

    store = MemoryStore(state["user_id"])
    all_memories = store.get_all_for_consolidation()
    if not all_memories:
        return {"summary": "", "preferences": {}}

    # Build a compact representation to send to Qwen
    memory_lines = "\n".join(
        f"[{m['memory_type']}|imp={m['importance']:.2f}] {m['content']}"
        for m in all_memories[:80]  # cap to avoid context overflow
    )

    prompt = (
        "You are a memory consolidation assistant. You receive a user's interaction history "
        "and must extract stable, actionable user preferences and write a concise long-term "
        "memory summary.\n\n"
        f"User ID: {state['user_id']}\n\n"
        f"Memory log:\n{memory_lines}\n\n"
        "Return a JSON object with:\n"
        "  \"summary\": string (≤300 words, plain English, captures who this user is and what they like/dislike)\n"
        "  \"preferences\": object (key-value pairs of named preferences, e.g. {\"budget\": \"low\", \"top_category\": \"electronics\"})\n"
        "Return ONLY valid JSON, no markdown fences."
    )

    try:
        llm = _get_qwen_llm()
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        import re
        stripped = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        parsed = json.loads(stripped)

        summary = str(parsed.get("summary", "")).strip()
        prefs = parsed.get("preferences", {})
        if not isinstance(prefs, dict):
            prefs = {}

        if summary:
            store.update_summary(summary)
        for k, v in prefs.items():
            store.set_preference(str(k), str(v))

        log.info(
            "MemoryAgent.consolidate: summary written (%d chars), %d prefs upserted for user %s",
            len(summary), len(prefs), state["user_id"],
        )
        return {"summary": summary, "preferences": prefs}

    except Exception as exc:
        log.warning("MemoryAgent.consolidate: Qwen consolidation failed: %s", exc)
        # Fallback: use existing summary unchanged
        return {"summary": store.get_summary(), "preferences": store.get_preferences()}


def prune_node(state: MemoryAgentState) -> dict:
    """Decay-prune stale memories and enforce the per-user hard cap."""
    if state.get("error"):
        return {}

    store = MemoryStore(state["user_id"])
    pruned = store.prune_stale()
    evicted = store.enforce_cap()
    snapshot = store.snapshot()

    log.info(
        "MemoryAgent.prune: pruned=%d evicted=%d remaining=%d for user %s",
        pruned, evicted, snapshot["memory_count"], state["user_id"],
    )
    return {
        "pruned": pruned,
        "evicted": evicted,
        "memories_after": snapshot["memory_count"],
    }


# ---------------------------------------------------------------------------
# Qwen LLM factory
# ---------------------------------------------------------------------------

def _get_qwen_llm():
    """
    Return a LangChain-compatible Qwen chat model via DashScope.

    DashScope exposes an OpenAI-compatible endpoint, so we use
    langchain_openai.ChatOpenAI pointed at the DashScope base URL.

    Falls back to the configured Gemini LLM if DASHSCOPE_API_KEY is absent.
    """
    dashscope_key = settings.DASHSCOPE_API_KEY.get_secret_value() if settings.DASHSCOPE_API_KEY else ""
    if dashscope_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.QWEN_MODEL,
                api_key=dashscope_key,  # type: ignore[arg-type]
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                temperature=0.3,
                timeout=60,
                max_retries=3,
            )
        except Exception as exc:
            log.warning("Qwen LLM init failed, falling back to Gemini: %s", exc)

    # Gemini fallback
    from core.llm import get_llm
    return get_llm(settings.LLM_MODEL, temperature=0.3)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_memory_agent_graph() -> StateGraph:
    g = StateGraph(MemoryAgentState)
    g.add_node("ingest_node", ingest_node)
    g.add_node("consolidate_node", consolidate_node)
    g.add_node("prune_node", prune_node)

    g.set_entry_point("ingest_node")
    g.add_edge("ingest_node", "consolidate_node")
    g.add_edge("consolidate_node", "prune_node")
    g.add_edge("prune_node", END)

    return g.compile()


memory_agent_graph = _build_memory_agent_graph()


# ---------------------------------------------------------------------------
# Public helper — used by API layer
# ---------------------------------------------------------------------------

def recall_for_user(
    user_id: str,
    query: str,
    max_results: int = 10,
    max_tokens: int = 600,
) -> dict:
    """
    Retrieve relevant memories + preferences + summary for a user.

    Returns a dict suitable for injecting into any LLM prompt:
      {
        "summary": str,
        "preferences": dict,
        "recent_memories": [{"content": str, "type": str, "score": float}, ...],
      }
    """
    store = MemoryStore(user_id)
    memories = store.recall(query=query, max_results=max_results, max_tokens=max_tokens)
    return {
        "summary": store.get_summary(),
        "preferences": store.get_preferences(),
        "recent_memories": [
            {
                "content": m["content"],
                "type": m["memory_type"],
                "score": m["score"],
            }
            for m in memories
        ],
    }
