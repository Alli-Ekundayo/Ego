import json
import re

from core.llm import get_llm
from core.config import settings
from langchain_core.prompts import ChatPromptTemplate


def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.
    Handles markdown code fences (```json ... ```) that LLMs frequently add.
    """
    # Strip markdown fences if present
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first {...} block anywhere in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in LLM response: {text[:200]!r}")


class RerankAgent:
    def __init__(self):
        self.llm = get_llm(settings.LLM_MODEL).bind(
            response_format={"type": "json_object"}
        )
        self.prompt = ChatPromptTemplate.from_template(
            "You are an expert recommendation reranker.\n"
            "User context/preferences: {context}\n\n"
            "Candidate items:\n{candidates}\n\n"
            "Rerank these items based on how well they match the user context. "
            "Return a JSON object with a single key 'reranked_items' containing a list of objects. "
            "Each object must have 'item_id', 'name', and 'reason' (explaining why it fits the user).\n"
            "Ensure the most relevant items appear first."
        )

    def rerank(self, context: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        formatted = "\n".join(
            f"ID: {c.get('item_id')} | Name: {c.get('name')} | Category: {c.get('category')}"
            for c in candidates
        )

        try:
            response = (self.prompt | self.llm).invoke({
                "context": context,
                "candidates": formatted,
            })
            output = _extract_json(response.content)
            return output.get("reranked_items", candidates)
        except Exception as exc:
            # Log and fall back gracefully to the original order
            import logging
            logging.getLogger(__name__).warning("Reranking failed: %s", exc)
            return candidates


rerank_agent = RerankAgent()


def rerank_candidates(profile, candidates: list[dict], context_text: str, session_history: list[dict] | None = None) -> list[dict]:
    """
    Adapter used by Task B graph.
    """
    _ = profile
    _ = session_history
    return rerank_agent.rerank(context_text, candidates)
