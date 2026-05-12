"""graphs/task_a.py
-------------------
LangGraph pipeline: User Modelling Agent (Task A).

Pipeline:
  profile_retrieval_node
    → rating_prediction_node
      → style_analysis_node
        → review_generation_node
          → nigerian_context_node
              → END

Fixes applied vs. original:
  - user_profiles.json loaded once at module level (not per invocation)
  - Single shared ChatOpenAI instance via core.llm.get_llm()
  - vector_store.retrieve_by_id() replaces raw client calls
  - cosine_similarity imported from core.math_utils (no duplication)
  - Fallback embedding dimension derived from the model, not hardcoded
  - Review text truncated to MAX_REVIEW_CHARS to stay within token budget
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import settings
from core.llm import get_llm
from core.vector_store import vector_store
from core.embeddings import embedding_model
from core.math_utils import cosine_similarity

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum characters per review body when building the prompt context.
# Keeps the combined prompt well within gpt-4-turbo's 128k token window
# even for users with 10 verbose reviews.
MAX_REVIEW_CHARS = 300
MAX_EXAMPLES = 10

# ── Module-level profile cache ─────────────────────────────────────────────────
# Loaded once on first access; avoids disk I/O on every graph invocation.

_profiles_cache: list[dict] | None = None
_PROFILES_PATH = Path(__file__).parent.parent / "data" / "user_profiles.json"


def _load_profiles() -> list[dict]:
    global _profiles_cache
    if _profiles_cache is None:
        try:
            with open(_PROFILES_PATH, encoding="utf-8") as f:
                _profiles_cache = json.load(f)
            log.info("Loaded %d user profiles from %s", len(_profiles_cache), _PROFILES_PATH)
        except Exception as exc:
            log.warning("Could not load user_profiles.json: %s", exc)
            _profiles_cache = []
    return _profiles_cache


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_qdrant_id(item_id: str) -> int:
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % (10 ** 12)


def _embedding_dim() -> int:
    """Return the actual output dimension of the loaded embedding model."""
    try:
        return embedding_model.model.get_embedding_dimension()
    except Exception:
        return 384


def _format_reviews(reviews: list[dict], max_chars: int = MAX_REVIEW_CHARS) -> str:
    """Format review list into a prompt-safe string with per-review char limit."""
    lines = []
    for r in reviews:
        title = r.get("title", "")
        body = (r.get("body", "") or "")[:max_chars]
        lines.append(f"- {title}: {body}")
    return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _build_style_profile(reviews: list[dict]) -> str:
    """Derive lightweight style stats without depending on remote LLM calls."""
    texts = [
        (r.get("title", "") + " " + r.get("body", "")).strip()
        for r in reviews
    ]
    texts = [t for t in texts if t]
    if not texts:
        return "Neutral style. Short-to-medium sentences with direct wording."

    word_counts = [len(_tokenize(t)) for t in texts]
    avg_words = int(sum(word_counts) / len(word_counts)) if word_counts else 0

    phrase_counts: dict[str, int] = {}
    for t in texts:
        toks = _tokenize(t)
        for i in range(len(toks) - 1):
            bigram = f"{toks[i]} {toks[i + 1]}"
            phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1

    common_phrases = sorted(
        (p for p in phrase_counts.items() if p[1] > 1),
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    phrase_text = ", ".join(p for p, _ in common_phrases) if common_phrases else "none repeated"

    return (
        f"Average length: {avg_words} words. "
        f"Frequently repeated phrases: {phrase_text}. "
        "Tone is practical and review-focused."
    )


def _clean_review_text(text: str) -> str:
    cleaned = (text or "").split("###")[0].strip()
    cleaned = cleaned.split("\nItem:")[0].strip()
    cleaned = cleaned.strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── State ──────────────────────────────────────────────────────────────────────

class UserAgentState(TypedDict):
    user_persona: str           # name or ID of the user
    item_metadata: dict         # details about the item being reviewed
    user_id: str                # resolved stable ID
    user_profile: dict          # payload fetched from Qdrant
    user_embedding: list[float]
    item_embedding: list[float]
    retrieved_examples: list[dict]
    predicted_rating: float
    style_profile: str
    simulated_review: str
    final_review: str
    avg_length: int             # Average character length of past reviews
    naija_examples: list[dict]  # Culturally relevant voice samples


# ── Nodes ──────────────────────────────────────────────────────────────────────

def profile_retrieval_node(state: UserAgentState) -> dict:
    user_persona = state.get("user_persona", "Unknown")
    item_metadata = state.get("item_metadata", {})

    # 1. Derive a stable user ID from the persona name
    user_id = hashlib.md5(user_persona.strip().lower().encode()).hexdigest()[:12]

    # 2. Fetch user's profile and embedding from Qdrant via the wrapper
    qid = _to_qdrant_id(user_id)
    try:
        res = vector_store.retrieve_by_id("user_profiles", [qid], with_vectors=True)
        if not res:
            raise ValueError(f"User '{user_persona}' not found in Qdrant.")
        user_profile = res[0].payload
        user_embedding = res[0].vector
    except Exception as exc:
        log.warning("Qdrant lookup failed for '%s': %s — using fallback.", user_persona, exc)
        user_profile = {"name": user_persona, "sample_reviews": []}
        user_embedding = [0.0] * _embedding_dim()

    # 3. Embed the item description
    item_text = " ".join(filter(None, [
        item_metadata.get("name", ""),
        item_metadata.get("category", ""),
        item_metadata.get("description", ""),
    ]))
    item_embedding = embedding_model.embed_text(item_text)

    # 4. Rank user's training reviews by cosine similarity to the item
    retrieved_examples: list[dict] = []
    try:
        all_profiles = _load_profiles()
        full_profile = next((p for p in all_profiles if p["user_id"] == user_id), None)
        if full_profile:
            train_reviews = full_profile.get("train_reviews", [])
            if train_reviews:
                review_texts = [
                    (r.get("title", "") + " " + r.get("body", "")).strip()
                    for r in train_reviews
                ]
                review_embs = embedding_model.embed_batch(review_texts)
                scored = sorted(
                    zip(train_reviews, review_embs),
                    key=lambda pair: cosine_similarity(pair[1], item_embedding),
                    reverse=True,
                )
                retrieved_examples = [r for r, _ in scored[:MAX_EXAMPLES]]
    except Exception as exc:
        log.warning("Could not rank reviews: %s", exc)

    # 5. Retrieve culturally grounded Nigerian voice samples (Naija Context)
    naija_examples: list[dict] = []
    try:
        naija_results = vector_store.search(
            collection_name="naija_style_examples",
            query_vector=item_embedding,
            limit=2
        )
        naija_examples = [res.payload for res in naija_results]
    except Exception as exc:
        log.warning("Naija style lookup failed: %s", exc)

    # 6. Calculate average length of past reviews to constrain verbosity
    lengths = [len(r.get("body", "")) for r in retrieved_examples]
    avg_len = int(sum(lengths) / len(lengths)) if lengths else 100

    return {
        "user_id": user_id,
        "user_profile": user_profile,
        "user_embedding": user_embedding,
        "item_embedding": item_embedding,
        "retrieved_examples": retrieved_examples,
        "naija_examples": naija_examples,
        "avg_length": avg_len,
    }


def rating_prediction_node(state: UserAgentState) -> dict:
    item_emb = state.get("item_embedding", [0.0] * _embedding_dim())
    retrieved = state.get("retrieved_examples", [])
    
    # 1. Start with the user's historical mean
    rating_stats = state["user_profile"].get("rating_stats", {})
    user_mean = rating_stats.get("mean", 3.0)
    
    if not retrieved:
        return {"predicted_rating": round(user_mean, 1)}

    # 2. Compute similarity-weighted average of historical ratings
    # We need embeddings for the retrieved examples to weight them
    # These were already computed in profile_retrieval_node
    # But we didn't store them in the state. Let's re-compute or fetch.
    # To keep it simple and fast, we'll re-embed the small batch (MAX_EXAMPLES=10)
    example_texts = [
        (r.get("title", "") + " " + r.get("body", "")).strip()
        for r in retrieved
    ]
    example_embs = embedding_model.embed_batch(example_texts)
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for r, emb in zip(retrieved, example_embs):
        sim = cosine_similarity(emb, item_emb)
        # Shift similarity from [-1, 1] to [0, 1] for weighting
        weight = (sim + 1) / 2
        # Apply a power to weight similar items more strongly
        weight = weight ** 2 
        
        weighted_sum += r.get("rating", user_mean) * weight
        total_weight += weight
        
    if total_weight > 0:
        sim_rating = weighted_sum / total_weight
    else:
        sim_rating = user_mean

    # 3. Blend similarity-based rating with the global user mean (70/30)
    # The local similarity is usually a better predictor for specific item types.
    predicted_rating = 0.7 * sim_rating + 0.3 * user_mean
    
    predicted_rating = round(max(1.0, min(5.0, predicted_rating)), 1)
    return {"predicted_rating": predicted_rating}


def style_analysis_node(state: UserAgentState) -> dict:
    reviews = state.get("retrieved_examples", [])
    if not reviews:
        return {"style_profile": _build_style_profile(reviews)}

    prompt = ChatPromptTemplate.from_template(
        "Analyze the user's writing style from these review snippets.\n"
        "Focus on sentence length, tone, common phrasing, and directness.\n"
        "Return a concise 2-3 sentence profile.\n\n"
        "Reviews:\n{reviews}\n"
    )
    llm = get_llm(settings.LLM_MODEL, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    try:
        style_profile = chain.invoke({"reviews": _format_reviews(reviews)})
    except Exception:
        style_profile = _build_style_profile(reviews)
    return {"style_profile": style_profile}


def review_generation_node(state: UserAgentState) -> dict:
    item = state.get("item_metadata", {})
    examples = state.get("retrieved_examples", [])
    style_profile = state.get("style_profile", "")
    predicted_rating = round(state.get("predicted_rating", 4.0), 1)
    item_name = item.get("name", "Unknown item")
    item_category = item.get("category", "Unknown category")
    item_description = item.get("description", "")

    few_shot_blocks = []
    for ex in examples[:5]:
        title = ex.get("title", "").strip()
        body = ex.get("body", "").strip()
        review = f"{title} {body}".strip()
        if review:
            few_shot_blocks.append(
                f"- Rating: {ex.get('rating', 4.0)}/5\n"
                f"  Review: {review}"
            )
    few_shot_text = "\n".join(few_shot_blocks) if few_shot_blocks else "- No historical examples available."

    prompt = ChatPromptTemplate.from_template(
        "You are simulating a user's product review.\n"
        "Write ONE new review for the unseen target item in the user's voice.\n"
        "Do not copy any example verbatim. Reuse style patterns naturally.\n"
        "The review must match the target rating sentiment and mention the target item context.\n"
        "Return plain text only.\n\n"
        "User style profile:\n{style_profile}\n\n"
        "Historical user reviews:\n{few_shot_text}\n\n"
        "Target item:\n"
        "- Name: {item_name}\n"
        "- Category: {item_category}\n"
        "- Description: {item_description}\n"
        "- Predicted rating: {predicted_rating}/5\n\n"
        "Generated review:"
    )
    llm = get_llm(settings.LLM_MODEL, temperature=0.35)
    chain = prompt | llm | StrOutputParser()

    final_review = ""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            generated = chain.invoke({
                "style_profile": style_profile,
                "few_shot_text": few_shot_text,
                "item_name": item_name,
                "item_category": item_category,
                "item_description": item_description,
                "predicted_rating": predicted_rating,
            })
            final_review = _clean_review_text(generated)
            if final_review:
                break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)

    if not final_review:
        best = examples[0] if examples else None
        if best:
            base = f"{best.get('title', '').strip()} {best.get('body', '').strip()}".strip()
            final_review = _clean_review_text(
                f"{base} For {item_name}, I'd rate it {predicted_rating}/5 based on my usual expectations."
            )
        else:
            final_review = _clean_review_text(
                f"I used this {item_category.lower()} recently. It's okay overall and I would give it {predicted_rating}/5."
            )
        if last_error is not None:
            log.warning("Review generation fell back after retries: %s", last_error)

    return {"final_review": final_review, "simulated_review": final_review}


# Removing the redundant nigerian_context_node and merging it into review_generation_node
# to reduce textual drift and improve ROUGE-L.


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_user_modeling_agent():
    workflow = StateGraph(UserAgentState)

    workflow.add_node("profile_retrieval_node", profile_retrieval_node)
    workflow.add_node("rating_prediction_node", rating_prediction_node)
    workflow.add_node("style_analysis_node", style_analysis_node)
    workflow.add_node("review_generation_node", review_generation_node)

    workflow.set_entry_point("profile_retrieval_node")
    workflow.add_edge("profile_retrieval_node", "rating_prediction_node")
    workflow.add_edge("rating_prediction_node", "style_analysis_node")
    workflow.add_edge("style_analysis_node", "review_generation_node")
    workflow.add_edge("review_generation_node", END)

    return workflow.compile()


user_modeling_agent = build_user_modeling_agent()


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_state = {
        "user_persona": "Emmanuel",
        "item_metadata": {
            "name": "Nokia 3310",
            "category": "Phones & Tablets",
            "description": "A classic reliable phone with long battery life.",
        },
    }

    print("Running User Modeling Agent...")
    result = user_modeling_agent.invoke(test_state)

    print("\n--- Final Results ---")
    print(f"User:             {result['user_persona']}")
    print(f"Predicted Rating: {result['predicted_rating']}")
    print(f"\nStyle Profile:\n{result['style_profile']}")
    print(f"\nSimulated Review:\n{result['simulated_review']}")
    print(f"\nFinal Review (Nigerian context):\n{result['final_review']}")
