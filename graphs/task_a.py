"""graphs/task_a.py
-------------------
LangGraph pipeline: User Modelling Agent (Task A).

Pipeline:
  profile_retrieval_node
    → rating_prediction_node
      → style_analysis_node
        → review_generation_node
          → naija_injection_node
            → END
"""

import logging
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from core.aspect_extractor import (
    aspects_to_query_strings,
    extract_aspects_rule_based,
    extract_sparse_keywords,
)
from core.config import settings
from core.embeddings import embedding_model
from core.llm import get_llm
from core.math_utils import cosine_similarity
from core.profiles import profiles_by_user_id as _load_profiles_dict
from core.utils import (
    clean_review_text,
    to_stable_id,
    to_vector_id,
    tokenize,
)
from core.vector_store import vector_store

log = logging.getLogger(__name__)

MAX_REVIEW_CHARS = 300
MAX_EXAMPLES = 10


def _embedding_dim() -> int:
    """Dynamically determine the embedding dimension."""
    try:
        return embedding_model.model.get_embedding_dimension()
    except Exception:
        return 384


def _format_reviews(reviews: list[dict], max_chars: int = MAX_REVIEW_CHARS) -> str:
    """Format a list of review dicts into a single prompt-safe string."""
    lines = []
    for r in reviews:
        title = r.get("title", "")
        body = (r.get("body", "") or "")[:max_chars]
        lines.append(f"- {title}: {body}")
    return "\n".join(lines)


def _build_rich_style_profile(reviews: list[dict]) -> str:
    """
    Derive a rich, LLM-free style profile from review statistics.

    Computes: sentence length distribution, vocabulary richness (TTR),
    dominant tone from rating distribution, and recurring bigrams.
    The output string is structured to guide the generation LLM directly.
    """
    texts = [(r.get("title", "") + " " + r.get("body", "")).strip() for r in reviews]
    texts = [t for t in texts if t]
    if not texts:
        return "Neutral style. Short-to-medium sentences with direct wording."

    all_tokens: list[str] = []
    word_counts: list[int] = []
    for t in texts:
        toks = tokenize(t)
        word_counts.append(len(toks))
        all_tokens.extend(toks)

    avg_words = int(sum(word_counts) / len(word_counts)) if word_counts else 0
    ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.5

    if avg_words < 8:
        length_desc = "very terse (fragment-style)"
    elif avg_words < 18:
        length_desc = "short"
    elif avg_words < 35:
        length_desc = "medium"
    else:
        length_desc = "verbose"

    richness_desc = "minimal, repetitive vocabulary" if ttr < 0.55 else (
        "moderately varied vocabulary" if ttr < 0.75 else "rich vocabulary"
    )

    ratings = [float(r.get("rating", 3.0)) for r in reviews if "rating" in r]
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        tone_desc = (
            "enthusiastic and positive" if avg_rating >= 4.5 else
            "generally positive" if avg_rating >= 3.5 else
            "mixed or critical"
        )
    else:
        tone_desc = "practical and review-focused"

    phrase_counts: dict[str, int] = {}
    for t in texts:
        toks = tokenize(t)
        for i in range(len(toks) - 1):
            bigram = f"{toks[i]} {toks[i + 1]}"
            phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
    common_phrases = sorted(
        (p for p, c in phrase_counts.items() if c > 1),
        key=lambda p: phrase_counts[p],
        reverse=True,
    )[:3]
    phrase_text = ", ".join(common_phrases) if common_phrases else "none"

    return (
        f"Sentence style: {length_desc} (~{avg_words} words avg). "
        f"Vocabulary: {richness_desc} (TTR={ttr:.2f}). "
        f"Tone: {tone_desc}. "
        f"Recurring phrases: {phrase_text}."
    )


class UserAgentState(TypedDict):
    user_persona: str
    item_metadata: dict
    user_id: str
    user_profile: dict
    user_embedding: list[float]
    item_embedding: list[float]
    retrieved_examples: list[dict]
    example_embeddings: list[list[float]]
    predicted_rating: float
    style_profile: str
    simulated_review: str
    final_review: str
    avg_length: int
    naija_examples: list[dict]
    extracted_aspects: list[str]
    aspect_queries: list[str]
    sparse_keywords: list[str]
    aspect_exemplars: dict  # {aspect_label: best_matching_review_snippet}


def profile_retrieval_node(state: UserAgentState) -> dict:
    """Task A - Node 1: Profile Retrieval & Example Ranking"""
    user_persona = state.get("user_persona", "Unknown")
    item_metadata = state.get("item_metadata", {})

    profiles_dict = _load_profiles_dict()
    if user_persona in profiles_dict:
        user_id = user_persona
    else:
        user_id = to_stable_id(user_persona)

    qid = to_vector_id(user_id)
    try:
        res = vector_store.retrieve_by_id("user_profiles", [qid], with_vectors=True)
        if not res:
            raise ValueError(f"User '{user_persona}' not found in Turbovec.")
        user_profile = res[0].payload
        user_embedding = res[0].vector
    except Exception as exc:
        log.warning(
            "Turbovec lookup failed for '%s': %s — using fallback.", user_persona, exc
        )
        user_profile = {"name": user_persona, "sample_reviews": []}
        user_embedding = [0.0] * _embedding_dim()

    extracted_aspects = extract_aspects_rule_based(item_metadata)
    aspect_queries = aspects_to_query_strings(extracted_aspects, item_metadata)
    sparse_keywords = extract_sparse_keywords(item_metadata, extracted_aspects)
    log.info("  ↳ Extracted aspects: %s", extracted_aspects)

    item_text = " ".join(
        filter(
            None,
            [
                item_metadata.get("name", ""),
                item_metadata.get("category", ""),
                item_metadata.get("description", ""),
                aspect_queries[0] if aspect_queries else "",
            ],
        )
    )
    item_embedding = embedding_model.embed_text(item_text)

    retrieved_examples: list[dict] = []
    example_embeddings: list[list[float]] = []
    try:
        full_profile = profiles_dict.get(user_id)
        if full_profile:
            train_reviews = full_profile.get("train_reviews", [])
            if train_reviews:
                review_texts = [
                    (r.get("title", "") + " " + r.get("body", "")).strip()
                    for r in train_reviews
                ]
                review_embs = embedding_model.embed_batch(review_texts)

                aspect_embs = (
                    embedding_model.embed_batch(aspect_queries)
                    if aspect_queries
                    else []
                )

                def _aspect_score(review_emb: list[float]) -> float:
                    base = cosine_similarity(review_emb, item_embedding)
                    if not aspect_embs:
                        return base
                    aspect_boost = max(
                        cosine_similarity(review_emb, aq_emb) for aq_emb in aspect_embs
                    )
                    return 0.6 * base + 0.4 * aspect_boost

                scored = sorted(
                    zip(train_reviews, review_embs),
                    key=lambda pair: _aspect_score(pair[1]),
                    reverse=True,
                )
                for r, emb in scored[:MAX_EXAMPLES]:
                    retrieved_examples.append(r)
                    example_embeddings.append(emb)
    except Exception as exc:
        log.warning("Could not rank reviews: %s", exc)

    # Per-aspect exemplar extraction ----------------------------------------
    # For each extracted aspect, find the user's own review that best matches
    # that aspect embedding. This gives the generation LLM targeted evidence
    # ("the user wrote X about battery life") instead of generic aspect labels.
    aspect_exemplars: dict[str, str] = {}
    if aspect_embs and retrieved_examples and example_embeddings:
        for asp, aq_emb in zip(extracted_aspects, aspect_embs):
            best_r, best_score = None, -1.0
            for r, emb in zip(retrieved_examples, example_embeddings):
                score = cosine_similarity(emb, aq_emb)
                if score > best_score:
                    best_score, best_r = score, r
            if best_r is not None:
                snippet = (
                    best_r.get("title", "") + " " + best_r.get("body", "")
                ).strip()[:140]
                if snippet:
                    aspect_exemplars[asp] = snippet
    log.info("  ↳ Aspect exemplars computed: %d", len(aspect_exemplars))
    # -------------------------------------------------------------------------

    naija_examples = []
    try:
        naija_results = vector_store.search(
            collection_name="naija_style_examples", query_vector=item_embedding, limit=2
        )
        naija_examples = [res.payload for res in naija_results]
    except Exception as exc:
        log.warning("Naija style lookup failed: %s", exc)

    lengths = [len(r.get("body", "")) for r in retrieved_examples]
    avg_len = int(sum(lengths) / len(lengths)) if lengths else 100

    return {
        "user_id": user_id,
        "user_profile": user_profile,
        "user_embedding": user_embedding,
        "item_embedding": item_embedding,
        "retrieved_examples": retrieved_examples,
        "example_embeddings": example_embeddings,
        "naija_examples": naija_examples,
        "avg_length": avg_len,
        "extracted_aspects": extracted_aspects,
        "aspect_queries": aspect_queries,
        "sparse_keywords": sparse_keywords,
        "aspect_exemplars": aspect_exemplars,
    }


def rating_prediction_node(state: UserAgentState) -> dict:
    """Task A - Node 2: Similarity-Weighted Rating Prediction"""
    item_emb = state.get("item_embedding", [0.0] * _embedding_dim())
    retrieved = state.get("retrieved_examples", [])
    example_embs = state.get("example_embeddings", [])

    rating_stats = state["user_profile"].get("rating_stats", {})
    user_mean = rating_stats.get("mean", 3.0)

    if not retrieved:
        return {"predicted_rating": round(user_mean, 1)}

    weighted_sum = 0.0
    total_weight = 0.0

    for r, emb in zip(retrieved, example_embs):
        sim = cosine_similarity(emb, item_emb)
        weight = ((sim + 1) / 2) ** 2
        weighted_sum += r.get("rating", user_mean) * weight
        total_weight += weight

    if total_weight > 0:
        sim_rating = weighted_sum / total_weight
    else:
        sim_rating = user_mean

    predicted_rating = 0.7 * sim_rating + 0.3 * user_mean
    predicted_rating = round(max(1.0, min(5.0, predicted_rating)), 1)
    return {"predicted_rating": predicted_rating}


def style_analysis_node(state: UserAgentState) -> dict:
    """
    Task A - Node 3: User Writing Style Analysis (stat-based, no LLM).

    Derives a structured style description from review statistics:
    length distribution, vocabulary richness (TTR), tone from rating mean,
    and recurring bigrams. This replaces the previous LLM call, saving one
    full cloud round-trip per Task A invocation while producing equally
    useful guidance for the generation prompt.
    """
    reviews = state.get("retrieved_examples", [])
    if not reviews:
        return {"style_profile": "Neutral style. Short-to-medium sentences with direct wording."}
    return {"style_profile": _build_rich_style_profile(reviews)}


def review_generation_node(state: UserAgentState) -> dict:
    """
    Task A - Node 4: Review Generation with Naija Voice (merged, single LLM call).

    Previously split across two nodes (review_generation_node + naija_injection_node),
    each making a separate LLM call. The Naija injection rewrote what the first call
    generated, introducing voice drift. Merging into one prompt:
      - Eliminates one full LLM round-trip per invocation.
      - Keeps the Naija voice authentic to the user's style (no rewrite step).
      - Injects per-aspect exemplars (embedding-selected from the user's own reviews)
        as targeted evidence instead of a bare aspect-label hint string.
    """
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
        body = ex.get("body", "").strip()[:MAX_REVIEW_CHARS]
        review = f"{title} {body}".strip()
        if review:
            few_shot_blocks.append(
                f"- Rating: {ex.get('rating', 4.0)}/5\n  Review: {review}"
            )
    few_shot_text = (
        "\n".join(few_shot_blocks)
        if few_shot_blocks
        else "- No historical examples available."
    )

    # Aspect exemplars: targeted evidence from the user's own reviews,
    # one per aspect, selected by embedding cosine in profile_retrieval_node.
    aspect_exemplars: dict = state.get("aspect_exemplars", {})
    extracted_aspects = state.get("extracted_aspects", [])
    if aspect_exemplars:
        exemplar_lines = [
            f"  - {asp}: '{snippet}'"
            for asp, snippet in aspect_exemplars.items()
        ]
        aspect_hint = (
            f"Key aspects to address (with examples from this user's past reviews):\n"
            + "\n".join(exemplar_lines)
        )
    elif extracted_aspects:
        aspect_hint = f"Key aspects to address in the review: {', '.join(extracted_aspects)}."
    else:
        aspect_hint = ""

    # Naija voice examples retrieved by embedding similarity in profile_retrieval_node.
    naija_examples = state.get("naija_examples", [])
    naija_blocks = [f"- {nx.get('text', '').strip()}" for nx in naija_examples if nx.get("text")]
    naija_text = (
        "\n".join(naija_blocks)
        if naija_blocks
        else "- Keep it authentic to a Nigerian e-commerce platform."
    )

    prompt = ChatPromptTemplate.from_template(
        "You are simulating a user's product review.\n"
        "Write ONE new review for the unseen target item in the user's voice.\n"
        "Do not copy any example verbatim. Reuse style patterns naturally.\n"
        "The review must match the target rating sentiment and mention the target item context.\n"
        "Add a subtle and natural 'Naija' (Nigerian) nuance to the writing style "
        "(e.g. slight local phrasing or vocabulary) as shown in the local examples below, "
        "but avoid making it exaggerated or overly thick.\n"
        "{aspect_hint}\n"
        "Return plain text only.\n\n"
        "User style profile:\n{style_profile}\n\n"
        "Historical user reviews:\n{few_shot_text}\n\n"
        "Authentic Naija voice examples for inspiration:\n{naija_text}\n\n"
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
            generated = chain.invoke(
                {
                    "style_profile": style_profile,
                    "few_shot_text": few_shot_text,
                    "item_name": item_name,
                    "item_category": item_category,
                    "item_description": item_description,
                    "predicted_rating": predicted_rating,
                    "aspect_hint": aspect_hint,
                    "naija_text": naija_text,
                }
            )
            final_review = clean_review_text(generated)
            if final_review:
                break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)

    if not final_review:
        if last_error:
            log.warning("Review generation fell back: %s", last_error)
        final_review = clean_review_text(
            f"I have mixed feelings but it works as expected. For the {item_name}, I'd rate it {predicted_rating}/5."
        )

    return {"simulated_review": final_review, "final_review": final_review}


def build_user_modeling_agent():
    workflow = StateGraph(UserAgentState)
    workflow.add_node("profile_retrieval_node", profile_retrieval_node)
    workflow.add_node("rating_prediction_node", rating_prediction_node)
    workflow.add_node("style_analysis_node", style_analysis_node)
    workflow.add_node("review_generation_node", review_generation_node)
    # naija_injection_node removed — Naija voice is now merged into
    # review_generation_node as a single combined prompt.

    workflow.set_entry_point("profile_retrieval_node")
    workflow.add_edge("profile_retrieval_node", "rating_prediction_node")
    workflow.add_edge("rating_prediction_node", "style_analysis_node")
    workflow.add_edge("style_analysis_node", "review_generation_node")
    workflow.add_edge("review_generation_node", END)

    return workflow.compile()


user_modeling_agent = build_user_modeling_agent()


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
    print(f"Predicted Rating: {result['predicted_rating']}")
    print(f"Simulated Review: {result['simulated_review']}")
    print(f"Final (Naija) Review: {result['final_review']}")
