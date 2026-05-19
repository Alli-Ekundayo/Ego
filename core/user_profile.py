from __future__ import annotations

from dataclasses import dataclass, field

from core.embeddings import embedding_model


@dataclass
class UserProfile:
    user_id: str
    history_vector: list[float] = field(default_factory=list)
    rating_stats: dict = field(default_factory=dict)
    category_pref: dict = field(default_factory=dict)
    sample_reviews: list[str] = field(default_factory=list)


def profile_from_payload(payload: dict) -> UserProfile:
    """
    Convert a Qdrant payload row into a lightweight UserProfile object.
    """
    return UserProfile(
        user_id=str(payload.get("id", "")),
        history_vector=list(payload.get("history_vector", [])),
        rating_stats=dict(payload.get("rating_stats", {})),
        category_pref=dict(payload.get("category_pref", {})),
        sample_reviews=list(payload.get("sample_reviews", [])),
    )


def build_profile(user_id: str, reviews: list[dict]) -> UserProfile:
    """
    Build an ephemeral profile from sparse review-like records.
    """
    texts = []
    ratings = []
    for review in reviews:
        text = str(review.get("text", "")).strip()
        if text:
            texts.append(text)
        if "rating" in review:
            ratings.append(float(review["rating"]))

    history_vector = embedding_model.embed_text(" ".join(texts)) if texts else []
    mean_rating = round(sum(ratings) / len(ratings), 2) if ratings else 3.0
    return UserProfile(
        user_id=user_id,
        history_vector=history_vector,
        rating_stats={"mean": mean_rating},
        category_pref={},
        sample_reviews=texts[:5],
    )
