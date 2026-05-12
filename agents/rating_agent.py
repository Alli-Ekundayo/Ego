from core.llm import get_llm
from core.config import settings
from core.math_utils import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate


class RatingAgent:
    def __init__(self):
        self.llm = get_llm(settings.LLM_MODEL)
        self.prompt = ChatPromptTemplate.from_template(
            "Based on the user's profile: {profile} and the item: {item},\n"
            "The predicted mathematical rating is {rating}/5.\n"
            "Provide a simulated review that matches this rating."
        )

    def predict_rating(
        self,
        user_emb: list[float],
        item_emb: list[float],
        rating_stats: dict,
    ) -> float:
        """
        ML algorithm combining content-based similarity with the user's
        historical rating distribution (mean, std, skew).
        """
        score = cosine_similarity(user_emb, item_emb)
        mean = rating_stats.get("mean", 3.0)
        std = rating_stats.get("std", 1.0)
        skew = rating_stats.get("skew", 0.0)

        # Special case: extremely consistent 5-star givers
        if std < 0.1 and mean >= 4.8:
            if score < 0.2:
                predicted = 3.0
            elif score < 0.4:
                predicted = 4.0
            else:
                predicted = 5.0
        else:
            # z is positive for a good match, negative for a bad one.
            z = (score - 0.5) * 2.0
            skew_adjustment = skew * 0.15
            predicted = mean + (z * std) - skew_adjustment

        return round(max(1.0, min(5.0, predicted)), 1)

    def predict_rating_and_review(
        self,
        profile: str,
        item: dict,
        user_emb: list[float],
        item_emb: list[float],
        rating_stats: dict,
    ) -> dict:
        """Calculate rating via ML algo, then generate review text via LLM."""
        calculated_rating = self.predict_rating(user_emb, item_emb, rating_stats)
        response = (self.prompt | self.llm).invoke({
            "profile": profile,
            "item": str(item),
            "rating": calculated_rating,
        })
        return {"rating": calculated_rating, "review": response.content}


rating_agent = RatingAgent()
