from core.llm import get_llm
from core.config import settings
from core.vector_store import vector_store
from core.embeddings import embedding_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

COL_NAIJA_REVIEWS = "naija_style_examples"


class NaijaAgent:
    def __init__(self):
        self.llm = get_llm(settings.LLM_MODEL)
        self.prompt = ChatPromptTemplate.from_template(
            "Review:\n{review}\n\n"
            "User's past reviews for context:\n{past_reviews}\n\n"
            "{style_guide}\n\n"
            "You are an expert at capturing the authentic 'Naija' (Nigerian) customer voice. "
            "Based on the user's past reviews and the provided style guide, rewrite the given review "
            "to sound like a real Nigerian customer. This includes:\n"
            "1. Using Naija Pidgin phrases where appropriate (e.g., 'abeg', 'e don tey', 'banger', 'correct').\n"
            "2. Injecting local cultural references or idioms if they fit the sentiment.\n"
            "3. Adjusting the structure to match the informal, direct tone typical of Jumia Nigeria reviews.\n\n"
            "CRITICAL: If the user already has a strong Nigerian voice in their past reviews, match it EXACTLY. "
            "If they write in standard English, add subtle local flavors that wouldn't feel out of place for someone living in Nigeria.\n"
            "Ensure the core sentiment and rating of the review remain unchanged. Output ONLY the final review text."
        )

    def retrieve_naija_examples(self, item_vec: list[float], top_k: int = 3) -> list[dict]:
        try:
            results = vector_store.search(
                collection_name=COL_NAIJA_REVIEWS,
                query_vector=item_vec,
                limit=top_k
            )
            return [res.payload for res in results]
        except Exception:
            # Fallback if collection doesn't exist or query fails
            return []

    def inject_context(self, review: str, item_text: str, past_reviews: list[str] | None = None) -> str:
        if not past_reviews:
            past_reviews_text = "No past reviews available. Assume general Nigerian context."
        else:
            past_reviews_text = "\n".join(f"- {r}" for r in past_reviews)

        item_vec = embedding_model.embed_text(item_text)
        naija_examples = self.retrieve_naija_examples(item_vec, top_k=3)

        if len(naija_examples) >= 2:
            style_guide = (
                "Style these examples show the authentic Nigerian review voice for similar products:\n"
                f"{naija_examples[0].get('text', '')}\n"
                f"{naija_examples[1].get('text', '')}\n"
                "Match this tone, structure, and code-switching pattern."
            )
        elif len(naija_examples) == 1:
            style_guide = (
                "Style this example shows the authentic Nigerian review voice for similar products:\n"
                f"{naija_examples[0].get('text', '')}\n"
                "Match this tone, structure, and code-switching pattern."
            )
        else:
            style_guide = ""

        response = (self.prompt | self.llm | StrOutputParser()).invoke({
            "review": review,
            "past_reviews": past_reviews_text,
            "style_guide": style_guide
        })
        return response


naija_agent = NaijaAgent()
