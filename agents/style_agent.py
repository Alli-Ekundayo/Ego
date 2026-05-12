from core.llm import get_llm
from core.config import settings
from langchain_core.prompts import ChatPromptTemplate


class StyleAgent:
    def __init__(self):
        self.llm = get_llm(settings.LLM_MODEL)
        self.prompt = ChatPromptTemplate.from_template(
            "Analyze the following reviews written by a user and extract their writing persona.\n"
            "Focus on:\n"
            "1. Sentence length distribution\n"
            "2. Positivity/negativity ratio\n"
            "3. Common phrases\n"
            "4. Formality level\n\n"
            "Reviews:\n{reviews}\n\n"
            "Provide a concise summary profile of their style."
        )

    def extract_style(self, reviews: list[str]) -> str:
        if not reviews:
            return "Standard neutral writing style, average length, objective."
        response = (self.prompt | self.llm).invoke({"reviews": "\n".join(reviews)})
        return response.content


style_agent = StyleAgent()
