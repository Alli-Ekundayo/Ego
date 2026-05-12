"""core/llm.py
-------------
Singleton LLM accessor used by all agents and graph nodes.
Avoids creating a new ChatOpenAI connection on every pipeline invocation.

The api_key falls back to "sk-dummy" when the environment variable is not set.
This keeps the object constructable in tests (where the key is absent) without
requiring a real credential — actual API calls will still fail if dummy is used.
"""

from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import settings


@lru_cache(maxsize=4)
def get_llm(model: str = "gemma-4-26b-a4b-it", temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """
    Return a cached ChatGoogleGenerativeAI instance for the given (model, temperature) pair.
    lru_cache ensures the same object is reused across the process lifetime.
    """
    raw_key = settings.GOOGLE_API_KEY.get_secret_value()
    # ChatGoogleGenerativeAI raises at __init__ if the key is empty; use a recognisable
    # placeholder so tests can import without a live credential.
    api_key = raw_key if raw_key else "dummy-key-for-testing"
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
