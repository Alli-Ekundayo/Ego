"""core/llm.py
-------------
Singleton LLM accessor with persistent disk-based caching for responses.
"""

from functools import lru_cache
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps, loads
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import settings


class SafeSQLiteCache(SQLiteCache):
    """SQLite cache with safe deserialization for cached generations."""

    def lookup(self, prompt: str, llm_string: str):  # type: ignore[override]
        stmt = (
            select(self.cache_schema.response)
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).fetchall()
            if not rows:
                return None
            results = []
            for row in rows:
                raw = row[0]
                try:
                    obj = loads(raw, allowed_objects="messages", secrets_from_env=False)
                    if isinstance(obj, BaseMessage):
                        results.append(ChatGeneration(message=obj))
                    else:
                        results.append(Generation(text=str(raw)))
                except Exception:
                    results.append(Generation(text=str(raw)))
            return results

    def update(self, prompt: str, llm_string: str, return_val):  # type: ignore[override]
        items = []
        for i, gen in enumerate(return_val):
            payload = None
            if (
                isinstance(gen, ChatGeneration)
                and getattr(gen, "message", None) is not None
            ):
                try:
                    payload = dumps(gen.message)
                except Exception:
                    payload = gen.text
            else:
                payload = getattr(gen, "text", "") or str(gen)
            items.append(
                self.cache_schema(
                    prompt=prompt, llm=llm_string, response=payload, idx=i
                )
            )
        with Session(self.engine) as session, session.begin():
            for item in items:
                session.merge(item)


# Initialize persistent SQLite cache for LangChain
cache_path = Path(__file__).parent.parent / "scratch" / "cache" / "llm_cache.db"
cache_path.parent.mkdir(parents=True, exist_ok=True)
set_llm_cache(SafeSQLiteCache(database_path=str(cache_path)))


@lru_cache(maxsize=4)
def get_llm(
    model: str = settings.LLM_MODEL, temperature: float = 0.7
) -> ChatGoogleGenerativeAI:
    """
    Return a cached ChatGoogleGenerativeAI instance.
    lru_cache ensures object reuse, while SQLiteCache ensures response persistence.
    """
    raw_key = settings.GOOGLE_API_KEY.get_secret_value()
    api_key = raw_key if raw_key else "dummy-key-for-testing"

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
