import hashlib
import re
import uuid
from typing import Any


def to_vector_id(item_id: str) -> int:
    """
    Convert a string ID to a Turbovec point ID (integer).
    Uses a 63-bit hash space (lower 63 bits of MD5) to fit in the 64-bit signed int
    range required by SQLite while remaining compatible with Turbovec.
    """
    return int(hashlib.md5(str(item_id).encode()).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF


def to_stable_id(name: str) -> str:
    """
    Derive a stable 12-char hex ID from a string (e.g. persona name).
    Normalises casing and whitespace before hashing.
    """
    clean_name = str(name or "unknown").strip().lower()
    return hashlib.md5(clean_name.encode()).hexdigest()[:12]


def normalise_category(category: str) -> str:
    """
    Normalise category strings for consistent matching.
    Lowercases and removes special characters.
    """
    if not category:
        return "general"
    return re.sub(r"[^a-z0-9]+", " ", category.lower()).strip()


def tokenize(text: str) -> list[str]:
    """
    Simple regex-based tokenizer for text analysis.
    Extracts alphanumeric tokens + apostrophes.
    """
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def clean_review_text(text: str) -> str:
    """
    Clean generated review text by removing markdown and extraneous sections.
    """
    if not text:
        return ""

    cleaned = re.sub(r"(?im)^###.*$", "", text)
    cleaned = re.sub(r"(?im)^Item:\s*.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^Rating:\s*.*$", "", cleaned)

    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def safe_loads_langchain(payload: str) -> Any:
    """
    Safely load LangChain serialized content, allowing only messages.
    Calls the underlying loads implementation directly to avoid the @beta
    decorator warning — this usage is intentional and well-understood.
    """
    from langchain_core.load import loads
    # Access __wrapped__ to bypass the @beta warning decorator.
    _loads_impl = getattr(loads, "__wrapped__", loads)
    return _loads_impl(payload, allowed_objects="messages", secrets_from_env=False)

