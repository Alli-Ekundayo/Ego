import hashlib
import re


def to_qdrant_id(item_id: str) -> int:
    """
    Convert a string ID to a Qdrant point ID (integer).
    Uses MD5 hashing to deterministically map any string ID to a valid integer
    in the range [0, 10^12).
    """
    return int(hashlib.md5(str(item_id).encode()).hexdigest(), 16) % (10**12)


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

    # If it starts with a markdown header, remove it to see the body
    if text.startswith("###"):
        lines = text.split("\n")
        if len(lines) > 1 and lines[0].startswith("###"):
            text = "\n".join(lines[1:])

    # Split on common delimiters LLMs use for metadata
    cleaned = text.split("###")[0].strip()
    cleaned = cleaned.split("\nItem:")[0].strip()
    cleaned = cleaned.split("\nRating:")[0].strip()

    # Strip quotes and normalize whitespace
    cleaned = cleaned.strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def safe_loads_langchain(payload_str: str) -> any:
    """
    Safely deserialize a LangChain payload, restricting allowed_objects to 'messages'
    only to prevent unauthorized class instantiation and code execution.
    """
    from langchain_core.load import loads

    return loads(
        payload_str,
        allowed_objects="messages",
        secrets_from_env=False,
    )


def safe_load_langchain(payload_file) -> any:
    """
    Safely deserialize a LangChain payload from a file-like object, restricting
    allowed_objects to 'messages' only.
    """
    from langchain_core.load import load

    return load(
        payload_file,
        allowed_objects="messages",
        secrets_from_env=False,
    )
