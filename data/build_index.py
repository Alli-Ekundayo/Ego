"""
data/build_index.py
-------------------
Reads data/items.json (produced by data/ingest.py + scripts/scrape_jumia.py)
and indexes every product into Qdrant for semantic retrieval.

The text embedded for each product is:
    "<name> <category> <description> <review1> | <review2> | …"

This means the vector space captures authentic Nigerian customer language
from real Jumia reviews, which is essential for the accent / NLP work.

Run:
    python data/build_index.py
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from core.embeddings import embedding_model
from core.vector_store import vector_store

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ITEMS_PATH = Path(__file__).parent / "items.json"
DEFAULT_COLLECTION = "user_profiles"   # changed from 'items' — we now model users
VECTOR_SIZE = 384          # all-MiniLM-L6-v2 output dimension
BATCH_SIZE = 8


def _to_qdrant_id(item_id: str) -> int:
    """
    Qdrant requires integer point IDs.
    We convert arbitrary string IDs to a stable integer via MD5.
    """
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % (10 ** 12)


def build_text(item: dict) -> str:
    """
    Compose the string that will be embedded.
    Reviews are included so Nigerian vernacular is captured in the vector.
    """
    parts = [
        item.get("name", ""),
        item.get("category", ""),
        item.get("description", ""),
    ]
    reviews = item.get("sample_reviews", [])
    if reviews:
        parts.append(" | ".join(reviews))
    return " ".join(filter(None, parts)).strip()


def build_index(collection_name: str = DEFAULT_COLLECTION) -> None:
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(
            f"{ITEMS_PATH} not found. Run 'python scripts/build_user_profiles.py' first."
        )

    with open(ITEMS_PATH, encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    if not items:
        log.warning("items.json is empty — nothing to index.")
        return

    log.info("Creating / recreating Qdrant collection '%s' (dim=%d)…", collection_name, VECTOR_SIZE)
    vector_store.create_collection(collection_name, VECTOR_SIZE)

    total = len(items)
    log.info("Embedding & indexing %d items in batches of %d…", total, BATCH_SIZE)

    indexed = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]

        texts   = [build_text(item) for item in batch]
        vectors = embedding_model.embed_batch(texts)
        ids     = [_to_qdrant_id(item["id"]) for item in batch]

        # Forward ALL keys except 'description' (too large for payload) as metadata
        payloads = [
            {
                k: v for k, v in item.items()
                if k != "description"
            }
            for item in batch
        ]

        vector_store.upsert(collection_name, ids, vectors, payloads)
        indexed += len(batch)
        log.info("  Indexed %d / %d", indexed, total)

    log.info("Done — %d items indexed into Qdrant collection '%s'.", indexed, collection_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed and index items.json into Qdrant.")
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    args = parser.parse_args()
    build_index(collection_name=args.collection)
