"""
scripts/seed_naija_examples.py
------------------------------
Reads data/jumia_reviews.json and indexes individual review bodies into the
'naija_style_examples' Qdrant collection. This allows the NaijaAgent to
retrieve authentic Nigerian review voices for RAG-based style transfer.
"""

import hashlib
import json
import logging
from pathlib import Path

from core.embeddings import embedding_model
from core.vector_store import vector_store

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

RAW_DATA_PATH = Path("data/jumia_reviews.json")
COLLECTION_NAME = "naija_style_examples"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2
BATCH_SIZE = 32


def _to_qdrant_id(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**12)


def seed_naija_examples():
    if not RAW_DATA_PATH.exists():
        log.error(f"{RAW_DATA_PATH} not found. Run scripts/scrape_jumia.py first.")
        return

    with open(RAW_DATA_PATH, encoding="utf-8") as f:
        products = json.load(f)

    all_reviews = []
    for p in products:
        for r in p.get("reviews", []):
            body = r.get("body", "").strip()
            if len(body) > 10:  # ignore very short reviews
                all_reviews.append(
                    {
                        "text": body,
                        "product_name": p.get("name", ""),
                        "category": p.get("category", ""),
                    }
                )

    if not all_reviews:
        log.warning("No reviews found in jumia_reviews.json.")
        return

    log.info(f"Creating collection '{COLLECTION_NAME}'...")
    vector_store.create_collection(COLLECTION_NAME, VECTOR_SIZE)

    total = len(all_reviews)
    log.info(f"Indexing {total} reviews into '{COLLECTION_NAME}'...")

    indexed = 0
    for i in range(0, total, BATCH_SIZE):
        batch = all_reviews[i : i + BATCH_SIZE]
        texts = [b["text"] for b in batch]
        vectors = embedding_model.embed_batch(texts)
        ids = [_to_qdrant_id(b["text"]) for b in batch]

        vector_store.upsert(COLLECTION_NAME, ids, vectors, batch)
        indexed += len(batch)
        if indexed % 100 == 0 or indexed == total:
            log.info(f"  Indexed {indexed} / {total}")

    log.info("Done seeding naija_style_examples.")


if __name__ == "__main__":
    seed_naija_examples()
