"""
data/ingest.py
--------------
Data ingestion layer for the Ego pipeline.

Supports two modes:
  1. Jumia (default) — loads data/jumia_reviews.json produced by scripts/scrape_jumia.py
  2. Fallback         — minimal synthetic items used for local dev / smoke tests

Run:
    python data/ingest.py                          # uses Jumia data if present
    python data/ingest.py --source fallback        # forces synthetic fallback data
    python data/ingest.py --source jumia --rebuild # re-generates items.json then re-indexes
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = Path("data")
JUMIA_RAW_PATH = DATA_DIR / "jumia_reviews.json"
ITEMS_PATH = DATA_DIR / "items.json"

# ---------------------------------------------------------------------------
# Fallback synthetic data (dev / smoke tests only)
# ---------------------------------------------------------------------------
SYNTHETIC_ITEMS = [
    {
        "id": "syn-001",
        "name": "Sony WH-1000XM5",
        "category": "Electronics",
        "description": "Noise cancelling headphones with 30-hour battery life.",
        "price": "N/A",
        "rating": 4.7,
        "reviews_count": 0,
        "sample_reviews": [],
    },
    {
        "id": "syn-002",
        "name": "Logitech MX Master 3S",
        "category": "Electronics",
        "description": "Ergonomic wireless mouse with silent click.",
        "price": "N/A",
        "rating": 4.6,
        "reviews_count": 0,
        "sample_reviews": [],
    },
    {
        "id": "syn-003",
        "name": "Instant Pot Duo",
        "category": "Home",
        "description": "7-in-1 electric pressure cooker.",
        "price": "N/A",
        "rating": 4.5,
        "reviews_count": 0,
        "sample_reviews": [],
    },
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_jumia_items() -> list[dict]:
    """
    Load items from data/items.json (already transformed by scrape_jumia.py).
    Falls back to loading raw jumia_reviews.json and converting on-the-fly.
    """
    if ITEMS_PATH.exists():
        with open(ITEMS_PATH, encoding="utf-8") as f:
            items = json.load(f)
        log.info("Loaded %d items from %s", len(items), ITEMS_PATH)
        return items

    if JUMIA_RAW_PATH.exists():
        log.info(
            "items.json not found; converting raw Jumia data from %s", JUMIA_RAW_PATH
        )
        with open(JUMIA_RAW_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return _convert_raw_to_items(raw)

    raise FileNotFoundError(
        "No Jumia data found. Run:\n"
        "  python scripts/scrape_jumia.py\n"
        "to scrape Jumia first, then re-run ingest."
    )


def _convert_raw_to_items(raw: list[dict]) -> list[dict]:
    """Convert full raw scraped records → pipeline-ready items."""
    items = []
    for p in raw:
        review_texts = [r["body"] for r in p.get("reviews", []) if r.get("body")]
        reviews_blob = " | ".join(review_texts[:5])
        enriched = " ".join(
            filter(None, [p.get("description", ""), reviews_blob])
        ).strip()
        items.append(
            {
                "id": p["id"],
                "name": p["name"],
                "category": p["category"],
                "description": enriched or p["name"],
                "price": p.get("price", "N/A"),
                "rating": p.get("rating", 0.0),
                "reviews_count": p.get("reviews_count", 0),
                "sample_reviews": review_texts[:5],
            }
        )

    # Persist so build_index.py can read it next time
    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    log.info("Converted & saved %d items → %s", len(items), ITEMS_PATH)
    return items


def load_synthetic_items() -> list[dict]:
    items = SYNTHETIC_ITEMS
    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d synthetic fallback items → %s", len(items), ITEMS_PATH)
    return items


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def ingest_data(source: str = "jumia", rebuild: bool = False) -> list[dict]:
    """
    Main ingestion entry-point.

    Args:
        source  : 'jumia' or 'fallback'
        rebuild : if True and source='jumia', trigger the scraper first

    Returns the list of pipeline-ready item dicts.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if source == "fallback":
        return load_synthetic_items()

    # -- Jumia path --
    if rebuild:
        log.info("Rebuild requested — launching scraper…")
        result = subprocess.run(
            [sys.executable, "scripts/scrape_jumia.py"],
            check=False,
        )
        if result.returncode != 0:
            log.error("Scraper exited with code %d", result.returncode)

    return load_jumia_items()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ego data ingestion layer.")
    parser.add_argument(
        "--source",
        choices=["jumia", "fallback"],
        default="jumia",
        help="Data source to ingest (default: jumia).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-run the Jumia scraper before ingestion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    items = ingest_data(source=args.source, rebuild=args.rebuild)
    log.info("Ingestion complete — %d items ready in %s", len(items), ITEMS_PATH)
    log.info("Next step → run: python data/build_index.py")
