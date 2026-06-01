"""
scripts/ingest.py
-----------------
Single entry-point for the Ego data pipeline.

Stage order
-----------
  1. [kaggle]   Download the public dataset from Kaggle (if local files are absent).
                Only KAGGLE_DATASET needs to be set — no API credentials required.
  2. [convert]  Convert jumia_reviews.json → items.json (skipped when
                items.json already exists, unless --rebuild is passed).
  3. [index]    Embed items.json and upsert into Turbovec (only when
                --index flag is set; otherwise call build_index.py separately).

One-time setup (free Kaggle account required even for public datasets)
----------------------------------------------------------------------
  1. Create a free account at https://www.kaggle.com
  2. Go to https://www.kaggle.com/settings → API → Create New Token
  3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json  (chmod 600)

Quick-start for contributors
------------------------------
  Pull data from Kaggle and build the Turbovec index in one command:
  python scripts/ingest.py --index

  Force a full re-download + re-index:
  python scripts/ingest.py --rebuild --index

  Local dev (no Kaggle / no Turbovec) — tiny synthetic dataset:
  python scripts/ingest.py --source fallback

  Run the scraper locally instead of downloading from Kaggle:
  python scripts/ingest.py --scrape --index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = _REPO_ROOT / "data"
JUMIA_RAW_PATH = DATA_DIR / "jumia_reviews.json"
ITEMS_PATH = DATA_DIR / "items.json"

KAGGLE_EXPECTED_FILES = [
    "jumia_reviews.json",
    "items.json",
    "user_profiles.json",
]

SYNTHETIC_ITEMS: list[dict] = [
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


def _all_raw_files_present() -> bool:
    return all((DATA_DIR / f).exists() for f in KAGGLE_EXPECTED_FILES)

def download_data(force: bool = False) -> None:
    """
    Download the raw dataset directly via HTTP.
    No credentials or third-party libraries required.
    """
    if not force and _all_raw_files_present():
        log.info("All data files already present – skipping download.")
        return

    base_url = os.getenv("DATASET_BASE_URL")
    if not base_url:
        log.error(
            "DATASET_BASE_URL is not set in .env.\n"
            "To allow credential-free downloads, set it to your Hugging Face dataset resolve URL."
        )
        sys.exit(1)

    import urllib.error
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in KAGGLE_EXPECTED_FILES:
        target_path = DATA_DIR / filename
        if not force and target_path.exists():
            log.info("  ✓ %s already exists.", filename)
            continue

        file_url = f"{base_url.rstrip('/')}/{filename}"
        log.info("Downloading %s → %s ...", filename, target_path)
        try:
            urllib.request.urlretrieve(file_url, target_path)
        except urllib.error.URLError as e:
            log.error("Failed to download %s: %s", filename, e)
            sys.exit(1)

    log.info("✅  Download complete.")


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
    ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    log.info("Converted & saved %d items → %s", len(items), ITEMS_PATH)
    return items


def load_jumia_items(rebuild: bool = False) -> list[dict]:
    """
    Return pipeline-ready items.

    Resolution order:
      1. data/items.json          (already converted – fastest path)
      2. data/jumia_reviews.json  (convert on-the-fly, saves items.json)
      3. FileNotFoundError        (guide the user to pull data)

    Pass ``rebuild=True`` to force re-conversion even when items.json exists.
    """
    if ITEMS_PATH.exists() and not rebuild:
        with open(ITEMS_PATH, encoding="utf-8") as f:
            items = json.load(f)
        log.info("Loaded %d items from %s", len(items), ITEMS_PATH)
        return items

    if JUMIA_RAW_PATH.exists():
        log.info(
            "%s – re-converting from %s …",
            "--rebuild requested" if rebuild else "items.json not found",
            JUMIA_RAW_PATH,
        )
        with open(JUMIA_RAW_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return _convert_raw_to_items(raw)

    raise FileNotFoundError(
        f"No Jumia data found in {DATA_DIR}.\n"
        "Options:\n"
        "  1. Set DATASET_BASE_URL in .env and run:  python scripts/ingest.py --index\n"
        "  2. Run the scraper locally:              python scripts/scrape_jumia.py\n"
        "  3. Use synthetic data:                   python scripts/ingest.py --source fallback"
    )


def load_synthetic_items() -> list[dict]:
    """Write and return the built-in synthetic fallback items."""
    ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(SYNTHETIC_ITEMS, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d synthetic items → %s", len(SYNTHETIC_ITEMS), ITEMS_PATH)
    return list(SYNTHETIC_ITEMS)


def run_build_index() -> None:
    """Invoke build_index directly to embed and upsert into Turbovec."""
    log.info("Running build_index.py …")
    try:
        from scripts.build_index import build_index as _build_index
        _build_index()
        log.info("✅  Turbovec index built successfully.")
    except Exception as e:
        log.error("build_index.py failed: %s", e)
        sys.exit(1)


def ingest_data(
    source: str = "jumia",
    rebuild: bool = False,
    scrape: bool = False,
    index: bool = False,
) -> list[dict]:
    """
    Main ingestion entry-point.

    Args:
        source:         ``'jumia'`` (default) or ``'fallback'``.
        rebuild:        Force re-download + re-conversion even if files exist.
        scrape:         Run scrape_jumia.py to regenerate data locally.
        index:          Run build_index.py after items are ready.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if source == "fallback":
        items = load_synthetic_items()
    else:
        base_url = os.getenv("DATASET_BASE_URL")
        if base_url:
            download_data(force=rebuild)
        elif not _all_raw_files_present() and not scrape:
            log.warning(
                "Data files not found and DATASET_BASE_URL is not set.\n"
                "Set DATASET_BASE_URL in .env, or pass --scrape to regenerate locally."
            )

        if scrape:
            log.info("Launching scrape_jumia.py …")
            import sys
            original_argv = sys.argv
            sys.argv = [original_argv[0]]
            try:
                from scripts.scrape_jumia import main as _scrape_jumia
                _scrape_jumia()
            except SystemExit as e:
                if getattr(e, 'code', 0) not in (0, None):
                    log.error("Scraper exited with code %s", getattr(e, 'code', 1))
            except Exception as e:
                log.error("Scraper failed: %s", e)
            finally:
                sys.argv = original_argv

        items = load_jumia_items(rebuild=rebuild)

    log.info("Ingestion complete — %d items ready in %s", len(items), ITEMS_PATH)

    if index:
        run_build_index()
    else:
        log.info("Next step → run: python scripts/build_index.py")

    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ego data pipeline entry-point: download from Kaggle, "
            "convert raw data, and optionally build the Turbovec index."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["jumia", "fallback"],
        default="jumia",
        help="Data source: 'jumia' (default) or 'fallback' (tiny synthetic dataset).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-download and re-conversion of raw → items.json.",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Run scrape_jumia.py locally instead of downloading from Kaggle.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Run build_index.py after items are ready to populate Turbovec.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_data(
        source=args.source,
        rebuild=args.rebuild,
        scrape=args.scrape,
        index=args.index,
    )
