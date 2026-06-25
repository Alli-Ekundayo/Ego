"""
scripts/enrich_prices.py
------------------------
Fast price enrichment for an existing dataset.

Instead of re-scraping all product pages (which takes hours), this script:
  1. Loads existing jumia_reviews.json + items.json
  2. For each product URL, re-fetches ONLY the product page to extract price
     metadata (one request per product, no review pagination)
  3. Patches both JSON files in-place with the new price fields
  4. Saves incrementally every 10 products

Run:
    PYTHONPATH=. python scripts/enrich_prices.py
    PYTHONPATH=. python scripts/enrich_prices.py --limit 200  # partial run
    PYTHONPATH=. python scripts/enrich_prices.py --skip-existing  # skip products that already have a price
"""

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Re-use helpers from the updated scraper ──────────────────────────────────
from scripts.scrape_jumia import (
    USER_AGENTS,
    _extract_card_price,
    _parse_price_text,
    _get,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
REVIEWS_PATH = DATA_DIR / "jumia_reviews.json"
ITEMS_PATH = DATA_DIR / "items.json"


def _extract_page_price(soup: BeautifulSoup, fallback: dict) -> dict:
    """
    Extract price from a Jumia product page.
    Falls back to `fallback` dict if page price is zero.
    """
    price_raw = ""
    price_value = 0.0
    old_price_raw = ""
    old_price_value = 0.0
    discount_percent = 0.0

    # 1. Try class-based selectors for both desktop/mobile (current layout)
    price_el = soup.find(class_=lambda c: c and "-ubpt" in c and "-b" in c)
    if price_el:
        price_raw = price_el.get_text(strip=True)
        price_value = _parse_price_text(price_raw)

    if price_value == 0:
        # Fallback to existing selectors
        for sel in (".-prc", ".price.-prc", ".prc", ".-prc-bx span", "[data-price]"):
            el = soup.select_one(sel)
            if el:
                # Skip empty divs/containers
                txt = el.get_text(strip=True)
                val = _parse_price_text(txt)
                if val > 0:
                    price_raw = txt
                    price_value = val
                    break

    # Old price
    old_price_el = soup.find(class_=lambda c: c and "-ubpt" in c and "-lthr" in c)
    if old_price_el:
        old_price_raw = old_price_el.get_text(strip=True)
        old_price_value = _parse_price_text(old_price_raw)
    else:
        slp = soup.select_one(".-slprc, .slprc")
        if slp:
            old_price_raw = slp.get_text(strip=True)
            old_price_value = _parse_price_text(old_price_raw)

    # Discount badge
    dsct = soup.find(class_=lambda c: c and ("-dsct" in c or "_dsct" in c))
    if dsct:
        m = re.search(r"(\d+(?:\.\d+)?)", dsct.get_text())
        if m:
            discount_percent = float(m.group(1))
    elif old_price_value > 0 and price_value > 0:
        discount_percent = round((old_price_value - price_value) / old_price_value * 100, 1)

    return {
        "price_raw": price_raw or fallback.get("price_raw", ""),
        "price_value": price_value or fallback.get("price_value", 0.0),
        "old_price_raw": old_price_raw or fallback.get("old_price_raw", ""),
        "old_price_value": old_price_value or fallback.get("old_price_value", 0.0),
        "discount_percent": discount_percent or fallback.get("discount_percent", 0.0),
        "currency": "NGN",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich existing dataset with Jumia price metadata.")
    parser.add_argument("--limit", type=int, default=0, help="Max products to process. 0 = all.")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests (s).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip products that already have a non-zero price_value in items.json.",
    )
    args = parser.parse_args()

    if not REVIEWS_PATH.exists():
        log.error("jumia_reviews.json not found at %s", REVIEWS_PATH)
        return
    if not ITEMS_PATH.exists():
        log.error("items.json not found at %s", ITEMS_PATH)
        return

    with open(REVIEWS_PATH, encoding="utf-8") as f:
        reviews_data: list[dict] = json.load(f)
    with open(ITEMS_PATH, encoding="utf-8") as f:
        items_data: list[dict] = json.load(f)

    log.info("Loaded %d products from jumia_reviews.json", len(reviews_data))
    log.info("Loaded %d items from items.json", len(items_data))

    # Build lookup by id for fast patching
    items_by_id = {item["id"]: item for item in items_data}

    session = requests.Session()
    session.headers.update({"Accept-Language": "en-US,en;q=0.9"})

    enriched = 0
    skipped = 0
    failed = 0

    for i, product in enumerate(reviews_data):
        if args.limit > 0 and enriched >= args.limit:
            break

        pid = product.get("id", "")
        url = product.get("url", "")
        if not url:
            skipped += 1
            continue

        # Skip if already has price and --skip-existing is set
        if args.skip_existing and product.get("price_value", 0.0) > 0:
            skipped += 1
            continue

        log.info("[%d/%d] %s", i + 1, len(reviews_data), product.get("name", "?")[:60])
        session.headers["User-Agent"] = random.choice(USER_AGENTS)

        soup = _get(session, url)
        if not soup:
            log.warning("  ✗ Failed to fetch — skipping")
            failed += 1
            time.sleep(args.delay)
            continue

        price_meta = _extract_page_price(soup, fallback={})

        # Patch reviews_data in-place
        product.update(price_meta)

        # Patch items_data if this product exists there
        if pid in items_by_id:
            items_by_id[pid].update(price_meta)

        enriched += 1
        price_str = price_meta.get("price_raw") or "N/A"
        log.info("  ✓ price=%s  old=%s  discount=%s%%",
                 price_str,
                 price_meta.get("old_price_raw") or "—",
                 price_meta.get("discount_percent") or 0)

        # Incremental save every 10 products
        if enriched % 10 == 0:
            with open(REVIEWS_PATH, "w", encoding="utf-8") as f:
                json.dump(reviews_data, f, ensure_ascii=False, indent=2)
            with open(ITEMS_PATH, "w", encoding="utf-8") as f:
                json.dump(list(items_by_id.values()), f, ensure_ascii=False, indent=2)
            log.info("  (incremental save: %d enriched)", enriched)

        time.sleep(args.delay + random.uniform(0, 0.8))

    # Final save
    with open(REVIEWS_PATH, "w", encoding="utf-8") as f:
        json.dump(reviews_data, f, ensure_ascii=False, indent=2)
    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(items_by_id.values()), f, ensure_ascii=False, indent=2)

    log.info("\n── Done ──────────────────────────────────────────────────────")
    log.info("Enriched : %d products", enriched)
    log.info("Skipped  : %d (no URL or already priced)", skipped)
    log.info("Failed   : %d (fetch errors)", failed)
    log.info("Saved    : %s  %s", REVIEWS_PATH, ITEMS_PATH)


if __name__ == "__main__":
    main()
