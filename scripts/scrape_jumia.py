"""
scripts/scrape_jumia.py
-----------------------
Scrapes product listings and customer reviews from Jumia Nigeria (jumia.com.ng)
using requests + BeautifulSoup.

Key findings from HTML inspection (2025 layout):
  - Category pages:  /phones-tablets/, /electronics/, etc.
  - Product cards:   article.prd  with  a.core (link) and .name (title)
  - SKU:             "sku":"<VALUE>" in the window.__STORE__ JSON blob
  - Reviews page:    /catalog/productratingsreviews/sku/<SKU>/
  - Review block:    article.-pvs.-hr._bet
      title ->  h3.-m.-fs16.-pvs
      body  ->  p.-pvs  (next sibling after h3)
      footer -> .-pvs.-gy7  (date, reviewer, verified badge)

Outputs:
  data/jumia_reviews.json   – full raw data
  data/items.json           – pipeline-ready (read by data/build_index.py)

Usage:
  python scripts/scrape_jumia.py --categories phones-tablets electronics
  python scripts/scrape_jumia.py --categories phones-tablets --limit 10
"""

import argparse
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.jumia.com.ng"

# Confirmed working category URL slugs (these are real Jumia nav paths)
DEFAULT_CATEGORIES: dict[str, str] = {
    "phones-tablets":  "Phones & Tablets",
    "electronics":     "Electronics",
    "computing":       "Computing",
    "home-office":     "Home & Office",
    "fashion":         "Fashion",
    "groceries":       "Grocery",
    "baby-products":   "Baby Products",
    "health-beauty":   "Health & Beauty",
    "sporting-goods":  "Sports",
    "mlp-appliances":  "Appliances",
    "video-games":     "Video Games",
}

# Rotating user-agents mimicking real Nigerian browser traffic
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; Tecno Camon 19) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Samsung SM-A325F) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/111.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Infinix X6831) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
]

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Review:
    title:    str
    body:     str
    reviewer: str
    date:     str
    verified: bool


@dataclass
class Product:
    id:           str
    name:         str
    category:     str
    url:          str
    description:  str
    reviews:      list[Review] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _get(session: requests.Session, url: str, retries: int = 3) -> BeautifulSoup | None:
    """GET a page with retry + exponential back-off. Returns BeautifulSoup or None."""
    session.headers["User-Agent"] = random.choice(USER_AGENTS)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            if resp.status_code == 429:
                wait = 5 * attempt + random.uniform(0, 2)
                log.warning("Rate-limited (429). Waiting %.1fs…", wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %s on attempt %d for %s", resp.status_code, attempt, url)
                time.sleep(2 * attempt)
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(2 * attempt)
    log.error("All retries exhausted for %s", url)
    return None


# ---------------------------------------------------------------------------
# Listing parser
# ---------------------------------------------------------------------------
def has_next_page(soup: BeautifulSoup) -> bool:
    """Return True if a 'Next Page' pagination link exists on the page."""
    return bool(soup.find("a", {"aria-label": "Next Page"}))


def parse_listing(soup: BeautifulSoup, category_label: str) -> list[dict]:
    """
    Extract product cards from a single Jumia category listing page.
    Confirmed selector: article.prd  /  a.core  /  .name
    """
    products = []
    for card in soup.select("article.prd"):
        link = card.select_one("a.core")
        if not link:
            continue
        url  = urljoin(BASE_URL, link["href"])
        name = card.select_one(".name")
        name = name.get_text(strip=True) if name else ""
        if not name:
            continue
        # Products with star ratings are more likely to have reviews
        has_stars = card.select_one(".stars") is not None
        products.append({
            "name":      name,
            "url":       url,
            "category":  category_label,
            "has_stars": has_stars,
        })
    return products


def scrape_listing_pages(
    session: requests.Session,
    cat_slug: str,
    cat_label: str,
    max_pages: int,
) -> list[dict]:
    """
    Crawl up to `max_pages` of a category listing, collecting all product cards.
    Stops early if Jumia returns no more pages (no 'Next Page' link).
    """
    listings: list[dict] = []
    seen_urls: set[str] = set()

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            url = f"{BASE_URL}/{cat_slug}/"
        else:
            url = f"{BASE_URL}/{cat_slug}/?page={page_num}"

        log.info("  Listing page %d/%d → %s", page_num, max_pages, url)
        soup = _get(session, url)
        if not soup:
            log.warning("  Failed to fetch listing page %d, stopping pagination.", page_num)
            break

        page_cards = parse_listing(soup, cat_label)
        new_cards = [c for c in page_cards if c["url"] not in seen_urls]
        seen_urls.update(c["url"] for c in new_cards)
        listings.extend(new_cards)

        if not has_next_page(soup):
            log.info("  No 'Next Page' link — reached last page at page %d.", page_num)
            break

        # Polite pause between listing pages
        time.sleep(random.uniform(0.8, 1.5))

    log.info("  Total unique products found across listing pages: %d", len(listings))
    return listings


# ---------------------------------------------------------------------------
# SKU extractor
# ---------------------------------------------------------------------------
def extract_sku(html_text: str) -> str | None:
    """
    Pull the product SKU from the window.__STORE__ JSON embedded in the page.
    Confirmed pattern: "sku":"IN717EA7WABPLNAFAMZ"
    """
    # The __STORE__ JSON has "sku":"<VALUE>" — grab the first occurrence
    match = re.search(r'"sku"\s*:\s*"([A-Z0-9]+)"', html_text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Review page parser
# ---------------------------------------------------------------------------
def parse_reviews(soup: BeautifulSoup) -> list[Review]:
    """
    Parse review blocks from /catalog/productratingsreviews/sku/<SKU>/.

    Confirmed 2025 HTML structure:
      <article class="-pvs -hr _bet">
        <div class="-df -i-ctr -pvs">
          <div class="stars _s" aria-label="5 out of 5">…</div>
          <h3 class="-m -fs16 -pvs">Review title</h3>
          <p class="-pvs">Review body text</p>
          <div class="-pvs -gy7">23-09-2025 by Francis · Verified Purchase</div>
        </div>
      </article>
    """
    reviews = []
    # Each review is an <article> with these three classes
    for article in soup.find_all("article", class_=lambda c: c and "-pvs" in c and "_bet" in c):
        try:
            # Title: h3.-m.-fs16.-pvs
            h3 = article.find("h3", class_="-m")
            title = h3.get_text(strip=True) if h3 else ""

            # Body: p.-pvs immediately after the h3
            p = h3.find_next_sibling("p") if h3 else article.find("p")
            body = p.get_text(strip=True) if p else ""
            if len(body) < 3:
                continue

            # Footer structure (confirmed from HTML inspection):
            #   <div class="-df -j-bet -i-ctr -gy5">
            #     <div class="-pvs">
            #       <span class="-prs">28-04-2026</span>
            #       <span>by Oluwole</span>
            #     </div>
            #     <div class="-df -i-ctr -gn5 -fsh0">...Verified Purchase</div>
            #   </div>
            footer = article.find("div", class_=lambda c: c and "-j-bet" in c and "-gy5" in c)

            reviewer = "Anonymous"
            date = ""
            verified = False

            if footer:
                info_div = footer.find("div", class_="-pvs")
                if info_div:
                    spans = info_div.find_all("span", recursive=False)
                    if len(spans) >= 1:
                        date = spans[0].get_text(strip=True)   # "28-04-2026"
                    if len(spans) >= 2:
                        by_text = spans[1].get_text(strip=True)  # "by Oluwole"
                        reviewer = re.sub(r'^by\s+', '', by_text, flags=re.I).strip() or "Anonymous"
                # Verified: green badge div has class -gn5
                verified = bool(footer.find(class_=lambda c: c and "-gn5" in c))

            # Star rating: div.stars text is "5 out of 5"
            stars_el = article.find("div", class_="stars")
            rating_text = stars_el.get_text(strip=True) if stars_el else "0"
            rating_match = re.search(r"(\d+\.?\d*)", rating_text)
            rating = float(rating_match.group(1)) if rating_match else 5.0

            reviews.append(Review(
                title=title,
                body=body,
                reviewer=reviewer,
                date=date,
                verified=verified,
            ))
        except Exception as exc:
            log.debug("Skipping review block: %s", exc)

    return reviews


# ---------------------------------------------------------------------------
# Product scraper
# ---------------------------------------------------------------------------
def paginate_reviews(
    session: requests.Session,
    reviews_base_url: str,
    max_review_pages: int = 10,
) -> list[Review]:
    """
    Fetch all pages of reviews for a product.
    Jumia paginates reviews with ?page=N — same 'Next Page' link pattern.
    Stops when there's no next page or after `max_review_pages`.
    """
    all_reviews: list[Review] = []

    for page_num in range(1, max_review_pages + 1):
        if page_num == 1:
            url = reviews_base_url
        else:
            url = f"{reviews_base_url}?page={page_num}"

        log.debug("    Reviews page %d → %s", page_num, url)
        soup = _get(session, url)
        if not soup:
            break

        page_reviews = parse_reviews(soup)
        all_reviews.extend(page_reviews)

        if not has_next_page(soup):
            break

        time.sleep(random.uniform(0.5, 1.0))

    return all_reviews


def scrape_product(
    product_url: str,
    session: requests.Session,
    max_review_pages: int = 10,
) -> tuple[list[Review], str]:
    """
    Fetch a product page, extract the SKU, then paginate through all review pages.
    Returns (reviews, description).
    """
    soup = _get(session, product_url)
    if not soup:
        return [], ""

    # ---- Description ----
    desc_el = (
        soup.select_one(".markup.-pvl")
        or soup.select_one("#productDescription")
        or soup.select_one(".col-s-6 p")
    )
    description = desc_el.get_text(" ", strip=True)[:1000] if desc_el else ""

    # ---- SKU → reviews URL ----
    sku = extract_sku(soup.get_text())   # search rendered text for speed
    if not sku:
        sku = extract_sku(str(soup))     # fallback: full HTML string

    if not sku:
        log.debug("No SKU found for %s", product_url)
        return [], description

    reviews_base = f"{BASE_URL}/catalog/productratingsreviews/sku/{sku}/"
    log.info("    → Reviews: %s", reviews_base)
    reviews = paginate_reviews(session, reviews_base, max_review_pages)
    return reviews, description


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Jumia Nigeria reviews for the Ego NLP/accent pipeline."
    )
    parser.add_argument(
        "--categories", nargs="+",
        default=["phones-tablets"],
        metavar="SLUG",
        help=f"Category slugs. Available: {', '.join(DEFAULT_CATEGORIES.keys())}",
    )
    parser.add_argument(
        "--pages", type=int, default=3,
        help="Number of listing pages to crawl per category (default: 3, ~120 products/page).",
    )
    parser.add_argument(
        "--review-pages", type=int, default=10,
        help="Max review pages to fetch per product (default: 10, ~100 reviews/product).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max products with reviews per category. 0 = no limit (default: 0).",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Base delay (seconds) between product requests (default: 2).",
    )
    args = parser.parse_args()

    session = requests.Session()
    all_products: list[dict] = []

    for cat_slug in args.categories:
        cat_label = DEFAULT_CATEGORIES.get(cat_slug, cat_slug.replace("-", " ").title())

        log.info("\n── Category: %s ──  (pages=1–%d)", cat_label, args.pages)

        # ---- Paginated listing crawl ----
        listings = scrape_listing_pages(session, cat_slug, cat_label, args.pages)
        # Put starred products first — they're more likely to have reviews
        listings.sort(key=lambda x: x["has_stars"], reverse=True)

        found = 0
        for item in listings:
            if args.limit > 0 and found >= args.limit:
                break

            log.info("  ↳ %s", item["name"])
            reviews, desc = scrape_product(
                item["url"], session, max_review_pages=args.review_pages
            )

            if not reviews:
                log.info("    (no reviews — skipping)")
                time.sleep(args.delay * 0.5 + random.uniform(0, 0.5))
                continue

            log.info("    ✓ %d review(s) across %d page(s)",
                     len(reviews),
                     min(args.review_pages, (len(reviews) // 10) + 1))
            product = Product(
                id=hashlib.md5(item["url"].encode()).hexdigest()[:12],
                name=item["name"],
                category=item["category"],
                url=item["url"],
                description=desc,
                reviews=reviews,
            )
            all_products.append(asdict(product))
            found += 1

            jitter = random.uniform(0.5, 1.5)
            time.sleep(args.delay + jitter)

        log.info("Category '%s' done: %d products collected", cat_label, found)

    # ---- Save raw output ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "jumia_reviews.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)
    log.info("\nSaved %d products (raw) → %s", len(all_products), raw_path)

    # ---- Save pipeline-ready items.json ----
    # Nigerian review text is intentionally stitched into `description` so
    # the embedding model captures authentic local language patterns.
    pipeline_items = []
    for p in all_products:
        review_bodies = [r["body"] for r in p["reviews"] if r.get("body")]
        reviews_blob  = " | ".join(review_bodies[:8])
        enriched_desc = " ".join(filter(None, [p.get("description", ""), reviews_blob])).strip()
        pipeline_items.append({
            "id":             p["id"],
            "name":           p["name"],
            "category":       p["category"],
            "description":    enriched_desc or p["name"],
            "url":            p["url"],
            "sample_reviews": review_bodies[:8],
        })

    items_path = DATA_DIR / "items.json"
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_items, f, ensure_ascii=False, indent=2)
    log.info("Saved %d pipeline-ready items → %s", len(pipeline_items), items_path)

    if all_products:
        log.info("\nNext → run:  PYTHONPATH=. python data/build_index.py")
    else:
        log.warning("0 products collected — check your network / category slugs.")


if __name__ == "__main__":
    main()
