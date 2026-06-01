"""
scripts/build_user_profiles.py
-------------------------------
Inverts the product-centric scraped data (jumia_reviews.json) into
a user-centric structure for the Ego user modelling agent.

Input:  data/jumia_reviews.json      (product → list of reviews)
Output: data/user_profiles.json      (user → list of reviews across products)
        data/items.json              (overwritten in user-profile format for Turbovec)

Each user profile contains:
  - user_id         : stable MD5 hash of the lowercase reviewer name
  - name            : reviewer name as captured from Jumia
  - review_count    : total reviews left by this user
  - verified_count  : how many were Verified Purchases
  - train_reviews   : first 80% of chronological reviews
  - test_reviews    : last 20% of chronological reviews (held out)
  - voice_sample    : concatenation of all train review bodies (used for embedding)
  - rating_stats    : mean, std, skew from train ratings
  - category_pref   : normalized frequency across item categories
  - vocab_fingerprint: top words sorted by TF-IDF

The vector indexed into Turbovec is the user's `voice_sample`, so semantic
search returns users with similar language patterns / Nigerian accent signals.

--min-reviews (default 5) keeps only users with enough reviews to build
an enriched, representative voice model. Run a wider scrape first if 0
users qualify.

Usage:
  python scripts/build_user_profiles.py
  python scripts/build_user_profiles.py --min-reviews 10
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from core.utils import to_stable_id as user_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")

STOPWORDS = {
    "the",
    "and",
    "for",
    "this",
    "that",
    "with",
    "from",
    "have",
    "been",
    "were",
    "they",
    "their",
    "what",
    "which",
    "when",
    "very",
    "just",
    "also",
    "more",
    "some",
    "would",
    "could",
    "then",
    "than",
    "into",
    "will",
    "your",
    "about",
    "like",
    "is",
    "it",
    "to",
    "i",
    "a",
    "of",
    "my",
    "in",
    "it's",
    "are",
    "but",
    "as",
    "on",
    "so",
    "not",
    "can",
    "be",
    "if",
    "or",
    "at",
    "you",
    "good",
    "very",
    "all",
    "out",
    "no",
    "yes",
    "do",
    "we",
    "he",
    "she",
    "an",
    "up",
    "was",
}


def parse_date(date_str: str) -> datetime.datetime:
    """Parse Jumia dates like '26-06-2025' or '05-01-2026'."""
    try:
        return datetime.datetime.strptime(date_str.strip(), "%d-%m-%Y")
    except ValueError:
        return datetime.datetime.min


def get_words(text: str) -> list[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def build_profiles(raw_products: list[dict], min_reviews: int = 1) -> list[dict]:
    user_reviews_map: dict[str, list[dict]] = defaultdict(list)
    display_name_map: dict[str, str] = {}

    for product in raw_products:
        product_context = {
            "product_id": product.get("id", ""),
            "product_name": product.get("name", ""),
            "product_url": product.get("url", ""),
            "category": product.get("category", ""),
        }
        for review in product.get("reviews", []):
            reviewer_raw = review.get("reviewer", "Anonymous").strip() or "Anonymous"
            reviewer_key = reviewer_raw.strip().lower()
            if not reviewer_key:
                reviewer_key = "anonymous"
            if reviewer_key not in display_name_map:
                display_name_map[reviewer_key] = reviewer_raw
            date_obj = parse_date(review.get("date", ""))

            user_reviews_map[reviewer_key].append(
                {
                    **product_context,
                    "title": review.get("title", ""),
                    "body": review.get("body", ""),
                    "rating": review.get("rating", 5.0),
                    "date": review.get("date", ""),
                    "date_obj": date_obj,
                    "verified": review.get("verified", False),
                }
            )

    valid_users = {k: v for k, v in user_reviews_map.items() if len(v) >= min_reviews}
    if not valid_users:
        valid_users = user_reviews_map

    document_frequency = Counter()
    total_users = 0

    user_data = {}
    for name, reviews in valid_users.items():
        reviews.sort(key=lambda r: r["date_obj"])

        test_size = max(1, int(len(reviews) * 0.2))
        train_reviews = reviews[:-test_size]
        test_reviews = reviews[-test_size:]

        for r in train_reviews + test_reviews:
            r.pop("date_obj", None)

        user_data[name] = {"train": train_reviews, "test": test_reviews}

        user_words = set()
        for r in train_reviews:
            user_words.update(get_words(r.get("title", "") + " " + r.get("body", "")))

        for w in user_words:
            document_frequency[w] += 1

        total_users += 1

    profiles = []
    for name, data in user_data.items():
        train_reviews = data["train"]
        test_reviews = data["test"]
        all_reviews = train_reviews + test_reviews

        ratings = [r["rating"] for r in train_reviews if "rating" in r]
        avg_rating = statistics.mean(ratings) if ratings else 0.0
        std_rating = statistics.stdev(ratings) if len(ratings) > 1 else 0.0

        if std_rating > 0:
            median_rating = statistics.median(ratings)
            skew_rating = 3 * (avg_rating - median_rating) / std_rating
        else:
            skew_rating = 0.0

        categories = [r["category"] for r in train_reviews if r.get("category")]
        cat_counts = Counter(categories)
        total_cats = sum(cat_counts.values())
        cat_pref = (
            {cat: count / total_cats for cat, count in cat_counts.items()}
            if total_cats > 0
            else {}
        )

        user_word_counts = Counter()
        for r in train_reviews:
            user_word_counts.update(
                get_words(r.get("title", "") + " " + r.get("body", ""))
            )

        tf_idf = {}
        total_words = sum(user_word_counts.values())
        for w, count in user_word_counts.items():
            tf = count / total_words if total_words > 0 else 0
            idf = math.log(total_users / (1 + document_frequency[w]))
            tf_idf[w] = tf * idf

        top_words = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:15]
        vocab_fingerprint = {w: round(score, 4) for w, score in top_words}

        voice_parts = []
        for r in train_reviews:
            part = " — ".join(filter(None, [r.get("title", ""), r.get("body", "")]))
            if part:
                voice_parts.append(part)
        voice_sample = "\n".join(voice_parts)

        profiles.append(
            {
                "user_id": user_id(name),
                "name": display_name_map.get(name, name),
                "review_count": len(all_reviews),
                "verified_count": sum(1 for r in all_reviews if r.get("verified")),
                "train_reviews": train_reviews,
                "test_reviews": test_reviews,
                "voice_sample": voice_sample,
                "rating_stats": {
                    "mean": round(avg_rating, 2),
                    "std": round(std_rating, 2),
                    "skew": round(skew_rating, 2),
                },
                "category_pref": cat_pref,
                "vocab_fingerprint": vocab_fingerprint,
            }
        )

    profiles.sort(key=lambda p: p["review_count"], reverse=True)
    return profiles


def print_distribution(profiles: list[dict], min_reviews: int) -> None:
    """Log a histogram of review-count distribution and filter impact."""
    thresholds = [1, 2, 3, 5, 10, 15, 20]
    log.info("\n── Review-count distribution across %d users ──", len(profiles))
    for t in thresholds:
        count = sum(1 for p in profiles if p["review_count"] >= t)
        bar = "█" * count
        log.info("  ≥%2d reviews: %3d users  %s", t, count, bar)
    qualifying = sum(1 for p in profiles if p["review_count"] >= min_reviews)
    log.info("")
    if qualifying == 0:
        log.warning(
            "0 users meet the --min-reviews=%d threshold. "
            "Run a larger scrape:\n"
            "  python scripts/scrape_jumia.py "
            "--categories phones-tablets electronics computing health-beauty "
            "--limit 50 --delay 1.5",
            min_reviews,
        )
    else:
        log.info(
            "%d / %d users qualify with ≥%d reviews and will be indexed.",
            qualifying,
            len(profiles),
            min_reviews,
        )


def profiles_to_items(profiles: list[dict]) -> list[dict]:
    """
    Convert user profiles to the items.json format consumed by build_index.py.
    The `description` field carries the voice_sample for embedding.
    """
    items = []
    for p in profiles:
        items.append(
            {
                "id": p["user_id"],
                "name": p["name"],
                "category": "User Profile",
                "description": p["voice_sample"],
                "review_count": p["review_count"],
                "verified_count": p["verified_count"],
                "rating_stats": p["rating_stats"],
                "category_pref": p["category_pref"],
                "vocab_fingerprint": list(p["vocab_fingerprint"].keys()),
                "sample_reviews": [r["body"] for r in p["train_reviews"][:5]],
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-user profiles from scraped Jumia reviews."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DATA_DIR / "jumia_reviews.json",
        help="Path to the raw product-centric scrape output.",
    )
    parser.add_argument(
        "--profiles-out",
        type=Path,
        default=DATA_DIR / "user_profiles.json",
        help="Output path for the user-centric profile data.",
    )
    parser.add_argument(
        "--items-out",
        type=Path,
        default=DATA_DIR / "items.json",
        help="Output path for pipeline-ready items (fed into build_index.py).",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Only index users with at least N reviews (default: 5). "
            "Higher = richer voice model. Run a wider scrape if 0 users qualify."
        ),
    )
    args = parser.parse_args()

    if not args.raw_path.exists():
        log.error("Raw data not found at %s. Run scrape_jumia.py first.", args.raw_path)
        return

    with open(args.raw_path, encoding="utf-8") as f:
        raw_products = json.load(f)
    log.info(
        "Loaded %d products (%d total reviews) from %s",
        len(raw_products),
        sum(len(p.get("reviews", [])) for p in raw_products),
        args.raw_path,
    )

    all_profiles = build_profiles(raw_products, min_reviews=1)
    log.info("Found %d unique reviewers in total", len(all_profiles))

    print_distribution(all_profiles, args.min_reviews)

    profiles = [p for p in all_profiles if p["review_count"] >= args.min_reviews]

    if not profiles:
        log.warning(
            "No users qualify. Saving ALL profiles without filter so you can inspect them.\n"
            "Tip: scrape more products to find repeat reviewers, then re-run with --min-reviews %d.",
            args.min_reviews,
        )
        profiles = all_profiles

    log.info("\nTop profiles after filtering (≥%d reviews):", args.min_reviews)
    for p in profiles[:5]:
        log.info(
            "  [%s] %-20s reviews=%-3d ratings: mean=%.2f std=%.2f skew=%.2f",
            p["user_id"],
            p["name"][:20],
            p["review_count"],
            p["rating_stats"]["mean"],
            p["rating_stats"]["std"],
            p["rating_stats"]["skew"],
        )

    args.profiles_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.profiles_out, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    log.info("Saved %d user profiles → %s", len(profiles), args.profiles_out)

    items = profiles_to_items(profiles)
    with open(args.items_out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    log.info("Saved %d pipeline items → %s", len(items), args.items_out)

    log.info(
        "\nNext → run:  PYTHONPATH=. python data/build_index.py --collection user_profiles"
    )


if __name__ == "__main__":
    main()
