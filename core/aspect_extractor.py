"""
core/aspect_extractor.py
------------------------
Aspect Extraction: Pull structured product aspects (e.g. "Battery Life",
"Price", "Build Quality") from new item metadata or a free-text product
description, using a lightweight rule-based pass + optional LLM enrichment.

These aspects drive both the dense semantic search queries
(find historical reviews discussing similar aspects) and the sparse keyword
search (match exact category / brand tokens).
"""

from __future__ import annotations

import logging

from core.utils import tokenize as _tokenize

log = logging.getLogger(__name__)

_ASPECT_PATTERNS: dict[str, list[str]] = {
    "Battery Life": ["battery", "charge", "charging", "power", "mah", "standby"],
    "Price / Value": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "affordable",
        "value",
        "worth",
    ],
    "Build Quality": [
        "build",
        "quality",
        "durable",
        "sturdy",
        "plastic",
        "metal",
        "material",
    ],
    "Performance": [
        "fast",
        "speed",
        "performance",
        "processor",
        "cpu",
        "lag",
        "smooth",
    ],
    "Display": [
        "screen",
        "display",
        "resolution",
        "bright",
        "colour",
        "color",
        "amoled",
    ],
    "Camera": ["camera", "photo", "picture", "megapixel", "mp", "selfie", "lens"],
    "Connectivity": [
        "wifi",
        "bluetooth",
        "5g",
        "4g",
        "nfc",
        "usb",
        "port",
        "headphone",
    ],
    "Design / Style": [
        "design",
        "slim",
        "lightweight",
        "colour",
        "color",
        "style",
        "look",
    ],
    "Brand": [
        "samsung",
        "apple",
        "nokia",
        "tecno",
        "infinix",
        "itel",
        "huawei",
        "xiaomi",
        "oppo",
        "vivo",
        "sony",
        "lg",
        "hp",
        "dell",
        "lenovo",
        "adidas",
        "nike",
        "puma",
    ],
    "Delivery": ["delivery", "shipping", "arrive", "package", "packaging", "box"],
    "Customer Service": [
        "service",
        "support",
        "refund",
        "return",
        "warranty",
        "customer",
    ],
    "Software": ["android", "ios", "app", "software", "update", "os", "interface"],
    "Size / Fit": ["size", "fit", "small", "large", "medium", "xl", "xxl", "fitting"],
    "Sound": ["sound", "audio", "bass", "treble", "speaker", "earphone", "noise"],
    "Storage": ["storage", "memory", "gb", "tb", "ram", "space", "card"],
}



def extract_aspects_rule_based(item_metadata: dict) -> list[str]:
    """
    Extract product aspects from item metadata using keyword matching.

    Combines: name, category, description, and any feature list fields.
    Returns a de-duplicated ordered list of matched aspect labels,
    with a "General" fallback if nothing matches.
    """
    text_parts: list[str] = [
        item_metadata.get("name", ""),
        item_metadata.get("category", ""),
        item_metadata.get("description", ""),
    ]
    for feat in item_metadata.get("features", []):
        text_parts.append(str(feat))

    combined = " ".join(text_parts)
    tokens = set(_tokenize(combined))

    matched: list[str] = []
    for aspect_label, keywords in _ASPECT_PATTERNS.items():
        if any(kw in tokens for kw in keywords):
            matched.append(aspect_label)

    if not matched:
        matched = ["General"]

    log.debug("Extracted aspects: %s", matched)
    return matched


def aspects_to_query_strings(aspects: list[str], item_metadata: dict) -> list[str]:
    """
    Convert aspect labels into natural-language query strings suitable
    for dense semantic search.

    Example:
        aspect="Battery Life", item_name="Tecno Spark 10"
        → "battery life charging experience Tecno Spark 10"
    """
    item_name = item_metadata.get("name", "")
    category = item_metadata.get("category", "")

    _ASPECT_TEMPLATES: dict[str, str] = {
        "Battery Life": "battery life charging experience {item}",
        "Price / Value": "price value for money affordable {item} {category}",
        "Build Quality": "build quality durability materials {item}",
        "Performance": "performance speed processor smooth {item}",
        "Display": "display screen quality brightness {item}",
        "Camera": "camera photo quality pictures {item}",
        "Connectivity": "connectivity wifi bluetooth ports {item}",
        "Design / Style": "design look style aesthetics {item}",
        "Brand": "{item} brand reputation {category}",
        "Delivery": "delivery packaging shipping experience {item}",
        "Customer Service": "customer service support warranty {item}",
        "Software": "software apps interface android {item}",
        "Size / Fit": "size fit comfort {item} {category}",
        "Sound": "sound audio quality bass speaker {item}",
        "Storage": "storage memory ram capacity {item}",
        "General": "{item} {category} review experience",
    }

    queries: list[str] = []
    for asp in aspects:
        tpl = _ASPECT_TEMPLATES.get(asp, "{item} {category} {asp}")
        q = tpl.format(item=item_name, category=category, asp=asp).strip()
        queries.append(q)
    return queries


def extract_sparse_keywords(item_metadata: dict, aspects: list[str]) -> list[str]:
    """
    Extract exact keywords for sparse BM25 keyword matching.

    Returns brand names, category tokens, product model tokens,
    and any aspect-specific high-signal keywords.
    """
    tokens: list[str] = []

    for field in ("name", "category", "brand"):
        val = item_metadata.get(field, "")
        tokens.extend(w for w in _tokenize(val) if len(w) >= 3)

    for asp in aspects:
        kws = _ASPECT_PATTERNS.get(asp, [])
        tokens.extend(kws[:3])

    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique
