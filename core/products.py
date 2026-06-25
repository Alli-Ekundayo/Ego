"""
core/products.py
----------------
Single source of truth for the items.json catalog data.

Provides shared, lazily-loaded caching for product metadata, allowing
fast retrieval by ID or as a full list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading

log = logging.getLogger(__name__)

_ITEMS_PATH = Path(__file__).parent.parent / "data" / "items.json"

_products_list: list[dict] | None = None
_products_by_id: dict[str, dict] | None = None
_lock = threading.RLock()


def load_products_list() -> list[dict]:
    """
    Load items.json and return as a raw list of product dictionaries.
    Cached after the first load.
    """
    global _products_list

    if _products_list is not None:
        return _products_list

    with _lock:
        if _products_list is not None:
            return _products_list

        try:
            if _ITEMS_PATH.exists():
                with _ITEMS_PATH.open(encoding="utf-8") as f:
                    raw = json.load(f)
                data = raw if isinstance(raw, list) else list(raw.values())
                log.info("Loaded %d products from %s", len(data), _ITEMS_PATH)
            else:
                log.warning("items.json not found at %s", _ITEMS_PATH)
                data = []
        except Exception as exc:
            log.error("Failed to load items.json: %s", exc)
            data = []

        _products_list = data
    return _products_list


def load_products_by_id() -> dict[str, dict]:
    """
    Return product catalog keyed by product ID for O(1) metadata enrichment.
    """
    global _products_by_id

    if _products_by_id is not None:
        return _products_by_id

    with _lock:
        if _products_by_id is not None:
            return _products_by_id

        items = load_products_list()
        _products_by_id = {str(item.get("id", "")).strip(): item for item in items if item.get("id")}
    return _products_by_id
