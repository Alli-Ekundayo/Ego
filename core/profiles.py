"""
core/profiles.py
----------------
Single source of truth for the user_profiles.json data.

Previously, graphs/task_a.py, graphs/task_b.py, and agents/retrieval_agent.py
each loaded user_profiles.json independently, creating three separate in-memory
copies of the same data and three separate load/parse paths to maintain.

This module provides one shared, lazily-loaded, mtime-invalidated store with
three access patterns to satisfy all existing consumers:

  profiles_list()           → list[dict]          (Task A — raw list)
  profiles_by_user_id()     → dict[str, dict]      (RetrievalAgent — keyed by user_id, last record wins)
  profiles_grouped_by_uid() → dict[str, list[dict]] (Task B — keyed by user_id, all records)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PROFILES_PATH = Path(__file__).parent.parent / "data" / "user_profiles.json"

_cache_list: list[dict] | None = None
_cache_mtime: float = 0.0


def _load() -> list[dict]:
    """
    Load user_profiles.json, returning a cached result unless the file has
    changed on disk (mtime-invalidated).
    """
    global _cache_list, _cache_mtime

    current_mtime = _PROFILES_PATH.stat().st_mtime if _PROFILES_PATH.exists() else 0.0
    if _cache_list is not None and current_mtime == _cache_mtime:
        return _cache_list

    try:
        with _PROFILES_PATH.open(encoding="utf-8") as f:
            data: list[dict] = json.load(f)
        log.info("Loaded %d user profiles from %s", len(data), _PROFILES_PATH)
    except Exception as exc:
        log.error("Failed to load user_profiles.json: %s", exc)
        data = []

    _cache_list = data
    _cache_mtime = current_mtime
    return _cache_list


def profiles_list() -> list[dict]:
    """
    Return all profiles as a raw list.

    Usage (Task A): iterate to find a specific user_id.
    """
    return _load()


def profiles_by_user_id() -> dict[str, dict]:
    """
    Return profiles keyed by user_id (last record wins for duplicates).

    Usage (RetrievalAgent): O(1) lookup by user_id.
    """
    return {str(p.get("user_id", "")): p for p in _load() if p.get("user_id")}


def profiles_grouped_by_uid() -> dict[str, list[dict]]:
    """
    Return profiles grouped by user_id, preserving all records per user.

    Usage (Task B load_profile_node): persona-name matching across all records
    for a user.
    """
    grouped: dict[str, list[dict]] = {}
    for p in _load():
        uid = str(p.get("user_id", "")).strip()
        if uid:
            grouped.setdefault(uid, []).append(p)
    return grouped
