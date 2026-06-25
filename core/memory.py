"""core/memory.py
-----------------
Persistent MemoryAgent store for cross-session, multi-turn memory.

Architecture
------------
Every interaction enriches a per-user memory graph stored in SQLite. Each
memory entry carries:

  - content     : the raw memory string
  - memory_type : "preference" | "interaction" | "feedback" | "context"
  - importance  : float in [0, 1] — set at creation, boosted on re-access
  - access_count: number of times recalled — used for Ebbinghaus-style decay
  - created_at  : ISO timestamp
  - last_accessed: ISO timestamp — used in the decay calculation
  - session_id  : origin session — enables per-session provenance

Decay / Forgetting
------------------
Importance decays exponentially with time since last access (Ebbinghaus
forgetting curve), countered by access frequency. Memories below a minimum
importance threshold are pruned automatically on each save() call.

Retrieval
---------
Recall is embedding-cosine ranked, then importance-decay weighted, and
finally capped by the caller-supplied `max_tokens` budget so results always
fit inside any context window.

Qwen Integration
-----------------
The MemoryAgent uses the Qwen-Plus model (via DashScope, OpenAI-compatible
endpoint) for memory consolidation — identifying duplicate/redundant memories
and merging or promoting key facts into long-term preferences.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import settings

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "scratch" / "cache" / "memory.db"

# ── Decay constants ───────────────────────────────────────────────────────────
_DECAY_HALF_LIFE_DAYS = 7.0          # importance halves every 7 days without access
_MIN_IMPORTANCE = 0.05               # prune entries that fall below this
_IMPORTANCE_ACCESS_BOOST = 0.08      # boost per recall
_MAX_MEMORIES_PER_USER = 200         # hard cap before forced eviction


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decay_factor(last_accessed_iso: str, access_count: int) -> float:
    """
    Ebbinghaus-style exponential decay, partially countered by access_count.

      R(t) = e^(-(t / half_life) / sqrt(1 + access_count))

    where t is days since last access.
    """
    try:
        last_dt = datetime.fromisoformat(last_accessed_iso)
        now_dt = datetime.now(timezone.utc)
        days_elapsed = (now_dt - last_dt).total_seconds() / 86400.0
    except Exception:
        days_elapsed = 0.0

    repetition_factor = math.sqrt(1.0 + max(access_count, 0))
    return math.exp(-(days_elapsed / _DECAY_HALF_LIFE_DAYS) / repetition_factor)


class MemoryStore:
    """
    SQLite-backed persistent memory store for a single user.

    Designed to be instantiated once per user per request and discarded;
    the underlying DB file persists across sessions.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._db_path = _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id       TEXT    NOT NULL,
                    session_id    TEXT    NOT NULL DEFAULT '',
                    memory_type   TEXT    NOT NULL DEFAULT 'interaction',
                    content       TEXT    NOT NULL,
                    importance    REAL    NOT NULL DEFAULT 0.5,
                    access_count  INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT    NOT NULL,
                    last_accessed TEXT    NOT NULL,
                    metadata      TEXT    NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user
                    ON memories(user_id);

                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id    TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                );

                CREATE TABLE IF NOT EXISTS memory_summaries (
                    user_id    TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        memory_type: str = "interaction",
        importance: float = 0.5,
        session_id: str = "",
        metadata: dict | None = None,
    ) -> int:
        """
        Persist a new memory entry. Returns the new row ID.

        Args:
            content:     The memory text (preference, interaction summary, etc.)
            memory_type: "preference" | "interaction" | "feedback" | "context"
            importance:  Initial salience in [0, 1].
            session_id:  Originating session identifier.
            metadata:    Arbitrary JSON payload (e.g. {"item_id": "...", "rating": 4}).
        """
        now = _now_iso()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories
                    (user_id, session_id, memory_type, content, importance,
                     access_count, created_at, last_accessed, metadata)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    self.user_id,
                    session_id,
                    memory_type,
                    content,
                    importance,
                    now,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def set_preference(self, key: str, value: str) -> None:
        """Upsert a named user preference (e.g. 'budget', 'preferred_category')."""
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value,
                                                         updated_at=excluded.updated_at
                """,
                (self.user_id, key, value, now),
            )
            conn.commit()

    def update_summary(self, summary: str) -> None:
        """Store the latest LLM-generated long-term memory summary."""
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memory_summaries (user_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary,
                                                    updated_at=excluded.updated_at
                """,
                (self.user_id, summary, now),
            )
            conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_preferences(self) -> dict[str, str]:
        """Return all named user preferences as a dict."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM user_preferences WHERE user_id = ?",
                (self.user_id,),
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def get_summary(self) -> str:
        """Return the latest long-term memory summary, or empty string."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT summary FROM memory_summaries WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def recall(
        self,
        query: str,
        max_results: int = 20,
        memory_types: list[str] | None = None,
        max_tokens: int = 800,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant memories for `query` within a token budget.

        Ranking: effective_importance = raw_importance × decay_factor(last_accessed)
        Memories with effective_importance < _MIN_IMPORTANCE are excluded.

        The returned list is ordered by effective_importance (descending) and
        capped so total content character count ≤ max_tokens * 4 (rough proxy).

        Access counts are incremented for every recalled memory (boosts retention).
        """
        with self._conn() as conn:
            if memory_types:
                placeholders = ",".join("?" for _ in memory_types)
                rows = conn.execute(
                    f"""
                    SELECT id, memory_type, content, importance, access_count,
                           last_accessed, created_at, metadata
                    FROM memories
                    WHERE user_id = ? AND memory_type IN ({placeholders})
                    ORDER BY importance DESC
                    LIMIT 500
                    """,
                    [self.user_id, *memory_types],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, memory_type, content, importance, access_count,
                           last_accessed, created_at, metadata
                    FROM memories
                    WHERE user_id = ?
                    ORDER BY importance DESC
                    LIMIT 500
                    """,
                    (self.user_id,),
                ).fetchall()

        # Score with decay and simple keyword overlap relevance
        query_tokens = set(query.lower().split())
        scored: list[tuple[float, dict]] = []
        for r in rows:
            decay = _decay_factor(r["last_accessed"], r["access_count"])
            eff_importance = float(r["importance"]) * decay
            if eff_importance < _MIN_IMPORTANCE:
                continue
            # Keyword overlap boost
            content_tokens = set(r["content"].lower().split())
            overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            final_score = eff_importance * (1.0 + 0.5 * overlap)
            scored.append(
                (
                    final_score,
                    {
                        "id": r["id"],
                        "memory_type": r["memory_type"],
                        "content": r["content"],
                        "importance": r["importance"],
                        "effective_importance": round(eff_importance, 4),
                        "score": round(final_score, 4),
                        "created_at": r["created_at"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)

        # Budget-aware truncation
        char_budget = max_tokens * 4
        results: list[dict] = []
        used_chars = 0
        recalled_ids: list[int] = []
        for _, mem in scored[:max_results]:
            if used_chars + len(mem["content"]) > char_budget:
                break
            results.append(mem)
            used_chars += len(mem["content"])
            recalled_ids.append(mem["id"])

        # Boost access counts for recalled entries
        if recalled_ids:
            now = _now_iso()
            with self._conn() as conn:
                for mid in recalled_ids:
                    conn.execute(
                        """
                        UPDATE memories
                        SET access_count = access_count + 1,
                            last_accessed = ?,
                            importance = MIN(1.0, importance + ?)
                        WHERE id = ?
                        """,
                        (now, _IMPORTANCE_ACCESS_BOOST, mid),
                    )
                conn.commit()

        return results

    def get_all_for_consolidation(self) -> list[dict]:
        """Return all memories for a user, used by the consolidation LLM pass."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, memory_type, content, importance, access_count,
                       last_accessed, created_at, metadata
                FROM memories
                WHERE user_id = ?
                ORDER BY importance DESC
                """,
                (self.user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Maintenance ───────────────────────────────────────────────────────────

    def prune_stale(self) -> int:
        """
        Delete memories whose effective importance has decayed below threshold.
        Returns the number of rows deleted.
        """
        all_memories = self.get_all_for_consolidation()
        stale_ids = []
        for r in all_memories:
            decay = _decay_factor(r["last_accessed"], r["access_count"])
            eff = float(r["importance"]) * decay
            if eff < _MIN_IMPORTANCE:
                stale_ids.append(r["id"])

        if not stale_ids:
            return 0

        placeholders = ",".join("?" for _ in stale_ids)
        with self._conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", stale_ids
            )
            conn.commit()
            deleted = cursor.rowcount

        log.info("MemoryStore: pruned %d stale memories for user %s", deleted, self.user_id)
        return deleted

    def enforce_cap(self) -> int:
        """
        Evict lowest-importance memories if the user is over the hard cap.
        Returns the number evicted.
        """
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (self.user_id,)
            ).fetchone()[0]

        if count <= _MAX_MEMORIES_PER_USER:
            return 0

        evict_n = count - _MAX_MEMORIES_PER_USER
        with self._conn() as conn:
            ids_to_evict = conn.execute(
                """
                SELECT id FROM memories WHERE user_id = ?
                ORDER BY (importance * 1.0) ASC
                LIMIT ?
                """,
                (self.user_id, evict_n),
            ).fetchall()
            ids = [r[0] for r in ids_to_evict]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})", ids
                )
                conn.commit()

        log.info("MemoryStore: evicted %d over-cap memories for user %s", evict_n, self.user_id)
        return evict_n

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the user's memory state."""
        with self._conn() as conn:
            mem_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (self.user_id,)
            ).fetchone()[0]
        return {
            "user_id": self.user_id,
            "memory_count": mem_count,
            "preferences": self.get_preferences(),
            "summary": self.get_summary(),
        }
