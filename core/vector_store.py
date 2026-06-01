"""
Vector Store Wrapper: Turbovec client interface for semantic search and storage.

This module provides a clean abstraction over the Turbovec vector index (using IdMapIndex)
and stores associated metadata payloads and raw embeddings in a SQLite side-car database.

All graph nodes use this wrapper for consistent vector storage and retrieval patterns.
"""

import json
import logging
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import ctypes
try:
    ctypes.CDLL("libopenblas.so.0", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import turbovec

from core.config import settings

log = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around turbovec.IdMapIndex with SQLite side-car for semantic search
    and metadata payload storage.
    """

    def __init__(self):
        """Initialize the storage directory, run one-time schema setup, and seed the index cache."""
        self.storage_dir = Path(settings.TURBOVEC_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # In-memory index cache
        self._indices: dict[str, turbovec.IdMapIndex] = {}
        # One-time DB schema initialisation
        self._init_db_schema()

    def _init_db_schema(self) -> None:
        """Run one-time schema initialisation (CREATE TABLE + WAL mode)."""
        db_path = self.storage_dir / "metadata.db"
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS payloads (
                    collection TEXT,
                    id INTEGER,
                    payload TEXT,
                    vector TEXT,
                    PRIMARY KEY (collection, id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _get_db(self) -> sqlite3.Connection:
        """Open a connection to the SQLite side-car database."""
        db_path = self.storage_dir / "metadata.db"
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _get_index(self, collection_name: str, vector_size: int = 384) -> turbovec.IdMapIndex:
        """Load the index from disk if it exists, otherwise create a new one."""
        index_path = self.storage_dir / f"{collection_name}.tvim"
        if index_path.exists():
            try:
                return turbovec.IdMapIndex.load(str(index_path))
            except Exception as e:
                log.warning(
                    "Failed to load index '%s' from %s: %s. Recreating.",
                    collection_name, index_path, e
                )
        return turbovec.IdMapIndex(dim=vector_size)

    def _clear_user_profile_cache(self) -> None:
        self._get_user_profile_payload.cache_clear()

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Create a Turbovec index file and initialize database table if it doesn't exist.
        """
        index_path = self.storage_dir / f"{collection_name}.tvim"
        if index_path.exists():
            return
        
        index = turbovec.IdMapIndex(dim=vector_size)
        index.write(str(index_path))
        
        # Initialize DB
        with self._get_db() as conn:
            pass

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Force-recreate a collection, completely wiping any existing data on disk and DB.
        """
        index_path = self.storage_dir / f"{collection_name}.tvim"
        if index_path.exists():
            try:
                os.remove(index_path)
            except OSError:
                pass

        # Clear from SQLite
        with self._get_db() as conn:
            conn.execute("DELETE FROM payloads WHERE collection = ?", (collection_name,))
            conn.commit()

        self.create_collection(collection_name, vector_size)
        self._clear_user_profile_cache()

    def upsert(
        self,
        collection_name: str,
        ids: list[int],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Insert or update points in a collection.

        Args:
            collection_name: Collection name (e.g., "user_profiles")
            ids: List of unique integer IDs
            vectors: List of embedding vectors
            payloads: List of metadata dicts
        """
        if not ids:
            return

        # Load existing index or instantiate a new one
        vector_size = len(vectors[0]) if vectors else 384
        index = self._get_index(collection_name, vector_size)

        # Deduplicate within the batch (keep the last occurrence)
        unique_indices = {}
        for idx, pid in enumerate(ids):
            unique_indices[pid] = idx
            
        dedup_indices = list(unique_indices.values())
        dedup_ids = [ids[i] for i in dedup_indices]
        dedup_vectors = [vectors[i] for i in dedup_indices]
        dedup_payloads = [payloads[i] for i in dedup_indices]

        # Normalize vectors for cosine similarity (dot product of normalized vectors)
        np_vectors = np.array(dedup_vectors, dtype=np.float32)
        norms = np.linalg.norm(np_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = np_vectors / norms

        np_ids = np.array(dedup_ids, dtype=np.uint64)

        # Remove existing IDs if they are present
        for pid in dedup_ids:
            index.remove(int(pid))

        # Add vectors with IDs
        index.add_with_ids(normalized_vectors, np_ids)

        # Write index to disk
        index_path = self.storage_dir / f"{collection_name}.tvim"
        index.write(str(index_path))

        # Save metadata and original raw vectors in SQLite side-car
        with self._get_db() as conn:
            for pid, vec, payload in zip(dedup_ids, dedup_vectors, dedup_payloads):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO payloads (collection, id, payload, vector)
                    VALUES (?, ?, ?, ?)
                    """,
                    (collection_name, int(pid), json.dumps(payload), json.dumps(vec)),
                )
            conn.commit()

        self._clear_user_profile_cache()

    def search(
        self,
        collection_name: str,
        query_vector: list,
        limit: int = 10,
    ) -> list[Any]:
        """
        Semantic similarity search: find nearest neighbors to query vector.
        """
        index_path = self.storage_dir / f"{collection_name}.tvim"
        if not index_path.exists():
            return []

        vector_size = len(query_vector)
        index = self._get_index(collection_name, vector_size)
        if len(index) == 0:
            return []

        # Normalize query vector
        np_query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(np_query)
        if norm > 0:
            np_query = np_query / norm

        # Reshape for search (1 query vector of shape (1, dim))
        queries = np.array([np_query], dtype=np.float32)
        
        search_k = min(limit, len(index))
        if search_k <= 0:
            return []

        scores, ids = index.search(queries, k=search_k)

        # Fetch payloads and raw vectors from SQLite
        res_ids = [int(pid) for pid in ids[0]]
        if not res_ids:
            return []

        with self._get_db() as conn:
            placeholders = ",".join("?" for _ in res_ids)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, payload, vector FROM payloads WHERE collection = ? AND id IN ({placeholders})",
                [collection_name] + res_ids,
            )
            rows = cursor.fetchall()

        payload_map = {}
        vector_map = {}
        for row in rows:
            pid, payload_str, vector_str = row
            payload_map[pid] = json.loads(payload_str)
            vector_map[pid] = json.loads(vector_str) if vector_str else None

        results = []
        for pid, score in zip(ids[0], scores[0]):
            pid = int(pid)
            payload = payload_map.get(pid, {})
            vector = vector_map.get(pid)
            results.append(
                SimpleNamespace(
                    id=pid,
                    score=float(score),
                    payload=payload,
                    vector=vector
                )
            )
        return results

    def retrieve_by_id(
        self,
        collection_name: str,
        ids: list[int],
        with_vectors: bool = True,
    ) -> list[Any]:
        """
        Direct lookup: fetch specific points by their integer IDs from SQLite.
        """
        if not ids:
            return []

        with self._get_db() as conn:
            placeholders = ",".join("?" for _ in ids)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, payload, vector FROM payloads WHERE collection = ? AND id IN ({placeholders})",
                [collection_name] + [int(pid) for pid in ids],
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            pid, payload_str, vector_str = row
            payload = json.loads(payload_str)
            vector = json.loads(vector_str) if (vector_str and with_vectors) else None
            results.append(
                SimpleNamespace(
                    id=pid,
                    payload=payload,
                    vector=vector
                )
            )
        return results

    @lru_cache(maxsize=2048)
    def _get_user_profile_payload(
        self, user_id: str, collection_name: str = "user_profiles"
    ) -> dict | None:
        """
        Fetch a single user profile payload by user_id string.
        Attempts fast ID lookup first, falling back to full scan of collection.
        """
        from core.utils import to_vector_id
        qid = to_vector_id(user_id)

        # Try direct integer ID lookup first
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload FROM payloads WHERE collection = ? AND id = ?",
                (collection_name, qid),
            )
            row = cursor.fetchone()

        if row:
            return json.loads(row[0])

        # Scan fallback
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload FROM payloads WHERE collection = ?",
                (collection_name,),
            )
            rows = cursor.fetchall()

        for r in rows:
            payload = json.loads(r[0])
            if payload.get("id") == user_id:
                return payload

        return None


vector_store = VectorStore()


def get_user_profile(
    user_id: str, collection_name: str = "user_profiles"
) -> dict | None:
    """
    Fetch a single user profile payload by logical user_id.
    """
    payload = vector_store._get_user_profile_payload(user_id, collection_name)
    return dict(payload) if payload is not None else None
