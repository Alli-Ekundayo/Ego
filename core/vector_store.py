"""
Vector Store Wrapper: Qdrant client interface for semantic search and storage.

This module provides a clean abstraction over the Qdrant vector database.
Qdrant stores and retrieves dense embeddings for:
1. User profiles with historical review history
2. Product items with descriptions
3. Naija-style examples for cultural voice injection

All graph nodes use this wrapper for consistent Qdrant access patterns.
This keeps implementation details (e.g., filter syntax, pagination) encapsulated.
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.config import settings


class VectorStore:
    """
    Wrapper around QdrantClient for semantic search and storage operations.

    Provides methods for:
    - Creating/managing collections
    - Upserting embeddings with metadata (payloads)
    - Semantic search by vector similarity
    - Direct ID-based lookups
    """

    def __init__(self):
        """Initialize Qdrant client connecting to remote/local server."""
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=60,
        )

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Create a Qdrant collection if it does not already exist.

        Collections are immutable in structure: once created, you can't change
        the vector size or distance metric. This method checks if the collection
        exists before attempting creation to avoid errors.

        Args:
            collection_name: Name of the collection (e.g., "user_profiles", "items")
            vector_size: Dimensionality of vectors (e.g., 384 for MiniLM)

        Distance metric is fixed to COSINE (best for embeddings from neural models).
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if collection_name in existing:
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._get_user_profile_payload.cache_clear()

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Force-recreate a collection, completely wiping any existing data.

        Use this for development/testing when you need to re-seed data.
        WARNING: This permanently deletes all data in the collection!

        Args:
            collection_name: Name of collection to delete and recreate
            vector_size: Dimensionality of new vectors
        """
        self.client.delete_collection(collection_name=collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._get_user_profile_payload.cache_clear()

    def upsert(
        self,
        collection_name: str,
        ids: list,
        vectors: list,
        payloads: list,
    ) -> None:
        """
        Insert or update points in a collection.

        Each point has:
        - id: Unique integer identifier
        - vector: Embedding (dense vector)
        - payload: Metadata dict (user profile, item info, etc.)

        "Upsert" means: if the ID exists, update it; otherwise, insert new.
        This is idempotent and safe for batch re-seeding.

        Args:
            collection_name: Collection to insert into
            ids: List of integer point IDs (unique within collection)
            vectors: List of embedding vectors (must match collection vector_size)
            payloads: List of metadata dicts to attach to each point
        """
        self.client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            ),
        )
        self._get_user_profile_payload.cache_clear()

    def search(
        self,
        collection_name: str,
        query_vector: list,
        limit: int = 10,
    ) -> list:
        """
        Semantic similarity search: find the nearest neighbors to a query vector.

        Uses cosine similarity to rank all points in the collection by how similar
        they are to the query. This is the core operation for finding relevant
        reviews, items, and examples.

        Args:
            collection_name: Collection to search in
            query_vector: Query embedding vector (must match collection's vector_size)
            limit: Max number of results to return (default 10)

        Returns:
            List of PointStruct objects with:
            - id: Point identifier
            - score: Similarity score (1.0 = perfect match, -1.0 = opposite)
            - payload: Associated metadata dict
            - vector: The point's embedding (if requested)
        """
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        ).points

    def retrieve_by_id(
        self,
        collection_name: str,
        ids: list,
        with_vectors: bool = True,
    ) -> list:
        """
        Direct lookup: fetch specific points by their integer IDs.

        This is faster than semantic search when you know exactly which
        points you want (e.g., fetching a specific user profile by stable ID).

        Args:
            collection_name: Collection to retrieve from
            ids: List of integer point IDs to fetch
            with_vectors: Include embedding vectors in results (True for re-ranking)

        Returns:
            List of PointStruct objects matching the IDs
            Empty list if IDs don't exist
        """
        return self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=with_vectors,
        )

    @lru_cache(maxsize=2048)
    def _get_user_profile_payload(
        self, user_id: str, collection_name: str = "user_profiles"
    ) -> dict | None:
        results, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        return dict(results[0].payload or {})


vector_store = VectorStore()


def get_user_profile(
    user_id: str, collection_name: str = "user_profiles"
) -> dict | None:
    """
    Fetch a single user profile payload by logical user_id.

    This is a convenience function for Task B's cold-start detection.
    It searches through "user_profiles" collection looking for a point
    with payload["id"] matching the given user_id.

    Note: This is a "scroll" operation (iterates all points with filter),
    which is slower than direct ID lookup but necessary because we're
    searching by a payload field rather than Qdrant point ID.

    Args:
        user_id: Logical user identifier (e.g., "john_doe_123")
        collection_name: Collection to search (default "user_profiles")

    Returns:
        Dict containing user profile (name, reviews, embeddings) or None if not found
    """
    payload = vector_store._get_user_profile_payload(user_id, collection_name)
    return dict(payload) if payload is not None else None
