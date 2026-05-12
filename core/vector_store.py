from qdrant_client import QdrantClient
from qdrant_client.http import models
from core.config import settings


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=60,
        )

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Create a Qdrant collection if it does not already exist.
        Uses the non-deprecated create_collection (replaces recreate_collection).
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

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        """Force-recreates the collection, wiping any existing data."""
        self.client.delete_collection(collection_name=collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(
        self,
        collection_name: str,
        ids: list,
        vectors: list,
        payloads: list,
    ) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            ),
        )

    def search(
        self,
        collection_name: str,
        query_vector: list,
        limit: int = 10,
    ) -> list:
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
        Retrieve specific points by their integer IDs.
        Centralises the raw client call so graph nodes go through the wrapper.
        """
        return self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=with_vectors,
        )


vector_store = VectorStore()


def get_user_profile(user_id: str, collection_name: str = "user_profiles") -> dict | None:
    """
    Fetch a single user profile payload by logical user_id stored in Qdrant payload["id"].
    Returns None when no matching profile is found.
    """
    results, _ = vector_store.client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="id",
                match=models.MatchValue(value=user_id),
            )]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        return None
    return results[0].payload
