"""
Embedding Model Wrapper: Converts text to dense vector representations.
Includes persistent disk-based caching to minimize re-computation.
"""

from pathlib import Path

from diskcache import Cache
from sentence_transformers import SentenceTransformer

from core.config import settings


class EmbeddingModel:
    """
    Wrapper around SentenceTransformers with persistent disk caching.
    """

    def __init__(self):
        """Initialize the embedding model and the persistent cache."""
        self._model = None

        cache_path = Path(__file__).parent.parent / "scratch" / "cache" / "embeddings"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(cache_path))

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    @staticmethod
    def _normalise_text(text: str) -> str:
        return str(text or "").strip()

    def embed_text(self, text: str) -> list[float]:
        """Convert a single text string to an embedding vector."""
        normalised = self._normalise_text(text)

        if normalised in self.cache:
            return list(self.cache[normalised])

        vector = self.model.encode(normalised).tolist()
        self.cache[normalised] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Efficiently embed multiple texts in a single batch with caching."""
        if not texts:
            return []

        normalised = [self._normalise_text(t) for t in texts]
        output: list[list[float] | None] = [None] * len(normalised)
        missing_positions: dict[str, list[int]] = {}

        for idx, text in enumerate(normalised):
            if text in self.cache:
                output[idx] = list(self.cache[text])
            else:
                missing_positions.setdefault(text, []).append(idx)

        if missing_positions:
            missing_texts = list(missing_positions.keys())
            encoded = self.model.encode(missing_texts).tolist()
            for text, vector in zip(missing_texts, encoded):
                self.cache[text] = vector
                for idx in missing_positions[text]:
                    output[idx] = list(vector)

        return [vec if vec is not None else [] for vec in output]

    def close(self):
        """Close the cache resource."""
        self.cache.close()


embedding_model = EmbeddingModel()
