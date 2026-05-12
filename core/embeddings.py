from sentence_transformers import SentenceTransformer
from core.config import settings

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed_text(self, text: str):
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]):
        return self.model.encode(texts).tolist()

embedding_model = EmbeddingModel()
