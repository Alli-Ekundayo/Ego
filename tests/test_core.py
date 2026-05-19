import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _fresh_import(module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


class TestEmbeddingCaching:
    def test_embed_text_reuses_cached_value(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            model = MagicMock()
            encoded_mock = MagicMock()
            encoded_mock.tolist.return_value = [0.1, 0.2, 0.3]
            model.encode.return_value = encoded_mock
            mock_st.return_value = model
            embeddings = _fresh_import("core.embeddings")
            embeddings.embedding_model.cache.clear()

        first = embeddings.embedding_model.embed_text("hello world")
        second = embeddings.embedding_model.embed_text("hello world")

        assert first == second
        assert model.encode.call_count == 1

    def test_embed_batch_only_encodes_missing_texts(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            model = MagicMock()

            class EncodedMock:
                def __init__(self, val):
                    self.val = val

                def tolist(self):
                    return self.val

            def _encode(texts):
                if isinstance(texts, list):
                    return EncodedMock([[float(len(t))] for t in texts])
                return EncodedMock([float(len(str(texts)))])

            model.encode.side_effect = _encode
            mock_st.return_value = model
            embeddings = _fresh_import("core.embeddings")
            embeddings.embedding_model.cache.clear()

        first = embeddings.embedding_model.embed_batch(["alpha", "beta", "alpha"])
        second = embeddings.embedding_model.embed_batch(["alpha", "beta"])

        assert first[0] == first[2]
        assert second[0] == first[0]
        assert model.encode.call_count == 1


class TestVectorStoreCaching:
    def test_get_user_profile_uses_cache(self):
        with patch("qdrant_client.QdrantClient") as mock_qdrant:
            client = MagicMock()
            client.scroll.return_value = (
                [SimpleNamespace(payload={"id": "u1", "name": "User"})],
                None,
            )
            mock_qdrant.return_value = client
            vector_store = _fresh_import("core.vector_store")

        first = vector_store.get_user_profile("u1")
        second = vector_store.get_user_profile("u1")

        assert first == second
        assert first is not second
        assert client.scroll.call_count == 1

    def test_upsert_clears_user_profile_cache(self):
        with patch("qdrant_client.QdrantClient") as mock_qdrant:
            client = MagicMock()
            client.scroll.return_value = ([SimpleNamespace(payload={"id": "u1"})], None)
            mock_qdrant.return_value = client
            vector_store = _fresh_import("core.vector_store")

        _ = vector_store.get_user_profile("u1")
        vector_store.vector_store.upsert(
            collection_name="user_profiles",
            ids=[1],
            vectors=[[0.1, 0.2]],
            payloads=[{"id": "u1"}],
        )
        _ = vector_store.get_user_profile("u1")

        assert client.scroll.call_count == 2


class TestSafeDeserialization:
    def test_deserializes_messages_successfully(self):
        from langchain_core.messages import HumanMessage

        from core.utils import safe_loads_langchain

        payload = """{
          "lc": 1,
          "type": "constructor",
          "id": ["langchain_core", "messages", "HumanMessage"],
          "kwargs": {
            "content": "Hello secure world"
          }
        }"""

        message = safe_loads_langchain(payload)
        assert isinstance(message, HumanMessage)
        assert message.content == "Hello secure world"

    def test_blocks_non_message_objects(self):
        import pytest

        from core.utils import safe_loads_langchain

        # Unsafe payload attempting to instantiate a chat model
        payload = """{
          "lc": 1,
          "type": "constructor",
          "id": ["langchain_google_genai", "ChatGoogleGenerativeAI"],
          "kwargs": {
            "model": "gemini-1.5-flash"
          }
        }"""

        with pytest.raises(ValueError) as excinfo:
            safe_loads_langchain(payload)
        assert "is not allowed" in str(excinfo.value)
