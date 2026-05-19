"""tests/test_api.py
-------------------
Unit and integration tests for the Ego pipeline.

LLM and Qdrant calls are mocked so the suite runs fast without
a live Gemini key or Qdrant instance.
"""

from unittest.mock import patch

import pytest

from agents.rerank_agent import _extract_json

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _agent_result():
    """Mock return value matching UserAgentState schema."""
    return {
        "user_persona": "Emmanuel",
        "user_id": "abc123def456",
        "user_profile": {"rating_stats": {"mean": 4.0, "std": 0.8, "skew": 0.1}},
        "user_embedding": [0.1] * 384,
        "item_embedding": [0.1] * 384,
        "retrieved_examples": [],
        "predicted_rating": 4.2,
        "style_profile": "Informal, short sentences.",
        "simulated_review": "This phone is really good.",
        "final_review": "Abeg, this phone na banger!",
    }


def _recommend_result():
    return {
        "user_id": "U123",
        "context": "looking for headphones",
        "n": 2,
        "candidates": [],
        "recommendations": [
            {"item_id": "p001", "name": "Sony WH-1000XM5", "reason": "Great ANC"},
            {"item_id": "p002", "name": "Jabra Evolve2", "reason": "Good call quality"},
        ],
    }


@pytest.fixture()
def api_client():
    """TestClient with all external dependencies patched out."""
    with (
        patch("qdrant_client.QdrantClient"),
        patch("sentence_transformers.SentenceTransformer"),
        patch("graphs.task_a.user_modeling_agent") as mock_agent,
        patch("graphs.task_b.task_b_graph") as mock_task_b,
        patch("core.llm.set_llm_cache"),  # Avoid SQLite setup in tests
    ):
        mock_agent.invoke.return_value = _agent_result()
        mock_task_b.invoke.return_value = _recommend_result()

        from fastapi.testclient import TestClient

        from api.main import app

        yield TestClient(app), mock_agent, mock_task_b


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: core/math_utils
# ─────────────────────────────────────────────────────────────────────────────


class TestCosimSimilarity:
    def test_identical_vectors_return_1(self):
        from core.math_utils import cosine_similarity

        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_0(self):
        from core.math_utils import cosine_similarity

        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector_returns_0(self):
        from core.math_utils import cosine_similarity

        assert cosine_similarity([0, 0], [1, 0]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: core/utils
# ─────────────────────────────────────────────────────────────────────────────


class TestUtils:
    def test_to_qdrant_id(self):
        from core.utils import to_qdrant_id

        assert isinstance(to_qdrant_id("test"), int)
        assert to_qdrant_id("test") == to_qdrant_id("test")

    def test_to_stable_id(self):
        from core.utils import to_stable_id

        assert len(to_stable_id("Emmanuel")) == 12
        assert to_stable_id("Emmanuel") == to_stable_id(" emmanuel ")

    def test_clean_review_text(self):
        from core.utils import clean_review_text

        assert clean_review_text('### Header\n"Review"') == "Review"
        assert clean_review_text("Review\nItem: Metadata") == "Review"


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: agents/rerank_agent — _extract_json
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractJson:
    def test_plain_json(self):
        data = _extract_json('{"reranked_items": []}')
        assert data == {"reranked_items": []}

    def test_markdown_fenced_json(self):
        text = '```json\n{"reranked_items": [{"item_id": "p1"}]}\n```'
        data = _extract_json(text)
        assert data["reranked_items"][0]["item_id"] == "p1"

    def test_json_list(self):
        data = _extract_json('[{"item_id": "p1"}]')
        assert data == [{"item_id": "p1"}]


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests: API endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestSimulateReviewEndpoint:
    def test_returns_200_on_valid_request(self, api_client):
        client, mock_agent, _ = api_client
        response = client.post(
            "/simulate-review",
            json={
                "user_id": "emmanuel",
                "item": {"name": "Nokia 3310", "category": "Phones & Tablets"},
            },
        )
        assert response.status_code == 200

    def test_response_contains_required_fields(self, api_client):
        client, _, _ = api_client
        data = client.post(
            "/simulate-review",
            json={
                "user_id": "emmanuel",
                "item": {"name": "Nokia 3310", "category": "Phones & Tablets"},
            },
        ).json()
        assert "rating" in data
        assert "review" in data


class TestRecommendEndpoint:
    def test_returns_200_on_valid_request(self, api_client):
        client, _, _ = api_client
        response = client.post(
            "/recommend",
            json={
                "user_id": "U123",
                "context": "looking for headphones",
                "n": 2,
            },
        )
        assert response.status_code == 200

    def test_response_has_recommendations_list(self, api_client):
        client, _, _ = api_client
        data = client.post(
            "/recommend",
            json={
                "user_id": "U123",
                "context": "looking for headphones",
                "n": 2,
            },
        ).json()
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) == 2
