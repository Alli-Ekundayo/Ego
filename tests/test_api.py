"""tests/test_api.py
-------------------
Unit and integration tests for the Ego pipeline.

LLM and Qdrant calls are mocked so the suite runs fast without
a live OpenAI key or Qdrant instance.

Run:
  pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agent_result():
    """Mock return value matching UserAgentState schema."""
    return {
        "user_persona":       "Emmanuel",
        "user_id":            "abc123def456",
        "user_profile":       {"rating_stats": {"mean": 4.0, "std": 0.8, "skew": 0.1}},
        "user_embedding":     [0.1] * 384,
        "item_embedding":     [0.1] * 384,
        "retrieved_examples": [],
        "predicted_rating":   4.2,
        "style_profile":      "Informal, short sentences.",
        "simulated_review":   "This phone is really good.",
        "final_review":       "Abeg, this phone na banger!",
    }


def _recommend_result():
    return {
        "user_id":         "U123",
        "context":         "looking for headphones",
        "n":               2,
        "candidates":      [],
        "recommendations": [
            {"item_id": "p001", "name": "Sony WH-1000XM5", "reason": "Great ANC"},
            {"item_id": "p002", "name": "Jabra Evolve2",    "reason": "Good call quality"},
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

    def test_empty_inputs_return_0(self):
        from core.math_utils import cosine_similarity
        assert cosine_similarity([], [1.0]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: agents/rating_agent — predict_rating (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestRatingAgentPredictRating:
    def setup_method(self):
        # Bypass __init__ (which calls get_llm) — only test predict_rating math
        from agents.rating_agent import RatingAgent
        self.agent = RatingAgent.__new__(RatingAgent)

    def test_output_always_within_1_to_5(self):
        import random
        for _ in range(20):
            v = [random.random() for _ in range(384)]
            rating = self.agent.predict_rating(
                v, v, {"mean": 3.0, "std": 1.0, "skew": 0.0}
            )
            assert 1.0 <= rating <= 5.0

    def test_consistent_5star_user_gets_5_for_similar_item(self):
        v = [1.0] + [0.0] * 383
        rating = self.agent.predict_rating(
            v, v, {"mean": 4.9, "std": 0.05, "skew": 0.0}
        )
        assert rating == 5.0

    def test_consistent_5star_user_penalised_for_dissimilar_item(self):
        user_v = [1.0] + [0.0] * 383
        item_v = [0.0, 1.0] + [0.0] * 382
        rating = self.agent.predict_rating(
            user_v, item_v, {"mean": 4.9, "std": 0.05, "skew": 0.0}
        )
        assert rating < 5.0

    def test_zero_vectors_returns_clamped_rating(self):
        z = [0.0] * 384
        rating = self.agent.predict_rating(
            z, z, {"mean": 3.0, "std": 1.0, "skew": 0.0}
        )
        assert 1.0 <= rating <= 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: agents/rerank_agent — _extract_json (pure function, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

# Import only the pure helper, NOT the module level rerank_agent singleton
# (which would call get_llm → ChatOpenAI → credential check)
from agents.rerank_agent import _extract_json


class TestExtractJson:
    def test_plain_json(self):
        data = _extract_json('{"reranked_items": []}')
        assert data == {"reranked_items": []}

    def test_markdown_fenced_json(self):
        text = '```json\n{"reranked_items": [{"item_id": "p1"}]}\n```'
        data = _extract_json(text)
        assert data["reranked_items"][0]["item_id"] == "p1"

    def test_json_embedded_in_prose(self):
        text = 'Here are the results: {"reranked_items": []} — enjoy!'
        data = _extract_json(text)
        assert "reranked_items" in data

    def test_invalid_raises_exception(self):
        with pytest.raises(Exception):
            _extract_json("no json here at all")


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests: API endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateReviewEndpoint:
    def test_returns_200_on_valid_request(self, api_client):
        client, mock_agent, _ = api_client
        mock_agent.invoke.return_value = _agent_result()
        response = client.post("/simulate-review", json={
            "user_id": "emmanuel",
            "item": {"name": "Nokia 3310", "category": "Phones & Tablets"},
        })
        assert response.status_code == 200

    def test_response_contains_required_fields(self, api_client):
        client, mock_agent, _ = api_client
        mock_agent.invoke.return_value = _agent_result()
        data = client.post("/simulate-review", json={
            "user_id": "emmanuel",
            "item": {"name": "Nokia 3310", "category": "Phones & Tablets"},
        }).json()
        assert "rating" in data
        assert "review" in data

    def test_rating_is_float(self, api_client):
        client, mock_agent, _ = api_client
        mock_agent.invoke.return_value = _agent_result()
        data = client.post("/simulate-review", json={
            "user_id": "emmanuel",
            "item": {"name": "Nokia 3310", "category": "Phones & Tablets"},
        }).json()
        assert isinstance(data["rating"], float)

    def test_missing_item_returns_422(self, api_client):
        client, _, _ = api_client
        response = client.post("/simulate-review", json={"user_id": "U123"})
        assert response.status_code == 422

    def test_naija_review_field_populated(self, api_client):
        client, mock_agent, _ = api_client
        mock_agent.invoke.return_value = _agent_result()
        data = client.post("/simulate-review", json={
            "user_id": "emmanuel",
            "item": {"name": "Nokia 3310", "category": "Phones & Tablets"},
        }).json()
        assert data.get("naija_review") == "Abeg, this phone na banger!"


class TestRecommendEndpoint:
    def test_returns_200_on_valid_request(self, api_client):
        client, _, mock_task_b = api_client
        mock_task_b.invoke.return_value = _recommend_result()
        response = client.post("/recommend", json={
            "user_id": "U123",
            "context": "looking for headphones",
            "n": 2,
        })
        assert response.status_code == 200

    def test_response_has_recommendations_list(self, api_client):
        client, _, mock_task_b = api_client
        mock_task_b.invoke.return_value = _recommend_result()
        data = client.post("/recommend", json={
            "user_id": "U123",
            "context": "looking for headphones",
            "n": 2,
        }).json()
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) == 2

    def test_recommendation_has_required_fields(self, api_client):
        client, _, mock_task_b = api_client
        mock_task_b.invoke.return_value = _recommend_result()
        data = client.post("/recommend", json={
            "user_id": "U123",
            "context": "electronics",
        }).json()
        rec = data["recommendations"][0]
        assert "item_id" in rec
        assert "name" in rec
        assert "reason" in rec

    def test_missing_context_returns_422(self, api_client):
        client, _, _ = api_client
        response = client.post("/recommend", json={"user_id": "U123"})
        assert response.status_code == 422
