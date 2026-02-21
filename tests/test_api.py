"""FastAPIエンドポイントのテスト."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """FastAPIテストクライアント（lifespanでモデルロード）."""
    from api import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """GET /health のテスト."""

    def test_health_returns_200(self, client):
        """ヘルスチェックが200を返すこと."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        """ヘルスチェックのレスポンスに必要なフィールドが含まれること."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "chromadb_chunks" in data
        assert "bm25_loaded" in data
        assert "reranker_loaded" in data
        assert "model" in data


class TestQueryEndpoint:
    """POST /query のテスト."""

    def test_query_validation_empty_question(self, client):
        """空の質問で422エラーが返ること."""
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_validation_invalid_mode(self, client):
        """不正なsearch_modeで422エラーが返ること."""
        response = client.post(
            "/query",
            json={"question": "test", "search_mode": "invalid"},
        )
        assert response.status_code == 422

    def test_query_validation_top_k_too_low(self, client):
        """top_k=0で422エラーが返ること."""
        response = client.post(
            "/query",
            json={"question": "test", "top_k": 0},
        )
        assert response.status_code == 422

    def test_query_validation_top_k_too_high(self, client):
        """top_k=100で422エラーが返ること."""
        response = client.post(
            "/query",
            json={"question": "test", "top_k": 100},
        )
        assert response.status_code == 422

    def test_query_response_fields(self, client):
        """クエリレスポンスに必要なフィールドが含まれること."""
        response = client.post(
            "/query",
            json={"question": "What was Tesla total revenue?", "search_mode": "vector"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "search_mode" in data
        assert "chunks_used" in data
        assert "crag_grades" in data
        assert "crag_retries" in data
        assert "processing_time_sec" in data

    def test_query_crag_mode_has_grades(self, client):
        """CRAGモードでcrag_gradesが空でないこと."""
        response = client.post(
            "/query",
            json={"question": "What was Tesla total revenue?", "search_mode": "crag"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["crag_grades"]) > 0

    def test_query_non_crag_mode_empty_grades(self, client):
        """CRAG以外のモードでcrag_gradesが空であること."""
        response = client.post(
            "/query",
            json={"question": "What was Tesla total revenue?", "search_mode": "rerank"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["crag_grades"] == []
        assert data["crag_retries"] == 0
