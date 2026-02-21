"""共通フィクスチャ: モデルやテストデータの準備."""

import sys
import os

import pytest

# srcディレクトリをPythonパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def embedding_model():
    """SentenceTransformerモデル（セッション全体で1回だけロード）."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def collection():
    """ChromaDBコレクション（セッション全体で1回だけ取得）."""
    from query import get_collection
    return get_collection()


@pytest.fixture(scope="session")
def searcher(collection):
    """HybridSearcherインスタンス（セッション全体で1回だけ初期化）."""
    from hybrid_search import HybridSearcher
    return HybridSearcher(collection)


@pytest.fixture(scope="session")
def reranker_model():
    """Rerankerインスタンス."""
    from reranker import Reranker
    return Reranker()


@pytest.fixture
def sample_text_chunk():
    """テスト用テキストチャンク."""
    return {
        "text": "Tesla's total revenue for Q4 2024 was $25.7 billion.",
        "metadata": {"content_type": "text", "page": 1},
    }


@pytest.fixture
def sample_table_chunk():
    """テスト用テーブルチャンク."""
    return {
        "text": "| Revenue | Q4 2024 | Q4 2023 |\n|---------|---------|---------|"
                "\n| Automotive | $19.8B | $21.6B |",
        "metadata": {"content_type": "financial_table", "page": 5},
    }
