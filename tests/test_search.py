"""検索関数のテスト."""

import pytest


class TestHybridSearch:
    """ハイブリッド検索のテスト."""

    def test_hybrid_search_returns_results(self, searcher, embedding_model):
        """ハイブリッド検索が結果を返すこと."""
        results = searcher.search("Tesla revenue", embedding_model, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_hybrid_search_result_has_content(self, searcher, embedding_model):
        """検索結果にcontentフィールドが含まれること."""
        results = searcher.search("Tesla revenue", embedding_model, top_k=1)
        assert "content" in results[0]
        assert len(results[0]["content"]) > 0

    def test_hybrid_search_with_reranker(self, searcher, embedding_model, reranker_model):
        """Reranker付きハイブリッド検索が結果を返すこと."""
        results = searcher.search(
            "Tesla revenue", embedding_model, top_k=3, reranker=reranker_model
        )
        assert len(results) > 0
        assert len(results) <= 3

    def test_table_query_returns_table_chunks(self, searcher, embedding_model):
        """テーブル関連クエリでテーブルチャンクが上位に含まれること."""
        results = searcher.search(
            "Tesla financial summary revenue table", embedding_model, top_k=5
        )
        content_types = [r.get("metadata", {}).get("content_type") for r in results]
        assert "financial_table" in content_types, (
            "テーブルクエリなのにテーブルチャンクが上位5件に含まれない"
        )
