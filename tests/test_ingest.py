"""チャンキング関数のテスト."""

import pytest


class TestChunking:
    """ingest.pyのチャンキング関連テスト."""

    def test_chromadb_chunk_count(self):
        """ChromaDBに112チャンクが格納されていること."""
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        assert collection.count() == 112

    def test_chunks_have_content_type_metadata(self):
        """全チャンクにcontent_typeメタデータが存在すること."""
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        for metadata in results["metadatas"]:
            assert "content_type" in metadata
            assert metadata["content_type"] in ("text", "financial_table")

    def test_table_chunks_exist(self):
        """テーブルチャンクが少なくとも1つ存在すること."""
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        table_chunks = [
            m for m in results["metadatas"]
            if m.get("content_type") == "financial_table"
        ]
        assert len(table_chunks) > 0, "テーブルチャンクが1つも存在しない"
