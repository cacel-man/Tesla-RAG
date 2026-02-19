"""BM25 + Vector のハイブリッド検索（RRF統合）."""

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import chromadb

from config import BM25_TOP_K_MULTIPLIER, RRF_K, TABLE_BOOST_FACTOR


class HybridSearcher:
    """BM25とベクトル検索をRRFで統合するハイブリッド検索器."""

    def __init__(self, collection: chromadb.Collection) -> None:
        self.collection = collection

        # ChromaDBから全ドキュメントを取得
        all_data = collection.get(include=["documents", "metadatas"])
        self.documents: list[str] = all_data["documents"]
        self.metadatas: list[dict] = all_data["metadatas"]
        self.ids: list[str] = all_data["ids"]

        # BM25インデックスを構築（スペース分割トークナイズ）
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(
        self,
        query: str,
        model: SentenceTransformer,
        top_k: int = 5,
    ) -> list[dict]:
        """ハイブリッド検索を実行し、RRFで統合した上位top_k件を返す."""
        fetch_k = top_k * BM25_TOP_K_MULTIPLIER

        # 1. ベクトル検索
        query_embedding = model.encode([query]).tolist()
        vector_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
        )
        vector_ids = vector_results["ids"][0]

        # 2. BM25検索
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]
        bm25_ids = [self.ids[i] for i in bm25_top_indices]

        # 3. RRF統合
        ranked_lists = [vector_ids, bm25_ids]
        fused_ids = self._rrf_fusion(ranked_lists, top_k=top_k)

        # 4. 結果を組み立て
        # 各IDがどの検索で引っかかったかを記録
        vector_set = set(vector_ids)
        bm25_set = set(bm25_ids)

        results = []
        for doc_id, score in fused_ids:
            idx = self.ids.index(doc_id)
            sources = []
            if doc_id in vector_set:
                sources.append("vector")
            if doc_id in bm25_set:
                sources.append("bm25")

            results.append({
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": score,
                "sources": sources,
            })

        return results

    def _rrf_fusion(
        self,
        ranked_lists: list[list[str]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusionでランキングを統合する.

        score(doc) = Σ 1 / (k + rank_i)
        k = RRF_K (デフォルト60)
        rank_i = i番目のリストでの順位（1始まり）
        """
        scores: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        # テーブルチャンクにブーストを適用
        for doc_id in scores:
            idx = self.ids.index(doc_id)
            if self.metadatas[idx].get("content_type") == "financial_table":
                scores[doc_id] *= TABLE_BOOST_FACTOR

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
