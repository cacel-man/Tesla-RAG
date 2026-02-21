"""Cross-Encoder Reranker: 検索結果をクエリとの関連度で再ランキングする."""

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        """Cross-Encoderモデルをロードする."""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """検索結果をクエリとの関連度で再ランキングする.

        Args:
            query: ユーザーのクエリ
            chunks: hybrid_search.pyから返されたチャンクリスト
                    各チャンク: {"content": str, "metadata": dict, "score": float, "sources": list}
            top_k: 最終的に返すチャンク数

        Returns:
            リランキングされたtop_k件のチャンクリスト
            各チャンクに "rerank_score" と "original_rank" が追加される
        """
        pairs = [[query, chunk["content"]] for chunk in chunks]
        scores = self.model.predict(pairs)

        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
            chunk["original_rank"] = i + 1

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
