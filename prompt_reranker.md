# V4: Reranker導入（Cross-Encoder）

## 目的
V2で発見したノイズチャンク問題を解決する。ベクトル検索が「意味的に近いだけで関連性が低い」チャンクを拾う問題（例: revenueクエリでSupercharger統計が混入）を、検索後のリランキングで除去する。

## 前提
- 既存コード: src/ingest.py, src/config.py, src/hybrid_search.py, src/query.py, src/evaluate.py
- ChromaDB: 112チャンク（V3テーブル対応チャンキング済み）
- 検索: ハイブリッド検索（BM25 + Vector + RRF + テーブルブースト）
- LangChain不使用方針を継続
- ingest.pyの変更は不要（チャンキングは変えない）

## アーキテクチャ上の位置

```
クエリ
  ├── ベクトル検索 → top_k*2件
  ├── BM25検索 → top_k*2件
  └── RRF統合 → top_k_rerank件（多めに取得）
        └── ★ Cross-Encoder Reranker → 最終top_k件 → Claude API → 回答
```

Rerankerは検索とLLMの間に入る「フィルター」。検索結果を受け取り、クエリとの関連度を再計算して並べ替える。

## 実装内容

### 1. cross-encoderのインストール
```bash
pip install sentence-transformers
# sentence-transformersは既にインストール済みだが、CrossEncoderクラスを使う
```

### 2. src/reranker.py を新規作成

```python
from sentence_transformers import CrossEncoder

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
        # Cross-Encoderにクエリとチャンクのペアを渡す
        pairs = [[query, chunk["content"]] for chunk in chunks]
        scores = self.model.predict(pairs)

        # スコアとチャンクを紐づけ
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
            chunk["original_rank"] = i + 1

        # リランクスコアで降順ソート
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
```

### 3. src/hybrid_search.py を修正

#### searchメソッドの変更:
- RRF統合後のtop_k取得数を増やす: top_k → RERANK_CANDIDATES件（多めに取得）
- Rerankerが有効な場合、RRF結果をRerankerに渡してフィルタリング

```python
def search(self, query, model, top_k=5, reranker=None):
    # ... 既存のベクトル検索 + BM25 + RRF統合 ...
    
    if reranker:
        # RRFからRERANK_CANDIDATES件取得 → Rerankerでtop_kに絞る
        rrf_results = self._get_rrf_results(ranked_lists, top_k=RERANK_CANDIDATES)
        results = reranker.rerank(query, rrf_results, top_k=top_k)
    else:
        # Rerankerなし（V3互換）
        results = self._get_rrf_results(ranked_lists, top_k=top_k)
    
    return results
```

### 4. src/query.py を修正
- --modeオプションを拡張: "vector" / "hybrid" / "rerank"
- "rerank"モード: HybridSearcher + Rerankerを使用
- チャンク可視化で "rerank_score" と "original_rank" も表示

### 5. src/evaluate.py を修正
- --mode rerank に対応
- 結果JSONにsearch_mode="rerank"を記録
- V3結果との比較表示

### 6. src/config.py に追加
```python
# --- V4: Reranker ---
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 軽量で高性能
RERANK_CANDIDATES: int = 15  # Rerankerに渡す候補数（top_k * 3）
SEARCH_MODE: str = "rerank"  # "vector" or "hybrid" or "rerank"（デフォルト更新）
```

#### モデル選定理由: cross-encoder/ms-marco-MiniLM-L-6-v2
- MS MARCOで学習済み（情報検索タスクに最適化）
- 軽量（MiniLM-L-6ベース）でローカル推論が速い
- sentence-transformersのCrossEncoderクラスでそのまま使える
- 実務でもよく使われる標準的なRerankerモデル

### 7. 変更しないもの
- src/ingest.py — 変更不要（チャンキングは変えない）
- chroma_db/ — 再構築不要（V3のチャンクをそのまま使用）

## テスト手順
1. `python src/query.py --mode rerank "What was Tesla's total revenue in Q4 2025?"` で動作確認
2. チャンク可視化で以下を確認:
   - rerank_scoreが表示されること
   - original_rankとrerank後の順位が異なること（リランキングが効いていること）
   - V2で問題だったノイズチャンク（Supercharger統計等）が除外されていること
3. `python src/query.py --mode hybrid "What was Tesla's total revenue in Q4 2025?"` でV3互換動作を確認
4. `python src/evaluate.py --mode rerank` で10問評価
5. V3結果と比較

## 成功基準
- ノイズチャンク（content_type: textの無関係チャンク）がtop_k=5から除外される
- 正答率: 6/10以上を維持（下がったらNG）
- Completeness: 4.5以上を維持
- Faithfulness: 5.0維持
- Relevancy: 5.0維持
- V3で正解だった問題がV4で不正解にならないこと（デグレ防止）

## 注意点
- Cross-Encoderは初回ロード時にモデルダウンロードが発生する（約90MB）
- Rerankerの推論はBi-Encoderより遅い。RERANK_CANDIDATES=15程度なら体感に影響なし
- reranker.pyは独立モジュール。hybrid_search.pyにreranker=Noneを渡せばV3と同じ挙動
- --mode hybrid でV3互換、--mode rerank でV4の動作。常にA/Bテスト可能
- RERANK_CANDIDATES=15は「RRFで15件取得→Rerankerで5件に絞る」という意味。少なすぎると良いチャンクを取りこぼし、多すぎるとRerankerが遅くなる

## 期待される改善の仕組み
1. RRFで15件の候補を広めに取得（今まではtop_k=5で即確定していた）
2. Cross-Encoderがクエリとチャンクを「ペアで」評価（Bi-Encoderより高精度）
3. クエリに本当に関連するチャンクだけがtop_k=5に残る
4. ノイズチャンクは低スコアになり除外される