# V2: ハイブリッド検索実装プロンプト

## 目的
Tesla-RAGの検索精度を改善する。V1ではベクトル検索単体でrevenueチャンク（21/144件）がtop_k=5に1件も入らなかった。BM25を追加しRRFで統合することで、キーワードマッチと意味検索の両方を活かす。

## 前提
- 既存コード: src/ingest.py, src/query.py, src/config.py, src/evaluate.py
- ChromaDB: chroma_db/ に144チャンク格納済み
- Embedding: all-MiniLM-L6-v2
- LangChain不使用。直接実装する

## 実装内容

### 1. rank_bm25のインストール
```bash
pip install rank-bm25
```

### 2. src/hybrid_search.py を新規作成

#### 必要なクラス/関数:
```
class HybridSearcher:
    def __init__(self, chroma_collection):
        - ChromaDBからall documentsを取得
        - BM25Okapiインデックスを構築（トークナイズはスペース分割でOK）
        - documentsとmetadataを保持

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        - vector_results: ChromaDBのquery()でtop_k*2件取得
        - bm25_results: BM25でtop_k*2件取得
        - RRFで統合してtop_k件返す

    def _rrf_fusion(self, ranked_lists: list[list], k: int = 60) -> list:
        - Reciprocal Rank Fusion
        - score = Σ 1/(k + rank) for each list
        - kはデフォルト60（標準値）
```

#### RRFの計算式:
```
RRF_score(doc) = Σ 1 / (k + rank_i)
- k = 60（定数）
- rank_i = そのdocのi番目のランキングリストでの順位（1始まり）
- ランキングリストに含まれない場合はスキップ
```

#### 返り値の形式（V1のquery.pyと互換）:
```python
[
    {
        "content": "チャンクのテキスト",
        "metadata": {"source": "...", "chunk_id": ...},
        "score": 0.032,  # RRFスコア
        "sources": ["vector", "bm25"]  # どの検索で引っかかったか
    },
    ...
]
```

### 3. src/query.py を修正
- HybridSearcherをimport
- 既存のChromaDB直接検索をHybridSearcherに差し替え
- --mode オプション追加: "vector"（V1互換）/ "hybrid"（V2デフォルト）
- チャンク可視化で「sources」も表示（どの検索で引っかかったか分かるように）

### 4. src/config.py に追加
```python
# V2: Hybrid Search
BM25_TOP_K_MULTIPLIER = 2  # BM25/Vectorそれぞれtop_k * この値を取得
RRF_K = 60                  # RRF定数
SEARCH_MODE = "hybrid"      # "vector" or "hybrid"
```

### 5. src/evaluate.py を修正
- search_mode パラメータ追加
- 結果JSONにsearch_modeを記録
- V1結果との比較表示機能（オプション）

## テスト手順
1. `python src/query.py --mode hybrid "What was Tesla's total revenue in Q4 2025?"` で動作確認
2. チャンク可視化でrevenueチャンクがtop_kに含まれることを確認
3. `python src/evaluate.py --mode hybrid` で10問評価
4. V1結果と比較: 正答率、Relevancy、Faithfulness、Completeness

## 成功基準
- revenueチャンクがtop_k=5に含まれる
- 正答率: 1/10 → 4/10以上
- Completeness: 3.3 → 4.0以上
- Faithfulness: 5.0を維持（下がったらNG）

## 注意点
- BM25のトークナイズは最初はシンプルに（split()）。改善はV2.1以降
- ChromaDBの既存データはそのまま使う。再ingest不要
- hybrid_search.pyは独立モジュールとして作る（query.pyから差し替え可能に）