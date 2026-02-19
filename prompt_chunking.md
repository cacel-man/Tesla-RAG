# V3: テーブル対応チャンキング + メタデータ強化

## 目的
V2で判明した問題を解決する：
- FINANCIAL SUMMARYテーブルがchunk_size=1000で分断され、ラベルと数値が別チャンクになる
- 検索がテーブルチャンクを優先的に拾えない
- Adjusted EBITDA（chunk_65）、Energy storage revenue（chunk_6）等のデータがチャンクに存在するのにtop_k=5に入らない

## 前提
- 既存コード: src/ingest.py, src/config.py, src/hybrid_search.py, src/query.py, src/evaluate.py
- チャンキング: RecursiveCharacterTextSplitter（chunk_size=1000, overlap=200）
- テーブルとテキストの区別なし（現状）
- LangChain不使用方針（チャンキングのtext-splittersのみ例外、これは継続）

## 実装内容

### 1. src/ingest.py の chunk_pages() を改修

#### 方針: テーブルページ検出 → テーブルは大きめチャンク、テキストは従来通り

```python
def is_table_page(text: str) -> bool:
    """テーブルページかどうかを判定する."""
    # FINANCIAL SUMMARYやRECONCILIATIONなどのテーブルページを検出
    table_indicators = [
        "F I N A N C I A L   S U M M A R Y",
        "F I N A N C I A L S U M M A R Y",
        "R E C O N C I L I A T I O N",
        "REVENUES",
        "In millions of USD",
    ]
    return any(indicator in text for indicator in table_indicators)
```

#### テーブルページの処理:
- テーブルページはchunk_size=2500, overlap=500で分割（テーブル全体が1チャンクに収まりやすい）
- メタデータに `"content_type": "financial_table"` を付与
- 通常テキストページは従来通り chunk_size=1000, overlap=200

```python
def chunk_pages(pages: list[dict]) -> list[dict]:
    """ページテキストをチャンクに分割し、メタデータを付与する."""
    # 2つのスプリッターを用意
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,           # 1000
        chunk_overlap=CHUNK_OVERLAP,     # 200
    )
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE,     # 2500
        chunk_overlap=TABLE_CHUNK_OVERLAP,  # 500
    )

    chunks = []
    chunk_index = 0

    for page_data in pages:
        text = page_data["text"]
        metadata = page_data["metadata"]
        section = extract_section_header(text)
        
        # テーブルページ判定
        is_table = is_table_page(text)
        splitter = table_splitter if is_table else text_splitter
        content_type = "financial_table" if is_table else "text"

        split_texts = splitter.split_text(text)
        for split_text in split_texts:
            chunks.append({
                "text": split_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "section": section,
                    "content_type": content_type,  # 新規追加
                },
            })
            chunk_index += 1

    return chunks
```

### 2. src/config.py に追加
```python
# --- V3: テーブル対応チャンキング ---
TABLE_CHUNK_SIZE: int = 2500
TABLE_CHUNK_OVERLAP: int = 500
```

### 3. src/hybrid_search.py の検索でテーブルチャンクをブースト

#### RRFスコアにcontent_typeベースのブーストを追加:
```python
def _rrf_fusion(self, ranked_lists, top_k=5):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    
    # テーブルチャンクにブーストを適用
    for doc_id in scores:
        idx = self.ids.index(doc_id)
        if self.metadatas[idx].get("content_type") == "financial_table":
            scores[doc_id] *= TABLE_BOOST_FACTOR  # 1.2倍
    
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]
```

### 4. src/config.py に追加（ブースト係数）
```python
TABLE_BOOST_FACTOR: float = 1.2  # テーブルチャンクのRRFスコアブースト
```

### 5. ChromaDBの再構築が必要
チャンキング戦略を変更するため、既存DBを再構築する必要がある：
```bash
python src/ingest.py --force
```

### 6. 変更しないもの
- src/query.py — 変更不要（hybrid_search.pyの内部変更で吸収）
- src/evaluate.py — 変更不要（同じ10問で評価）
- 評価基準のexact_match判定 — V3では変更しない（別トラックで対応）

## テスト手順
1. `python src/ingest.py --force` でDB再構築
2. 再構築後のチャンク数を確認（144から変化するはず）
3. `python src/search_chunks.py "adjusted ebitda"` でchunk内にラベル+数値が同居してるか確認
4. `python src/search_chunks.py "energy generation and storage revenue"` で同様に確認
5. `python src/query.py --mode hybrid "What was Tesla's Adjusted EBITDA in 2025?"` で動作確認
6. `python src/evaluate.py --mode hybrid` で10問評価
7. V2結果と比較

## 成功基準
- テーブルページのFINANCIAL SUMMARYが1チャンクに収まる
- Adjusted EBITDA $14,596Mがラベルと一緒に1チャンク内に存在
- Energy storage revenue $3,837Mがラベルと一緒に1チャンク内に存在
- Completeness: 4.2 → 4.5以上
- Faithfulness: 5.0維持
- 正答率: 3/10 → 4/10以上（exact_match正規化なしの厳しい基準で）

## 注意点
- ingest.py --force で再構築するとチャンク数が変わる。BM25インデックスはHybridSearcherの__init__で自動再構築されるので問題なし
- TABLE_BOOST_FACTOR=1.2は控えめな値。効果が薄ければ1.3-1.5に上げる余地あり
- テーブル判定のtable_indicatorsはTesla IRに特化している。V4でマルチ企業対応する時に汎用化が必要