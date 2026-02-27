# Tesla-RAG: 精度10%→80%への体系的なRAG改善プロジェクト

TeslaのIRレポート（決算報告書）に対する質問応答RAGシステム。8つのバージョンを通じて段階的に改善し、各バージョンで1つの改善レバーだけを変更して効果を計測・記録しています。

> **基本方針**: 1バージョン1変更。すべて計測。データ品質はアルゴリズムより先。

---

## 精度推移

| Version | 変更内容 | 正答率 | 変化 | Relevancy | Faithfulness | Completeness |
|---------|----------|--------|------|-----------|--------------|--------------|
| **V1** | ベクトル検索ベースライン | 10% (1/10) | — | 4.7 | 5.0 | 3.3 |
| **V2** | + BM25ハイブリッド検索 (RRF) | 30% (3/10) | +20pp | 4.9 | 5.0 | 4.2 |
| **V3** | + テーブル対応チャンキング | 60% (6/10) | +30pp | 5.0 | 5.0 | 4.5 |
| **V4** | + Cross-Encoderリランキング | 80% (8/10) | +20pp | 5.0 | 5.0 | 4.8 |
| **V5** | + CRAG（自己修正ループ） | 80% (8/10) | +0pp | 5.0 | 5.0 | 4.7 |
| **V6** | FastAPI化 | — | — | — | — | — |
| **V7** | pytest（21テスト、バグ発見） | — | — | — | — | — |
| **V8** | Docker化 | — | — | — | — | — |

**評価方法**: 財務に関する10問のQ&Aデータセット。完全一致判定 + LLM-as-a-Judgeによる3指標（Relevancy / Faithfulness / Completeness、各5点満点）で自動評価。

---

## アーキテクチャ（V8最終版）

```
┌──────────────────────────────────────────────────────────────┐
│                      Dockerコンテナ                            │
│                                                               │
│  PDF ──→ テーブル/テキスト判定 ──→ チャンキング ──→ ChromaDB   │
│          (is_table_page())          テーブル: 2500文字         │
│                                     テキスト: 1000文字         │
│                                                               │
│  クエリ ─┬─→ ベクトル検索 (top_k×2) ──┐                       │
│          └─→ BM25検索 (top_k×2) ──────┤                       │
│                                        ▼                      │
│                               RRF統合 (k=60)                  │
│                            + テーブルブースト (×1.2)           │
│                                        │                      │
│                                        ▼                      │
│                          Cross-Encoder Reranker               │
│                          (15件候補 → 5件に絞込)               │
│                                        │                      │
│                                        ▼                      │
│                            CRAG品質ゲート                     │
│                     (CORRECT / AMBIGUOUS / INCORRECT)         │
│                          ↓                ↓                   │
│                       通過          クエリ書換→再検索          │
│                          ↓          (最大1回)                 │
│                                        │                      │
│                                        ▼                      │
│                          Claude API ──→ 回答                  │
│                                                               │
│  FastAPI: POST /query, GET /health                            │
│  pytest: 21テスト（5モック、16結合テスト）                      │
└──────────────────────────────────────────────────────────────┘
```

---

## バージョン履歴

### V1: ベースライン — ベクトル検索のみ（10%）
**課題**: LangChainを使わずにRAGパイプラインをゼロから構築し、各レイヤーの責務を理解する。  
**結果**: Faithfulnessは満点（5.0）— LLMはハルシネーションゼロ。しかし正答率は1/10。ベクトル検索が財務データのチャンクを取得できず、"revenue"の質問に「Superchargerの統計」を返していた。  
**気づき**: ボトルネックは生成（LLM）ではなく検索（Retrieval）にあった。

### V2: ハイブリッド検索 — BM25 + RRF（30%）
**課題**: ベクトル検索単体ではキーワード依存の財務データ（"revenue"等）を拾えない。  
**解決策**: BM25キーワード検索を追加し、Reciprocal Rank Fusion（k=60）でスコアを統合。  
**なぜRRFを選んだか**: RRFはスコアではなく順位ベースで統合するため、BM25とコサイン類似度のスケール差を正規化せずに使える。実装3行、ハイパーパラメータなし。  
**結果**: +20pp改善。BM25のキーワードマッチでrevenueチャンクが確実にヒットするようになった。

### V3: テーブル対応チャンキング（60%）⭐
**課題**: 残り7問の不正解のうち3問は、データがChromaDB内に存在するがチャンク境界で分断されていた。テーブルのラベルと数値が別チャンクに分かれていた。  
**当初の計画**: V3はCRAG（Active RAG）を実装する予定だった。しかし調査の結果、優先順位を変更。  
**解決策**: ルールベースでテーブルページを検出し（`is_table_page()`）、テーブルには大きめのチャンクサイズ（2500文字 vs 1000文字）を適用してテーブル全体を1チャンクに保持。  
**結果**: チャンク数が22%減少（144→112）したにもかかわらず、+30pp改善。**「データ品質 > アルゴリズム」を証明** — プロジェクト全体で最大の精度向上はデータの修正から生まれた。

### V4: Cross-Encoderリランキング（80%）
**課題**: ハイブリッド検索でもノイズチャンク（意味的に近いだけで無関係なデータ）が混入する問題が残っていた。  
**解決策**: 2段階検索を導入。RRFで15件の候補を取得 → `cross-encoder/ms-marco-MiniLM-L-6-v2`で5件に絞り込み。Cross-Encoderはクエリとチャンクをペアで同時評価するため、Bi-Encoderより高精度な関連度判定が可能。  
**結果**: +20pp改善。デバッグツール（`search_chunks.py`）にハードコードされたバグも発見・修正。

### V5: CRAG — Corrective RAG（80%）
**課題**: パイプラインに検索品質の自己修正機能がなかった。  
**解決策**: LLMが検索結果の品質を判定（CORRECT / AMBIGUOUS / INCORRECT）。不十分な場合はクエリを書き換えて再検索。  
**結果**: 全10問でCRAG判定がCORRECT、リトライ発生0回。**これはV4のRerankerが十分な品質のチャンクを返していたことの証明。** CRAGは未知のクエリに対するセーフティネットとして残す価値がある。  
**設計判断**: パース失敗時はINCORRECT（リトライ実行）にデフォルト。余計なAPI呼び出し1回のコストより、不十分な検索結果で回答するリスクの方が大きい。

### V6: FastAPI化
パイプラインを`POST /query`（4つの検索モード対応）と`GET /health`でHTTP API化。コアロジックの変更なし — V2で始めたモジュール設計が機能した。

### V7: pytest — 21の自動テスト
- **重大バグを発見**: `crag.py`の品質判定で部分一致を使っていたため、"INCORRECT"に含まれる"CORRECT"で誤判定が発生。判定順序を修正。
- CRAGテストはClaude APIをモック化（APIキー不要、高速、再現可能）。
- APIテストはFastAPIの`TestClient`で実パイプラインとの結合テスト。

### V8: Docker化
`docker build && docker run`だけでパイプライン全体が動作。Dockerのクリーンビルドで`rank-bm25`のrequirements.txt漏れを発見 — ローカルのconda環境に隠れていた依存関係の不備が炙り出された。

---

## 主要な設計判断

### 1. LangChain不使用
LangChainの`EnsembleRetriever`を使えばハイブリッド検索は数分で実装できた。あえて自前実装を選んだのは、面接で「RRFの計算式は？」と聞かれた時に答えられる必要があったから。トレードオフ: 開発速度は遅くなるが、理解の深さが違う。

### 2. 「データ品質 > アルゴリズム」によるV3のロードマップ変更
当初のロードマップではV3 = CRAG。しかし失敗原因を調査した結果、チャンク境界がテーブルを分断しているのが根本原因だと判明。CRAGで再検索しても同じ壊れたチャンクを拾うだけ。データ品質の修正を優先する判断をした。この1つの意思決定がプロジェクト全体で最大の精度向上（+30pp）を生んだ。

### 3. 独立モジュール設計
主要機能（hybrid_search.py, reranker.py, crag.py）はすべて独立モジュール。`query.py`が`--mode`フラグで切り替える:
```bash
python src/query.py --mode vector   # V1
python src/query.py --mode hybrid   # V2-V3
python src/query.py --mode rerank   # V4
python src/query.py --mode crag     # V5
```
全バージョンのA/Bテストがコード変更なしで可能。

### 4. 安全側に倒す設計
CRAGの品質判定でパース失敗時 → INCORRECT（リトライ実行）にデフォルト。API呼び出し1回のコスト増より、不十分な検索結果で回答を返すリスクの方が大きい。本番RAGでは「間違った回答」は「遅い回答」より遥かに危険。

---

## 学んだこと

### 技術面
- **2段階検索が業界標準**: 高速リコール（BM25 + Vector）→ 高精度リランキング（Cross-Encoder）。Azure AI SearchのSemantic Rankerも内部でこの構造を使っている。
- **評価指標がボトルネックを教えてくれる**: Faithfulness低い → LLMの問題。Completeness低い → 検索の問題。V1の分析（Faithfulness=5.0, Completeness=3.3）で即座に検索がボトルネックだと特定できた。
- **効果がないことも成果**: V5のCRAGはリトライ0回。これはV5の無意味さではなく、V4のパイプライン品質の証明。

### プロセス面
- **1バージョン1変更の原則**: 各バージョンで1つだけ変えることで、精度向上の原因を特定の改善に帰属できた。
- **デバッグツールにもバグはある**: V4でデバッグ用スクリプトにハードコードされたキーワードを発見。バグを探すツールがバグっていた。
- **Dockerは依存関係の監査役**: V8のクリーンビルドでrequirements.txtの漏れを発見。ローカル環境は依存関係の不備を隠す。

---

## 技術スタック

| コンポーネント | 技術 |
|----------------|------|
| 言語 | Python 3.11 |
| ベクトルDB | ChromaDB（永続化モード） |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 |
| BM25 | rank_bm25 (BM25Okapi) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Claude API (claude-sonnet-4-20250514) |
| API | FastAPI + Uvicorn |
| テスト | pytest + unittest.mock |
| コンテナ | Docker (python:3.11-slim) |
| スコア統合 | Reciprocal Rank Fusion (k=60) |

---

## クイックスタート

### Docker（推奨）
```bash
docker build -t tesla-rag .
docker run -d -p 8000:8000 --env-file .env tesla-rag

# ヘルスチェック
curl http://localhost:8000/health

# クエリ実行
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Tesla total revenue in Q4 2025?"}'
```

### ローカル実行
```bash
pip install -r requirements.txt

# PDF取り込み（初回のみ）
python src/ingest.py

# クエリ実行（検索モード選択可能）
python src/query.py --mode crag "What was Tesla's Adjusted EBITDA in 2025?"

# 評価パイプライン実行
python src/evaluate.py --mode crag

# テスト実行
PYTHONPATH=src pytest tests/ -v
```

---

## プロジェクト構成

```
Tesla-RAG/
├── src/
│   ├── api.py              # FastAPIサーバー（POST /query, GET /health）
│   ├── config.py           # 全ハイパーパラメータを集約
│   ├── ingest.py           # PDF → チャンク → ChromaDB
│   ├── query.py            # オーケストレーター（4つの検索モード）
│   ├── hybrid_search.py    # BM25 + Vector + RRF統合
│   ├── reranker.py         # Cross-Encoderリランキング
│   ├── crag.py             # 品質ゲート + クエリ書換
│   ├── evaluate.py         # 10問ベンチマーク
│   └── search_chunks.py    # デバッグ用キーワード検索
├── tests/
│   ├── conftest.py         # 共通フィクスチャ（session-scopeモデルロード）
│   ├── test_ingest.py      # チャンク数、メタデータ、テーブル検出
│   ├── test_search.py      # ハイブリッド検索、Reranker、テーブルブースト
│   ├── test_crag.py        # 品質判定パース（モック化、APIキー不要）
│   └── test_api.py         # エンドポイント検証（TestClient）
├── data/                   # Tesla IRレポートPDF（Q3, Q4 2025）
├── results/                # 評価結果JSON
├── Dockerfile
├── requirements.txt
└── CLAUDE.md               # AIアシスタント指示書
```

---

## 評価の詳細

**データセット**: Teslaの2025年Q3/Q4 IRレポートから、収益・利益率・EPS・EBITDA・フリーキャッシュフロー・セグメント別内訳をカバーする10問の財務質問。

**評価指標**:
- **Exact Match**: 期待される回答との文字列一致
- **Relevancy** (1-5): 回答が質問に対応しているか
- **Faithfulness** (1-5): 回答が取得したコンテキストに基づいているか（ハルシネーションがないか）
- **Completeness** (1-5): 回答に必要な情報がすべて含まれているか

**V4/V5の残り2問について**: (1) Q6: フリーキャッシュフローが12四半期分のRECONCILIATIONテーブルに埋もれている — テキストベースRAGの構造的限界。(2) Q8: 正しい数値（$28,095M→$24,901M, decreased）を含む回答だがexact_matchの表記揺れで不正解判定。実質的な精度は約90%。