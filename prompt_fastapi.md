# V6: FastAPI化（RAGパイプラインのAPI化）

## 目的
V5まではCLIベースで動かしていた（`python src/query.py --mode crag "質問"`）。これは開発・デバッグには十分だが、実際のプロダクトではAPIとして他のシステムやフロントエンドから呼び出される。V6ではFastAPIでHTTPエンドポイントを提供し、RAGパイプラインをAPIサーバーとして動かせるようにする。Kasanareの業務要件「FastAPI（Python）を用いた機能実装」に直結する。

## 前提
- 既存コード: src/ingest.py, src/config.py, src/hybrid_search.py, src/reranker.py, src/crag.py, src/query.py, src/evaluate.py
- ChromaDB: 112チャンク（V3テーブル対応チャンキング済み）
- 全検索モード動作確認済み: vector / hybrid / rerank / crag
- LangChain不使用方針を継続
- 既存モジュール（ingest.py, hybrid_search.py, reranker.py, crag.py）の変更は不要

## アーキテクチャ上の位置

```
【V5まで: CLI】
ターミナル → python src/query.py --mode crag "質問" → 標準出力

【V6: API】
HTTPクライアント → POST /query → FastAPIサーバー
                                      ├── vector: ベクトル検索 → 回答生成
                                      ├── hybrid: BM25+Vector+RRF → 回答生成
                                      ├── rerank: hybrid + Cross-Encoder → 回答生成
                                      └── crag: rerank + CRAG品質判定ループ → 回答生成
                 → GET /health → ヘルスチェック（モデルロード状態確認）
```

FastAPIはCLIの上に薄いHTTPレイヤーを被せるだけ。検索・リランク・CRAG・回答生成のロジックは既存モジュールをそのまま使う。query.pyの`process_query()`にあるロジックをAPI用に再構成する。

## 実装内容

### 1. src/api.py を新規作成

```python
"""FastAPI server for Tesla-RAG pipeline."""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

from config import TOP_K, CLAUDE_MODEL, MAX_RETRY
from hybrid_search import HybridSearcher
from reranker import Reranker
from crag import CRAGProcessor

# --- Pydanticモデル ---

class SearchMode(str, Enum):
    vector = "vector"
    hybrid = "hybrid"
    rerank = "rerank"
    crag = "crag"

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="質問文")
    search_mode: SearchMode = Field(default=SearchMode.crag, description="検索モード")
    top_k: int = Field(default=TOP_K, ge=1, le=20, description="取得チャンク数")

class ChunkInfo(BaseModel):
    rank: int
    content_preview: str  # 先頭200文字
    content_type: str     # "text" or "table"
    score: float | None = None

class CRAGLog(BaseModel):
    retry: int
    query: str
    grade: str  # "CORRECT" / "AMBIGUOUS" / "INCORRECT"

class QueryResponse(BaseModel):
    answer: str
    search_mode: str
    chunks_used: int
    crag_grades: list[str] = []      # 各ステップの判定結果
    crag_retries: int = 0            # 実際のリトライ回数
    crag_log: list[CRAGLog] = []     # 詳細ログ
    chunks: list[ChunkInfo] = []     # 使用チャンク情報
    processing_time_sec: float       # 処理時間

class HealthResponse(BaseModel):
    status: str  # "healthy" / "unhealthy"
    chromadb_chunks: int
    bm25_loaded: bool
    reranker_loaded: bool
    model: str


# --- モデルのグローバル変数（lifespanで初期化） ---
searcher: HybridSearcher | None = None
reranker_model: Reranker | None = None
crag_processor: CRAGProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバー起動時にモデルをロードする."""
    global searcher, reranker_model, crag_processor
    print("🚀 モデルをロード中...")

    # SentenceTransformerの埋め込みモデルはHybridSearcher内で自動ロード
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    searcher = HybridSearcher(model=model)
    reranker_model = Reranker()
    crag_processor = CRAGProcessor()

    print(f"✅ ロード完了: ChromaDB {searcher.collection.count()} チャンク")
    yield
    print("👋 サーバーシャットダウン")


app = FastAPI(
    title="Tesla-RAG API",
    description="Tesla 10-K filing RAG pipeline with hybrid search, reranking, and CRAG",
    version="0.6.0",
    lifespan=lifespan,
)


# --- ヘルパー関数 ---

def build_context_from_results(results: list[dict]) -> tuple[str, list[ChunkInfo]]:
    """検索結果からコンテキスト文字列とチャンク情報を構築する.

    Args:
        results: hybrid_search or rerankerから返された検索結果リスト

    Returns:
        (context文字列, ChunkInfoリスト)
    """
    context_parts = []
    chunks_info = []
    for i, r in enumerate(results):
        text = r["text"]
        context_parts.append(text)
        chunks_info.append(ChunkInfo(
            rank=i + 1,
            content_preview=text[:200],
            content_type=r.get("metadata", {}).get("content_type", "text"),
            score=r.get("score"),
        ))
    context = "\n\n---\n\n".join(context_parts)
    return context, chunks_info


def ask_claude(question: str, context: str) -> str:
    """Claudeに回答を生成させる.

    既存query.pyのask_claude()と同じロジック。
    """
    import anthropic
    from config import ANTHROPIC_API_KEY

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system_prompt = (
        "You are a financial analyst assistant. Answer questions about Tesla's SEC filings "
        "based only on the provided context. If the context doesn't contain enough information, "
        "say so clearly. Be precise with numbers and cite specific data from the context."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return response.content[0].text


# --- エンドポイント ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """ヘルスチェック: モデルのロード状態を返す."""
    return HealthResponse(
        status="healthy" if searcher is not None else "unhealthy",
        chromadb_chunks=searcher.collection.count() if searcher else 0,
        bm25_loaded=searcher is not None,
        reranker_loaded=reranker_model is not None,
        model=CLAUDE_MODEL,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """RAGパイプラインを実行して回答を返す."""
    if searcher is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    start_time = time.time()
    mode = req.search_mode
    question = req.question
    top_k = req.top_k

    # SentenceTransformerモデルの取得（searcher内に保持）
    model = searcher.model

    crag_grades: list[str] = []
    crag_retries: int = 0
    crag_log: list[CRAGLog] = []
    chunks_info: list[ChunkInfo] = []
    context: str = ""

    if mode == SearchMode.vector:
        # V1: ベクトル検索のみ
        results = searcher.vector_search(question, model, top_k=top_k)
        context, chunks_info = build_context_from_results(results)

    elif mode == SearchMode.hybrid:
        # V2: ハイブリッド検索（BM25 + Vector + RRF）
        results = searcher.search(question, model, top_k=top_k)
        context, chunks_info = build_context_from_results(results)

    elif mode == SearchMode.rerank:
        # V4: ハイブリッド + Reranker
        results = searcher.search(question, model, top_k=top_k, reranker=reranker_model)
        context, chunks_info = build_context_from_results(results)

    elif mode == SearchMode.crag:
        # V5: Rerank + CRAG品質判定ループ
        current_query = question
        retry_count = 0

        while True:
            results = searcher.search(current_query, model, top_k=top_k, reranker=reranker_model)
            context, chunks_info = build_context_from_results(results)

            grade = crag_processor.grade_results(question, context)  # 常に元の質問で判定
            crag_grades.append(grade)
            crag_log.append(CRAGLog(retry=retry_count, query=current_query, grade=grade))

            if grade == "CORRECT" or retry_count >= MAX_RETRY:
                break

            current_query = crag_processor.rewrite_query(question, current_query, context)
            retry_count += 1

        crag_retries = retry_count

    # 回答生成（全モード共通）
    answer = ask_claude(question, context)

    processing_time = time.time() - start_time

    return QueryResponse(
        answer=answer,
        search_mode=mode.value,
        chunks_used=len(chunks_info),
        crag_grades=crag_grades,
        crag_retries=crag_retries,
        crag_log=crag_log,
        chunks=chunks_info,
        processing_time_sec=round(processing_time, 2),
    )
```

### 2. src/config.py に追加
```python
# --- V6: FastAPI ---
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
```

### 3. requirements.txt に追加
```
fastapi
uvicorn[standard]
```

### 4. 変更しないもの
- src/ingest.py — 変更不要
- src/hybrid_search.py — 変更不要
- src/reranker.py — 変更不要
- src/crag.py — 変更不要
- src/query.py — CLIとして残す（API化はquery.pyの置き換えではなく追加）
- src/evaluate.py — CLIのまま残す（評価はバッチ処理のためAPI経由にしない）
- chroma_db/ — 再構築不要

**重要な設計判断:**
- query.pyは削除しない。CLIとAPIを共存させる。開発時のデバッグはCLI、プロダクション利用はAPIという使い分け
- evaluate.pyはAPI経由にしない。評価は10問バッチ処理なのでCLI直接実行の方が効率的。APIのレスポンス形式が変わっても評価ロジックに影響しない
- ask_claude()はapi.py内に再定義する（query.pyからimportすると循環参照やCLI引数パースの問題が起きる可能性がある）。ロジックは同一なので、将来的にはutils.pyに切り出してもよい
- lifespanパターンでモデルロード。FastAPIの推奨パターンで、deprecated な@app.on_event("startup")は使わない
- SentenceTransformerモデルはHybridSearcher外で明示的にロードし、searcher.modelに保持する。これはquery.pyの既存パターンに合わせる（HybridSearcherの実装次第で調整必要）

## テスト手順

### 1. サーバー起動
```bash
cd ~/product/Tesla_rag
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
※ `--reload`は開発時のみ。起動ログで「✅ ロード完了: ChromaDB 112 チャンク」が出ることを確認。

### 2. ヘルスチェック
```bash
curl http://localhost:8000/health | python -m json.tool
```
確認項目:
- `status: "healthy"`
- `chromadb_chunks: 112`
- `bm25_loaded: true`
- `reranker_loaded: true`

### 3. 各モードでクエリ実行
```bash
# CRAGモード（デフォルト）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Tesla total revenue in Q4 2025?"}' \
  | python -m json.tool

# Rerankモード
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Tesla total revenue in Q4 2025?", "search_mode": "rerank"}' \
  | python -m json.tool

# ハイブリッドモード
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Tesla total revenue in Q4 2025?", "search_mode": "hybrid"}' \
  | python -m json.tool

# ベクトルモード
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Tesla total revenue in Q4 2025?", "search_mode": "vector"}' \
  | python -m json.tool
```
確認項目:
- 全モードでanswerが返ること
- search_modeが指定通りであること
- cragモードでcrag_gradesとcrag_logが入っていること
- crag以外のモードでcrag_grades, crag_logが空リストであること
- processing_time_secが記録されていること

### 4. バリデーション確認
```bash
# 空の質問 → 422エラー
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": ""}' \
  | python -m json.tool

# 不正なモード → 422エラー
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "search_mode": "invalid"}' \
  | python -m json.tool
```

### 5. Swagger UI確認
ブラウザで `http://localhost:8000/docs` にアクセス:
- エンドポイント一覧が表示されること
- リクエスト/レスポンスのスキーマが表示されること
- Try it outで実際にリクエストが送れること

### 6. CLIとの回答一致確認
```bash
# 同じ質問をCLIとAPIで実行し、回答が同等であることを確認
python src/query.py --mode crag "What was Tesla total revenue in Q4 2025?"
# ↑の回答と POST /query の回答を比較（完全一致は不要、同じ数値を含むこと）
```

## 壊す実験

### 実験: モデル未ロード状態でのリクエスト
lifespan内のモデルロードをコメントアウトして起動し、`POST /query` が503を返すことを確認する。
→ ヘルスチェックとエラーハンドリングが正しく動作する証拠。

### 実験: 不正なtop_k値
```bash
# top_k=0 → 422バリデーションエラー
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 0}'

# top_k=100 → 422バリデーションエラー（上限20）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 100}'
```

## 成功基準
- `GET /health` が200でモデル状態を返すこと
- `POST /query` が全4モード（vector/hybrid/rerank/crag）で正しく回答を返すこと
- CRAGモードでcrag_grades, crag_retries, crag_logが正しく含まれること
- Pydanticバリデーションが効くこと（空文字、不正モード、top_k範囲外 → 422）
- Swagger UI（/docs）でスキーマが確認できること
- CLIの`query.py --mode crag`と同等の回答品質であること
- 既存のquery.py, evaluate.pyが引き続きCLIで動作すること（デグレ防止）

## 注意点
- api.pyのimportパス: `uvicorn src.api:app` で起動するため、src/内の他モジュールのimportは `from config import ...` の形（src.config ではない）。sys.pathの調整が必要な場合はapi.py冒頭で対応
- HybridSearcherの初期化方法: 既存のquery.pyでどうモデルを渡しているか確認し、api.pyのlifespan内でも同じパターンにすること。`searcher.model`で参照できない場合はlifespanのスコープ内でmodelを保持する方法を検討
- ask_claude()の重複: query.pyとapi.pyに同じロジックが存在する。V6時点では許容。気になる場合はsrc/llm.pyに切り出してもよいが、スコープ外
- 環境変数: .envのANTHROPIC_API_KEYはFastAPIサーバーのプロセスでも必要。uvicorn起動前に`source .env`するか、python-dotenvで読み込むこと（config.pyが既に対応しているはず）
- `--reload`モード: モデルロードが毎回走るので開発時はやや重い。本番では`--reload`を外す