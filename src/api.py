"""FastAPI server for Tesla-RAG pipeline."""

import time
from contextlib import asynccontextmanager
from enum import Enum

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    EMBEDDING_MODEL,
    MAX_RETRY,
    SYSTEM_PROMPT,
    TOP_K,
)
from crag import CRAGProcessor
from hybrid_search import HybridSearcher
from query import get_collection, search, build_context
from reranker import Reranker


# --- Pydanticモデル ---


class SearchMode(str, Enum):
    vector = "vector"
    hybrid = "hybrid"
    rerank = "rerank"
    crag = "crag"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="質問文")
    search_mode: SearchMode = Field(
        default=SearchMode.crag, description="検索モード"
    )
    top_k: int = Field(default=TOP_K, ge=1, le=20, description="取得チャンク数")


class ChunkInfo(BaseModel):
    rank: int
    content_preview: str
    content_type: str
    score: float | None = None


class CRAGLog(BaseModel):
    retry: int
    query: str
    grade: str


class QueryResponse(BaseModel):
    answer: str
    search_mode: str
    chunks_used: int
    crag_grades: list[str] = []
    crag_retries: int = 0
    crag_log: list[CRAGLog] = []
    chunks: list[ChunkInfo] = []
    processing_time_sec: float


class HealthResponse(BaseModel):
    status: str
    chromadb_chunks: int
    bm25_loaded: bool
    reranker_loaded: bool
    model: str


# --- モデルのグローバル変数（lifespanで初期化） ---
embedding_model: SentenceTransformer | None = None
collection = None
searcher: HybridSearcher | None = None
reranker_model: Reranker | None = None
crag_processor: CRAGProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """サーバー起動時にモデルをロードする."""
    global embedding_model, collection, searcher, reranker_model, crag_processor
    print("モデルをロード中...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("ChromaDBに接続中...")
    collection = get_collection()
    chunk_count = collection.count()

    print("BM25インデックスを構築中...")
    searcher = HybridSearcher(collection)

    print("Rerankerモデルをロード中...")
    reranker_model = Reranker()

    print("CRAGプロセッサを初期化中...")
    crag_processor = CRAGProcessor()

    print(f"ロード完了: ChromaDB {chunk_count} チャンク")
    yield
    print("サーバーシャットダウン")


app = FastAPI(
    title="Tesla-RAG API",
    description="Tesla 10-K filing RAG pipeline with hybrid search, reranking, and CRAG",
    version="0.6.0",
    lifespan=lifespan,
)


# --- ヘルパー関数 ---


def build_chunks_info(hybrid_results: list[dict]) -> list[ChunkInfo]:
    """ハイブリッド検索結果からChunkInfoリストを構築する."""
    chunks_info = []
    for i, r in enumerate(hybrid_results):
        content = r["content"]
        meta = r.get("metadata", {})
        chunks_info.append(
            ChunkInfo(
                rank=i + 1,
                content_preview=content[:200],
                content_type=meta.get("content_type", "text"),
                score=r.get("score"),
            )
        )
    return chunks_info


def build_chunks_info_from_vector(results: dict) -> list[ChunkInfo]:
    """ベクトル検索結果からChunkInfoリストを構築する."""
    chunks_info = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        chunks_info.append(
            ChunkInfo(
                rank=i + 1,
                content_preview=doc[:200],
                content_type=meta.get("content_type", "text"),
                score=None,
            )
        )
    return chunks_info


def generate_answer(question: str, context: str) -> str:
    """Claudeに回答を生成させる."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = (
        f"以下のコンテキストに基づいて質問に回答してください。\n\n"
        f"## コンテキスト\n{context}\n\n"
        f"## 質問\n{question}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# --- エンドポイント ---


@app.get("/health", response_model=HealthResponse)
def health_check():
    """ヘルスチェック: モデルのロード状態を返す."""
    return HealthResponse(
        status="healthy" if searcher is not None else "unhealthy",
        chromadb_chunks=collection.count() if collection else 0,
        bm25_loaded=searcher is not None,
        reranker_loaded=reranker_model is not None,
        model=CLAUDE_MODEL,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """RAGパイプラインを実行して回答を返す."""
    if searcher is None or embedding_model is None or collection is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    start_time = time.time()
    mode = req.search_mode
    question = req.question
    top_k = req.top_k

    crag_grades: list[str] = []
    crag_retries: int = 0
    crag_log: list[CRAGLog] = []
    chunks_info: list[ChunkInfo] = []
    context: str = ""

    if mode == SearchMode.vector:
        results = search(question, collection, embedding_model, top_k=top_k)
        context, _ = build_context(results)
        chunks_info = build_chunks_info_from_vector(results)

    elif mode == SearchMode.hybrid:
        hybrid_results = searcher.search(question, embedding_model, top_k=top_k)
        from query import build_context_from_hybrid

        context, _ = build_context_from_hybrid(hybrid_results)
        chunks_info = build_chunks_info(hybrid_results)

    elif mode == SearchMode.rerank:
        hybrid_results = searcher.search(
            question, embedding_model, top_k=top_k, reranker=reranker_model
        )
        from query import build_context_from_hybrid

        context, _ = build_context_from_hybrid(hybrid_results)
        chunks_info = build_chunks_info(hybrid_results)

    elif mode == SearchMode.crag:
        current_query = question
        retry_count = 0

        while True:
            hybrid_results = searcher.search(
                current_query,
                embedding_model,
                top_k=top_k,
                reranker=reranker_model,
            )
            from query import build_context_from_hybrid

            context, _ = build_context_from_hybrid(hybrid_results)
            chunks_info = build_chunks_info(hybrid_results)

            grade = crag_processor.grade_results(question, context)
            crag_grades.append(grade)
            crag_log.append(
                CRAGLog(retry=retry_count, query=current_query, grade=grade)
            )

            if grade == "CORRECT" or retry_count >= MAX_RETRY:
                break

            current_query = crag_processor.rewrite_query(
                question, current_query, context
            )
            retry_count += 1

        crag_retries = retry_count

    answer = generate_answer(question, context)
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
