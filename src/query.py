"""検索 + Claude APIで回答生成."""

import sys

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    ANTHROPIC_API_KEY,
    CHROMA_DIR,
    CLAUDE_MODEL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_RETRY,
    SEARCH_MODE,
    SYSTEM_PROMPT,
    TOP_K,
)
from hybrid_search import HybridSearcher
from reranker import Reranker
from crag import CRAGProcessor


def get_collection() -> chromadb.Collection:
    """ChromaDBのコレクションを取得する."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def search(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> dict:
    """クエリに関連するチャンクをChromaDBから検索する（V1ベクトル検索）."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results


def hybrid_search(
    query: str,
    searcher: HybridSearcher,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """ハイブリッド検索を実行する（V2）."""
    return searcher.search(query, model, top_k=top_k)


def build_context(results: dict) -> tuple[str, list[dict]]:
    """検索結果からClaudeに渡すコンテキスト文字列と参照情報を構築する."""
    context_parts: list[str] = []
    references: list[dict] = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        quarter = meta.get("quarter", "?")
        section = meta.get("section", "")

        header = f"[参照{i+1}] {source} p.{page} ({quarter})"
        if section:
            header += f" - {section}"

        context_parts.append(f"{header}\n{doc}")
        references.append({"source": source, "page": page, "quarter": quarter})

    return "\n\n---\n\n".join(context_parts), references


def build_context_from_hybrid(results: list[dict]) -> tuple[str, list[dict]]:
    """ハイブリッド検索結果からコンテキスト文字列と参照情報を構築する."""
    context_parts: list[str] = []
    references: list[dict] = []

    for i, result in enumerate(results):
        meta = result["metadata"]
        doc = result["content"]
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        quarter = meta.get("quarter", "?")
        section = meta.get("section", "")

        header = f"[参照{i+1}] {source} p.{page} ({quarter})"
        if section:
            header += f" - {section}"

        context_parts.append(f"{header}\n{doc}")
        references.append({"source": source, "page": page, "quarter": quarter})

    return "\n\n---\n\n".join(context_parts), references


def ask_claude(query: str, context: str) -> str:
    """コンテキストと質問をClaudeに渡して回答を生成する."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = (
        f"以下のコンテキストに基づいて質問に回答してください。\n\n"
        f"## コンテキスト\n{context}\n\n"
        f"## 質問\n{query}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def display_references(references: list[dict]) -> None:
    """参照元情報を表示する."""
    print("\n📎 参照元:")
    seen: set[str] = set()
    for ref in references:
        key = f"{ref['source']} p.{ref['page']}"
        if key not in seen:
            seen.add(key)
            print(f"  - {ref['source']} (p.{ref['page']}, {ref['quarter']})")


def main() -> None:
    """インタラクティブな質問応答ループ."""
    if not ANTHROPIC_API_KEY:
        print("[ERROR] ANTHROPIC_API_KEY 環境変数を設定してください")
        return

    # --mode オプション解析
    mode = SEARCH_MODE
    args = sys.argv[1:]
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args):
            mode = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    # 位置引数があればワンショットモード
    oneshot_query = " ".join(args) if args else None

    print("Embeddingモデルをロード中...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("ChromaDBに接続中...")
    collection = get_collection()
    count = collection.count()
    print(f"準備完了! ({count}チャンクをインデックス済み)")
    print(f"検索モード: {mode}\n")

    # hybrid/rerank/cragモード時はHybridSearcherを初期化
    searcher = None
    reranker = None
    crag_processor = None
    if mode in ("hybrid", "rerank", "crag"):
        print("BM25インデックスを構築中...")
        searcher = HybridSearcher(collection)
        if mode in ("rerank", "crag"):
            print("Rerankerモデルをロード中...")
            reranker = Reranker()
        if mode == "crag":
            print("CRAGプロセッサを初期化中...")
            crag_processor = CRAGProcessor()

    def _display_hybrid_chunks(hybrid_results: list[dict], label: str) -> None:
        """ハイブリッド/rerank/crag検索結果のチャンクを表示する."""
        print("\n" + "=" * 60)
        print(f"検索で取得したチャンク ({label})")
        print("=" * 60)
        for i, result in enumerate(hybrid_results):
            meta = result["metadata"]
            print(f"\n--- チャンク {i+1}/{len(hybrid_results)} ---")
            print(f"  source : {meta.get('source', 'unknown')}")
            print(f"  page   : {meta.get('page', '?')}")
            print(f"  quarter: {meta.get('quarter', '?')}")
            print(f"  sources: {', '.join(result['sources'])}")
            print(f"  score  : {result['score']:.4f}")
            if "rerank_score" in result:
                print(f"  rerank : {result['rerank_score']:.4f}")
                print(f"  元順位  : {result['original_rank']}")
            print(f"  テキスト: {result['content'][:200]}...")
        print("\n" + "=" * 60)

    def process_query(query: str) -> None:
        if mode == "crag" and searcher is not None and reranker is not None and crag_processor is not None:
            current_query = query
            retry_count = 0
            crag_log = []

            while True:
                hybrid_results = searcher.search(
                    current_query, model, top_k=TOP_K, reranker=reranker,
                )
                context, references = build_context_from_hybrid(hybrid_results)

                grade = crag_processor.grade_results(query, context)
                crag_log.append({
                    "retry": retry_count,
                    "query": current_query,
                    "grade": grade,
                })

                print(f"\n🔍 CRAG判定 (retry={retry_count}): {grade}")
                print(f"   検索クエリ: {current_query}")

                if grade == "CORRECT" or retry_count >= MAX_RETRY:
                    if retry_count >= MAX_RETRY and grade != "CORRECT":
                        print(f"   ⚠️ MAX_RETRY({MAX_RETRY})到達。現在の検索結果で回答生成。")
                    break

                print(f"   → クエリリライト中...")
                current_query = crag_processor.rewrite_query(query, current_query, context)
                print(f"   → 新クエリ: {current_query}")
                retry_count += 1

            _display_hybrid_chunks(hybrid_results, "crag")

            print(f"\n📊 CRAGログ:")
            for log in crag_log:
                print(f"   retry={log['retry']}: [{log['grade']}] {log['query']}")

            print("\n回答を生成中...\n")
            answer = ask_claude(query, context)
            print(f"{answer}")
            display_references(references)
            print()
            return

        elif mode in ("hybrid", "rerank") and searcher is not None:
            hybrid_results = searcher.search(
                query, model, top_k=TOP_K, reranker=reranker,
            )
            context, references = build_context_from_hybrid(hybrid_results)
            _display_hybrid_chunks(hybrid_results, mode)
        else:
            results = search(query, collection, model)
            context, references = build_context(results)

            # チャンク表示
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            print("\n" + "=" * 60)
            print("検索で取得したチャンク (vector)")
            print("=" * 60)
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                print(f"\n--- チャンク {i+1}/{len(documents)} ---")
                print(f"  source : {meta.get('source', 'unknown')}")
                print(f"  page   : {meta.get('page', '?')}")
                print(f"  quarter: {meta.get('quarter', '?')}")
                print(f"  テキスト: {doc[:200]}...")
            print("\n" + "=" * 60)

        # Claude APIで回答生成
        print("\n回答を生成中...\n")
        answer = ask_claude(query, context)

        print(f"{answer}")
        display_references(references)
        print()

    # ワンショットモード
    if oneshot_query:
        process_query(oneshot_query)
        return

    # インタラクティブループ
    print('質問を入力してください ("quit"で終了)\n')
    while True:
        query = input("質問> ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("終了します。")
            break
        process_query(query)


if __name__ == "__main__":
    main()
