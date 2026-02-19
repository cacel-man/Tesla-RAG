"""RAG評価パイプライン: 正解一致チェック + LLM-as-a-Judge."""

import json
import os
import re
import sys
from datetime import datetime

import anthropic
from sentence_transformers import SentenceTransformer

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, EMBEDDING_MODEL, SEARCH_MODE
from query import (
    get_collection,
    search,
    build_context,
    build_context_from_hybrid,
    ask_claude,
)
from hybrid_search import HybridSearcher

# --- 評価データ（10問） ---
EVAL_DATA = [
    {"q": "What was Tesla's total revenues in 2025?", "answer": "$94,827M"},
    {"q": "What was Tesla's operating margin in 2025?", "answer": "4.6%"},
    {"q": "What was Tesla's GAAP net income in 2025?", "answer": "$3,794M"},
    {"q": "What was Tesla's Energy generation and storage revenue in Q4-2025?", "answer": "$3,837M"},
    {"q": "What was Tesla's total automotive revenues in Q4-2025?", "answer": "$17,693M"},
    {"q": "What was Tesla's free cash flow in Q3-2025?", "answer": "$3,990M"},
    {"q": "What was Tesla's Adjusted EBITDA in 2025?", "answer": "$14,596M"},
    {"q": "How did Tesla's total revenues change from Q3-2025 to Q4-2025?", "answer": "$28,095M to $24,901M, decreased"},
    {"q": "What was the YoY growth rate of Tesla's Energy generation and storage revenue in 2025?", "answer": "27%"},
    {"q": "What was Tesla's GAAP EPS diluted in Q4-2025?", "answer": "$0.24"},
]

JUDGE_PROMPT = """\
You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.
Given a question, the retrieved context, and the RAG system's answer, score the answer on three criteria.

Each criterion is scored 1-5:
- **Relevancy**: How relevant is the answer to the question? (1=completely irrelevant, 5=directly answers the question)
- **Faithfulness**: Is the answer faithful to the retrieved context? Does it avoid hallucination? (1=fabricated, 5=fully grounded in context)
- **Completeness**: Does the answer provide sufficient information? (1=missing key info, 5=comprehensive)

Respond ONLY with a JSON object in this exact format, no other text:
{{"relevancy": <int>, "faithfulness": <int>, "completeness": <int>}}

Question: {question}

Retrieved Context:
{context}

RAG Answer:
{answer}
"""


def run_rag(
    query: str,
    collection,
    model: SentenceTransformer,
    search_mode: str = "vector",
    searcher: HybridSearcher | None = None,
) -> tuple[str, str, list[dict]]:
    """RAGパイプラインを実行し、(回答, コンテキスト, 参照情報)を返す."""
    if search_mode == "hybrid" and searcher is not None:
        hybrid_results = searcher.search(query, model)
        context, references = build_context_from_hybrid(hybrid_results)
    else:
        results = search(query, collection, model)
        context, references = build_context(results)

    answer = ask_claude(query, context)
    return answer, context, references


def normalize_text(text: str) -> str:
    """比較用にテキストを正規化する（$, カンマ, M, 空白を除去し小文字化）."""
    text = text.lower()
    text = text.replace("$", "").replace(",", "").replace("m", "").replace(" ", "")
    return text


def check_exact_match(answer: str, expected: str) -> bool:
    """正解キーワードがRAG回答に含まれるかチェック.

    表記揺れを考慮:
    - "$94,827M" → "94,827" "94827" のバリエーションをチェック
    - "4.6%" → "4.6%" をチェック
    - カンマ区切り回答 ("$28,095M to $24,901M, decreased") は各パートを個別チェック
    """
    answer_lower = answer.lower()
    answer_normalized = normalize_text(answer)

    # カンマ区切りの複数キーワードを分割
    parts = [p.strip() for p in expected.split(",")]

    for part in parts:
        # そのまま含まれるかチェック
        if part.lower() in answer_lower:
            continue

        # $やMを除去したバージョンでチェック
        stripped = part.replace("$", "").replace("M", "").strip()
        if stripped.lower() in answer_lower:
            continue

        # カンマも除去したバージョンでチェック
        no_comma = stripped.replace(",", "")
        if no_comma.lower() in answer_lower:
            continue

        # 正規化済み同士で比較
        if normalize_text(part) in answer_normalized:
            continue

        # %の数値部分を抽出してチェック
        pct_match = re.search(r"([\d.]+)%", part)
        if pct_match and pct_match.group(0) in answer:
            continue

        return False

    return True


def judge_with_llm(query: str, answer: str, context: str) -> dict:
    """Claude APIで回答を採点する. Relevancy/Faithfulness/Completenessを各1-5で返す."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = JUDGE_PROMPT.format(question=query, context=context, answer=answer)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text.strip()

    # JSONをパース（コードブロックで囲まれている場合に対応）
    json_match = re.search(r"\{[^}]+\}", response_text)
    if json_match:
        scores = json.loads(json_match.group())
    else:
        scores = json.loads(response_text)

    return {
        "relevancy": int(scores.get("relevancy", 0)),
        "faithfulness": int(scores.get("faithfulness", 0)),
        "completeness": int(scores.get("completeness", 0)),
    }


def save_results(results: list[dict], summary: dict) -> str:
    """結果をresults/フォルダにタイムスタンプ付きJSONで保存する."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": summary,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return filepath


def main() -> None:
    """評価パイプラインのメイン処理."""
    if not ANTHROPIC_API_KEY:
        print("[ERROR] ANTHROPIC_API_KEY 環境変数を設定してください")
        sys.exit(1)

    # --mode オプション解析
    search_mode = SEARCH_MODE
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            search_mode = sys.argv[idx + 1]

    print("=" * 60)
    print("RAG 評価パイプライン")
    print(f"検索モード: {search_mode}")
    print("=" * 60)

    print("\nEmbeddingモデルをロード中...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("ChromaDBに接続中...")
    collection = get_collection()
    print(f"準備完了! ({collection.count()}チャンクをインデックス済み)")

    # hybridモード時はHybridSearcherを初期化
    searcher = None
    if search_mode == "hybrid":
        print("BM25インデックスを構築中...")
        searcher = HybridSearcher(collection)

    print()

    results = []
    exact_match_count = 0
    total_relevancy = 0
    total_faithfulness = 0
    total_completeness = 0

    for i, item in enumerate(EVAL_DATA):
        query = item["q"]
        expected = item["answer"]

        print(f"--- 質問 {i + 1}/{len(EVAL_DATA)} ---")
        print(f"Q: {query}")
        print(f"正解: {expected}")

        # RAG実行
        print("  RAG実行中...")
        answer, context, references = run_rag(
            query, collection, model,
            search_mode=search_mode,
            searcher=searcher,
        )

        # 正解一致チェック
        match = check_exact_match(answer, expected)
        if match:
            exact_match_count += 1

        # LLM-as-a-Judge
        print("  LLM-as-a-Judge 採点中...")
        scores = judge_with_llm(query, answer, context)

        total_relevancy += scores["relevancy"]
        total_faithfulness += scores["faithfulness"]
        total_completeness += scores["completeness"]

        # 結果表示
        match_icon = "\u2705" if match else "\u274c"
        print(f"  RAG回答: {answer[:200]}...")
        print(f"  正解一致: {match_icon}")
        print(
            f"  LLM Score: Relevancy={scores['relevancy']} "
            f"Faithfulness={scores['faithfulness']} "
            f"Completeness={scores['completeness']}"
        )
        print()

        results.append({
            "question": query,
            "expected": expected,
            "answer": answer,
            "exact_match": match,
            "scores": scores,
            "references": references,
        })

    # サマリー
    n = len(EVAL_DATA)
    summary = {
        "search_mode": search_mode,
        "exact_match": f"{exact_match_count}/{n}",
        "avg_relevancy": round(total_relevancy / n, 1),
        "avg_faithfulness": round(total_faithfulness / n, 1),
        "avg_completeness": round(total_completeness / n, 1),
    }

    print("=" * 60)
    print("サマリー")
    print("=" * 60)
    print(f"  検索モード: {search_mode}")
    print(f"  正答率: {summary['exact_match']}")
    print(
        f"  平均スコア: Relevancy {summary['avg_relevancy']} / "
        f"Faithfulness {summary['avg_faithfulness']} / "
        f"Completeness {summary['avg_completeness']}"
    )

    # V1結果との比較（hybridモード時）
    if search_mode == "hybrid":
        print("\n  --- V1ベースラインとの比較 ---")
        print(f"  正答率:      V1=1/10  → V2={summary['exact_match']}")
        print(f"  Relevancy:   V1=4.7   → V2={summary['avg_relevancy']}")
        print(f"  Faithfulness:V1=5.0   → V2={summary['avg_faithfulness']}")
        print(f"  Completeness:V1=3.3   → V2={summary['avg_completeness']}")

    # 結果保存
    filepath = save_results(results, summary)
    print(f"\n結果を保存しました: {filepath}")


if __name__ == "__main__":
    main()
