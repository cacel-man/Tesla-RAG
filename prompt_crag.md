# V5: Active RAG（CRAG: Corrective RAG）

## 目的
V4までは検索結果の品質に関わらず、取得したチャンクをそのままLLMに渡して回答していた。検索結果が不十分な場合（例: Q6 Free cash flowでRECONCILIATIONテーブルの構造的問題により正しいチャンクが取れない）でも、LLMは「与えられた情報で無理やり答える」しかなかった。V5では検索結果の品質をLLMが判定し、不十分ならクエリをリライトして再検索する自己修正ループを導入する。

## 前提
- 既存コード: src/ingest.py, src/config.py, src/hybrid_search.py, src/reranker.py, src/query.py, src/evaluate.py
- ChromaDB: 112チャンク（V3テーブル対応チャンキング済み）
- 検索: ハイブリッド検索 + Reranker（V4パイプライン）
- LangChain不使用方針を継続
- ingest.py, hybrid_search.py, reranker.pyの変更は不要

## アーキテクチャ上の位置

```
クエリ
  ├── ベクトル検索 → top_k*2件
  ├── BM25検索 → top_k*2件
  └── RRF統合 → RERANK_CANDIDATES件
        └── Cross-Encoder Reranker → top_k件
              └── ★ LLM品質判定（CORRECT / AMBIGUOUS / INCORRECT）
                    ├── CORRECT → そのまま回答生成
                    └── AMBIGUOUS/INCORRECT → クエリリライト → 再検索（ループ先頭に戻る）
                          └── MAX_RETRY回まで繰り返し → 回答生成
```

CRAGは検索とLLMの間に「品質ゲート」を挟む。検索結果で質問に答えられるかをLLMが判定し、不十分ならクエリを書き換えて再検索する。人間がデバッグ時にやる「このチャンク、質問に関係ある？」をLLMに自動化させるイメージ。

## 実装内容

### 1. src/crag.py を新規作成

```python
"""Corrective RAG: 検索結果の品質判定とクエリリライトによる自己修正ループ."""

import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_RETRY

GRADING_PROMPT = """あなたは検索結果の品質を判定するグレーダーです。

## タスク
ユーザーの質問に対して、検索結果に回答に必要な情報が含まれているかを判定してください。

## 判定基準
- CORRECT: 検索結果に質問の回答に必要な具体的な数値・事実が含まれている
- AMBIGUOUS: 関連する情報は一部あるが、正確に回答するには不十分
- INCORRECT: 検索結果が質問と無関係、または回答に必要な情報がまったくない

## 出力形式
1行目に判定結果（CORRECT, AMBIGUOUS, INCORRECTのいずれか）のみを出力してください。
"""

REWRITE_PROMPT = """あなたは検索クエリの最適化を行うアシスタントです。

## タスク
元の質問に対する検索結果が不十分でした。より良い検索結果を得るために、検索クエリを書き換えてください。

## ルール
- 元の質問の意図を変えないこと
- より具体的なキーワード（数値、年度、テーブル名など）を含めること
- 英語で出力すること（検索対象が英語のため）
- クエリのみを1行で出力すること（説明不要）

## 元の質問
{question}

## 前回の検索クエリ
{previous_query}

## 検索結果の要約（不十分だった内容）
{context_summary}
"""


class CRAGProcessor:
    def __init__(self):
        """CRAG用のClaudeクライアントを初期化する."""
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def grade_results(self, question: str, context: str) -> str:
        """検索結果の品質を判定する.

        Args:
            question: ユーザーの元の質問
            context: 検索結果から構築されたコンテキスト文字列

        Returns:
            "CORRECT", "AMBIGUOUS", "INCORRECT" のいずれか
        """
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=50,
            system=GRADING_PROMPT,
            messages=[{
                "role": "user",
                "content": f"## 質問\n{question}\n\n## 検索結果\n{context}"
            }],
        )
        grade = response.content[0].text.strip().upper()

        # 安全なパース: 先頭の判定語を抽出
        for valid in ["CORRECT", "AMBIGUOUS", "INCORRECT"]:
            if valid in grade:
                return valid
        return "INCORRECT"  # パース失敗時は安全側に倒す

    def rewrite_query(self, question: str, previous_query: str, context: str) -> str:
        """クエリをリライトする.

        Args:
            question: ユーザーの元の質問
            previous_query: 前回使った検索クエリ
            context: 前回の検索結果のコンテキスト（不十分だったもの）

        Returns:
            リライトされた検索クエリ
        """
        # コンテキストが長すぎる場合は先頭500文字に要約
        context_summary = context[:500] + "..." if len(context) > 500 else context

        prompt = REWRITE_PROMPT.format(
            question=question,
            previous_query=previous_query,
            context_summary=context_summary,
        )

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
```

### 2. src/config.py に追加
```python
# --- V5: CRAG ---
MAX_RETRY: int = 1  # クエリリライト→再検索の最大回数（実験で変更可能）
SEARCH_MODE: str = "crag"  # デフォルトを更新: "vector" / "hybrid" / "rerank" / "crag"
```

### 3. src/query.py を修正

#### --mode crag の追加:
- "crag"モード: V4パイプライン（hybrid+rerank） + CRAG品質判定ループ
- process_query内にCRAGループを追加

```python
# cragモード時の初期化
crag_processor = None
if mode == "crag":
    print("CRAGプロセッサを初期化中...")
    crag_processor = CRAGProcessor()

def process_query(query: str) -> None:
    if mode == "crag" and searcher is not None and reranker is not None and crag_processor is not None:
        current_query = query
        retry_count = 0
        crag_log = []  # CRAGの判定ログ

        while True:
            # 検索実行（hybrid + rerank）
            hybrid_results = searcher.search(current_query, model, top_k=TOP_K, reranker=reranker)
            context, references = build_context_from_hybrid(hybrid_results)

            # LLM品質判定
            grade = crag_processor.grade_results(query, context)  # 常に元の質問で判定
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

            # クエリリライト→再検索
            print(f"   → クエリリライト中...")
            current_query = crag_processor.rewrite_query(query, current_query, context)
            print(f"   → 新クエリ: {current_query}")
            retry_count += 1

        # チャンク表示（既存のrerank表示と同じ + CRAGログ追加）
        # ... 既存のチャンク表示コード ...

        # CRAGログ表示
        print(f"\n📊 CRAGログ:")
        for log in crag_log:
            print(f"   retry={log['retry']}: [{log['grade']}] {log['query']}")

        # 回答生成
        answer = ask_claude(query, context)  # 常に元の質問で回答生成
        # ...

    elif mode in ("hybrid", "rerank") and searcher is not None:
        # 既存のV3/V4処理（変更なし）
        ...
```

**重要な設計判断:**
- `grade_results()`には常に元の`query`を渡す（リライト後のクエリではなく）。判定基準は「元の質問に答えられるか」
- `ask_claude()`にも常に元の`query`を渡す。リライトはあくまで検索用
- MAX_RETRY到達時は最後の検索結果で回答を生成する（エラーにしない）

### 4. src/evaluate.py を修正
- --mode crag に対応
- 結果JSONに追加フィールド:
  - `search_mode: "crag"`
  - `max_retry: 1`（設定値を記録）
  - 各問題に `crag_retries: int`（実際に何回リトライしたか）
  - 各問題に `crag_grades: list[str]`（各ステップの判定結果）
- V4結果との比較表示

### 5. 変更しないもの
- src/ingest.py — 変更不要
- src/hybrid_search.py — 変更不要
- src/reranker.py — 変更不要
- chroma_db/ — 再構築不要

## テスト手順
1. `python src/query.py --mode crag "What was Tesla's total revenue in Q4 2025?"` で動作確認
2. 以下を確認:
   - CRAG判定が表示されること（CORRECT / AMBIGUOUS / INCORRECT）
   - CORRECT判定時: リトライなしで回答が生成されること
   - AMBIGUOUS/INCORRECT判定時: クエリリライト→再検索が実行されること
   - CRAGログが表示されること
3. `python src/query.py --mode crag "What was Tesla's free cash flow in 2024?"` でQ6（V4で不正解）をテスト
   - クエリリライトが発生するか確認
   - リライト後の検索で正しいチャンクが取得されるか確認
4. `python src/query.py --mode rerank "What was Tesla's total revenue in Q4 2025?"` でV4互換動作を確認
5. `python src/evaluate.py --mode crag` で10問評価

## 壊す実験（MAX_RETRY比較）

### 実験の目的
MAX_RETRYの最適値を定量的に決定する。

### 実験手順
```bash
# 実験1: MAX_RETRY=0（リライトなし、品質判定のみ → V4相当のベースライン）
# config.pyのMAX_RETRYを0に変更
python src/evaluate.py --mode crag
# → results/に保存

# 実験2: MAX_RETRY=1
# config.pyのMAX_RETRYを1に変更
python src/evaluate.py --mode crag
# → results/に保存

# 実験3: MAX_RETRY=2
# config.pyのMAX_RETRYを2に変更
python src/evaluate.py --mode crag
# → results/に保存
```

### 記録する指標
| 指標 | MAX_RETRY=0 | MAX_RETRY=1 | MAX_RETRY=2 |
|------|-------------|-------------|-------------|
| 正答率 | /10 | /10 | /10 |
| Relevancy | | | |
| Faithfulness | | | |
| Completeness | | | |
| 平均リトライ回数 | 0 | | |
| LLM API呼び出し回数 | | | |

### 予想
- MAX_RETRY=0→1で改善がある（特にQ6）
- MAX_RETRY=1→2の改善は小さい（収穫逓減）
- LLM API呼び出し回数はMAX_RETRYに比例して増加（コスト増）
- 最適解: MAX_RETRY=1（コスト対効果のバランス）

## 成功基準
- CRAG品質判定が正しく動作すること（全問CORRECTにならない、かつ全問INCORRECTにもならない）
- 正答率: 8/10以上を維持（V4から下がらない）
- MAX_RETRY実験の結果が記録できること
- V4で正解だった問題がV5で不正解にならないこと（デグレ防止）
- --mode rerank でV4互換動作が維持されること

## 注意点
- CRAGはLLM APIを追加で呼び出す（品質判定 + クエリリライト）。MAX_RETRY=1で最大3回のAPI呼び出し（判定→リライト→判定）+ 最後の回答生成1回
- grade_results()のパース失敗時は"INCORRECT"に倒す（再検索する方が安全）
- クエリリライトは英語で出力させる（検索対象のPDFが英語のため）
- crag.pyは独立モジュール。--mode rerankならCRAGを完全にスキップ
- evaluate.pyでMAX_RETRYの値を結果JSONに記録すること（実験の再現性）