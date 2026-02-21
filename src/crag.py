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

        for valid in ["INCORRECT", "AMBIGUOUS", "CORRECT"]:
            if valid in grade:
                return valid
        return "INCORRECT"

    def rewrite_query(self, question: str, previous_query: str, context: str) -> str:
        """クエリをリライトする.

        Args:
            question: ユーザーの元の質問
            previous_query: 前回使った検索クエリ
            context: 前回の検索結果のコンテキスト（不十分だったもの）

        Returns:
            リライトされた検索クエリ
        """
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
