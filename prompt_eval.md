評価パイプラインを作って。

## ファイル: src/evaluate.py

## 概要
10問の質問+正解のセットでRAGの回答精度を自動測定するスクリプト。
2つの評価方式を実装する。

## 評価方式1: 正解一致チェック
- 質問に対するRAGの回答に、正解の数値/キーワードが含まれているかチェック
- 含まれていれば正解、なければ不正解
- 正答率を算出

## 評価方式2: LLM-as-a-Judge
- RAGの回答を別のClaude API呼び出しで採点
- 採点基準（各1-5点）:
  - Relevancy: 質問に関連する回答か
  - Faithfulness: 取得したチャンクの内容に忠実か
  - Completeness: 情報が十分か
- 各質問のスコアと、全体の平均スコアを出力

## 評価データ（10問）
questions = [
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

## 出力形式
各質問ごとに:
- 質問文
- 正解
- RAGの回答（最初の200文字）
- 正解一致: ✅ or ❌
- LLM-as-a-Judge: Relevancy/Faithfulness/Completeness のスコア

最後にサマリー:
- 正答率: X/10
- 平均スコア: Relevancy X.X / Faithfulness X.X / Completeness X.X

## 技術的な要件
- query.pyの検索+回答生成ロジックを関数として呼び出す（importできるようにquery.pyも必要なら修正して）
- LLM-as-a-Judgeの評価プロンプトは英語で書く
- 結果をresults/フォルダにタイムスタンプ付きJSONで保存
- LangChainは使わない
- config.pyの設定値を使う