# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tesla IR (Investor Relations) レポートを分析するRAGアプリケーション。Q3/Q4 2025のTesla決算PDFをベクトルDBに格納し、自然言語で質問→回答を生成する。

## Commands

```bash
# 依存関係インストール
pip install -r requirements.txt

# PDF取り込み（初回のみ。~/Downloads からPDFをdata/にコピーし、ChroamDBへ格納）
python src/ingest.py          # 既存DB検出時はスキップ
python src/ingest.py --force  # 既存DBを削除して再構築

# 対話型質問応答
python src/query.py

# RAG評価パイプライン（10問の自動評価）
python src/evaluate.py
```

全スクリプトはプロジェクトルートから `python src/<script>.py` で実行する（src/ 内の相対importのため）。

## Architecture

**パイプライン**: PDF → PyMuPDF抽出 → RecursiveCharacterTextSplitter → sentence-transformers Embedding → ChromaDB → Claude API回答生成

```
src/config.py      # 全設定値の一元管理（パス、チャンク設定、モデル名、TOP_K等）
src/ingest.py      # データ取り込み: PDF→チャンク→Embedding→ChromaDB保存
src/query.py       # 検索+回答: ChromaDB検索→コンテキスト構築→Claude API呼び出し
src/evaluate.py    # 評価: 正解一致チェック + LLM-as-a-Judge（Relevancy/Faithfulness/Completeness）
src/search_chunks.py  # デバッグ用: ChromaDB内チャンクのキーワード検索
```

- **ベクトルDB**: ChromaDB永続化モード (`chroma_db/`), コレクション名 `tesla_ir`
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Claude Sonnet (`anthropic` SDK直接使用、LangChain不使用)
- **APIキー**: 環境変数 `ANTHROPIC_API_KEY`

## Key Design Decisions

- LangChainは意図的に不使用（チャンキングの`langchain-text-splitters`のみ例外）。anthropic SDKを直接使用して内部動作を理解する方針。
- `query.py` の関数（`get_collection`, `search`, `build_context`, `ask_claude`）は独立関数として設計されており、`evaluate.py` 等から直接importして使用する。
- チャンクメタデータ: `source`(ファイル名), `page`, `quarter`(Q3/Q4), `year`, `chunk_index`, `section`
- 評価結果は `results/eval_YYYYMMDD_HHMMSS.json` に自動保存される。

## Current Phase: V2 - ハイブリッド検索

### V1評価結果（ベースライン）
- 正答率: 1/10（10%）
- Relevancy: 4.7/5.0
- Faithfulness: 5.0/5.0
- Completeness: 3.3/5.0
- ボトルネック: ベクトル検索精度。revenueチャンク21/144件あるがtop_k=5に0件

### V2目標
- BM25 + Vector のハイブリッド検索（RRF統合）
- 正答率4/10以上、Completeness 4.0以上、Faithfulness 5.0維持

### 開発ルール
- 変更前後で同じ10問評価を必ず回す
- インタラクティブスクリプトはClaude Code外のターミナルで実行

