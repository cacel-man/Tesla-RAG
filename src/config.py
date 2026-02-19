"""Tesla IR RAG アプリケーションの設定値."""

import os
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"

# --- コピー元PDF ---
SOURCE_PDFS: list[Path] = [
    Path.home() / "Downloads" / "TSLA-Q4-2025-Update.pdf",
    Path.home() / "Downloads" / "TSLA-Q3-2025-Update.pdf",
]

# --- チャンキング設定 ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# --- Embedding設定 ---
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- ChromaDB設定 ---
COLLECTION_NAME: str = "tesla_ir"

# --- 検索設定 ---
TOP_K: int = 5

# --- V2: Hybrid Search ---
BM25_TOP_K_MULTIPLIER: int = 2  # BM25/Vectorそれぞれtop_k * この値を取得
RRF_K: int = 60                  # RRF定数
SEARCH_MODE: str = "hybrid"      # "vector" or "hybrid"

# --- V3: テーブル対応チャンキング ---
TABLE_CHUNK_SIZE: int = 2500
TABLE_CHUNK_OVERLAP: int = 500
TABLE_BOOST_FACTOR: float = 1.2  # テーブルチャンクのRRFスコアブースト

# --- LLM設定 ---
CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# --- システムプロンプト ---
SYSTEM_PROMPT: str = (
    "あなたはTeslaのIRレポート分析アシスタントです。"
    "提供されたコンテキストのみに基づいて回答してください。"
    "情報が不足している場合はその旨を伝えてください。"
)
