"""PDF読み込み + チャンキング + Embedding + ChromaDB保存."""

import re
import shutil
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    SOURCE_PDFS,
    TABLE_CHUNK_OVERLAP,
    TABLE_CHUNK_SIZE,
)


def copy_pdfs_to_data() -> list[Path]:
    """~/Downloads からPDFをdata/フォルダにコピーする."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in SOURCE_PDFS:
        dst = DATA_DIR / src.name
        if not src.exists():
            print(f"[WARN] ソースPDFが見つかりません: {src}")
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"[COPY] {src.name} -> data/")
        else:
            print(f"[SKIP] {src.name} は既にdata/に存在します")
        copied.append(dst)
    return copied


def extract_quarter(filename: str) -> str:
    """ファイル名からQ3/Q4等の四半期情報を抽出する."""
    match = re.search(r"Q(\d)", filename)
    return f"Q{match.group(1)}" if match else "Unknown"


def extract_section_header(text: str) -> str:
    """テキストの先頭行からセクション名を推定する."""
    lines = text.strip().split("\n")
    for line in lines[:3]:
        stripped = line.strip()
        if stripped and len(stripped) < 100 and not stripped[0].isdigit():
            return stripped
    return ""


def load_pdf(pdf_path: Path) -> list[dict]:
    """PDFをページ単位で読み込み、テキストとメタデータを返す."""
    doc = fitz.open(str(pdf_path))
    pages: list[dict] = []
    filename = pdf_path.name
    quarter = extract_quarter(filename)

    print(f"[READ] {filename}: {len(doc)}ページ")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,
                    "quarter": quarter,
                    "year": 2025,
                },
            })
    doc.close()
    return pages


def is_table_page(text: str) -> bool:
    """テーブルページかどうかを判定する."""
    table_indicators = [
        "F I N A N C I A L   S U M M A R Y",
        "F I N A N C I A L S U M M A R Y",
        "R E C O N C I L I A T I O N",
        "REVENUES",
        "In millions of USD",
    ]
    return any(indicator in text for indicator in table_indicators)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """ページテキストをチャンクに分割し、メタデータを付与する."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE,
        chunk_overlap=TABLE_CHUNK_OVERLAP,
    )

    chunks: list[dict] = []
    chunk_index = 0

    for page_data in pages:
        text = page_data["text"]
        metadata = page_data["metadata"]
        section = extract_section_header(text)

        is_table = is_table_page(text)
        splitter = table_splitter if is_table else text_splitter
        content_type = "financial_table" if is_table else "text"

        split_texts = splitter.split_text(text)
        for split_text in split_texts:
            chunks.append({
                "text": split_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "section": section,
                    "content_type": content_type,
                },
            })
            chunk_index += 1

    return chunks


def store_in_chromadb(chunks: list[dict], model: SentenceTransformer) -> None:
    """チャンクをEmbeddingしてChromaDBに保存する."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"[EMBED] {len(texts)}チャンクをEmbedding中...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"[DONE] ChromaDBに{len(texts)}チャンクを保存しました")


def ingest(force: bool = False) -> None:
    """メインのIngest処理. force=Trueで既存DBを再構築する."""
    # 既存DBチェック
    if CHROMA_DIR.exists() and not force:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                print(f"[SKIP] 既存DB検出 ({count}チャンク). --forceで再構築できます")
                return
        except Exception:
            pass

    # PDFコピー
    pdf_files = copy_pdfs_to_data()
    if not pdf_files:
        print("[ERROR] 処理するPDFがありません")
        return

    # PDF読み込み + チャンキング
    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        print(f"[CHUNK] {pdf_path.name}: {len(chunks)}チャンク生成")

    print(f"[TOTAL] 合計: {len(all_chunks)}チャンク")

    # Embeddingモデルの読み込み
    print(f"[MODEL] {EMBEDDING_MODEL} をロード中...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ChromaDBに保存
    if force and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("[CLEAN] 既存DBを削除しました")

    store_in_chromadb(all_chunks, model)


if __name__ == "__main__":
    import sys

    force_flag = "--force" in sys.argv
    ingest(force=force_flag)
