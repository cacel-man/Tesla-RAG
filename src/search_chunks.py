"""ChromaDBの全チャンクから指定キーワードを含むものを表示する."""

import chromadb
from config import CHROMA_DIR, COLLECTION_NAME


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    # 全チャンクを取得
    all_data = collection.get(include=["documents", "metadatas"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]

    import sys
    keyword = sys.argv[1] if len(sys.argv) > 1 else "revenue"
    matches = []

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        if keyword.lower() in doc.lower():
            matches.append((i, doc, meta))

    print(f'全 {len(documents)} チャンク中、"{keyword}" を含むチャンク: {len(matches)} 件\n')
    print("=" * 60)

    for rank, (idx, doc, meta) in enumerate(matches, 1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        quarter = meta.get("quarter", "?")
        print(f"\n--- [{rank}] チャンク番号: {idx} (ID: chunk_{idx}) ---")
        print(f"  source : {source}")
        print(f"  page   : {page}")
        print(f"  quarter: {quarter}")
        print(f"  テキスト: {doc[:300]}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
