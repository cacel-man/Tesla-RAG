Tesla IRレポートを分析するRAGアプリのベースラインをゼロから構築して。

## プロジェクト構成
tesla-rag/
├── data/           # PDFファイル格納
├── src/
│   ├── ingest.py   # PDF読み込み + チャンキング + Embedding + ChromaDB保存
│   ├── query.py    # 検索 + Claude APIで回答生成
│   └── config.py   # 設定値まとめ
├── requirements.txt
└── README.md

## データ
- ~/Downloads/TSLA-Q4-2025-Update.pdf（5.6MB）
- ~/Downloads/TSLA-Q3-2025-Update.pdf（6MB）
- 起動時にdata/フォルダにコピーする処理を入れて

## 技術スタック
- PDF読み込み: PyMuPDF (fitz)
- チャンキング: RecursiveCharacterTextSplitter（chunk_size=1000, overlap=200）
- Embedding: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- ベクトルDB: ChromaDB（永続化モード）
- LLM: Claude API (claude-sonnet-4-20250514) via anthropic SDK
- APIキーは環境変数 ANTHROPIC_API_KEY から読む

## メタデータ設計（重要）
各チャンクに以下のメタデータを付与:
- source: ファイル名
- page: ページ番号
- quarter: Q3 or Q4（ファイル名から自動判定）
- year: 2025
- chunk_index: チャンク通し番号
- section: ページ上部のセクション名（取れる範囲で）

## ingest.py の仕様
- PDFをページ単位で読み込み
- メタデータ付きでチャンキング
- ChromaDBに保存（collection名: tesla_ir）
- 処理ログを出力（何ページ読んだ、何チャンク生成した等）
- 既にDBが存在する場合はスキップするオプション付き

## query.py の仕様
- CLIで質問を入力 → 関連チャンクをtop_k=5で検索
- 検索結果のチャンク + メタデータをClaudeに渡して回答生成
- システムプロンプト: 「あなたはTeslaのIRレポート分析アシスタントです。提供されたコンテキストのみに基づいて回答してください。情報が不足している場合はその旨を伝えてください。」
- 回答と一緒に参照元（ファイル名、ページ番号）を表示
- "quit"で終了するインタラクティブループ

## config.py
- 全ての設定値（chunk_size, overlap, top_k, model名等）を1箇所にまとめる
- 後から簡単に変更できるように

## 重要な制約
- LangChainは使わない（V1はシンプルに直接実装で中身を理解する）
- エラーハンドリングは最低限でOK
- 型ヒントをつける
- 各関数にdocstringをつける

まず必要なパッケージをpip installするコマンドを出して、
その後全ファイルを生成して。