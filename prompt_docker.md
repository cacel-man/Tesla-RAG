# V8: Docker化（コンテナ化によるポータビリティ確保）

## 目的
V7まではローカル環境（mamba環境: tesla-rag）でしか動かせなかった。Python版、依存パッケージ、ChromaDBのパスなど、環境に依存する要素が多く、他の人のマシンや本番サーバーで動かすには環境構築から始める必要がある。V8ではDockerfileを作成し、`docker build && docker run`だけでRAGパイプライン全体が動く状態にする。Kasanareの歓迎要件「Docker等を用いたコンテナ技術利用経験」に直結する。

## 前提
- 既存コード: src/配下全ファイル + tests/配下全ファイル
- ChromaDB: 112チャンク（chroma_db/ディレクトリ）
- FastAPI: POST /query + GET /health（V6で実装済み）
- pytest: 全21テストPASSED（V7で実装済み）
- 環境変数: ANTHROPIC_API_KEY（.envファイル）
- LangChain不使用方針を継続
- 既存モジュールの変更は不要

## アーキテクチャ上の位置

```
【V7まで: ローカル環境依存】
mamba activate tesla-rag → uvicorn src.api:app → localhost:8000

【V8: Docker化】
docker build -t tesla-rag . → docker run -p 8000:8000 --env-file .env tesla-rag
                                  │
                                  └── コンテナ内部:
                                        ├── Python 3.11
                                        ├── 依存パッケージ（requirements.txt）
                                        ├── src/ + tests/
                                        ├── chroma_db/（112チャンク）
                                        └── uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Dockerは「環境ごとコードを包む」だけ。アプリケーションロジックは一切変更しない。

## 実装内容

### 1. Dockerfile を新規作成（プロジェクトルート）

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 依存パッケージインストール（キャッシュ活用のためrequirements.txtを先にコピー）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY src/ ./src/
COPY chroma_db/ ./chroma_db/
COPY tests/ ./tests/

# 環境変数（PYTHONPATHでsrc/配下のimportを解決）
ENV PYTHONPATH=/app/src

# ポート公開
EXPOSE 8000

# ヘルスチェック（V6のGET /healthを活用）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# サーバー起動
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. .dockerignore を新規作成（プロジェクトルート）

```
.env
.git/
.gitignore
__pycache__/
*.pyc
*.pyo
results/
data/
*.md
.claude/
prompt_*.md
```

### 3. 変更しないもの
- src/ 配下の全ファイル — 変更不要
- tests/ 配下の全ファイル — 変更不要
- chroma_db/ — 変更不要（Dockerイメージに含める）
- requirements.txt — 変更不要（V7でpytest追加済み）

**重要な設計判断:**
- `python:3.11-slim`を使用。fullイメージは不要に大きく、alpineはC拡張のビルドで問題が出やすい。slimがバランス最良
- chroma_db/をイメージに含める。ボリュームマウントでも可能だが、「docker runだけで動く」を優先。データが大きくなった場合はボリュームに切り替える
- .envはDockerイメージに含めない（.dockerignoreで除外）。実行時に`--env-file .env`で渡す。APIキーをイメージに焼き込むのはセキュリティリスク
- HEALTHCHECKにV6のGET /healthを使用。start-period=60sはモデルロード時間を考慮（SentenceTransformer + Rerankerのロードに数十秒かかる可能性）
- requirements.txtにcurlは含まれないが、python:3.11-slimにはcurlが入っていない場合がある。HEALTHCHECKが失敗する場合は`RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*`を追加するか、pythonベースのヘルスチェックに変更

## テスト手順

### 1. Dockerイメージのビルド
```bash
cd ~/product/Tesla_rag
docker build -t tesla-rag .
```
確認項目:
- ビルドが成功すること（エラーなし）
- `docker images` でtesla-ragが表示されること

### 2. コンテナ起動
```bash
docker run -d -p 8000:8000 --env-file .env --name tesla-rag-server tesla-rag
```
確認項目:
- `docker ps` でtesla-rag-serverが表示されること
- `docker logs tesla-rag-server` で「✅ ロード完了: ChromaDB 112 チャンク」が出ること

### 3. APIテスト（ローカルと同じ）
```bash
# ヘルスチェック
curl http://localhost:8000/health | python -m json.tool

# クエリ
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Tesla total revenue in Q4 2025?"}' \
  | python -m json.tool
```
確認項目:
- V6と同じレスポンスが返ること（コンテナ内でもローカルと同じ動作）

### 4. コンテナ内でpytest実行
```bash
docker exec tesla-rag-server pytest tests/ -v
```
確認項目:
- 全21テストがPASS（コンテナ内でもテストが通る）
- ※ test_api.pyのClaude API呼び出しテストは環境変数が渡っていれば動作する

### 5. ヘルスチェック確認
```bash
docker inspect --format='{{.State.Health.Status}}' tesla-rag-server
```
確認項目:
- `healthy` が表示されること（start-period後）

### 6. クリーンアップ
```bash
docker stop tesla-rag-server
docker rm tesla-rag-server
```

## 壊す実験

### 実験: .envなしでコンテナ起動
```bash
docker run -d -p 8001:8000 --name tesla-rag-no-env tesla-rag
```
- ヘルスチェックは通るか？ → モデルロードは成功するはず（APIキー不要）
- POST /queryは？ → ask_claude()でAPIキーエラーが出るはず
- つまり「検索は動くがLLM回答生成だけ失敗する」状態になるか確認

### 実験: ポート変更
```bash
docker run -d -p 9000:8000 --env-file .env --name tesla-rag-port tesla-rag
curl http://localhost:9000/health
```
- ポートマッピングが正しく動作するか確認

## 成功基準
- `docker build` がエラーなく完了すること
- `docker run` でコンテナが起動し、APIが応答すること
- GET /health が200で返ること（コンテナ内）
- POST /query が正しい回答を返すこと（コンテナ内）
- `docker exec ... pytest` で全テストがPASSすること
- HEALTHCHECKが`healthy`になること
- .envがDockerイメージに含まれていないこと（セキュリティ）
- 既存のローカル実行（uvicorn直接起動、pytest直接実行）が引き続き動作すること

## 注意点
- Dockerデーモンが起動していること: `docker info` で確認。起動してなければ `open -a Docker`（Mac）または `sudo systemctl start docker`（Linux）
- ディスク容量: python:3.11-slimベースでも依存パッケージ含めると数GBになる可能性（sentence-transformers, torch等）。ディスク空き容量に注意
- ビルド時間: 初回はpip installに時間がかかる（torch, sentence-transformers等）。2回目以降はDockerのレイヤーキャッシュが効く（requirements.txtを先にCOPYしている理由）
- ARM/x86: M1/M2 Macの場合、一部パッケージでアーキテクチャの問題が出る可能性。`--platform linux/amd64`を付ける場合がある
- HEALTHCHECKのcurl: python:3.11-slimにcurlが入っていない場合は、Dockerfile内で`apt-get install -y curl`を追加するか、`CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"` に変更