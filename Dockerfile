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

# ヘルスチェック（V6のGET /healthを活用、python:3.11-slimにcurlがないためpythonで実行）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# サーバー起動
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
