# V7: pytest（ユニットテスト導入）

## 目的
V6まではコードの正しさを手動テスト（curlで叩いて目で確認）に頼っていた。これは「今動いている」ことは確認できるが、将来のコード変更で既存機能が壊れた（デグレ）ことを検知できない。V7ではpytestでユニットテストを導入し、パイプラインの各コンポーネントの動作を自動で検証できるようにする。Kasanareの業務要件「ユニットテストの実装を完遂」に直結する。

## 前提
- 既存コード: src/ingest.py, src/config.py, src/hybrid_search.py, src/reranker.py, src/crag.py, src/query.py, src/evaluate.py, src/api.py
- ChromaDB: 112チャンク（V3テーブル対応チャンキング済み）
- FastAPI: POST /query + GET /health（V6で実装済み）
- LangChain不使用方針を継続
- 既存モジュールの変更は不要

## アーキテクチャ上の位置

```
【テスト対象のレイヤー】

src/
├── ingest.py          ← テスト対象: チャンキング関数
├── hybrid_search.py   ← テスト対象: 検索関数
├── reranker.py        ← テスト対象: リランキング
├── crag.py            ← テスト対象: 品質判定パース
├── api.py             ← テスト対象: FastAPIエンドポイント
├── query.py           ← テスト対象外（CLIラッパー）
├── evaluate.py        ← テスト対象外（評価スクリプト）
└── config.py          ← テスト対象外（設定値のみ）

tests/
├── conftest.py        ← 共通フィクスチャ（モデルロード等）
├── test_ingest.py     ← チャンキングのテスト
├── test_search.py     ← 検索のテスト
├── test_crag.py       ← CRAG品質判定のテスト
└── test_api.py        ← FastAPIエンドポイントのテスト
```

ユニットテストはパイプラインの各コンポーネントを個別に検証する。外部API（Claude）に依存するテストはモック化して、テスト実行にAPIキー不要・高速実行を実現する。

## 実装内容

### 1. tests/conftest.py を新規作成

```python
"""共通フィクスチャ: モデルやテストデータの準備."""

import sys
import os
import pytest

# srcディレクトリをPythonパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def embedding_model():
    """SentenceTransformerモデル（セッション全体で1回だけロード）."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def searcher(embedding_model):
    """HybridSearcherインスタンス（セッション全体で1回だけ初期化）."""
    from hybrid_search import HybridSearcher
    return HybridSearcher(model=embedding_model)


@pytest.fixture(scope="session")
def reranker_model():
    """Rerankerインスタンス."""
    from reranker import Reranker
    return Reranker()


@pytest.fixture
def sample_text_chunk():
    """テスト用テキストチャンク."""
    return {
        "text": "Tesla's total revenue for Q4 2024 was $25.7 billion.",
        "metadata": {"content_type": "text", "page": 1},
    }


@pytest.fixture
def sample_table_chunk():
    """テスト用テーブルチャンク."""
    return {
        "text": "| Revenue | Q4 2024 | Q4 2023 |\n|---------|---------|---------|"
                "\n| Automotive | $19.8B | $21.6B |",
        "metadata": {"content_type": "table", "page": 5},
    }
```

### 2. tests/test_ingest.py を新規作成

```python
"""チャンキング関数のテスト."""

import pytest


class TestChunking:
    """ingest.pyのチャンキング関連テスト."""

    def test_chromadb_chunk_count(self):
        """ChromaDBに112チャンクが格納されていること."""
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        assert collection.count() == 112

    def test_chunks_have_content_type_metadata(self):
        """全チャンクにcontent_typeメタデータが存在すること."""
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        for metadata in results["metadatas"]:
            assert "content_type" in metadata
            assert metadata["content_type"] in ("text", "table")

    def test_table_chunks_exist(self):
        """テーブルチャンクが少なくとも1つ存在すること."""
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        table_chunks = [m for m in results["metadatas"] if m.get("content_type") == "table"]
        assert len(table_chunks) > 0, "テーブルチャンクが1つも存在しない"
```

### 3. tests/test_search.py を新規作成

```python
"""検索関数のテスト."""

import pytest


class TestVectorSearch:
    """ベクトル検索のテスト."""

    def test_vector_search_returns_results(self, searcher, embedding_model):
        """ベクトル検索が結果を返すこと."""
        results = searcher.vector_search("Tesla revenue", embedding_model, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_vector_search_result_has_text(self, searcher, embedding_model):
        """検索結果にtextフィールドが含まれること."""
        results = searcher.vector_search("Tesla revenue", embedding_model, top_k=1)
        assert "text" in results[0]
        assert len(results[0]["text"]) > 0


class TestHybridSearch:
    """ハイブリッド検索のテスト."""

    def test_hybrid_search_returns_results(self, searcher, embedding_model):
        """ハイブリッド検索が結果を返すこと."""
        results = searcher.search("Tesla revenue", embedding_model, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_hybrid_search_with_reranker(self, searcher, embedding_model, reranker_model):
        """Reranker付きハイブリッド検索が結果を返すこと."""
        results = searcher.search(
            "Tesla revenue", embedding_model, top_k=3, reranker=reranker_model
        )
        assert len(results) > 0
        assert len(results) <= 3

    def test_table_query_returns_table_chunks(self, searcher, embedding_model):
        """テーブル関連クエリでテーブルチャンクが上位に含まれること."""
        results = searcher.search("Tesla revenue table", embedding_model, top_k=5)
        content_types = [r.get("metadata", {}).get("content_type") for r in results]
        assert "table" in content_types, "テーブルクエリなのにテーブルチャンクが上位5件に含まれない"
```

### 4. tests/test_crag.py を新規作成

```python
"""CRAG品質判定のテスト（Claude APIはモック化）."""

import pytest
from unittest.mock import patch, MagicMock


class TestCRAGGrading:
    """品質判定のパースロジックのテスト."""

    def _create_mock_response(self, text):
        """Claude APIレスポンスのモックを作成."""
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = text
        mock_response.content = [mock_content]
        return mock_response

    @patch("crag.anthropic.Anthropic")
    def test_grade_correct(self, mock_anthropic_class):
        """CORRECT判定が正しくパースされること."""
        from crag import CRAGProcessor

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = self._create_mock_response("CORRECT")

        processor = CRAGProcessor()
        grade = processor.grade_results("test question", "test context")
        assert grade == "CORRECT"

    @patch("crag.anthropic.Anthropic")
    def test_grade_ambiguous(self, mock_anthropic_class):
        """AMBIGUOUS判定が正しくパースされること."""
        from crag import CRAGProcessor

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = self._create_mock_response("AMBIGUOUS")

        processor = CRAGProcessor()
        grade = processor.grade_results("test question", "test context")
        assert grade == "AMBIGUOUS"

    @patch("crag.anthropic.Anthropic")
    def test_grade_incorrect(self, mock_anthropic_class):
        """INCORRECT判定が正しくパースされること."""
        from crag import CRAGProcessor

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = self._create_mock_response("INCORRECT")

        processor = CRAGProcessor()
        grade = processor.grade_results("test question", "test context")
        assert grade == "INCORRECT"

    @patch("crag.anthropic.Anthropic")
    def test_grade_parse_with_extra_text(self, mock_anthropic_class):
        """判定結果に余分なテキストが含まれていてもパースできること."""
        from crag import CRAGProcessor

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = self._create_mock_response(
            "CORRECT\nThe search results contain the relevant information."
        )

        processor = CRAGProcessor()
        grade = processor.grade_results("test question", "test context")
        assert grade == "CORRECT"

    @patch("crag.anthropic.Anthropic")
    def test_grade_unparseable_falls_back_to_incorrect(self, mock_anthropic_class):
        """パース不能な応答はINCORRECTにフォールバックすること."""
        from crag import CRAGProcessor

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = self._create_mock_response(
            "I'm not sure about this one."
        )

        processor = CRAGProcessor()
        grade = processor.grade_results("test question", "test context")
        assert grade == "INCORRECT"
```

### 5. tests/test_api.py を新規作成

```python
"""FastAPIエンドポイントのテスト."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """FastAPIテストクライアント."""
    from api import app
    return TestClient(app)


class TestHealthEndpoint:
    """GET /health のテスト."""

    def test_health_returns_200(self, client):
        """ヘルスチェックが200を返すこと."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        """ヘルスチェックのレスポンスに必要なフィールドが含まれること."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "chromadb_chunks" in data
        assert "bm25_loaded" in data
        assert "reranker_loaded" in data
        assert "model" in data


class TestQueryEndpoint:
    """POST /query のテスト."""

    def test_query_validation_empty_question(self, client):
        """空の質問で422エラーが返ること."""
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_validation_invalid_mode(self, client):
        """不正なsearch_modeで422エラーが返ること."""
        response = client.post(
            "/query",
            json={"question": "test", "search_mode": "invalid"},
        )
        assert response.status_code == 422

    def test_query_validation_top_k_too_low(self, client):
        """top_k=0で422エラーが返ること."""
        response = client.post(
            "/query",
            json={"question": "test", "top_k": 0},
        )
        assert response.status_code == 422

    def test_query_validation_top_k_too_high(self, client):
        """top_k=100で422エラーが返ること."""
        response = client.post(
            "/query",
            json={"question": "test", "top_k": 100},
        )
        assert response.status_code == 422

    def test_query_response_fields(self, client):
        """クエリレスポンスに必要なフィールドが含まれること."""
        response = client.post(
            "/query",
            json={"question": "What was Tesla total revenue?", "search_mode": "vector"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "search_mode" in data
        assert "chunks_used" in data
        assert "crag_grades" in data
        assert "crag_retries" in data
        assert "processing_time_sec" in data

    def test_query_crag_mode_has_grades(self, client):
        """CRAGモードでcrag_gradesが空でないこと."""
        response = client.post(
            "/query",
            json={"question": "What was Tesla total revenue?", "search_mode": "crag"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["crag_grades"]) > 0

    def test_query_non_crag_mode_empty_grades(self, client):
        """CRAG以外のモードでcrag_gradesが空であること."""
        response = client.post(
            "/query",
            json={"question": "What was Tesla total revenue?", "search_mode": "rerank"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["crag_grades"] == []
        assert data["crag_retries"] == 0
```

### 6. requirements.txt に追加
```
pytest
```

### 7. 変更しないもの
- src/ 配下の全ファイル — 変更不要
- chroma_db/ — 再構築不要

**重要な設計判断:**
- test_crag.pyではClaude APIをモック化。テスト実行にAPIキーが不要で、高速かつ再現性がある。テストの目的は「CRAGのパースロジックが正しいか」であり「Claudeが正しく答えるか」ではない
- test_api.pyではFastAPIのTestClientを使用。実際にサーバーを起動せずにエンドポイントをテストできる。ただしask_claude()は実際のAPIを呼ぶため、APIキーが必要（モック化する場合はスコープ拡大）
- conftest.pyのフィクスチャはscope="session"でモデルを1回だけロード。テスト毎にロードすると遅すぎる
- test_api.pyのquery応答テスト（test_query_response_fields, test_query_crag_mode_has_grades等）は実際にClaude APIを呼ぶ「結合テスト」に近い。純粋なユニットテストにするならask_claude()もモック化するが、V7の目的は「パイプライン全体が正しく動くことの自動検証」なのでこのまま進める

## テスト手順

### 1. pytest実行（全テスト）
```bash
cd ~/product/Tesla_rag
PYTHONPATH=src pytest tests/ -v
```
確認項目:
- 全テストがPASS（緑）であること
- FAILEDやERRORがないこと

### 2. 個別テスト実行
```bash
# チャンキングテストのみ
PYTHONPATH=src pytest tests/test_ingest.py -v

# 検索テストのみ
PYTHONPATH=src pytest tests/test_search.py -v

# CRAGテストのみ（APIキー不要、高速）
PYTHONPATH=src pytest tests/test_crag.py -v

# APIテストのみ
PYTHONPATH=src pytest tests/test_api.py -v
```

### 3. テストカバレッジ確認（オプション）
```bash
pip install pytest-cov
PYTHONPATH=src pytest tests/ -v --cov=src --cov-report=term-missing
```

## 壊す実験

### 実験: ChromaDBのチャンク数を変えたらテストが検知するか
```bash
# 1. 現在のテストが通ることを確認
PYTHONPATH=src pytest tests/test_ingest.py::TestChunking::test_chromadb_chunk_count -v

# 2. test_ingest.pyのassert値を113に変更（意図的に壊す）
# assert collection.count() == 113

# 3. テスト実行 → FAILEDになること確認
PYTHONPATH=src pytest tests/test_ingest.py::TestChunking::test_chromadb_chunk_count -v

# 4. 元に戻す（112）
```
→ テストがデグレを検知できる証拠。

### 実験: CRAGのパースロジックを壊したらテストが検知するか
```bash
# 1. crag.pyのgrade_results()のフォールバックを"CORRECT"に変更（意図的に壊す）
# return "CORRECT"  # 本来は "INCORRECT"

# 2. テスト実行
PYTHONPATH=src pytest tests/test_crag.py::TestCRAGGrading::test_grade_unparseable_falls_back_to_incorrect -v
# → FAILEDになること確認

# 3. 元に戻す
```

## 成功基準
- `pytest tests/ -v` で全テストがPASSすること
- test_crag.pyがAPIキー不要で実行できること（モック化の確認）
- テストが既存コード（src/配下）を一切変更せずに動くこと
- 壊す実験でテストがデグレを検知できること
- 既存のquery.py, evaluate.py, api.pyが引き続き動作すること（デグレ防止）

## 注意点
- PYTHONPATHの設定: `PYTHONPATH=src pytest tests/ -v` を忘れるとimportエラーになる
- test_api.pyのクエリ応答テスト（test_query_response_fields等）は実際にClaude APIを呼ぶため、APIキーが必要かつ実行に数秒かかる。CI/CDに組み込む場合はモック化を検討
- conftest.pyのsys.path追加: testsディレクトリからsrc/配下のモジュールをimportするために必要
- HybridSearcherの初期化方法: conftest.pyのフィクスチャでembedding_modelを渡しているが、実際のHybridSearcherのコンストラクタに合わせて調整が必要な場合がある
- test_search.pyのtest_table_query_returns_table_chunks: テーブルブーストの効果を確認するテストだが、クエリ次第で結果が変わる可能性あり。失敗した場合はクエリを調整