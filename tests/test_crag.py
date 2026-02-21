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
