"""Tests for BlueCastAI entry point (__init__.py)"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bluecast.ai import BlueCastAI
from bluecast.ai.result import BlueCastAIResult


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})


class TestBlueCastAI:
    def test_run_success(self, sample_df):
        ai = BlueCastAI(api_key="test", provider="gemini")

        mock_result = BlueCastAIResult(metrics={"auc": 0.9})

        with patch("bluecast.ai.orchestrator.Orchestrator") as mock_orch:
            mock_orch_instance = MagicMock()
            mock_orch_instance.run.return_value = mock_result
            mock_orch.return_value = mock_orch_instance

            result = ai.run(sample_df, "target", prompt="test prompt")

            assert isinstance(result, BlueCastAIResult)
            assert result.metrics["auc"] == 0.9
            mock_orch_instance.run.assert_called_once_with(
                "test prompt", max_iterations=None
            )

    def test_run_no_api_key_openai(self):
        with patch.dict("os.environ", clear=True):
            with pytest.raises(ValueError, match="API key required"):
                BlueCastAI(provider="openai")

    def test_run_vertex_auth_error(self, sample_df):
        with patch("bluecast.ai.providers.vertexai_provider.VertexAIProvider"):
            # Let initialization succeed but run fail
            ai = BlueCastAI(provider="vertexai", project_id="test", location="test")
            with patch("bluecast.ai.orchestrator.Orchestrator") as mock_orch:
                mock_orch_instance = MagicMock()
                from google.api_core.exceptions import DefaultCredentialsError

                mock_orch_instance.run.side_effect = DefaultCredentialsError(
                    "Auth failed"
                )
                mock_orch.return_value = mock_orch_instance

                result = ai.run(sample_df, "target")
                assert "Authentication failed" in result.error
