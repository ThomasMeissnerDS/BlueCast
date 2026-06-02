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
            mock_orch_instance.run.assert_called_once_with()


