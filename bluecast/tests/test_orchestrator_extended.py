from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.config import AIConfig
from bluecast.ai.orchestrator import Orchestrator


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.simple_chat.return_value = '```json\n{"tuning_rounds": 10}\n```'
    return llm


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "cat": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        }
    )


def test_step_feature_engineer(mock_llm, sample_df, tmpdir):
    config = AIConfig(api_key="test", verbose=False, checkpoint_dir=str(tmpdir))
    orch = Orchestrator(mock_llm, config, sample_df.copy(), "target", "test")

    orch.context.feature_code_snippets = ["df['new_col'] = df['num1'] * 2"]

    plan = {"feature_engineering_hints": ["hint 1"]}

    with patch(
        "bluecast.ai.orchestrator.Orchestrator._get_critique_rounds", return_value=0
    ):
        orch._step_feature_engineer(plan)

    assert orch.context.engineered_df is not None
    assert "new_col" in orch.context.engineered_df.columns
    assert len(orch.context.feature_code_snippets) == 1


def test_compare_results(mock_llm, sample_df, tmpdir):
    config = AIConfig(api_key="test", verbose=False, checkpoint_dir=str(tmpdir))
    orch = Orchestrator(mock_llm, config, sample_df.copy(), "target", "test")

    res1 = {"metrics": {"mae": 5.0}}
    res2 = {"metrics": {"mae": 3.0}}
    assert orch._compare_results(res2, res1) is True
    assert orch._compare_results(res1, res2) is False

    res3 = {"metrics": {"roc_auc": 0.8}}
    res4 = {"metrics": {"roc_auc": 0.9}}
    assert orch._compare_results(res4, res3) is True
    assert orch._compare_results(res3, res4) is False


def test_evaluate_for_arch(mock_llm, sample_df, tmpdir):
    config = AIConfig(api_key="test", verbose=False, checkpoint_dir=str(tmpdir))
    orch = Orchestrator(mock_llm, config, sample_df.copy(), "target", "test")

    orch.context.arch_feature_importances["xgboost"] = {"num1": 0.5, "cat": 0.1}
    orch.context.arch_errors["xgboost"] = "Test error"

    result = {"success": True, "metrics": {"mae": 5.0}, "config_used": {}}

    with patch(
        "bluecast.ai.orchestrator.Orchestrator._get_critique_rounds", return_value=0
    ):
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = '```json\n{"tuning_rounds": 10}\n```'
        sugg = orch._evaluate_for_arch(
            "xgboost", "XGBoost", result, iteration=0, total_iterations=2
        )

    assert sugg.get("tuning_rounds") == 10


def test_assemble_result(mock_llm, sample_df, tmpdir):
    config = AIConfig(api_key="test", verbose=False, checkpoint_dir=str(tmpdir))
    orch = Orchestrator(mock_llm, config, sample_df.copy(), "target", "test")

    orch.context.class_problem = "binary"

    p1 = MagicMock()
    p1.oof_predictions_ = np.array([0.1, 0.9])
    p1.oof_valid_mask_ = np.array([True, True])

    p2 = MagicMock()
    p2.oof_predictions_ = np.array([0.2, 0.8])
    p2.oof_valid_mask_ = np.array([True, True])

    orch.context.best_pipelines = [p1, p2]
    orch.context.df_train = pd.DataFrame({"target": [0, 1]})

    with patch("bluecast.ensemble.hill_climbing.HillClimbingEnsemble.fit") as mock_hc:
        res = orch._assemble_result()
        assert mock_hc.called
        assert len(res.pipelines) == 2
