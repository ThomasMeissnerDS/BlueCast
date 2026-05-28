from unittest.mock import patch
from bluecast.ai.orchestrator import Orchestrator
from bluecast.ai.config import AIConfig
from bluecast.tests.test_ai_mock_provider import MockLLMProvider, make_tool_response, make_text_response
import pandas as pd
import numpy as np

sample_df = pd.DataFrame({"num": [1,2,3], "target": [0,1,0]})
mock_llm = MockLLMProvider()
mock_llm.enqueue_response(make_tool_response("build_and_run_pipeline", {"class_problem": "binary"}))
mock_llm.enqueue_response(make_text_response("Done"))

config = AIConfig(api_key="test", mode="fast", verbose=True, max_iterations=1, critique_max_rounds=0)
orch = Orchestrator(mock_llm, config, sample_df, "target", "test")

plan = {
    "class_problem": "binary",
    "use_cv": True,
    "ensemble_strategy": "stacking",
    "n_folds": 3,
}

with patch("bluecast.ai.orchestrator.tool_build_and_run_pipeline") as mock_build:
    mock_build.return_value = {
        "success": True,
        "metrics": {"roc_auc": 0.85},
        "config_used": {"class_problem": "binary"},
        "pipeline": None,
        "error": None,
    }
    orch._step_build_loop(plan, max_iterations=1)
