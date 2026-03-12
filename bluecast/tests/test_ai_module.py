"""Tests for the AI module (config, context, tools, result) - no LLM API keys needed."""

import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.config import AIConfig
from bluecast.ai.context import AgentLogEntry, SharedContext
from bluecast.ai.result import BlueCastAIResult
from bluecast.ai.tools import (
    TOOL_DEFINITIONS,
    _serialize_metrics,
    tool_check_correlations,
    tool_check_leakage,
    tool_create_feature,
    tool_describe_data,
)


# --- AIConfig ---
class TestAIConfig:
    def test_defaults(self):
        config = AIConfig(api_key="test")
        assert config.provider == "gemini"
        assert config.temperature == 0.2
        assert config.max_rows_for_agents == 50_000
        assert config.checkpoint_dir is None

    def test_get_model_name_default(self):
        assert (
            AIConfig(api_key="t", provider="gemini").get_model_name()
            == "gemini-2.5-flash"
        )
        assert AIConfig(api_key="t", provider="openai").get_model_name() == "gpt-4o"
        assert (
            AIConfig(api_key="t", provider="anthropic").get_model_name()
            == "claude-sonnet-4-20250514"
        )

    def test_get_model_name_custom(self):
        config = AIConfig(api_key="t", model="custom-model")
        assert config.get_model_name() == "custom-model"

    def test_get_max_iterations(self):
        assert AIConfig(api_key="t", max_iterations=7).get_max_iterations() == 7
        assert (
            AIConfig(api_key="t", max_iterations=0, mode="fast").get_max_iterations()
            == 1
        )
        assert (
            AIConfig(api_key="t", max_iterations=0, mode="precise").get_max_iterations()
            == 5
        )


# --- AgentLogEntry ---
class TestAgentLogEntry:
    def test_creation(self):
        entry = AgentLogEntry(
            timestamp=time.time(),
            agent="TestAgent",
            event_type="task",
            content="hello",
            metadata={"key": "val"},
        )
        assert entry.agent == "TestAgent"
        assert entry.event_type == "task"
        assert "TestAgent" in str(entry)
        assert "task" in str(entry)


# --- SharedContext ---
class TestSharedContext:
    def test_log(self):
        ctx = SharedContext()
        ctx.log("Agent1", "message", event_type="info", metadata={"x": 1})
        assert len(ctx.structured_log) == 1
        assert ctx.structured_log[0].agent == "Agent1"

    def test_agent_log_backward_compat(self):
        ctx = SharedContext()
        ctx.log("A", "msg")
        assert len(ctx.agent_log) == 1
        assert "[A]" in ctx.agent_log[0]

    def test_get_data_summary_empty(self):
        ctx = SharedContext()
        assert ctx.get_data_summary() == "No data loaded."

    def test_get_data_summary_classification(self):
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        ctx = SharedContext(df_train=df, target_col="target")
        summary = ctx.get_data_summary()
        assert "3 rows" in summary
        assert "target" in summary

    def test_get_data_summary_regression(self):
        df = pd.DataFrame({"a": range(100), "target": np.random.randn(100)})
        ctx = SharedContext(df_train=df, target_col="target")
        summary = ctx.get_data_summary()
        assert "continuous" in summary

    def test_get_data_summary_sampled(self):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        ctx = SharedContext(
            df_train=df,
            target_col="target",
            df_sample=df.head(1),
            was_sampled=True,
            original_shape=(1000, 5),
        )
        summary = ctx.get_data_summary()
        assert "sample" in summary.lower()

    def test_get_working_df(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        sample = pd.DataFrame({"a": [1]})
        ctx = SharedContext(df_train=df, df_sample=sample)
        assert len(ctx.get_working_df()) == 1

    def test_get_working_df_no_sample(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx = SharedContext(df_train=df)
        assert len(ctx.get_working_df()) == 3

    def test_get_working_df_no_data_raises(self):
        ctx = SharedContext()
        with pytest.raises(ValueError):
            ctx.get_working_df()

    def test_get_full_df(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx = SharedContext(df_train=df, df_sample=df.head(1))
        assert len(ctx.get_full_df()) == 3


# --- Tools ---
class TestTools:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "num1": rng.normal(0, 1, 100),
                "num2": rng.normal(5, 2, 100),
                "cat": rng.choice(["a", "b", "c"], 100),
                "target": rng.choice([0, 1], 100),
            }
        )

    def test_describe_data(self, sample_df):
        result = tool_describe_data(sample_df, "target")
        assert "Shape" in result
        assert "target" in result.lower()
        assert "binary" in result.lower() or "2" in result

    def test_check_correlations(self, sample_df):
        result = tool_check_correlations(sample_df, "target")
        assert "correlation" in result.lower() or "correlations" in result.lower()

    def test_check_leakage(self, sample_df):
        result = tool_check_leakage(sample_df, "target")
        assert "leakage" in result.lower()

    def test_create_feature_success(self, sample_df):
        result = tool_create_feature(
            sample_df, "df['ratio'] = df['num1'] / (df['num2'] + 1)"
        )
        assert result["success"] is True
        assert "ratio" in result["new_columns"]

    def test_create_feature_failure(self, sample_df):
        result = tool_create_feature(sample_df, "df['bad'] = df['nonexistent'] * 2")
        assert result["success"] is False
        assert result["error"] is not None

    def test_tool_definitions(self):
        assert len(TOOL_DEFINITIONS) >= 5
        for name, td in TOOL_DEFINITIONS.items():
            assert td.name == name
            assert len(td.description) > 10
            assert "type" in td.parameters

    def test_serialize_metrics_dict(self):
        m = {"roc_auc": 0.85, "accuracy": 0.9, "report": "text"}
        result = _serialize_metrics(m)
        assert result["roc_auc"] == 0.85

    def test_serialize_metrics_tuple(self):
        result = _serialize_metrics((0.85, 0.02))
        assert result["oof_mean"] == 0.85

    def test_serialize_metrics_numpy(self):
        m = {"score": np.float64(0.85)}
        result = _serialize_metrics(m)
        assert isinstance(result["score"], float)


# --- BlueCastAIResult ---
class TestBlueCastAIResult:
    def test_repr(self):
        result = BlueCastAIResult(metrics={"auc": 0.9})
        r = repr(result)
        assert "no pipeline" in r
        assert "auc" in r

    def test_repr_with_report(self):
        result = BlueCastAIResult(report_markdown="# Report")
        assert "report=yes" in repr(result)

    def test_predict_no_pipeline_raises(self):
        result = BlueCastAIResult()
        with pytest.raises(RuntimeError, match="No trained pipeline"):
            result.predict(pd.DataFrame())

    def test_save_code(self):
        result = BlueCastAIResult(
            pipeline_code="pipeline.fit(df)",
            feature_engineering_code="df['new'] = 1",
        )
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            path = f.name
        try:
            result.save_code(path)
            with open(path) as f:
                code = f.read()
            assert "pipeline.fit" in code
            assert "df['new']" in code
            assert "Auto-generated" in code
        finally:
            os.unlink(path)

    def test_save_report(self):
        result = BlueCastAIResult(report_markdown="# My Report\nGreat results.")
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            path = f.name
        try:
            result.save_report(path)
            with open(path) as f:
                content = f.read()
            assert "# My Report" in content
        finally:
            os.unlink(path)

    def test_save_log(self):
        entries = [
            AgentLogEntry(
                timestamp=1000.0, agent="A", event_type="task", content="hello"
            )
        ]
        result = BlueCastAIResult(structured_log=entries)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            result.save_log(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["agent"] == "A"
        finally:
            os.unlink(path)

    def test_show_report_with_markdown(self, capsys):
        result = BlueCastAIResult(report_markdown="# Report Content")
        result.show_report()
        captured = capsys.readouterr()
        assert "# Report Content" in captured.out

    def test_show_report_without_markdown(self, capsys):
        result = BlueCastAIResult(
            metrics={"roc_auc": 0.85},
            run_history=[{"success": True, "metrics": {"roc_auc": 0.85}}],
            class_problem="binary",
        )
        result.show_report()
        captured = capsys.readouterr()
        assert "binary" in captured.out
        assert "0.85" in captured.out
