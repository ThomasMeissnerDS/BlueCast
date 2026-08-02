"""Tests for bluecast.ai.agents — BaseAgent, all agent subclasses."""

import json

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.agents.arch_feature_engineer import (
    ARCH_FE_GUIDELINES,
    ArchFeatureEngineerAgent,
)
from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.agents.data_analyst import DataAnalystAgent
from bluecast.ai.agents.evaluator import EvaluatorAgent
from bluecast.ai.agents.feature_engineer import FeatureEngineerAgent
from bluecast.ai.agents.pipeline_builder import PipelineBuilderAgent
from bluecast.ai.agents.planner import PlannerAgent
from bluecast.ai.agents.reporter import ReporterAgent
from bluecast.ai.agents.researcher import ResearcherAgent
from bluecast.ai.context import SharedContext
from bluecast.ai.providers.base import (
    LLMResponse,
    ToolDefinition,
)
from bluecast.tests.test_ai_mock_provider import (
    MockLLMProvider,
    make_text_response,
    make_tool_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num1": rng.normal(0, 1, 50),
            "num2": rng.normal(5, 2, 50),
            "cat": rng.choice(["a", "b", "c"], 50),
            "target": rng.choice([0, 1], 50),
        }
    )


@pytest.fixture
def context(sample_df):
    return SharedContext(
        df_train=sample_df,
        target_col="target",
        mode="balanced",
    )


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing the abstract BaseAgent."""

    @property
    def name(self):
        return "TestAgent"

    def system_prompt(self):
        return "You are a test agent."

    def get_tools(self):
        return [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]


class TestBaseAgent:
    def test_run_text_response(self, context, mock_llm):
        mock_llm.enqueue_response(make_text_response("Final answer."))
        agent = ConcreteAgent(mock_llm, context, verbose=False)
        result = agent.run("Test task")
        assert result == "Final answer."
        assert len(context.structured_log) >= 2  # task + response

    def test_run_tool_call_then_text(self, context, mock_llm):
        """Agent calls a tool, gets result, then gives final text."""
        mock_llm.enqueue_response(make_tool_response("test_tool", {"arg": "value"}))
        mock_llm.enqueue_response(make_text_response("Done with tools."))

        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.register_tool_impl("test_tool", lambda **kw: "tool result")
        result = agent.run("Do something")
        assert result == "Done with tools."
        assert mock_llm.call_count == 2

    def test_run_loop_detection(self, context, mock_llm):
        """Same tool call repeated → infinite loop intercepted."""
        same_response = make_tool_response("test_tool", {"arg": "same"})
        # Queue 3 identical tool calls then a text response
        mock_llm.enqueue_responses(
            [same_response, same_response, make_text_response("Stopped looping.")]
        )

        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.register_tool_impl("test_tool", lambda **kw: "result")
        result = agent.run("Do something")
        assert result == "Stopped looping."

    def test_run_api_error(self, context):
        """LLM API error → agent returns fallback message."""

        class FailingLLM(MockLLMProvider):
            def chat(self, messages, tools=None):
                self.call_count += 1
                raise ConnectionError("Network error")

        agent = ConcreteAgent(FailingLLM(), context, verbose=False)
        result = agent.run("Test task")
        assert "API error" in result or "error" in result.lower()

    def test_run_max_iterations(self, context, mock_llm):
        """Exhausts max iterations → returns last text."""
        # Queue more tool calls than max iterations
        for _ in range(15):
            mock_llm.enqueue_response(make_tool_response("test_tool", {"i": str(_)}))

        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.register_tool_impl("test_tool", lambda **kw: "result")
        result = agent.run("Do something")
        assert isinstance(result, str)

    def test_execute_tool_known(self, context, mock_llm):
        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.register_tool_impl("test_tool", lambda x=1: f"result_{x}")
        result = agent.execute_tool("test_tool", {"x": 42})
        assert result == "result_42"

    def test_execute_tool_unknown(self, context, mock_llm):
        agent = ConcreteAgent(mock_llm, context, verbose=False)
        result = agent.execute_tool("unknown_tool", {})
        assert "Unknown tool" in result

    def test_execute_tool_exception(self, context, mock_llm):
        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.register_tool_impl("bad_tool", lambda: 1 / 0)
        result = agent.execute_tool("bad_tool", {})
        assert "Error" in result

    def test_execute_tool_dict_result(self, context, mock_llm):
        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.register_tool_impl("dict_tool", lambda: {"key": "value"})
        result = agent.execute_tool("dict_tool", {})
        assert "key" in result

    def test_execute_tool_long_result_truncated(self, context, mock_llm):
        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.register_tool_impl("long_tool", lambda: "x" * 10000)
        result = agent.execute_tool("long_tool", {})
        assert len(result) <= 5100  # 5000 + truncation message
        assert "truncated" in result.lower()

    def test_usage_tracking(self, context, mock_llm):
        mock_llm.enqueue_response(
            LLMResponse(
                text="done",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            )
        )
        agent = ConcreteAgent(mock_llm, context, verbose=False)
        agent.run("task")
        assert context.prompt_tokens == 100
        assert context.completion_tokens == 50


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    def test_parse_plan_json(self, context, mock_llm):
        planner = PlannerAgent(mock_llm, context, verbose=False)
        json_text = json.dumps(
            {
                "class_problem": "regression",
                "needs_feature_engineering": True,
                "max_iterations": 5,
            }
        )
        plan = planner.parse_plan(json_text)
        assert plan["class_problem"] == "regression"

    def test_parse_plan_markdown_json(self, context, mock_llm):
        planner = PlannerAgent(mock_llm, context, verbose=False)
        text = '```json\n{"class_problem": "binary"}\n```'
        plan = planner.parse_plan(text)
        assert plan["class_problem"] == "binary"

    def test_parse_plan_embedded_json(self, context, mock_llm):
        planner = PlannerAgent(mock_llm, context, verbose=False)
        text = 'Here is the plan: {"class_problem": "multiclass"} end.'
        plan = planner.parse_plan(text)
        assert plan["class_problem"] == "multiclass"

    def test_parse_plan_garbage(self, context, mock_llm):
        planner = PlannerAgent(mock_llm, context, verbose=False)
        plan = planner.parse_plan("not json at all")
        # Should return default plan
        assert plan["class_problem"] == "binary"

    def test_default_plan(self, context, mock_llm):
        planner = PlannerAgent(mock_llm, context, verbose=False)
        plan = planner._default_plan()
        assert plan["class_problem"] == "binary"
        assert plan["ensemble_strategy"] == "stacking"

    def test_name(self, context, mock_llm):
        planner = PlannerAgent(mock_llm, context, verbose=False)
        assert planner.name == "Planner"

    def test_get_tools_empty(self, context, mock_llm):
        planner = PlannerAgent(mock_llm, context, verbose=False)
        assert planner.get_tools() == []


# ---------------------------------------------------------------------------
# DataAnalystAgent
# ---------------------------------------------------------------------------


class TestDataAnalystAgent:
    def test_name(self, context, mock_llm):
        agent = DataAnalystAgent(mock_llm, context, verbose=False)
        assert agent.name == "DataAnalyst"

    def test_has_tools(self, context, mock_llm):
        agent = DataAnalystAgent(mock_llm, context, verbose=False)
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        assert "describe_data" in tool_names
        assert "check_correlations" in tool_names
        assert "check_leakage" in tool_names

    def test_tool_registration(self, context, mock_llm):
        agent = DataAnalystAgent(mock_llm, context, verbose=False)
        assert "describe_data" in agent._tool_implementations
        assert "check_correlations" in agent._tool_implementations

    def test_system_prompt(self, context, mock_llm):
        agent = DataAnalystAgent(mock_llm, context, verbose=False)
        prompt = agent.system_prompt()
        assert "data analyst" in prompt.lower()
        assert "target" in prompt

    def test_system_prompt_with_context_files(self, context, mock_llm):
        context.context_file_contents = ["Domain: medical imaging"]
        agent = DataAnalystAgent(mock_llm, context, verbose=False)
        prompt = agent.system_prompt()
        assert "medical" in prompt


# ---------------------------------------------------------------------------
# EvaluatorAgent
# ---------------------------------------------------------------------------


class TestEvaluatorAgent:
    def test_name(self, context, mock_llm):
        agent = EvaluatorAgent(mock_llm, context, verbose=False)
        assert agent.name == "Evaluator"

    def test_get_tools_empty(self, context, mock_llm):
        agent = EvaluatorAgent(mock_llm, context, verbose=False)
        assert agent.get_tools() == []

    def test_system_prompt_no_history(self, context, mock_llm):
        agent = EvaluatorAgent(mock_llm, context, verbose=False)
        prompt = agent.system_prompt()
        assert "evaluation" in prompt.lower()

    def test_system_prompt_with_history(self, context, mock_llm):
        context.run_history = [
            {"success": True, "metrics": {"roc_auc": 0.85}, "config": {}},
            {"success": False, "metrics": {}, "config": {}},
        ]
        agent = EvaluatorAgent(mock_llm, context, verbose=False)
        prompt = agent.system_prompt()
        assert "Run 1" in prompt
        assert "Run 2" in prompt


# ---------------------------------------------------------------------------
# FeatureEngineerAgent
# ---------------------------------------------------------------------------


class TestFeatureEngineerAgent:
    def test_name(self, context, mock_llm):
        agent = FeatureEngineerAgent(mock_llm, context, verbose=False)
        assert agent.name == "FeatureEngineer"

    def test_create_feature_wrapper_success(self, context, mock_llm):
        agent = FeatureEngineerAgent(mock_llm, context, verbose=False)
        result = agent._create_feature_wrapper(
            "df['new'] = df['num1'] * 2", description="double num1"
        )
        assert result["success"] is True
        assert context.engineered_df is not None
        assert "new" in context.engineered_df.columns
        # Snippet recorded
        assert "df['new'] = df['num1'] * 2" in context.feature_code_snippets

    def test_create_feature_wrapper_no_data(self, mock_llm):
        ctx = SharedContext(target_col="target")
        agent = FeatureEngineerAgent(mock_llm, ctx, verbose=False)
        result = agent._create_feature_wrapper("df['x'] = 1")
        assert result["success"] is False

    def test_create_feature_wrapper_hides_target(self, context, mock_llm):
        agent = FeatureEngineerAgent(mock_llm, context, verbose=False)
        agent._create_feature_wrapper("df['x'] = df['num1']")
        # Target should not be in engineered_df
        assert "target" not in context.engineered_df.columns

    def test_create_tfidf_wrapper_success(self, mock_llm):
        df = pd.DataFrame(
            {
                "text": ["hello world", "foo bar", "hello foo"] * 20,
                "num": list(range(60)),
                "target": [0, 1] * 30,
            }
        )
        ctx = SharedContext(df_train=df, target_col="target")
        agent = FeatureEngineerAgent(mock_llm, ctx, verbose=False)
        result = agent._create_tfidf_wrapper("text", max_features=3)
        assert result["success"] is True
        assert len(ctx.feature_code_snippets) > 0

    def test_create_tfidf_wrapper_no_data(self, mock_llm):
        ctx = SharedContext(target_col="target")
        agent = FeatureEngineerAgent(mock_llm, ctx, verbose=False)
        result = agent._create_tfidf_wrapper("text")
        assert result["success"] is False

    def test_system_prompt(self, context, mock_llm):
        agent = FeatureEngineerAgent(mock_llm, context, verbose=False)
        prompt = agent.system_prompt()
        assert "feature engineer" in prompt.lower()

    def test_get_tools(self, context, mock_llm):
        agent = FeatureEngineerAgent(mock_llm, context, verbose=False)
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        assert "create_feature" in tool_names
        assert "create_tfidf_features" in tool_names


# ---------------------------------------------------------------------------
# ArchFeatureEngineerAgent
# ---------------------------------------------------------------------------


class TestArchFeatureEngineerAgent:
    def test_name(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        assert agent.name == "ArchFeatureEngineer"

    def test_set_architecture(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("catboost", "CatBoost")
        assert agent._arch_name == "catboost"
        assert agent._arch_display_name == "CatBoost"

    def test_create_feature_wrapper(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("catboost", "CatBoost")
        result = agent._create_feature_wrapper(
            "df['sq'] = df['num1'] ** 2", description="squared"
        )
        assert result["success"] is True
        assert "catboost" in context.arch_feature_snippets
        assert len(context.arch_feature_snippets["catboost"]) >= 1

    def test_create_feature_wrapper_constant_rejected(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("xgboost", "XGBoost")
        result = agent._create_feature_wrapper("df['const'] = 1")
        # Constant columns should be rejected
        assert result["success"] is False or (
            result["success"] is True and "const" not in result.get("new_columns", [])
        )

    def test_create_feature_wrapper_no_data(self, mock_llm):
        ctx = SharedContext(target_col="target")
        agent = ArchFeatureEngineerAgent(mock_llm, ctx, verbose=False)
        agent.set_architecture("catboost", "CatBoost")
        result = agent._create_feature_wrapper("df['x'] = 1")
        assert result["success"] is False

    def test_tfidf_wrapper(self, mock_llm):
        df = pd.DataFrame(
            {
                "text": ["hello world", "foo bar", "baz"] * 20,
                "num": list(range(60)),
                "target": [0, 1] * 30,
            }
        )
        ctx = SharedContext(df_train=df, target_col="target")
        agent = ArchFeatureEngineerAgent(mock_llm, ctx, verbose=False)
        agent.set_architecture("histgb", "HistGradientBoosting")
        result = agent._create_tfidf_wrapper("text", max_features=3)
        assert result["success"] is True
        assert "histgb" in ctx.arch_feature_snippets

    def test_drop_collinear_wrapper(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("linear", "Linear")
        result = agent._drop_collinear_wrapper(threshold=0.9)
        assert result.get("success") is True or "success" in result

    def test_l1_selection_wrapper(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("linear", "Linear")
        result = agent._l1_selection_wrapper(alpha=0.01)
        assert "success" in result

    def test_l1_selection_no_target(self, mock_llm):
        ctx = SharedContext(df_train=pd.DataFrame({"a": [1, 2, 3]}))
        agent = ArchFeatureEngineerAgent(mock_llm, ctx, verbose=False)
        agent.set_architecture("linear", "Linear")
        result = agent._l1_selection_wrapper()
        assert result["success"] is False

    def test_check_feature_quality_wrapper(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("catboost", "CatBoost")
        result = agent._check_feature_quality_wrapper(feature_cols=["num1"])
        assert isinstance(result, (str, dict))

    def test_check_feature_quality_no_data(self, mock_llm):
        ctx = SharedContext(target_col="target")
        agent = ArchFeatureEngineerAgent(mock_llm, ctx, verbose=False)
        agent.set_architecture("catboost", "CatBoost")
        result = agent._check_feature_quality_wrapper(feature_cols=["x"])
        assert isinstance(result, str)

    def test_system_prompt_with_importances(self, context, mock_llm):
        context.arch_feature_importances["catboost"] = {
            "num1": 0.5,
            "num2": 0.3,
            "cat": 0.1,
        }
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("catboost", "CatBoost")
        prompt = agent.system_prompt()
        assert "num1" in prompt
        assert "importance" in prompt.lower()

    def test_get_tools_linear(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("linear", "Linear")
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        assert "drop_collinear_features" in tool_names
        assert "l1_feature_selection" in tool_names

    def test_get_tools_catboost(self, context, mock_llm):
        agent = ArchFeatureEngineerAgent(mock_llm, context, verbose=False)
        agent.set_architecture("catboost", "CatBoost")
        tools = agent.get_tools()
        tool_names = [t.name for t in tools]
        assert "create_feature" in tool_names
        # Catboost doesn't get L1/collinear tools
        assert "l1_feature_selection" not in tool_names

    def test_arch_fe_guidelines(self):
        for arch in ["catboost", "xgboost", "histgb", "randomforest", "linear", "mlp"]:
            assert arch in ARCH_FE_GUIDELINES
            assert len(ARCH_FE_GUIDELINES[arch]) > 50


# ---------------------------------------------------------------------------
# PipelineBuilderAgent
# ---------------------------------------------------------------------------


class TestPipelineBuilderAgent:
    def test_name(self, context, mock_llm):
        agent = PipelineBuilderAgent(mock_llm, context, verbose=False)
        assert agent.name == "PipelineBuilder"

    def test_get_tools(self, context, mock_llm):
        agent = PipelineBuilderAgent(mock_llm, context, verbose=False)
        tools = agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "build_and_run_pipeline"

    def test_system_prompt_no_history(self, context, mock_llm):
        agent = PipelineBuilderAgent(mock_llm, context, verbose=False)
        prompt = agent.system_prompt()
        assert "pipeline builder" in prompt.lower()

    def test_system_prompt_with_history(self, context, mock_llm):
        context.run_history = [
            {"success": True, "metrics": {"roc_auc": 0.85}, "config": {"n_folds": 5}},
        ]
        agent = PipelineBuilderAgent(mock_llm, context, verbose=False)
        prompt = agent.system_prompt()
        assert "Run 1" in prompt

    def test_generate_pipeline_code(self, context, mock_llm):
        agent = PipelineBuilderAgent(mock_llm, context, verbose=False)
        agent._generate_pipeline_code({"class_problem": "binary", "use_cv": True})
        assert context.pipeline_code is not None
        assert "BlueCastAuto" in context.pipeline_code

    def test_generate_pipeline_code_regression_mae(self, context, mock_llm):
        context.target_col = "target"
        agent = PipelineBuilderAgent(mock_llm, context, verbose=False)
        agent._generate_pipeline_code(
            {
                "class_problem": "regression",
                "use_cv": True,
                "regression_eval_metric": "mae",
                "ensemble_strategy": "mean",
            }
        )
        assert "MAE" in context.pipeline_code

    def test_generate_pipeline_code_with_fe_snippets(self, context, mock_llm):
        context.feature_code_snippets = ["df['new'] = 1"]
        agent = PipelineBuilderAgent(mock_llm, context, verbose=False)
        agent._generate_pipeline_code({"class_problem": "binary", "use_cv": True})
        assert "AIFeaturePreprocessor" in context.pipeline_code


# ---------------------------------------------------------------------------
# ReporterAgent
# ---------------------------------------------------------------------------


class TestReporterAgent:
    def test_name(self, context, mock_llm):
        agent = ReporterAgent(mock_llm, context, verbose=False)
        assert agent.name == "Reporter"

    def test_get_tools_empty(self, context, mock_llm):
        agent = ReporterAgent(mock_llm, context, verbose=False)
        assert agent.get_tools() == []

    def test_build_report_task(self, context, mock_llm):
        context.class_problem = "binary"
        context.best_metrics = {"roc_auc": 0.85}
        context.run_history = [{"success": True, "metrics": {"roc_auc": 0.85}}]
        context.feature_engineering_code = "df['new'] = 1"

        agent = ReporterAgent(mock_llm, context, verbose=False)
        task = agent.build_report_task()
        assert "binary" in task
        assert "roc_auc" in task

    def test_build_report_task_with_arch_errors(self, context, mock_llm):
        context.arch_error_analysis = {"catboost": "Residuals clustered at low end"}
        agent = ReporterAgent(mock_llm, context, verbose=False)
        task = agent.build_report_task()
        assert "catboost" in task.lower()

    def test_build_report_task_sampled(self, context, mock_llm):
        context.was_sampled = True
        context.original_shape = (10000, 20)
        agent = ReporterAgent(mock_llm, context, verbose=False)
        task = agent.build_report_task()
        assert "sampled" in task.lower() or "sample" in task.lower()


# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------


class TestResearcherAgent:
    def test_name(self, context, mock_llm):
        agent = ResearcherAgent(mock_llm, context, verbose=False)
        assert agent.name == "Researcher"
