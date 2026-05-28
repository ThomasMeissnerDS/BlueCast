"""Tests for the Orchestrator class — full coverage with mocked LLM."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.config import AIConfig
from bluecast.ai.orchestrator import Orchestrator
from bluecast.ai.result import BlueCastAIResult
from bluecast.tests.test_ai_mock_provider import (
    MockLLMProvider,
    make_planner_response,
    make_text_response,
    make_tool_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """Small synthetic dataset for testing."""
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
def regression_df():
    """Small synthetic regression dataset."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 50),
            "x2": rng.normal(5, 2, 50),
            "target": rng.normal(10, 3, 50),
        }
    )


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def config():
    return AIConfig(
        api_key="test",
        provider="gemini",
        mode="fast",
        verbose=False,
        max_iterations=1,
        critique_max_rounds=0,
    )


@pytest.fixture
def orchestrator(sample_df, mock_llm, config):
    return Orchestrator(
        llm=mock_llm,
        config=config,
        df=sample_df,
        target_col="target",
        prompt="Test classification",
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_creates_agents(self, orchestrator):
        assert orchestrator.planner is not None
        assert orchestrator.analyst is not None
        assert orchestrator.engineer is not None
        assert orchestrator.builder is not None
        assert orchestrator.evaluator is not None
        assert orchestrator.researcher is not None
        assert orchestrator.reporter is not None
        assert orchestrator.arch_engineer is not None

    def test_context_initialized(self, orchestrator, sample_df):
        ctx = orchestrator.context
        assert ctx.df_train is not None
        assert ctx.target_col == "target"
        assert ctx.mode == "fast"
        assert ctx.user_prompt == "Test classification"
        assert ctx.original_shape == sample_df.shape


# ---------------------------------------------------------------------------
# Context file loading
# ---------------------------------------------------------------------------


class TestContextFiles:
    def test_load_txt_file(self, sample_df, mock_llm):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Domain context: medical data")
            path = f.name
        try:
            config = AIConfig(
                api_key="test", mode="fast", verbose=False, context_files=[path]
            )
            orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
            assert len(orch.context.context_file_contents) == 1
            assert "medical" in orch.context.context_file_contents[0]
        finally:
            os.unlink(path)

    def test_load_csv_file(self, sample_df, mock_llm):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            path = f.name
        try:
            config = AIConfig(
                api_key="test", mode="fast", verbose=False, context_files=[path]
            )
            orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
            assert len(orch.context.context_file_contents) == 1
            assert "col1" in orch.context.context_file_contents[0]
        finally:
            os.unlink(path)

    def test_load_missing_file(self, sample_df, mock_llm):
        config = AIConfig(
            api_key="test",
            mode="fast",
            verbose=False,
            context_files=["/nonexistent/file.txt"],
        )
        # Should not raise
        orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
        assert len(orch.context.context_file_contents) == 0

    def test_extract_pdf_import_error(self):
        with patch.dict("sys.modules", {"pypdf": None}):
            with pytest.raises(ImportError, match="pypdf"):
                Orchestrator._extract_pdf_text("/fake.pdf")

    def test_extract_docx_import_error(self):
        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(ImportError, match="python-docx"):
                Orchestrator._extract_docx_text("/fake.docx")


# ---------------------------------------------------------------------------
# Smart sampling
# ---------------------------------------------------------------------------


class TestSmartSampling:
    def test_no_sample_needed(self, orchestrator):
        orchestrator._apply_smart_sampling()
        assert orchestrator.context.df_sample is None
        assert orchestrator.context.was_sampled is False

    def test_row_sampling_classification(self, mock_llm):
        rng = np.random.default_rng(42)
        big_df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 1000),
                "target": rng.choice([0, 1], 1000),
            }
        )
        config = AIConfig(
            api_key="test",
            mode="fast",
            verbose=False,
            max_rows_for_agents=100,
        )
        orch = Orchestrator(mock_llm, config, big_df, "target", "test")
        orch._apply_smart_sampling()
        assert orch.context.was_sampled is True
        assert len(orch.context.df_sample) <= 100

    def test_row_sampling_regression(self, mock_llm):
        rng = np.random.default_rng(42)
        big_df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 1000),
                "target": rng.normal(0, 1, 1000),
            }
        )
        config = AIConfig(
            api_key="test",
            mode="fast",
            verbose=False,
            max_rows_for_agents=100,
        )
        orch = Orchestrator(mock_llm, config, big_df, "target", "test")
        orch._apply_smart_sampling()
        assert orch.context.was_sampled is True
        assert len(orch.context.df_sample) == 100

    def test_column_warning(self, mock_llm):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, 10) for i in range(250)})
        df["target"] = rng.choice([0, 1], 10)
        config = AIConfig(
            api_key="test",
            mode="fast",
            verbose=False,
            max_columns_for_agents=10,
        )
        orch = Orchestrator(mock_llm, config, df, "target", "test")
        orch._apply_smart_sampling()
        assert len(orch.context.data_warnings) > 0
        assert "columns" in orch.context.data_warnings[0].lower()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_checkpoint_save_load_clear(self, sample_df, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AIConfig(
                api_key="test",
                mode="fast",
                verbose=False,
                checkpoint_dir=tmpdir,
            )
            orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
            orch._save_checkpoint("plan")
            assert "plan" in orch.context.completed_steps

            # Create new orchestrator and load checkpoint
            orch2 = Orchestrator(mock_llm, config, sample_df, "target", "test")
            loaded = orch2._load_checkpoint()
            assert loaded is True
            assert "plan" in orch2.context.completed_steps

            # Clear checkpoint
            orch2._clear_checkpoint()
            path = orch2._checkpoint_path()
            assert not os.path.exists(path)

    def test_checkpoint_no_dir(self, orchestrator):
        # No checkpoint_dir → nothing happens
        orchestrator._save_checkpoint("test")
        assert orchestrator._checkpoint_path() is None

    def test_is_step_done(self, orchestrator):
        assert orchestrator._is_step_done("plan") is False
        orchestrator.context.completed_steps.append("plan")
        assert orchestrator._is_step_done("plan") is True


# ---------------------------------------------------------------------------
# Critique rounds
# ---------------------------------------------------------------------------


class TestCritiqueRounds:
    def test_mode_defaults(self, sample_df, mock_llm):
        for mode, expected in [
            ("fast", 0),
            ("balanced", 1),
            ("precise", 2),
            ("ultimate", 5),
        ]:
            config = AIConfig(api_key="test", mode=mode, verbose=False)
            orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
            assert orch._get_critique_rounds() == expected

    def test_explicit_override(self, sample_df, mock_llm):
        config = AIConfig(
            api_key="test", mode="ultimate", verbose=False, critique_max_rounds=3
        )
        orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
        assert orch._get_critique_rounds() == 3

    def test_explicit_zero(self, sample_df, mock_llm):
        config = AIConfig(
            api_key="test", mode="precise", verbose=False, critique_max_rounds=0
        )
        orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
        assert orch._get_critique_rounds() == 0


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------


class TestStepPlan:
    def test_step_plan_parses_json(self, orchestrator, mock_llm):
        mock_llm.enqueue_response(
            make_planner_response(class_problem="binary", max_iterations=2)
        )
        plan = orchestrator._step_plan()
        assert plan["class_problem"] == "binary"
        assert orchestrator.context.class_problem == "binary"

    def test_step_plan_fallback_on_error(self, orchestrator, mock_llm):
        # Simulate LLM returning garbage
        mock_llm.enqueue_response(make_text_response("not valid json at all!!!"))
        plan = orchestrator._step_plan()
        # Should fall back to default plan
        assert "class_problem" in plan

    def test_step_plan_api_exception(self, sample_df):
        """When the LLM raises, the planner catches and returns default plan."""

        class FailingLLM(MockLLMProvider):
            def chat(self, messages, tools=None):
                raise ConnectionError("API down")

        config = AIConfig(api_key="test", mode="fast", verbose=False)
        orch = Orchestrator(FailingLLM(), config, sample_df, "target", "test")
        plan = orch._step_plan()
        assert plan["class_problem"] == "binary"


class TestStepAnalyze:
    def test_step_analyze(self, orchestrator, mock_llm):
        mock_llm.enqueue_response(
            make_text_response(
                "Dataset analysis: found leakage in col1, missing values detected, "
                "class imbalance noted, 5 outlier rows."
            )
        )
        orchestrator._step_analyze()
        assert orchestrator.context.data_profile is not None
        assert "summary" in orchestrator.context.data_profile
        # Should extract warnings from keywords
        assert any("leakage" in w for w in orchestrator.context.data_warnings)
        assert any("missing" in w for w in orchestrator.context.data_warnings)

    def test_step_analyze_imputation_extraction(self, orchestrator, mock_llm):
        mock_llm.enqueue_response(
            make_text_response(
                "Analysis complete.\n"
                "## Detected Sentinel Values\n"
                "Column x5 uses 999.0 as missing.\n"
                "## Next Section\n"
                "More content."
            )
        )
        orchestrator._step_analyze()
        assert orchestrator.context.imputation_recommendations is not None
        assert "Sentinel" in orchestrator.context.imputation_recommendations


class TestStepResearch:
    def test_step_research(self, orchestrator, mock_llm):
        mock_llm.enqueue_response(make_text_response("Found relevant papers."))
        orchestrator._step_research(["query1", "query2"])
        assert orchestrator.context.web_research == "Found relevant papers."


class TestReconstructPlan:
    def test_reconstruct_from_log(self, orchestrator):
        plan = {"class_problem": "regression", "max_iterations": 3}
        orchestrator.context.log(
            "Orchestrator",
            "plan",
            event_type="plan",
            metadata={"plan": plan},
        )
        result = orchestrator._reconstruct_plan()
        assert result["class_problem"] == "regression"

    def test_reconstruct_fallback(self, orchestrator):
        result = orchestrator._reconstruct_plan()
        assert result["class_problem"] == "binary"  # default


# ---------------------------------------------------------------------------
# Build loop
# ---------------------------------------------------------------------------


class TestStepBuildLoop:
    def test_build_loop_fast_mode(self, orchestrator, mock_llm):
        """Fast mode: one iteration, no evaluator."""
        # Builder response
        mock_llm.enqueue_response(
            make_tool_response(
                "build_and_run_pipeline",
                {"class_problem": "binary", "use_cv": True},
            )
        )
        # After tool call, builder returns text
        mock_llm.enqueue_response(make_text_response("Pipeline built successfully."))

        plan = {
            "class_problem": "binary",
            "use_cv": True,
            "ensemble_strategy": "stacking",
            "n_folds": 3,
        }

        with patch(
            "bluecast.ai.agents.pipeline_builder.tool_build_and_run_pipeline"
        ) as mock_build:
            mock_build.return_value = {
                "success": True,
                "metrics": {"roc_auc": 0.85},
                "config_used": {"class_problem": "binary"},
                "pipeline": MagicMock(),
                "error": None,
            }
            orchestrator._step_build_loop(plan, max_iterations=1)

        assert len(orchestrator.context.run_history) >= 1


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


class TestBuildArchConfig:
    def test_basic_config(self, orchestrator):
        plan = {
            "class_problem": "binary",
            "use_cv": True,
            "ensemble_strategy": "stacking",
            "n_folds": 5,
            "tuning_rounds": 100,
            "tuning_max_runtime": 120,
        }
        config = orchestrator._build_arch_config(plan, "catboost")
        assert config["class_problem"] == "binary"
        assert config["use_cv"] is True
        assert config["tuning_rounds"] == 100

    def test_linear_config(self, orchestrator):
        plan = {"class_problem": "regression", "tuning_rounds": 100}
        config = orchestrator._build_arch_config(plan, "linear")
        assert config["tuning_rounds"] == 1
        assert config["tuning_max_runtime"] == 30
        assert config["cat_encoding_via_ml_algorithm"] is False

    def test_regression_metric_propagation(self, orchestrator):
        plan = {
            "class_problem": "regression",
            "regression_eval_metric": "mae",
        }
        config = orchestrator._build_arch_config(plan, "histgb")
        assert config["regression_eval_metric"] == "mae"

    def test_classification_metric_not_for_regression(self, orchestrator):
        plan = {
            "class_problem": "regression",
            "classification_eval_metric": "roc_auc",
        }
        config = orchestrator._build_arch_config(plan, "catboost")
        assert "classification_eval_metric" not in config


class TestEnforceConfigConstraints:
    def test_clamps_tuning_rounds(self, orchestrator):
        config = {"tuning_rounds": 500, "tuning_max_runtime": 9999}
        original = {"tuning_rounds": 200, "tuning_max_runtime": 1800}
        result = orchestrator._enforce_config_constraints(config, original)
        assert result["tuning_rounds"] == 200
        assert result["tuning_max_runtime"] == 1800

    def test_disables_feature_selection(self, orchestrator):
        config = {"enable_feature_selection": True}
        result = orchestrator._enforce_config_constraints(config, {})
        assert result["enable_feature_selection"] is False

    def test_nn_max_iter_capped(self, orchestrator):
        config = {"nn_max_iter": 5000}
        result = orchestrator._enforce_config_constraints(config, {})
        assert result["nn_max_iter"] == 1000

    def test_linear_cat_encoding(self, orchestrator):
        config = {}
        result = orchestrator._enforce_config_constraints(
            config, {}, arch_name="linear"
        )
        assert result["cat_encoding_via_ml_algorithm"] is False


# ---------------------------------------------------------------------------
# Full run (fast mode, everything mocked)
# ---------------------------------------------------------------------------


class TestFullRun:
    def test_run_fast_mode(self, sample_df):
        mock_llm = MockLLMProvider()

        # Plan response
        mock_llm.enqueue_response(
            make_planner_response(
                class_problem="binary", needs_fe=False, max_iterations=1
            )
        )
        # Analyst response
        mock_llm.enqueue_response(make_text_response("Data looks clean. No issues."))
        # Builder tool call
        mock_llm.enqueue_response(
            make_tool_response("build_and_run_pipeline", {"class_problem": "binary"})
        )
        # Builder text response after tool
        mock_llm.enqueue_response(make_text_response("Pipeline complete."))
        # Reporter response
        mock_llm.enqueue_response(make_text_response("# Report\nGreat results."))

        config = AIConfig(
            api_key="test",
            mode="fast",
            verbose=False,
            max_iterations=1,
            critique_max_rounds=0,
        )
        orch = Orchestrator(mock_llm, config, sample_df, "target", "test")

        with patch(
            "bluecast.ai.agents.pipeline_builder.tool_build_and_run_pipeline"
        ) as mock_build:
            mock_build.return_value = {
                "success": True,
                "metrics": {"roc_auc": 0.85},
                "config_used": {"class_problem": "binary"},
                "pipeline": MagicMock(),
                "error": None,
            }
            result = orch.run()

        assert isinstance(result, BlueCastAIResult)
        assert result.class_problem == "binary"

    def test_run_resumes_from_checkpoint(self, sample_df):
        """Run completes even when steps are pre-completed in context."""
        mock_llm = MockLLMProvider()
        mock_llm.enqueue_response(make_text_response("# Report\nDone."))

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AIConfig(
                api_key="test",
                mode="fast",
                verbose=False,
                max_iterations=1,
                checkpoint_dir=tmpdir,
                critique_max_rounds=0,
            )

            # First run: complete sampling and plan
            orch1 = Orchestrator(mock_llm, config, sample_df, "target", "test")
            orch1.context.completed_steps = [
                "sampling",
                "plan",
                "analyze",
                "build_loop",
            ]
            # Save a plan in the log so reconstruct works
            orch1.context.log(
                "Orchestrator",
                "plan",
                event_type="plan",
                metadata={
                    "plan": {
                        "class_problem": "binary",
                        "needs_feature_engineering": False,
                    }
                },
            )
            orch1._save_checkpoint("build_loop")

            # Second run: should skip to report
            mock_llm2 = MockLLMProvider()
            mock_llm2.enqueue_response(make_text_response("# Final Report"))
            orch2 = Orchestrator(mock_llm2, config, sample_df, "target", "test")

            with patch(
                "bluecast.ai.agents.pipeline_builder.tool_build_and_run_pipeline"
            ) as mock_build:
                mock_build.return_value = {
                    "success": True,
                    "metrics": {},
                    "config_used": {},
                    "pipeline": None,
                    "error": None,
                }
                result = orch2.run()

            assert isinstance(result, BlueCastAIResult)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


class TestAssembleResult:
    def test_assemble_result(self, orchestrator):
        orchestrator.context.class_problem = "binary"
        orchestrator.context.best_metrics = {"roc_auc": 0.9}
        orchestrator.context.best_pipeline = MagicMock()
        orchestrator.context.pipeline_code = "pipeline.fit(df)"
        orchestrator.context.feature_engineering_code = "df['new'] = 1"
        orchestrator.context.report_markdown = "# Report"
        orchestrator.context.run_history = [
            {"success": True, "metrics": {"roc_auc": 0.9}}
        ]

        result = orchestrator._assemble_result()
        assert isinstance(result, BlueCastAIResult)
        assert result.class_problem == "binary"
        assert result.metrics == {"roc_auc": 0.9}
        assert result.pipeline_code == "pipeline.fit(df)"
        assert result.report_markdown == "# Report"

    def test_run_ultimate_mode(self, mock_llm, sample_df, tmpdir):
        config = AIConfig(
            api_key="test",
            mode="ultimate",
            verbose=False,
            checkpoint_dir=str(tmpdir),
            global_tuning_budget=0,
        )
        orch = Orchestrator(mock_llm, config, sample_df, "target", "test")

        # Mock out the steps so it doesn't do LLM calls
        orch._step_plan = MagicMock(
            return_value={
                "class_problem": "binary",
                "needs_feature_engineering": False,
                "max_iterations": 1,
                "ensemble_strategy": "nelder_mead",
            }
        )
        orch._step_analyze = MagicMock()
        orch._step_research = MagicMock()
        orch._step_ultimate_build_loop = MagicMock()

        orch.run()

        orch._step_plan.assert_called_once()
        orch._step_analyze.assert_called_once()
        orch._step_ultimate_build_loop.assert_called_once()


def test_step_ultimate_build_loop(mock_llm, sample_df, tmpdir):
    config = AIConfig(
        api_key="test",
        mode="ultimate",
        verbose=False,
        checkpoint_dir=str(tmpdir),
    )
    orch = Orchestrator(mock_llm, config, sample_df, "target", "test")

    # Mock the internal architecture building
    orch._build_single_arch = MagicMock()
    orch._build_single_arch.return_value = {
        "success": True,
        "metrics": {"mae": 0.5},
        "config_used": {},
        "pipeline": MagicMock(),
        "model": MagicMock(),
        "oof_preds": [0.1, 0.9],
        "val_score": 0.8,
    }

    plan = {"class_problem": "binary"}
    orch._step_ultimate_build_loop(plan)

    # Check that architectures were built
    assert orch._build_single_arch.call_count > 0

def test_create_arch_fe_task(mock_llm, sample_df, tmpdir):
    config = AIConfig(api_key="test", mode="ultimate", verbose=False)
    orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
    
    # test iteration 0
    task0 = orch._create_arch_fe_task("xgboost", "XGBoost", 0, 2, None)
    assert "FIRST iteration" in task0
    
    # test iteration 1 with inheritance
    task1 = orch._create_arch_fe_task("xgboost", "XGBoost", 1, 3, ["df['a'] = 1"])
    assert "improve on the best so far" in task1
    assert "df['a'] = 1" in task1
    
    # test final iteration
    task2 = orch._create_arch_fe_task("xgboost", "XGBoost", 2, 3, ["df['a'] = 1"])
    assert "FINAL iteration" in task2

@patch("bluecast.ai.tools.tool_build_and_run_pipeline")
def test_build_single_arch(mock_tool, mock_llm, sample_df, tmpdir):
    config = AIConfig(api_key="test", mode="ultimate", verbose=False)
    orch = Orchestrator(mock_llm, config, sample_df, "target", "test")
    
    mock_tool.return_value = {
        "success": True,
        "metrics": {"roc_auc": 0.8},
        "oof_preds": [0.1, 0.9],
        "pipeline": MagicMock(),
    }
    
    plan = {"class_problem": "binary"}
    plan_config = orch._build_arch_config(plan, "xgboost")
    res = orch._build_single_arch(plan_config, "xgboost", use_xgboost=True)
    
    assert res["success"] is True
    assert mock_tool.call_count == 1
    
    # Test linear fallback to linear architecture
    plan_config_linear = orch._build_arch_config(plan, "linear")
    res_linear = orch._build_single_arch(plan_config_linear, "linear", use_xgboost=False)
    assert res_linear["success"] is True
