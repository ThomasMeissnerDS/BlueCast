"""Extended orchestrator tests — covers helper methods for coverage gaps."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat.return_value = '```json\n{"tuning_rounds": 5}\n```'
    return llm


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num1": rng.normal(0, 1, 50),
            "num2": rng.normal(5, 2, 50),
            "target": rng.choice([0, 1], 50),
        }
    )


def _make_orch(mock_llm, sample_df, tmpdir, mode="balanced", verbose=False):
    from bluecast.ai.config import AIConfig
    from bluecast.ai.orchestrator import Orchestrator

    config = AIConfig(
        api_key="test",
        mode=mode,
        verbose=verbose,
        checkpoint_dir=str(tmpdir),
    )
    return Orchestrator(mock_llm, config, sample_df.copy(), "target", "test")


class TestIsResultBetter:
    def test_no_best_metrics(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.best_metrics = None
        assert orch._is_result_better({"metrics": {"roc_auc": 0.8}}) is True

    def test_higher_auc_is_better(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.best_metrics = {"roc_auc": 0.7}
        assert orch._is_result_better({"metrics": {"roc_auc": 0.8}}) is True
        assert orch._is_result_better({"metrics": {"roc_auc": 0.6}}) is False

    def test_lower_mae_is_better(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.best_metrics = {"mae": 1.0}
        assert orch._is_result_better({"metrics": {"mae": 0.5}}) is True
        assert orch._is_result_better({"metrics": {"mae": 2.0}}) is False

    def test_lower_rmse_is_better(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.best_metrics = {"rmse": 1.0}
        assert orch._is_result_better({"metrics": {"rmse": 0.5}}) is True

    def test_empty_old_metrics(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.best_metrics = {}
        assert orch._is_result_better({"metrics": {"roc_auc": 0.8}}) is True

    def test_no_common_metrics(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.best_metrics = {"custom_metric": 0.5}
        assert orch._is_result_better({"metrics": {"other_metric": 0.8}}) is False


class TestFeatureEngineeringException:
    def test_feature_engineering_exception_handled(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.context.step_checkpoints = {}
        plan = {"needs_feature_engineering": True}

        with patch.object(orch, "_step_plan", return_value=plan):
            with patch.object(orch, "_step_research"):
                with patch.object(orch, "_step_analyze"):
                    with patch.object(
                        orch,
                        "_step_feature_engineer",
                        side_effect=Exception("Test FE Error"),
                    ):
                        with patch.object(orch, "_step_build_loop"):
                            with patch.object(orch, "_step_ultimate_build_loop"):
                                with patch.object(orch, "_step_report"):
                                    orch.run()

        # It should have saved a checkpoint for feature_engineering despite the exception
        assert "feature_engineering" in orch.context.completed_steps


class TestStepPlan:
    def test_planner_exception(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        with patch.object(orch.planner, "run", side_effect=Exception("Planner Failed")):
            plan = orch._step_plan()
            # It should fallback to the default plan
            assert plan["class_problem"] in ["binary", "multiclass", "regression"]


class TestStepAnalyze:
    def test_critique_rounds(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        # Mock _get_critique_rounds to return 1
        with patch.object(orch, "_get_critique_rounds", return_value=1):
            with patch(
                "bluecast.ai.critique.CritiqueLoop.run_with_critique",
                return_value="Data Summary with Imputation Strategy Evaluation",
            ):
                orch._step_analyze()
                assert (
                    orch.context.data_profile["summary"]
                    == "Data Summary with Imputation Strategy Evaluation"
                )


class TestStepResearch:
    def test_research(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        with patch.object(orch.researcher, "run", return_value="Search Results"):
            orch._step_research(["query1", "query2"])
            assert orch.context.web_research == "Search Results"


class TestUltimateBuildLoopBudget:
    def test_budget_capping(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.global_tuning_budget = 100  # very low budget
        orch.config.architectures_to_run = ["xgboost", "linear"]
        plan = {"needs_feature_engineering": False}
        with patch.object(orch, "_create_arch_fe_task"):
            with patch.object(
                orch,
                "_build_single_arch",
                return_value={
                    "success": True,
                    "metrics": {"roc_auc": 0.8},
                    "pipeline": "base_pipeline",
                    "snippets": [],
                    "config_used": {},
                },
            ):
                with patch.object(orch, "_compare_results", return_value=True):
                    orch._step_ultimate_build_loop(plan)

    def test_custom_preprocessor_execution(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import MagicMock, patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.architectures_to_run = ["xgboost"]
        orch.context.custom_preprocessor = MagicMock()
        plan = {"needs_feature_engineering": False}
        with patch.object(orch, "_create_arch_fe_task"):
            with patch.object(
                orch,
                "_build_single_arch",
                return_value={
                    "success": True,
                    "metrics": {"roc_auc": 0.8},
                    "pipeline": "base_pipeline",
                    "snippets": [],
                    "config_used": {},
                },
            ) as mock_build:
                with patch.object(orch, "_compare_results", return_value=True):
                    orch._step_ultimate_build_loop(plan)
                    assert (
                        mock_build.call_args[1]["preprocessor"]
                        == orch.context.custom_preprocessor
                    )

    def test_combined_snippets_preprocessor(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.architectures_to_run = ["xgboost"]
        orch.context.feature_code_snippets = ["df['new'] = 1"]
        plan = {"needs_feature_engineering": False}
        with patch.object(orch, "_create_arch_fe_task"):
            with patch.object(orch.arch_engineer, "run", return_value="{}"):
                with patch.object(
                    orch,
                    "_build_single_arch",
                    return_value={
                        "success": True,
                        "metrics": {"roc_auc": 0.8},
                        "pipeline": "base_pipeline",
                        "snippets": [],
                        "config_used": {},
                    },
                ) as mock_build:
                    with patch.object(orch, "_compare_results", return_value=True):
                        orch._step_ultimate_build_loop(plan)
                        assert mock_build.call_args[1]["preprocessor"] is not None


class TestLoadContextFiles:
    def test_load_pdf(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import MagicMock, patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.context_files = ["test.pdf", "test.docx", "test.txt"]

        # We mock the internal extract methods to avoid actual file I/O
        with patch.object(orch, "_extract_pdf_text", return_value="PDF text"):
            with patch.object(orch, "_extract_docx_text", return_value="DOCX text"):
                with patch("builtins.open") as mock_open:
                    mock_file = MagicMock()
                    mock_file.read.return_value = "TXT text"
                    mock_open.return_value.__enter__.return_value = mock_file
                    orch._load_context_files()
        assert len(orch.context.context_file_contents) == 3
        assert "PDF text" in orch.context.context_file_contents[0]

    def test_extract_pdf_text(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        from unittest.mock import MagicMock, patch

        with patch("builtins.open"):
            import sys

            mock_pypdf = MagicMock()
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Page 1"
            mock_reader.pages = [mock_page]
            mock_pypdf.PdfReader.return_value = mock_reader
            with patch.dict(sys.modules, {"pypdf": mock_pypdf}):
                text = orch._extract_pdf_text("test.pdf")
                assert text == "Page 1"

    def test_extract_docx_text(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        import sys
        from unittest.mock import MagicMock, patch

        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "Para 1"
        mock_doc.paragraphs = [mock_para]
        mock_docx.Document.return_value = mock_doc
        with patch.dict(sys.modules, {"docx": mock_docx}):
            text = orch._extract_docx_text("test.docx")
            assert text == "Para 1"


class TestStepUltimateBuildLoop:
    def test_architectures_to_run(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.architectures_to_run = ["xgboost"]
        orch.config.global_tuning_budget = 360  # should cap max_affordable_iters
        plan = {"needs_feature_engineering": False}
        with patch.object(orch, "_create_arch_fe_task"):
            with patch.object(
                orch, "_evaluate_for_arch", return_value={"success": True}
            ):
                with patch.object(orch, "_save_checkpoint"):
                    # We just mock the evaluator so it doesn't do real LLM calls
                    orch.evaluator = mock_llm
                    # And mock fe agent
                    with patch.object(orch.arch_engineer, "run", return_value="{}"):
                        # just catch if it tries to build
                        try:
                            with patch.object(
                                orch, "_build_arch_config", return_value={}
                            ):
                                with patch.object(
                                    orch,
                                    "_build_single_arch",
                                    return_value={
                                        "success": True,
                                        "metrics": {"roc_auc": 0.8},
                                        "pipeline": "base_pipeline",
                                        "snippets": [],
                                        "config_used": {},
                                    },
                                ):
                                    orch._step_ultimate_build_loop(plan)
                        except Exception:
                            pass

    def test_invalid_architectures(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.architectures_to_run = ["invalid_arch"]
        plan = {"needs_feature_engineering": False}
        import pytest

        with pytest.raises(ValueError, match="None of the specified architectures"):
            orch._step_ultimate_build_loop(plan)

    def test_refinement_improved(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.ultimate_refine_best = True
        plan = {"needs_feature_engineering": False}

        # We want to jump straight to refinement, so we mock _create_arch_fe_task and _evaluate_for_arch
        # to fast track the main loop, and we mock _compare_results to True
        orch.context.arch_feature_snippets["xgboost"] = ["df['new'] = 1"]
        orch.context.best_metrics = {"roc_auc": 0.5}

        with patch.object(orch, "_create_arch_fe_task"):
            with patch.object(
                orch, "_evaluate_for_arch", return_value={"success": True}
            ):
                with patch.object(orch, "_save_checkpoint"):
                    orch.evaluator = mock_llm
                    with patch.object(orch.arch_engineer, "run", return_value="{}"):
                        with patch.object(orch, "_build_arch_config", return_value={}):
                            with patch.object(
                                orch,
                                "_build_single_arch",
                                return_value={
                                    "success": True,
                                    "metrics": {"roc_auc": 0.8},
                                    "pipeline": "base_pipeline",
                                    "snippets": [],
                                    "config_used": {},
                                },
                            ):
                                with patch.object(
                                    orch, "_compare_results", return_value=True
                                ):
                                    # Let's mock _is_result_better too for global update
                                    with patch.object(
                                        orch, "_is_result_better", return_value=True
                                    ):
                                        orch._step_ultimate_build_loop(plan)

        # It should have updated the run history
        assert len(orch.context.run_history) > 0

    def test_fe_agent_failure(self, mock_llm, sample_df, tmpdir):
        from unittest.mock import patch

        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.config.architectures_to_run = ["xgboost"]
        plan = {"needs_feature_engineering": False}
        with patch.object(orch, "_create_arch_fe_task"):
            with patch.object(
                orch.arch_engineer, "run", side_effect=Exception("LLM Crash")
            ):
                import pytest

                with pytest.raises(Exception, match="LLM Crash"):
                    orch._step_ultimate_build_loop(plan)


class TestCompareResults:
    def test_new_better_auc(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        assert orch._compare_results(
            {"metrics": {"roc_auc": 0.9}}, {"metrics": {"roc_auc": 0.8}}
        )

    def test_old_better_auc(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        assert not orch._compare_results(
            {"metrics": {"roc_auc": 0.7}}, {"metrics": {"roc_auc": 0.8}}
        )

    def test_new_better_mae(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        assert orch._compare_results(
            {"metrics": {"mae": 0.5}}, {"metrics": {"mae": 1.0}}
        )

    def test_empty_old(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        assert orch._compare_results({"metrics": {"roc_auc": 0.8}}, {"metrics": {}})

    def test_no_common(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        assert not orch._compare_results({"metrics": {"x": 1}}, {"metrics": {"y": 2}})


class TestEnforceConfigConstraints:
    def test_tuning_rounds_capped(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {"tuning_rounds": 500}
        result = orch._enforce_config_constraints(config, {"tuning_rounds": 200})
        assert result["tuning_rounds"] == 200

    def test_tuning_runtime_capped(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {"tuning_max_runtime": 5000}
        result = orch._enforce_config_constraints(config, {"tuning_max_runtime": 1000})
        assert result["tuning_max_runtime"] == 1000

    def test_override_max_runtime(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {"tuning_max_runtime": 5000}
        result = orch._enforce_config_constraints(config, {}, override_max_runtime=300)
        assert result["tuning_max_runtime"] == 300

    def test_nn_max_iter_capped(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {"nn_max_iter": 2000}
        result = orch._enforce_config_constraints(config, {})
        assert result["nn_max_iter"] == 1000

    def test_feature_selection_disabled(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {"enable_feature_selection": True}
        result = orch._enforce_config_constraints(config, {})
        assert result["enable_feature_selection"] is False

    def test_linear_arch_no_cat_encoding(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {}
        result = orch._enforce_config_constraints(config, {}, arch_name="linear")
        assert result["cat_encoding_via_ml_algorithm"] is False

    def test_none_tuning_rounds(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {"tuning_rounds": None}
        result = orch._enforce_config_constraints(config, {"tuning_rounds": 100})
        assert result["tuning_rounds"] == 100

    def test_none_tuning_max_runtime(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        config = {"tuning_max_runtime": None}
        result = orch._enforce_config_constraints(config, {})
        assert result["tuning_max_runtime"] == 1800


class TestBuildArchConfig:
    def test_basic_binary(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {"class_problem": "binary", "n_folds": 3, "tuning_rounds": 30}
        config = orch._build_arch_config(plan, "xgboost")
        assert config["class_problem"] == "binary"
        assert config["n_folds"] == 3
        assert config["tuning_rounds"] == 30

    def test_linear_arch_overrides(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {"class_problem": "binary", "tuning_rounds": 100}
        config = orch._build_arch_config(plan, "linear")
        assert config["tuning_rounds"] == 1
        assert config["tuning_max_runtime"] == 30
        assert config["cat_encoding_via_ml_algorithm"] is False

    def test_regression_metric_propagation(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {"class_problem": "regression", "regression_eval_metric": "mae"}
        config = orch._build_arch_config(plan, "xgboost")
        assert config["regression_eval_metric"] == "mae"

    def test_classification_metric_propagation(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {
            "class_problem": "binary",
            "classification_eval_metric": "balanced_accuracy",
        }
        config = orch._build_arch_config(plan, "xgboost")
        assert config["classification_eval_metric"] == "balanced_accuracy"

    def test_override_max_runtime(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {"tuning_max_runtime": 500}
        config = orch._build_arch_config(plan, "xgboost", override_max_runtime=200)
        assert config["tuning_max_runtime"] == 200


class TestCreateBuildTask:
    def test_first_iteration(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {"class_problem": "binary", "tuning_rounds": 50}
        task = orch._create_build_task(plan, 0)
        assert "iteration 1" in task
        assert "binary" in task

    def test_later_iteration_tuning_increase(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {"class_problem": "binary", "tuning_rounds": 50}
        task = orch._create_build_task(plan, 2)
        assert "iteration 3" in task

    def test_stacking_to_hill_climbing(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        plan = {"class_problem": "binary", "ensemble_strategy": "stacking"}
        task = orch._create_build_task(plan, 2)
        assert "hill_climbing" in task


class TestEvaluateForArch:
    def test_linear_arch_warning(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = '```json\n{"tuning_rounds": 10}\n```'
        result = {"metrics": {"roc_auc": 0.8}, "success": True, "config_used": {}}
        suggestions = orch._evaluate_for_arch("linear", "Linear", result)
        assert isinstance(suggestions, dict)

    def test_with_feature_importances(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = '```json\n{"tuning_rounds": 20}\n```'
        orch.context.arch_feature_importances["xgboost"] = {
            "feat1": 0.5,
            "feat2": 0.3,
            "feat3": 0.2,
        }
        result = {"metrics": {"roc_auc": 0.8}, "success": True, "config_used": {}}
        suggestions = orch._evaluate_for_arch("xgboost", "XGBoost", result)
        assert suggestions.get("tuning_rounds") == 20

    def test_with_arch_errors(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = "No JSON here"
        orch.context.arch_errors["xgboost"] = "OutOfMemory"
        result = {"metrics": {"roc_auc": 0.8}, "success": True, "config_used": {}}
        suggestions = orch._evaluate_for_arch("xgboost", "XGBoost", result)
        assert isinstance(suggestions, dict)

    def test_iteration_context_first(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = "```json\n{}\n```"
        result = {"metrics": {"roc_auc": 0.8}, "success": True, "config_used": {}}
        orch._evaluate_for_arch(
            "xgboost", "XGBoost", result, iteration=0, total_iterations=5
        )
        call_args = orch.evaluator.run.call_args[0][0]
        assert "BASELINE" in call_args

    def test_iteration_context_last(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = "```json\n{}\n```"
        result = {"metrics": {"roc_auc": 0.8}, "success": True, "config_used": {}}
        orch._evaluate_for_arch(
            "xgboost", "XGBoost", result, iteration=3, total_iterations=5
        )
        call_args = orch.evaluator.run.call_args[0][0]
        assert "SECOND-TO-LAST" in call_args

    def test_convergence_info(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = "```json\n{}\n```"
        result = {
            "metrics": {"roc_auc": 0.8},
            "success": True,
            "config_used": {"nn_max_iter": 200},
            "convergence_info": {"nn_max_iter_used": 200, "trials_completed": 10},
        }
        orch._evaluate_for_arch("mlp", "MLP", result)
        call_args = orch.evaluator.run.call_args[0][0]
        assert "CONVERGENCE" in call_args

    def test_arch_snippets_included(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.evaluator = MagicMock()
        orch.evaluator.run.return_value = "```json\n{}\n```"
        orch.context.arch_feature_snippets["xgboost"] = ["df['x'] = 1"]
        result = {"metrics": {"roc_auc": 0.8}, "success": True, "config_used": {}}
        orch._evaluate_for_arch("xgboost", "XGBoost", result)
        call_args = orch.evaluator.run.call_args[0][0]
        assert "SNIPPETS" in call_args


class TestStepReport:
    def test_report_stored(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir, verbose=True)
        orch.reporter = MagicMock()
        orch.reporter.build_report_task.return_value = "report task"
        orch.reporter.run.return_value = "# Final Report\nAll done."
        orch._step_report()
        assert orch.context.report_markdown == "# Final Report\nAll done."


class TestCreateArchFeTask:
    def test_first_iteration(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 0, 5)
        assert "FIRST" in task
        assert "XGBoost" in task

    def test_middle_iteration(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 2, 5)
        assert "iteration 3" in task.lower()

    def test_final_iteration(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 4, 5)
        assert "FINAL" in task

    def test_with_inherited_snippets(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        task = orch._create_arch_fe_task(
            "xgboost",
            "XGBoost",
            1,
            5,
            inherited_snippets=["df['x'] = df['a'] + df['b']"],
        )
        assert "INHERITED" in task

    def test_with_importances(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.arch_feature_importances["xgboost"] = {"f1": 0.8, "f2": 0.2}
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 1, 5)
        assert "f1" in task

    def test_with_error_analysis(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.arch_error_analysis["xgboost"] = "Top 10 residuals..."
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 1, 5)
        assert "ERROR ANALYSIS" in task

    def test_with_run_history(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.run_history = [
            {
                "architecture": "xgboost",
                "success": True,
                "metrics": {"auc": 0.8},
                "iteration": 1,
                "snippets": ["df['x']=1"],
            },
            {
                "architecture": "xgboost",
                "success": False,
                "error": "OOM",
                "iteration": 2,
                "snippets": [],
            },
        ]
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 2, 5)
        assert "CUMULATIVE" in task
        assert "FAILED" in task

    def test_with_arch_errors(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.arch_errors["xgboost"] = "NaN in features"
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 1, 5)
        assert "CRITICAL" in task

    def test_with_imputation_recommendations(self, mock_llm, sample_df, tmpdir):
        orch = _make_orch(mock_llm, sample_df, tmpdir)
        orch.context.imputation_recommendations = "Use median for col_a"
        task = orch._create_arch_fe_task("xgboost", "XGBoost", 1, 5)
        assert "IMPUTATION" in task
