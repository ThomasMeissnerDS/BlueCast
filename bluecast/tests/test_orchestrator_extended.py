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
