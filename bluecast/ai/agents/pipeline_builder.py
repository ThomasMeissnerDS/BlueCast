"""Pipeline builder agent: generates and runs BlueCast pipelines."""

from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition
from bluecast.ai.tools import TOOL_DEFINITIONS, tool_build_and_run_pipeline


class PipelineBuilderAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_tool_impl(
            "build_and_run_pipeline",
            self._build_wrapper,
        )

    def _build_wrapper(self, **config):
        # Always use the original training data.
        # Feature engineering is replayed via the custom preprocessor.
        df = self.context.df_train

        preprocessor = None
        if self.context.feature_code_snippets:
            from bluecast.ai.fe_preprocessor import AIFeaturePreprocessor

            preprocessor = AIFeaturePreprocessor(
                list(self.context.feature_code_snippets)
            )

        # Pop ml_model out of config so it doesn't go to TrainingConfig
        ml_model = config.pop("ml_model", None)

        result = tool_build_and_run_pipeline(
            df,
            self.context.target_col,
            config,
            custom_preprocessor=preprocessor,
            ml_model=ml_model,
        )

        run_record = {
            "success": result["success"],
            "metrics": result["metrics"],
            "config": result["config_used"],
            "error": result.get("error"),
        }
        self.context.run_history.append(run_record)

        if result["success"] and result["pipeline"] is not None:
            is_better = False
            if self.context.best_metrics is None:
                is_better = True
            else:
                new_m = result["metrics"]
                old_m = self.context.best_metrics
                # Compare OOF scores, classification metrics, or regression metrics
                eval_metrics = [
                    "roc_auc", "oof_mean", "r2_score", "mae", "rmse",
                    "mse", "mean_absolute_error", "mean_squared_error",
                    "median_absolute_error", "mean_squared_log_error"
                ]
                error_metrics = [
                    "oof_mean", "mae", "rmse", "mse", "mean_absolute_error",
                    "mean_squared_error", "median_absolute_error",
                    "mean_squared_log_error"
                ]
                for key in eval_metrics:
                    if key in new_m and key in old_m:
                        if key in error_metrics:
                            is_better = abs(new_m[key]) < abs(old_m[key])
                        else:
                            is_better = new_m[key] > old_m[key]
                        break
                if not is_better and not old_m:
                    is_better = True

            if is_better:
                self.context.best_pipeline = result["pipeline"]
                self.context.best_metrics = result["metrics"]

        # Generate reproducible code
        self._generate_pipeline_code(config)

        serializable = {k: v for k, v in result.items() if k != "pipeline"}
        return serializable

    def _generate_pipeline_code(self, config: dict) -> None:
        """Generate Python code that reproduces this pipeline."""
        lines = [
            "import numpy as np",
            "import pandas as pd",
            "",
            "from bluecast.blueprints.unified import BlueCastAuto",
            "from bluecast.config.training_config import TrainingConfig",
            "from bluecast.ensemble.ensemble_config import EnsembleConfig",
        ]

        # Include FE preprocessor if snippets were used
        if self.context.feature_code_snippets:
            lines.append(
                "from bluecast.ai.fe_preprocessor import AIFeaturePreprocessor"
            )
            lines.append("")
            lines.append("# Feature engineering code captured from AI agents")
            lines.append("fe_code_snippets = [")
            for snippet in self.context.feature_code_snippets:
                escaped = snippet.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'    """{escaped}""",')
            lines.append("]")
            lines.append("")
            lines.append("preprocessor = AIFeaturePreprocessor(fe_code_snippets)")

        lines.append("")
        lines.extend(
            [
                "training_config = TrainingConfig(",
                f"    hyperparameter_tuning_rounds={config.get('tuning_rounds', 50)},",
                f"    hyperparameter_tuning_max_runtime_secs={config.get('tuning_max_runtime', 120)},",
                f"    hypertuning_cv_folds={config.get('hypertuning_cv_folds', 3)},",
                f'    autotune_on_device="{config.get("autotune_on_device", "cpu")}",',
                f"    bluecast_cv_train_n_model=({config.get('n_folds', 5)}, {config.get('n_repeats', 1)}),",
                "    calculate_shap_values=False,",
                "    plot_hyperparameter_tuning_overview=False,",
            ]
        )

        if "out_of_fold_dataset_store_path" in config:
            lines.append(f'    out_of_fold_dataset_store_path="{config["out_of_fold_dataset_store_path"]}",')
        lines.append(")")
        lines.append("")

        strategy = config.get("ensemble_strategy", "mean")
        reg_metric = config.get("regression_eval_metric", "rmse")
        if config.get("use_cv", True):
            lines.append(
                f'ensemble_config = EnsembleConfig(ensemble_strategy="{strategy}", regression_eval_metric="{reg_metric}")'
            )


        if config.get("class_problem") == "regression" and reg_metric == "mae":
            lines.append("from bluecast.config.training_config import CatboostTuneParamsRegressionConfig")
            lines.append("conf_tuning = CatboostTuneParamsRegressionConfig()")
            lines.append('conf_tuning.catboost_loss_function = "MAE"')
            lines.append('conf_tuning.catboost_eval_metric = "MAE"')
            lines.append("")

        lines.append("pipeline = BlueCastAuto(")
        lines.append(f"    class_problem=\"{config.get('class_problem', 'binary')}\",")
        lines.append(f"    use_cross_validation={config.get('use_cv', True)},")
        lines.append("    conf_training=training_config,")
        if config.get("use_cv", True):
            lines.append("    ensemble_config=ensemble_config,")
        if config.get("class_problem") == "regression" and reg_metric == "mae":
            lines.append("    conf_tuning=conf_tuning,")
        if self.context.feature_code_snippets:
            lines.append("    custom_preprocessor=preprocessor,")
        lines.append(")")
        lines.append("")
        lines.append(
            f'pipeline.fit_eval(df_train, target_col="{self.context.target_col}")'
        )

        self.context.pipeline_code = "\n".join(lines)

    @property
    def name(self) -> str:
        return "PipelineBuilder"

    def system_prompt(self) -> str:
        history = ""
        if self.context.run_history:
            history = "\n\nPrevious runs:\n"
            for i, run in enumerate(self.context.run_history):
                metrics_val = run.get('metrics', 'N/A')
                history += f"  Run {i + 1}: success={run.get('success', False)}, metrics={metrics_val}\n"
                if run.get("config"):
                    history += f"    config: {run['config']}\n"

        data_summary = self.context.get_data_summary()

        return f"""You are a pipeline builder for the BlueCast AutoML framework.

Your job is to configure and run a BlueCast ML pipeline using the build_and_run_pipeline tool.
You must call the tool with appropriate configuration parameters.

Available parameters:
- class_problem: "binary", "multiclass", or "regression"
- use_cv: true/false (use cross-validation)
- ensemble_strategy: "mean", "stacking", or "hill_climbing"
- regression_eval_metric: "rmse" or "mae" (for regression tasks)
- n_folds: number of CV folds (3-10)
- n_repeats: CV repeats (1-3)
- tuning_rounds: hyperparameter tuning rounds (20-200)
- tuning_max_runtime: max tuning time in seconds
- enable_feature_selection: boolean (can disable to speed up evaluation)
- columns_to_drop: list of string column names to drop to eliminate noise
- rf_max_depth_min, rf_max_depth_max, rf_estimators_min, rf_estimators_max: integer bounds for RandomForest
- histgb_max_iter_max, histgb_depth_max: integer bounds for HistGradientBoosting
- autotune_on_device: "cpu" or "gpu"
- out_of_fold_dataset_store_path: (optional) path to save OOF parquet predictions

After running the pipeline, analyze the results and suggest improvements.
{history}

Dataset:
{data_summary}"""

    def get_tools(self) -> List[ToolDefinition]:
        return [TOOL_DEFINITIONS["build_and_run_pipeline"]]
