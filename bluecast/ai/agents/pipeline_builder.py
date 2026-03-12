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
        df = (
            self.context.engineered_df
            if self.context.engineered_df is not None
            else self.context.df_train
        )
        result = tool_build_and_run_pipeline(df, self.context.target_col, config)

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
                # Compare OOF scores or classification metrics
                for key in ["roc_auc", "oof_mean", "r2_score"]:
                    if key in new_m and key in old_m:
                        if key == "oof_mean":
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
            "from bluecast.blueprints.unified import BlueCastAuto",
            "from bluecast.config.training_config import TrainingConfig",
            "from bluecast.ensemble.ensemble_config import EnsembleConfig",
            "",
            "training_config = TrainingConfig(",
            f"    hyperparameter_tuning_rounds={config.get('tuning_rounds', 50)},",
            f"    hyperparameter_tuning_max_runtime_secs={config.get('tuning_max_runtime', 120)},",
            f"    hypertuning_cv_folds={config.get('hypertuning_cv_folds', 3)},",
            f"    autotune_on_device=\"{config.get('autotune_on_device', 'cpu')}\",",
            f"    bluecast_cv_train_n_model=({config.get('n_folds', 5)}, {config.get('n_repeats', 1)}),",
            "    calculate_shap_values=False,",
            "    plot_hyperparameter_tuning_overview=False,",
            ")",
            "",
        ]

        strategy = config.get("ensemble_strategy", "mean")
        if config.get("use_cv", True):
            lines.append(
                f'ensemble_config = EnsembleConfig(ensemble_strategy="{strategy}")'
            )
            lines.append("")

        lines.append("pipeline = BlueCastAuto(")
        lines.append(f"    class_problem=\"{config.get('class_problem', 'binary')}\",")
        lines.append(f"    use_cross_validation={config.get('use_cv', True)},")
        lines.append("    conf_training=training_config,")
        if config.get("use_cv", True):
            lines.append("    ensemble_config=ensemble_config,")
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
                history += f"  Run {i + 1}: success={run['success']}, metrics={run['metrics']}\n"
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
- n_folds: number of CV folds (3-10)
- n_repeats: CV repeats (1-3)
- tuning_rounds: hyperparameter tuning rounds (20-200)
- tuning_max_runtime: max tuning time in seconds
- autotune_on_device: "cpu" or "gpu"

After running the pipeline, analyze the results and suggest improvements.
{history}

Dataset:
{data_summary}"""

    def get_tools(self) -> List[ToolDefinition]:
        return [TOOL_DEFINITIONS["build_and_run_pipeline"]]
