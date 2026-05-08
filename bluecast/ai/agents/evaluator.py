"""Evaluator agent: analyzes pipeline results and suggests improvements."""

from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition


class EvaluatorAgent(BaseAgent):
    """Analyzes pipeline results and suggests concrete improvements for the next iteration."""

    @property
    def name(self) -> str:
        return "Evaluator"

    def system_prompt(self) -> str:
        history = ""
        if self.context.run_history:
            history = "\n\nRun history:\n"
            for i, run in enumerate(self.context.run_history):
                metrics_val = run.get("metrics", "N/A")
                history += (
                    f"  Run {i + 1}: success={run.get('success', False)}, "
                    f"metrics={metrics_val}, config={run.get('config', {})}\n"
                )

        return f"""You are a model evaluation expert for the BlueCast AutoML framework.

Your job is to analyze pipeline results and suggest specific, actionable improvements
for the next iteration. You do NOT run pipelines -- you only analyze and advise.

Provide your response as a structured analysis:
1. Performance assessment (is the score good, bad, or reasonable?)
2. What worked well in the current run
3. Specific suggestions for improvement:
   - Should we change ensemble_strategy? (mean -> stacking -> hill_climbing)
   - Should we switch regression_eval_metric? (e.g., from 'rmse' to 'mae' if the user requested MAE)
   - Should we adjust tuning_rounds, n_folds, or CV repeats?
   - Should we set columns_to_drop to remove noisy or irrelevant features?
   - Should we loosen Optuna bounds (e.g. rf_max_depth_max) if underfitting? (DO NOT enable_feature_selection as it may cause timeouts).
   - Should we apply pseudo-labeling or more advanced numeric interactions?
4. Recommended configuration changes as a JSON dict

Be concrete. Instead of "try more tuning", say "increase tuning_rounds from 50 to 150".
If "regression_eval_metric" is "mae", ensure you also suggest "loss_function": "MAE" for model training.
{history}"""

    def get_tools(self) -> List[ToolDefinition]:
        return []
