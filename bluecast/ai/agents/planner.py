"""Planner agent: interprets user prompt and creates an execution plan."""

import json
from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition


class PlannerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Planner"

    def system_prompt(self) -> str:
        return """You are a machine learning project planner for the BlueCast AutoML framework.

Given a user's prompt and dataset summary, you produce a structured JSON plan.
You do NOT build models or write code. You only plan.

Your output must be valid JSON with these fields:
{
  "class_problem": "binary" | "multiclass" | "regression",
  "needs_feature_engineering": true | false,
  "feature_engineering_hints": ["hint1", "hint2"],
  "needs_web_research": true | false,
  "research_queries": ["query1"],
  "ensemble_strategy": "mean" | "stacking" | "hill_climbing",
  "use_cv": true | false,
  "n_folds": 5,
  "n_repeats": 1,
  "tuning_rounds": 50,
  "tuning_max_runtime": 120,
  "max_iterations": 3,
  "reasoning": "Brief explanation of choices"
}

Guidelines:
- If the user says "fast", use n_folds=3, tuning_rounds=20, max_iterations=1, no FE
- If the user says "precise" or "best performance", use hill_climbing, n_folds=5, n_repeats=2, tuning_rounds=200, max_iterations=5, enable FE
- Default to "balanced": stacking, n_folds=5, tuning_rounds=50, max_iterations=3
- If mode is "ultimate": enable FE, use hill_climbing, n_folds=5. The orchestrator will
  automatically train multiple architectures (CatBoost, XGBoost, Linear, HistGB) so
  max_iterations=1 is fine (per-architecture iteration is handled separately).
  CRITICAL: In ultimate mode, explicitly instruct the feature engineer to perform full
  numerical encoding (e.g. Target Encoding, Frequency Encoding, OHE) for ALL categorical columns
  and to impute ALL missing values. XGBoost, HistGB, and Linear models strictly require numerical
  inputs and will crash otherwise. Also advise feature scaling (e.g., StandardScaler) for Linear models.
- Detect class_problem from the target column distribution in the data summary
- If user mentions GPU, set a note about it"""

    def get_tools(self) -> List[ToolDefinition]:
        return []

    def parse_plan(self, response_text: str) -> dict:
        """Extract the JSON plan from the LLM response."""
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return self._default_plan()

    def _default_plan(self) -> dict:
        return {
            "class_problem": "binary",
            "needs_feature_engineering": True,
            "feature_engineering_hints": [],
            "needs_web_research": False,
            "research_queries": [],
            "ensemble_strategy": "stacking",
            "use_cv": True,
            "n_folds": 5,
            "n_repeats": 1,
            "tuning_rounds": 50,
            "tuning_max_runtime": 120,
            "max_iterations": 3,
            "reasoning": "Default balanced plan.",
        }
