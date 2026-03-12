"""Reporter agent: writes a polished summary report of the entire run."""

import json
from typing import List

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.providers.base import ToolDefinition


class ReporterAgent(BaseAgent):
    """Produces a comprehensive, human-readable report summarizing
    the full BlueCastAI run: data findings, feature engineering,
    model performance, and actionable recommendations."""

    @property
    def name(self) -> str:
        return "Reporter"

    def system_prompt(self) -> str:
        return """You are a technical report writer for the BlueCast AutoML framework.

You receive a structured summary of an automated ML pipeline run and produce
a clear, well-organized Markdown report. The audience is a data scientist who
wants to understand what happened and what to do next.

Your report MUST include these sections:

## 1. Executive Summary
One paragraph: problem type, dataset size, best metric, key insight.

## 2. Data Profile
Key dataset characteristics, data quality issues, class balance.

## 3. Feature Engineering
What features were created and why. Show the code if available.

## 4. Model Pipeline
Configuration used (ensemble strategy, CV folds, tuning rounds).
Performance across iterations.

## 5. Results
Best metrics achieved. Comparison across iterations if multiple.

## 6. Issues & Warnings
Any data quality issues, leakage risks, class imbalance.

## 7. Recommendations
Concrete next steps to improve performance further.

Write in Markdown. Be concise but thorough. Use bullet points and tables."""

    def get_tools(self) -> List[ToolDefinition]:
        return []

    def build_report_task(self) -> str:
        """Assemble all context into a single prompt for the reporter."""
        sections = []

        sections.append(f"User prompt: {self.context.user_prompt}")
        sections.append(f"Mode: {self.context.mode}")
        sections.append(f"Problem type: {self.context.class_problem}")

        if self.context.was_sampled:
            sections.append(
                f"Data was sampled from {self.context.original_shape} "
                f"to {self.context.get_working_df().shape} for agent analysis."
            )

        if self.context.data_profile:
            sections.append(
                f"\nData profile:\n{json.dumps(self.context.data_profile, indent=2, default=str)[:3000]}"
            )

        if self.context.data_warnings:
            sections.append(
                "\nWarnings:\n"
                + "\n".join(f"- {w}" for w in self.context.data_warnings)
            )

        if self.context.feature_engineering_code:
            sections.append(
                f"\nFeature engineering code:\n```python\n{self.context.feature_engineering_code}\n```"
            )
        else:
            sections.append("\nNo feature engineering was applied.")

        if self.context.run_history:
            sections.append("\nRun history:")
            for i, run in enumerate(self.context.run_history):
                sections.append(
                    f"  Run {i + 1}: success={run.get('success')}, "
                    f"metrics={run.get('metrics')}, "
                    f"config={run.get('config', {})}"
                )

        if self.context.best_metrics:
            sections.append(f"\nBest metrics: {self.context.best_metrics}")

        if self.context.web_research:
            sections.append(
                f"\nWeb research findings:\n{self.context.web_research[:1000]}"
            )

        if self.context.pipeline_code:
            sections.append(
                f"\nGenerated pipeline code:\n```python\n{self.context.pipeline_code}\n```"
            )

        # Include a selection of the structured log
        log_entries = self.context.structured_log[-30:]
        if log_entries:
            sections.append("\nAgent activity log (last 30 entries):")
            for entry in log_entries:
                sections.append(f"  {entry}")

        return (
            "Write a comprehensive Markdown report for this BlueCastAI run.\n\n"
            + "\n".join(sections)
        )
