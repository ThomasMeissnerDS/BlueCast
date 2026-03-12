"""Orchestrator: coordinates agents with sampling, checkpoints, and reporting."""

import json
import logging
import os
import time
from typing import Optional

import dill
import numpy as np
import pandas as pd

from bluecast.ai.agents.data_analyst import DataAnalystAgent
from bluecast.ai.agents.evaluator import EvaluatorAgent
from bluecast.ai.agents.feature_engineer import FeatureEngineerAgent
from bluecast.ai.agents.pipeline_builder import PipelineBuilderAgent
from bluecast.ai.agents.planner import PlannerAgent
from bluecast.ai.agents.reporter import ReporterAgent
from bluecast.ai.agents.researcher import ResearcherAgent
from bluecast.ai.config import AIConfig
from bluecast.ai.context import SharedContext
from bluecast.ai.providers.base import BaseLLMProvider
from bluecast.ai.result import BlueCastAIResult

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "bluecast_ai_checkpoint.pkl"


class Orchestrator:
    """Coordinates the multi-agent pipeline from planning to final report.

    Features:
    - Smart sampling for large datasets (agents work on a sample, final
      model trains on full data)
    - Checkpointing after each step (resume after crashes)
    - Structured I/O logging for all agent interactions
    - Reporter agent writes a polished summary at the end

    Flow:
    1. Sample data (if large)
    2. Planner interprets user prompt -> execution plan
    3. Researcher searches web for domain knowledge (optional)
    4. DataAnalyst profiles the dataset
    5. FeatureEngineer creates features (optional)
    6. PipelineBuilder + Evaluator loop: build -> evaluate -> improve
    7. Reporter writes a comprehensive summary
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        config: AIConfig,
        df: pd.DataFrame,
        target_col: str,
        prompt: str,
    ):
        self.llm = llm
        self.config = config
        self.context = SharedContext(
            df_train=df,
            target_col=target_col,
            user_prompt=prompt,
            mode=config.mode,
            original_shape=df.shape,
        )

        for path in config.context_files:
            try:
                with open(path, "r") as f:
                    self.context.context_file_contents.append(
                        f"--- {path} ---\n{f.read()[:5000]}"
                    )
            except Exception as e:
                logger.warning(f"Could not load context file {path}: {e}")

        verbose = config.verbose

        self.planner = PlannerAgent(llm, self.context, verbose=verbose)
        self.analyst = DataAnalystAgent(llm, self.context, verbose=verbose)
        self.engineer = FeatureEngineerAgent(llm, self.context, verbose=verbose)
        self.builder = PipelineBuilderAgent(llm, self.context, verbose=verbose)
        self.evaluator = EvaluatorAgent(llm, self.context, verbose=verbose)
        self.researcher = ResearcherAgent(llm, self.context, verbose=verbose)
        self.reporter = ReporterAgent(llm, self.context, verbose=verbose)

    # ------------------------------------------------------------------
    # Smart sampling
    # ------------------------------------------------------------------

    def _apply_smart_sampling(self) -> None:
        """Downsample the dataset if it exceeds configured limits.

        Uses stratified sampling for classification to preserve class balance.
        Logs sampling details to context.
        """
        df = self.context.df_train
        if df is None:
            return

        n_rows, n_cols = df.shape
        max_rows = self.config.max_rows_for_agents
        max_cols = self.config.max_columns_for_agents

        needs_row_sample = n_rows > max_rows
        needs_col_note = n_cols > max_cols

        if not needs_row_sample and not needs_col_note:
            self.context.df_sample = None
            self.context.was_sampled = False
            return

        sample_df = df

        if needs_row_sample:
            target = self.context.target_col
            if target in df.columns and df[target].nunique() <= 20:
                # Stratified sampling for classification
                sample_df = df.groupby(target, group_keys=False).apply(
                    lambda x: x.sample(
                        n=min(len(x), max(1, int(max_rows * len(x) / n_rows))),
                        random_state=42,
                    )
                ).reset_index(drop=True)
            else:
                sample_df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

            msg = (
                f"Dataset sampled: {n_rows} -> {len(sample_df)} rows "
                f"(limit: {max_rows}). Full data used for final training."
            )
            self.context.log("Orchestrator", msg, event_type="info")
            if self.config.verbose:
                print(f"  {msg}")

        if needs_col_note:
            msg = (
                f"Dataset has {n_cols} columns (limit for detailed profiling: "
                f"{max_cols}). Agents will focus on top features."
            )
            self.context.log("Orchestrator", msg, event_type="info")
            self.context.data_warnings.append(msg)
            if self.config.verbose:
                print(f"  {msg}")

        self.context.df_sample = sample_df
        self.context.was_sampled = True

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Optional[str]:
        if not self.config.checkpoint_dir:
            return None
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        return os.path.join(self.config.checkpoint_dir, CHECKPOINT_FILENAME)

    def _save_checkpoint(self, step_name: str) -> None:
        """Save current context to disk after completing a step."""
        path = self._checkpoint_path()
        if path is None:
            return
        self.context.completed_steps.append(step_name)
        self.context.current_step = None
        try:
            with open(path, "wb") as f:
                dill.dump(self.context, f)
            self.context.log(
                "Orchestrator", f"Checkpoint saved after '{step_name}'",
                event_type="checkpoint",
            )
            if self.config.verbose:
                print(f"  [Checkpoint] Saved after '{step_name}'")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> bool:
        """Try to load a checkpoint. Returns True if resumed."""
        path = self._checkpoint_path()
        if path is None or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                saved_context = dill.load(f)

            completed = saved_context.completed_steps
            self.context.log(
                "Orchestrator",
                f"Resuming from checkpoint. Completed steps: {completed}",
                event_type="checkpoint",
            )
            if self.config.verbose:
                print(f"  [Checkpoint] Resuming. Already completed: {completed}")

            # Restore context fields (keep df_train from current init)
            df_train = self.context.df_train
            original_shape = self.context.original_shape
            self.context = saved_context
            self.context.df_train = df_train
            self.context.original_shape = original_shape

            # Re-attach context to all agents
            for agent in [
                self.planner, self.analyst, self.engineer,
                self.builder, self.evaluator, self.researcher, self.reporter,
            ]:
                agent.context = self.context

            return True
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return False

    def _is_step_done(self, step_name: str) -> bool:
        return step_name in self.context.completed_steps

    def _clear_checkpoint(self) -> None:
        path = self._checkpoint_path()
        if path and os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> BlueCastAIResult:
        """Execute the full multi-agent pipeline."""
        start_time = time.time()

        if self.config.verbose:
            print("=" * 60)
            print("BlueCastAI - Multi-Agent AutoML Pipeline")
            print("=" * 60)

        resumed = self._load_checkpoint()

        # --- Step 0: Smart sampling ---
        if not self._is_step_done("sampling"):
            self._apply_smart_sampling()
            self._save_checkpoint("sampling")

        # --- Step 1: Plan ---
        if not self._is_step_done("plan"):
            plan = self._step_plan()
            self._save_checkpoint("plan")
        else:
            # Reconstruct plan from context log for downstream steps
            plan = self._reconstruct_plan()

        # --- Step 2: Research (optional) ---
        if (
            not self._is_step_done("research")
            and plan.get("needs_web_research")
            and self.config.enable_web_search
        ):
            self._step_research(plan.get("research_queries", []))
            self._save_checkpoint("research")

        # --- Step 3: Analyze data ---
        if not self._is_step_done("analyze"):
            self._step_analyze()
            self._save_checkpoint("analyze")

        # --- Step 4: Feature engineering (optional) ---
        if (
            not self._is_step_done("feature_engineering")
            and plan.get("needs_feature_engineering", False)
            and self.config.mode != "fast"
        ):
            self._step_feature_engineer(plan)
            self._save_checkpoint("feature_engineering")

        # --- Step 5: Build-Evaluate-Improve loop ---
        if not self._is_step_done("build_loop"):
            max_iterations = plan.get("max_iterations", self.config.get_max_iterations())
            self._step_build_loop(plan, max_iterations)
            self._save_checkpoint("build_loop")

        # --- Step 6: Report ---
        if not self._is_step_done("report"):
            self._step_report()
            self._save_checkpoint("report")

        elapsed = time.time() - start_time

        self.context.log(
            "Orchestrator",
            f"Pipeline complete in {elapsed:.1f}s",
            event_type="info",
            metadata={"elapsed_seconds": elapsed},
        )

        result = self._assemble_result()

        self._clear_checkpoint()

        return result

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def _step_plan(self) -> dict:
        if self.config.verbose:
            n_steps = 7 if self.config.mode != "fast" else 5
            print(f"\nStep 1/{n_steps}: Planning...")

        context_info = ""
        if self.context.context_file_contents:
            context_info = "\n\nContext files provided:\n" + "\n".join(
                self.context.context_file_contents
            )

        task = (
            f"User prompt: {self.context.user_prompt}\n"
            f"Mode: {self.config.mode}\n\n"
            f"Dataset summary:\n{self.context.get_data_summary()}"
            f"{context_info}"
        )

        response = self.planner.run(task)
        plan = self.planner.parse_plan(response)

        self.context.class_problem = plan.get("class_problem", "binary")
        self.context.log(
            "Orchestrator", f"Plan: {json.dumps(plan, indent=2)}",
            event_type="plan", metadata={"plan": plan},
        )

        if self.config.verbose:
            print(
                f"  Plan: problem={plan.get('class_problem')}, "
                f"ensemble={plan.get('ensemble_strategy')}, "
                f"FE={plan.get('needs_feature_engineering')}, "
                f"iterations={plan.get('max_iterations')}"
            )

        return plan

    def _reconstruct_plan(self) -> dict:
        """Reconstruct the plan from structured log metadata."""
        for entry in self.context.structured_log:
            if entry.event_type == "plan" and entry.metadata and "plan" in entry.metadata:
                return entry.metadata["plan"]
        return self.planner._default_plan()

    def _step_research(self, queries: list) -> None:
        if self.config.verbose:
            print("\nStep 2: Researching...")
        task = "Search for information relevant to this ML task:\n"
        task += "\n".join(f"- {q}" for q in queries[:3])
        result = self.researcher.run(task)
        self.context.web_research = result

    def _step_analyze(self) -> None:
        if self.config.verbose:
            print("\nStep 3: Analyzing data...")
        result = self.analyst.run(
            "Thoroughly profile this dataset. Use all your tools to understand "
            "the data quality, distributions, correlations, and potential issues."
        )
        self.context.data_profile = {"summary": result}

        for keyword in ["leakage", "imbalance", "missing", "null", "duplicate", "constant"]:
            if keyword in result.lower():
                self.context.data_warnings.append(
                    f"Data analyst flagged: {keyword} detected"
                )

    def _step_feature_engineer(self, plan: dict) -> None:
        if self.config.verbose:
            print("\nStep 4: Engineering features...")

        hints = plan.get("feature_engineering_hints", [])
        hint_text = "\n".join(f"- {h}" for h in hints) if hints else "Use your judgment."

        task = (
            f"Create useful features for this {self.context.class_problem} problem.\n"
            f"Hints from the planner:\n{hint_text}\n\n"
            f"Create 3-5 strong features. Call create_feature for each one."
        )
        self.engineer.run(task)

        if self.context.engineered_df is not None and self.config.verbose:
            orig_cols = len(self.context.df_train.columns)
            new_cols = len(self.context.engineered_df.columns)
            print(f"  Features: {orig_cols} -> {new_cols} columns")

    def _step_build_loop(self, plan: dict, max_iterations: int) -> None:
        if self.config.verbose:
            print(f"\nStep 5: Building pipeline (up to {max_iterations} iterations)...")

        for iteration in range(max_iterations):
            if self.config.verbose:
                print(f"\n  Iteration {iteration + 1}/{max_iterations}:")

            build_task = self._create_build_task(plan, iteration)
            self.builder.run(build_task)

            latest_run = self.context.run_history[-1] if self.context.run_history else None
            if latest_run and self.config.verbose:
                status = "OK" if latest_run["success"] else "FAILED"
                print(f"    Result [{status}]: {latest_run.get('metrics', {})}")

            if self.config.mode == "fast" or iteration >= max_iterations - 1:
                break

            eval_result = self.evaluator.run(
                f"Analyze the results of iteration {iteration + 1} and suggest "
                f"specific improvements for the next run."
            )

            try:
                if "```json" in eval_result:
                    json_text = eval_result.split("```json")[1].split("```")[0]
                    suggestions = json.loads(json_text)
                    plan.update(suggestions)
            except (json.JSONDecodeError, IndexError):
                pass

    def _step_report(self) -> None:
        """Have the Reporter agent write a polished summary."""
        if self.config.verbose:
            print("\nStep 6: Writing report...")

        task = self.reporter.build_report_task()
        report = self.reporter.run(task)
        self.context.report_markdown = report

        if self.config.verbose:
            print("  Report generated.")

    def _create_build_task(self, plan: dict, iteration: int) -> str:
        config_hints = {
            "class_problem": plan.get(
                "class_problem", self.context.class_problem or "binary"
            ),
            "use_cv": plan.get("use_cv", True),
            "ensemble_strategy": plan.get("ensemble_strategy", "mean"),
            "n_folds": plan.get("n_folds", 5),
            "n_repeats": plan.get("n_repeats", 1),
            "tuning_rounds": plan.get("tuning_rounds", 50),
            "tuning_max_runtime": plan.get("tuning_max_runtime", 120),
        }

        if iteration > 0:
            config_hints["tuning_rounds"] = min(
                config_hints["tuning_rounds"] * (iteration + 1), 500
            )
        if iteration >= 2 and config_hints["ensemble_strategy"] == "mean":
            config_hints["ensemble_strategy"] = "stacking"

        return (
            f"Build and run a BlueCast pipeline (iteration {iteration + 1}).\n"
            f"Recommended configuration: {json.dumps(config_hints)}\n"
            f"Call build_and_run_pipeline with these parameters."
        )

    def _assemble_result(self) -> BlueCastAIResult:
        result = BlueCastAIResult(
            pipeline=self.context.best_pipeline,
            pipeline_code=self.context.pipeline_code or "",
            feature_engineering_code=self.context.feature_engineering_code or "",
            metrics=self.context.best_metrics or {},
            data_profile=self.context.data_profile,
            run_history=self.context.run_history,
            agent_log=self.context.agent_log,
            class_problem=self.context.class_problem or "binary",
            report_markdown=self.context.report_markdown or "",
            structured_log=self.context.structured_log,
        )

        if self.config.verbose:
            print("\n" + "=" * 60)
            result.show_report()

        return result
