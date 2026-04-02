"""Orchestrator: coordinates agents with sampling, checkpoints, and reporting."""

import concurrent.futures
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
            callbacks=config.callbacks,
        )

        self._load_context_files()

        verbose = config.verbose

        self.planner = PlannerAgent(llm, self.context, verbose=verbose)
        self.analyst = DataAnalystAgent(llm, self.context, verbose=verbose)
        self.engineer = FeatureEngineerAgent(llm, self.context, verbose=verbose)
        self.builder = PipelineBuilderAgent(llm, self.context, verbose=verbose)
        self.evaluator = EvaluatorAgent(llm, self.context, verbose=verbose)
        self.researcher = ResearcherAgent(llm, self.context, verbose=verbose)
        self.reporter = ReporterAgent(llm, self.context, verbose=verbose)

    # ------------------------------------------------------------------
    # Context file loading (PDF, docx, CSV, txt, md)
    # ------------------------------------------------------------------

    def _load_context_files(self) -> None:
        """Load domain knowledge files into context.

        Supports: .pdf (via PyPDF2), .docx (via python-docx),
        .csv/.tsv (first 100 rows), .txt/.md/.rst (raw text).
        """
        from pathlib import Path

        for file_path in self.config.context_files:
            try:
                p = Path(file_path)
                ext = p.suffix.lower()
                name = p.name

                if ext == ".pdf":
                    text = self._extract_pdf_text(file_path)
                elif ext == ".docx":
                    text = self._extract_docx_text(file_path)
                elif ext in (".csv", ".tsv"):
                    sep = "\t" if ext == ".tsv" else ","
                    df = pd.read_csv(file_path, nrows=100, sep=sep)
                    text = f"CSV/TSV sample ({len(df)} rows):\n{df.to_string()}"
                else:
                    with open(file_path, "r", errors="replace") as f:
                        text = f.read()

                self.context.context_file_contents.append(
                    f"--- Domain knowledge from {name} ---\n{text[:10000]}"
                )

                if self.config.verbose:
                    print(f"  Loaded context file: {name} ({len(text)} chars)")

            except Exception as e:
                logger.warning(f"Could not load context file {file_path}: {e}")

    @staticmethod
    def _extract_pdf_text(path: str) -> str:
        """Extract text from a PDF file."""
        try:
            import PyPDF2

            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF support. "
                "Install it with: pip install PyPDF2"
            )

    @staticmethod
    def _extract_docx_text(path: str) -> str:
        """Extract text from a Word document."""
        try:
            import docx

            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ImportError(
                "python-docx is required for .docx support. "
                "Install it with: pip install python-docx"
            )

    def _get_critique_rounds(self) -> int:
        """Return the number of critique rounds based on mode."""
        if self.config.critique_max_rounds <= 0:
            return 0

        mode_rounds = {
            "fast": 0,
            "balanced": 1,
            "precise": 2,
            "ultimate": 2,
        }
        return min(
            mode_rounds.get(self.config.mode, 0),
            self.config.critique_max_rounds,
        )

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
                sample_df = (
                    df.groupby(target, group_keys=False)
                    .apply(
                        lambda x: x.sample(
                            n=min(len(x), max(1, int(max_rows * len(x) / n_rows))),
                            random_state=42,
                        )
                    )
                    .reset_index(drop=True)
                )
            else:
                sample_df = df.sample(n=max_rows, random_state=42).reset_index(
                    drop=True
                )

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
                "Orchestrator",
                f"Checkpoint saved after '{step_name}'",
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
                self.planner,
                self.analyst,
                self.engineer,
                self.builder,
                self.evaluator,
                self.researcher,
                self.reporter,
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

        self._load_checkpoint()

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

        # --- Step 2 & 3: Research and Analyze Data Concurrent ---
        needs_research = (
            not self._is_step_done("research")
            and plan.get("needs_web_research")
            and self.config.enable_web_search
        )
        needs_analyze = not self._is_step_done("analyze")

        if needs_research or needs_analyze:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                if needs_research:
                    futures[
                        executor.submit(
                            self._step_research, plan.get("research_queries", [])
                        )
                    ] = "research"
                if needs_analyze:
                    futures[executor.submit(self._step_analyze)] = "analyze"

                for future in concurrent.futures.as_completed(futures):
                    step_name = futures[future]
                    try:
                        future.result()
                        self._save_checkpoint(step_name)
                    except Exception as e:
                        logger.error(f"Error during {step_name}: {e}")

        # --- Step 4: Feature engineering (optional) ---
        if (
            not self._is_step_done("feature_engineering")
            and plan.get("needs_feature_engineering", False)
            and self.config.mode != "fast"
        ):
            try:
                self._step_feature_engineer(plan)
                self._save_checkpoint("feature_engineering")
            except Exception as e:
                logger.warning(f"Feature engineering failed, skipping: {e}")
                if self.config.verbose:
                    print(f"  ⚠️ Feature engineering skipped due to error: {e}")
                self._save_checkpoint("feature_engineering")

        # --- Step 5: Build-Evaluate-Improve loop ---
        if not self._is_step_done("build_loop"):
            try:
                if self.config.mode == "ultimate":
                    self._step_ultimate_build_loop(plan)
                else:
                    max_iterations = plan.get(
                        "max_iterations", self.config.get_max_iterations()
                    )
                    self._step_build_loop(plan, max_iterations)
                self._save_checkpoint("build_loop")
            except Exception as e:
                logger.warning(f"Build loop failed: {e}")
                if self.config.verbose:
                    print(f"  ⚠️ Build loop encountered an error: {e}")
                self._save_checkpoint("build_loop")

        # --- Step 6: Report ---
        if not self._is_step_done("report"):
            try:
                self._step_report()
            except Exception as e:
                logger.warning(f"Report generation failed, skipping: {e}")
                if self.config.verbose:
                    print(f"  ⚠️ Report skipped due to error: {e}")
            self._save_checkpoint("report")

        elapsed = time.time() - start_time

        self.context.log(
            "Orchestrator",
            f"Pipeline complete in {elapsed:.1f}s",
            event_type="info",
            metadata={"elapsed_seconds": elapsed},
        )

        if self.config.verbose:
            print(
                f"  Tokens usage: Prompt={self.context.prompt_tokens}, Completion={self.context.completion_tokens}"
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

        try:
            response = self.planner.run(task)
            plan = self.planner.parse_plan(response)
        except Exception as e:
            logger.warning(
                f"Planner failed, using default plan: {type(e).__name__}: {e}"
            )
            if self.config.verbose:
                print(
                    f"  ⚠️ Planner encountered an error: {type(e).__name__}. "
                    f"Using default plan."
                )
            plan = self.planner._default_plan()

        self.context.class_problem = plan.get("class_problem", "binary")
        self.context.log(
            "Orchestrator",
            f"Plan: {json.dumps(plan, indent=2)}",
            event_type="plan",
            metadata={"plan": plan},
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
            if (
                entry.event_type == "plan"
                and entry.metadata
                and "plan" in entry.metadata
            ):
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

        task = (
            "Thoroughly profile this dataset. Use all your tools to understand "
            "the data quality, distributions, correlations, potential issues, "
            "outliers, cardinality, and temporal patterns. "
            "Look for anything unusual or unexpected."
        )

        critique_rounds = self._get_critique_rounds()

        if critique_rounds > 0:
            from bluecast.ai.critique import CritiqueLoop

            critic = CritiqueLoop(
                self.llm,
                self.context,
                max_rounds=critique_rounds,
                verbose=self.config.verbose,
            )
            result = critic.run_with_critique(
                agent=self.analyst,
                agent_task=task,
                mode=self.config.mode,
            )
        else:
            result = self.analyst.run(task)

        self.context.data_profile = {"summary": result}

        for keyword in [
            "leakage",
            "imbalance",
            "missing",
            "null",
            "duplicate",
            "constant",
            "outlier",
        ]:
            if keyword in result.lower():
                self.context.data_warnings.append(
                    f"Data analyst flagged: {keyword} detected"
                )

    def _step_feature_engineer(self, plan: dict) -> None:
        if self.config.verbose:
            print("\nStep 4: Engineering features...")

        hints = plan.get("feature_engineering_hints", [])
        hint_text = (
            "\n".join(f"- {h}" for h in hints) if hints else "Use your judgment."
        )

        task = (
            f"Create useful features for this {self.context.class_problem} problem.\n"
            f"Hints from the planner:\n{hint_text}\n\n"
            f"Create 3-5 strong features. Call create_feature for each one.\n"
            f"If any column contains free text, use create_tfidf_features."
        )

        critique_rounds = self._get_critique_rounds()

        if critique_rounds > 0 and self.config.mode in ("precise", "ultimate"):
            from bluecast.ai.critique import CritiqueLoop

            critic = CritiqueLoop(
                self.llm,
                self.context,
                max_rounds=critique_rounds,
                verbose=self.config.verbose,
            )
            critic.run_with_critique(
                agent=self.engineer,
                agent_task=task,
                mode=self.config.mode,
            )
        else:
            self.engineer.run(task)

        if self.context.feature_code_snippets and self.context.df_train is not None:
            if self.config.verbose:
                print("  [Validation] Verifying feature engineering snippets...")

            df_test = self.context.df_train.copy()
            valid_snippets = []

            for i, code in enumerate(self.context.feature_code_snippets):
                try:
                    local_vars = {"df": df_test, "np": np, "pd": pd}
                    exec(code, {}, local_vars)  # noqa: S102
                    df_test = local_vars.get("df", df_test)
                    valid_snippets.append(code)
                except Exception as e:
                    if self.config.verbose:
                        print(
                            f"    Warning: FE snippet {i + 1} failed validation ({e}). Pruning."
                        )

            self.context.feature_code_snippets = valid_snippets
            self.context.engineered_df = df_test

            if self.config.verbose:
                orig_cols = len(self.context.df_train.columns)
                new_cols = len(self.context.engineered_df.columns)
                print(
                    f"  Features finalized: {orig_cols} -> {new_cols} columns ({len(valid_snippets)} snippets)"
                )

    def _step_build_loop(self, plan: dict, max_iterations: int) -> None:
        if self.config.verbose:
            print(f"\nStep 5: Building pipeline (up to {max_iterations} iterations)...")

        override_max_runtime = None
        if self.config.global_tuning_budget and self.config.global_tuning_budget > 0:
            n_folds_expected = plan.get("n_folds", 5)
            total_jobs = max_iterations * n_folds_expected
            override_max_runtime = max(10, int(self.config.global_tuning_budget / total_jobs))
            plan["tuning_max_runtime"] = override_max_runtime

        original_plan_limits = {
            "tuning_rounds": plan.get("tuning_rounds", 200),
            "tuning_max_runtime": plan.get("tuning_max_runtime", 1800),
        }

        for iteration in range(max_iterations):
            if self.config.verbose:
                print(f"\n  Iteration {iteration + 1}/{max_iterations}:")

            build_task = self._create_build_task(plan, iteration)
            self.builder.run(build_task)

            latest_run = (
                self.context.run_history[-1] if self.context.run_history else None
            )
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
                    plan = self._enforce_config_constraints(plan, original_plan_limits)
            except (json.JSONDecodeError, IndexError):
                pass

    def _step_ultimate_build_loop(self, plan: dict) -> None:
        """Train multiple model architectures with per-arch iterative improvement."""
        from bluecast.ai.architectures import get_architectures_for_problem

        problem = self.context.class_problem or "binary"
        archs = get_architectures_for_problem(problem)
        iters = self.config.ultimate_iterations_per_arch
        total_archs = len(archs)
        
        override_max_runtime = None
        if self.config.global_tuning_budget and self.config.global_tuning_budget > 0:
            n_folds_expected = plan.get("n_folds", 5)
            # Rough estimation: divide budget evenly across all tunable jobs
            tunable_archs = max(1, total_archs - 1)  # Linear models usually don't tune heavily
            total_jobs = tunable_archs * iters * n_folds_expected
            override_max_runtime = max(10, int(self.config.global_tuning_budget / total_jobs))

        if self.config.verbose:
            print(
                f"\nStep 5: Ultimate build loop "
                f"({total_archs} architectures × {iters} iterations)..."
            )

        for arch_idx, (arch_name, arch_info) in enumerate(archs.items(), 1):
            if self._is_step_done(f"build_arch_{arch_name}"):
                if self.config.verbose:
                    print(
                        f"\n  [{arch_idx}/{total_archs}] Skipping {arch_info['name']} (Loaded from checkpoint)"
                    )
                continue

            if self.config.verbose:
                print(f"\n  [{arch_idx}/{total_archs}] " f"=== {arch_info['name']} ===")

            ml_model = arch_info["factory"](problem)

            # XGBoost uses BlueCast's native pipeline, not ml_model injection.
            # For XGBoost, we set use_xgboost_native in the config so the
            # tool knows to pass conf_xgboost instead.
            use_xgboost = arch_info.get("use_xgboost_native", False)

            arch_config = self._build_arch_config(plan, arch_name, override_max_runtime)

            for iteration in range(iters):
                if self.config.verbose:
                    print(f"    Iteration {iteration + 1}/{iters}:")

                # Build the pipeline config
                config = dict(arch_config)  # copy
                if not use_xgboost:
                    config["ml_model"] = ml_model

                result = self._build_single_arch(config, arch_name, use_xgboost)

                if not result["success"]:
                    if self.config.verbose:
                        print(f"      FAILED: {result.get('error', 'unknown')}")
                    break

                if self.config.verbose:
                    print(f"      Metrics: {result['metrics']}")

                # Update global best pipeline
                is_better = False
                if self.context.best_metrics is None:
                    is_better = True
                else:
                    new_m = result["metrics"]
                    old_m = self.context.best_metrics
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
                    if self.config.verbose:
                        print(
                            f"      [Best so far] New best pipeline: {arch_name} with {result['metrics']}"
                        )

                # Record in run history
                self.context.run_history.append(
                    {
                        "success": True,
                        "metrics": result["metrics"],
                        "config": result["config_used"],
                        "architecture": arch_name,
                        "iteration": iteration + 1,
                    }
                )

                # Ask Evaluator for arch-specific improvements (unless last)
                if iteration < iters - 1:
                    suggestions = self._evaluate_for_arch(
                        arch_name, arch_info["name"], result
                    )
                    arch_config.update(suggestions)
                    arch_config = self._enforce_config_constraints(arch_config, plan, arch_name)

            self._save_checkpoint(f"build_arch_{arch_name}")

    def _enforce_config_constraints(self, config: dict, original_plan: dict, arch_name: str = "") -> dict:
        """Enforce strict bounds on tuning rounds and runtime to prevent runaway LLM configs."""
        max_rounds = original_plan.get("tuning_rounds", 200)
        max_runtime = original_plan.get("tuning_max_runtime", 1800)

        if "tuning_rounds" in config:
            config["tuning_rounds"] = min(
                config.get("tuning_rounds", max_rounds), max_rounds
            )
        if "tuning_max_runtime" in config:
            config["tuning_max_runtime"] = min(
                config.get("tuning_max_runtime", max_runtime), max_runtime
            )
            
        if arch_name in ["linear", "randomforest"]:
            config["cat_encoding_via_ml_algorithm"] = False
            
        return config

    def _build_arch_config(self, plan: dict, arch_name: str, override_max_runtime: Optional[int] = None) -> dict:
        """Build a base pipeline config for a specific architecture."""
        config = {
            "class_problem": plan.get(
                "class_problem", self.context.class_problem or "binary"
            ),
            "use_cv": plan.get("use_cv", True),
            "ensemble_strategy": plan.get("ensemble_strategy", "mean"),
            "n_folds": plan.get("n_folds", 5),
            "n_repeats": plan.get("n_repeats", 1),
            "tuning_rounds": plan.get("tuning_rounds", 50),
            "tuning_max_runtime": override_max_runtime if override_max_runtime else plan.get("tuning_max_runtime", 120),
        }

        # Linear models don't benefit from gradient boosting tuning
        if arch_name == "linear":
            config["tuning_rounds"] = 1
            config["tuning_max_runtime"] = 30
            
        if arch_name in ["linear", "randomforest"]:
            config["cat_encoding_via_ml_algorithm"] = False

        return config

    def _build_single_arch(
        self, config: dict, arch_name: str, use_xgboost: bool
    ) -> dict:
        """Build a single architecture pipeline."""
        from bluecast.ai.tools import tool_build_and_run_pipeline

        preprocessor = None
        if self.context.feature_code_snippets:
            from bluecast.ai.fe_preprocessor import AIFeaturePreprocessor

            preprocessor = AIFeaturePreprocessor(
                list(self.context.feature_code_snippets)
            )

        ml_model = config.pop("ml_model", None)

        # For XGBoost, let BlueCast handle it natively by passing
        # conf_xgboost/conf_params_xgboost (via the default non-catboost path).
        # We leave ml_model=None which means BlueCast will use its CatBoost
        # default — but for xgboost we need to create the XgboostBaseModel.
        if use_xgboost:
            from bluecast.ml_modelling.xgboost import XgboostModel
            from bluecast.ml_modelling.xgboost_regression import XgboostModelRegression

            problem = config.get(
                "class_problem", self.context.class_problem or "binary"
            )
            if problem == "regression":
                ml_model = XgboostModelRegression(class_problem="regression")
            else:
                ml_model = XgboostModel(class_problem=problem)

        return tool_build_and_run_pipeline(
            self.context.df_train,
            self.context.target_col,
            config,
            custom_preprocessor=preprocessor,
            ml_model=ml_model,
        )

    def _evaluate_for_arch(
        self, arch_name: str, arch_display_name: str, result: dict
    ) -> dict:
        """Ask the Evaluator for architecture-specific improvements."""
        extra_info = ""
        if arch_name == "linear":
            extra_info = "\nWARNING: Linear models strictly require rigorous missing value imputation, categorical encoding, and feature scaling to perform well."
        elif arch_name in ("histgb", "xgboost"):
            extra_info = "\nWARNING: This architecture strictly requires categorical features to be numerically encoded and missing values to be imputed."

        arch_context = (
            f"You are evaluating the **{arch_display_name}** architecture track.\n"
            f"Current metrics: {result['metrics']}\n\n"
            f"Available levers for this architecture:\n"
            f"- enable_feature_selection: true/false (recursive feature elimination)\n"
            f"- tuning_rounds: integer (hyperparameter tuning iterations)\n"
            f"- n_folds / n_repeats: cross-validation settings\n"
            f"- ensemble_strategy: mean / stacking / hill_climbing\n"
            f"{extra_info}\n\n"
            f"Suggest specific improvements as a JSON dict."
        )

        eval_result = self.evaluator.run(arch_context)

        suggestions: dict = {}
        try:
            if "```json" in eval_result:
                json_text = eval_result.split("```json")[1].split("```")[0]
                suggestions = json.loads(json_text)
        except (json.JSONDecodeError, IndexError):
            pass

        return suggestions

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
