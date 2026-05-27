"""Orchestrator: coordinates agents with sampling, checkpoints, and reporting."""

import concurrent.futures
import json
import logging
import os
import time
from typing import List, Optional

import dill
import numpy as np
import pandas as pd

from bluecast.ai.agents.arch_feature_engineer import ArchFeatureEngineerAgent
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
        custom_preprocessor=None,
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
            custom_preprocessor=custom_preprocessor,
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
        self.arch_engineer = ArchFeatureEngineerAgent(
            llm, self.context, verbose=verbose
        )

    # ------------------------------------------------------------------
    # Context file loading (PDF, docx, CSV, txt, md)
    # ------------------------------------------------------------------

    def _load_context_files(self) -> None:
        """Load domain knowledge files into context.

        Supports: .pdf (via pypdf), .docx (via python-docx),
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
            import pypdf

            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF support. "
                "Install it with: pip install pypdf"
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
        """Return the number of critique rounds based on mode.

        If the user set ``critique_max_rounds`` explicitly, that value is
        used directly.  Otherwise mode-based defaults apply:
        fast=0, balanced=1, precise=2, ultimate=5.
        """
        mode_defaults = {
            "fast": 0,
            "balanced": 1,
            "precise": 2,
            "ultimate": 5,
        }

        if self.config.critique_max_rounds is not None:
            # User explicitly set the value — honour it
            return max(0, self.config.critique_max_rounds)

        return mode_defaults.get(self.config.mode, 1)

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

        # Enable incremental log flushing so logs survive crashes / timeouts
        if not self.context.log_file_path:
            log_dir = os.getcwd()
            self.context.log_file_path = os.path.join(
                log_dir, "bluecastai_agent_log.jsonl"
            )
            logger.info(f"Incremental log file: {self.context.log_file_path}")

        if self.config.verbose:
            print("=" * 60)
            print("BlueCastAI - Multi-Agent AutoML Pipeline")
            print(f"  Log file: {self.context.log_file_path}")
            print("=" * 60)

        self._load_checkpoint()

        self.context.pipeline_start_time = start_time

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
        # Tool-based agents communicate through tool calls, not text.
        # The critic only sees truncated text and almost always returns
        # NEEDS_IMPROVEMENT, wasting the entire LLM budget.  Cap at 1.
        critique_rounds = min(critique_rounds, 1)

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

        # Extract imputation recommendations (sentinel detection + strategy evaluation)
        # so downstream FE agents receive them as context
        imputation_markers = [
            "Per-Column Imputation Recommendation",
            "Detected Sentinel Values",
            "Imputation Strategy Evaluation",
        ]
        for marker in imputation_markers:
            if marker in result:
                # Store the full imputation section for FE agents
                idx = result.index(marker)
                # Extract from the marker to end of the section (next ### or end)
                section = result[idx:]
                # Find the next top-level section boundary
                next_section = section.find("\n## ", 3)
                if next_section > 0:
                    section = section[:next_section]
                self.context.imputation_recommendations = section
                if self.config.verbose:
                    print(
                        "    [DataAnalyst] Imputation recommendations extracted and stored."
                    )
                break

        for keyword in [
            "leakage",
            "imbalance",
            "missing",
            "null",
            "duplicate",
            "constant",
            "outlier",
            "sentinel",
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
            f"Use your judgment to decide the right number of features based on the data.\n"
            f"Start with the simplest, most impactful features first.\n"
            f"If there are categorical columns with strong predictive potential, consider creating group-level aggregations (mean/std of numeric columns) using StateAwareGroupbyAggregator.\n"
            f"Call create_feature for each feature separately.\n"
            f"If any column contains free text, use create_tfidf_features."
        )

        critique_rounds = self._get_critique_rounds()
        # Same rationale as _step_analyze: FE agents express work via
        # tool calls.  Excessive critique rounds reset state and throw
        # away 80%+ of feature engineering effort.
        critique_rounds = min(critique_rounds, 1)

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

            df_val = self.context.df_train.copy()
            if self.context.target_col and self.context.target_col in df_val.columns:
                df_val = df_val.drop(columns=[self.context.target_col])

            valid_snippets = []

            for i, code in enumerate(self.context.feature_code_snippets):
                try:
                    # Pass 1: Simulate train time (is_fit=True)
                    state_train: dict = {}
                    df_train_pass = df_val.copy()
                    local_vars_train = {
                        "df": df_train_pass,
                        "np": np,
                        "pd": pd,
                        "state": state_train,
                        "is_fit": True,
                    }
                    exec(code, local_vars_train)  # noqa: S102

                    # Pass 2: Simulate inference time (is_fit=False)
                    # Use only 2 rows to ensure methods like qcut that fail on small sets are caught
                    df_infer_pass = df_val.head(2).copy()
                    local_vars_infer = {
                        "df": df_infer_pass,
                        "np": np,
                        "pd": pd,
                        "state": state_train,
                        "is_fit": False,
                    }
                    exec(code, local_vars_infer)  # noqa: S102

                    valid_snippets.append(code)
                except Exception as e:
                    if self.config.verbose:
                        print(
                            f"    Warning: FE snippet {i + 1} failed validation ({e}). Pruning."
                        )

            self.context.feature_code_snippets = valid_snippets

            # Re-run valid snippets on full df_train (with target) to build engineered_df
            df_final = self.context.df_train.copy()
            final_state: dict = {}
            for code in valid_snippets:
                local_vars = {
                    "df": df_final,
                    "np": np,
                    "pd": pd,
                    "state": final_state,
                    "is_fit": True,
                }
                exec(code, local_vars)  # noqa: S102
                df_final = local_vars.get("df", df_final)

            self.context.engineered_df = df_final

            # Prune snippets referencing constant columns (nunique <= 1).
            # These columns are typically dropped by BlueCast's internal
            # pipeline, causing KeyError during CV folds.
            # Check the engineered df (df_test) to catch junk columns
            # created by the FE agent (e.g. dummy_shape_check = 1).
            const_cols = [
                c
                for c in df_final.columns
                if c != self.context.target_col and df_final[c].nunique() <= 1
            ]
            if const_cols:
                pruned = []
                for code in valid_snippets:
                    refs = [c for c in const_cols if c in code]
                    if refs:
                        if self.config.verbose:
                            print(
                                f"    Pruning FE snippet referencing constant column(s): {refs}"
                            )
                    else:
                        pruned.append(code)
                valid_snippets = pruned
                self.context.feature_code_snippets = valid_snippets

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
            override_max_runtime = max(
                10, int(self.config.global_tuning_budget / total_jobs)
            )
            plan["tuning_max_runtime"] = override_max_runtime

        original_plan_limits = {
            "tuning_rounds": plan.get("tuning_rounds", 200),
            "tuning_max_runtime": plan.get("tuning_max_runtime", 1800),
        }

        for iteration in range(max_iterations):
            if (
                self.config.global_tuning_budget
                and self.config.global_tuning_budget > 0
            ):
                elapsed = time.time() - getattr(
                    self.context, "pipeline_start_time", time.time()
                )
                if elapsed > self.config.global_tuning_budget * 0.8:
                    if self.config.verbose:
                        print(
                            f"\n  [TIMEOUT] Global budget nearly exhausted ({elapsed:.0f}s). Stopping build loop early."
                        )
                    break

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

    def _step_ultimate_build_loop(self, plan: dict) -> None:  # noqa: C901
        """Train multiple architectures with coupled FE + model iteration.

        For each architecture:
          1. Architecture-specific FE agent creates model-tailored features
          2. Model is built + tuned
          3. Evaluator reviews metrics + feature importances
          4. Critique recommends changes → next iteration rebuilds FE
        """
        from bluecast.ai.architectures import get_architectures_for_problem
        from bluecast.ai.fe_preprocessor import AIFeaturePreprocessor

        problem = self.context.class_problem or "binary"
        archs = get_architectures_for_problem(problem)

        if self.config.architectures_to_run is not None:
            # Iterate in the user-specified order so strong architectures
            # (e.g. CatBoost) run first and get the tuning budget.
            archs = {
                k: archs[k] for k in self.config.architectures_to_run if k in archs
            }
            if not archs:
                raise ValueError(
                    f"None of the specified architectures {self.config.architectures_to_run} "
                    f"support problem type '{problem}' or exist in the registry."
                )

        iters = self.config.ultimate_iterations_per_arch
        total_archs = len(archs)

        # Cap iterations per arch based on budget so every architecture gets
        # enough compute.  Each iteration needs ~180s for FE + tuning + CV.
        if self.config.global_tuning_budget and self.config.global_tuning_budget > 0:
            max_affordable_iters = max(
                2, int(self.config.global_tuning_budget / (total_archs * 180))
            )
            if max_affordable_iters < iters:
                if self.config.verbose:
                    print(
                        f"  [BUDGET] Capping iterations from {iters} to "
                        f"{max_affordable_iters} per arch "
                        f"(budget={self.config.global_tuning_budget}s, "
                        f"{total_archs} archs)"
                    )
                iters = max_affordable_iters

        override_max_runtime = None
        if self.config.global_tuning_budget and self.config.global_tuning_budget > 0:
            tunable_archs = max(1, total_archs - 1)
            n_folds = plan.get("n_folds", 5)
            total_jobs = max(1, tunable_archs * iters * n_folds)
            override_max_runtime = max(
                10, int(self.config.global_tuning_budget / total_jobs)
            )

        if self.config.verbose:
            print(
                f"\nStep 5: Ultimate build loop "
                f"({total_archs} architectures × {iters} iterations)..."
            )

        for arch_idx, (arch_name, arch_info) in enumerate(archs.items(), 1):
            if self._is_step_done(f"build_arch_{arch_name}"):
                if self.config.verbose:
                    print(
                        f"\n  [{arch_idx}/{total_archs}] Skipping "
                        f"{arch_info['name']} (Loaded from checkpoint)"
                    )
                continue

            if self.config.verbose:
                print(f"\n  [{arch_idx}/{total_archs}] " f"=== {arch_info['name']} ===")

            ml_model = arch_info["factory"](problem)
            use_xgboost = arch_info.get("use_xgboost_native", False)
            arch_config = self._build_arch_config(plan, arch_name, override_max_runtime)

            # Set scoring for custom architectures if they support it
            if ml_model is not None and hasattr(ml_model, "scoring"):
                # Regression: dynamic scoring
                if arch_config.get("regression_eval_metric"):
                    from bluecast.ai.metrics import get_regression_metric_config

                    metric_config = get_regression_metric_config(
                        str(arch_config.get("regression_eval_metric", "mae"))
                    )
                    ml_model.scoring = metric_config["sklearn_scoring"]
                # Classification: balanced_accuracy scoring
                clf_metric = arch_config.get("classification_eval_metric")
                if clf_metric == "balanced_accuracy":
                    ml_model.scoring = "balanced_accuracy"
                elif clf_metric == "log_loss":
                    ml_model.scoring = "neg_log_loss"

            # Reset arch FE state for this architecture
            self.context.arch_feature_snippets[arch_name] = []
            self.arch_engineer.set_architecture(arch_name, arch_info["name"])

            # Track the best exploration iteration for this architecture
            best_arch_result = None
            best_arch_config = None
            best_arch_snippets = None
            arch_best_pipeline = None
            base_rounds = plan.get("tuning_rounds", 50)

            # --- N Exploration Iterations (10% tuning rounds, full FE) ---
            for iteration in range(iters):
                if (
                    self.config.global_tuning_budget
                    and self.config.global_tuning_budget > 0
                ):
                    elapsed = time.time() - getattr(
                        self.context, "pipeline_start_time", time.time()
                    )
                    if elapsed > self.config.global_tuning_budget * 0.8:
                        if self.config.verbose:
                            print(
                                f"\n  [TIMEOUT] Global budget nearly exhausted ({elapsed:.0f}s). Skipping further iterations for {arch_name}."
                            )
                        break

                if self.config.verbose:
                    print(f"    Iteration {iteration + 1}/{iters} (exploration):")

                # --- Exploration uses 10% of tuning rounds ---
                explore_rounds = max(5, int(base_rounds * 0.1))
                arch_config["tuning_rounds"] = explore_rounds
                if self.config.verbose:
                    print(f"      Tuning rounds: {explore_rounds} (10% exploration)")

                # --- 1. Architecture-specific Feature Engineering ---

                # Seed this iteration with the best-performing snippets
                # from previous iterations instead of starting from scratch.
                # This lets the LLM focus on adding NEW features rather than
                # wasting tool calls recreating the same proven features.
                global_snip_set = set(self.context.feature_code_snippets)
                inherited_snippets: list[str] = []
                if best_arch_snippets is not None:
                    inherited_snippets = [
                        s for s in best_arch_snippets if s not in global_snip_set
                    ]

                self.context.arch_feature_snippets[arch_name] = list(inherited_snippets)

                arch_fe_task = self._create_arch_fe_task(
                    arch_name,
                    arch_info["name"],
                    iteration,
                    iters,
                    inherited_snippets=inherited_snippets,
                )

                # Initialize engineered_df with global + inherited features
                # so arch FE builds on top of them
                seed_snippets = list(self.context.feature_code_snippets) + list(
                    inherited_snippets
                )
                if seed_snippets and self.context.df_train is not None:
                    seed_prep = AIFeaturePreprocessor(seed_snippets)
                    df_base, _ = seed_prep.fit_transform(
                        self.context.df_train.copy(), target=None
                    )
                    self.context.engineered_df = df_base
                else:
                    self.context.engineered_df = None

                # Give later iterations a larger tool call budget since the
                # LLM has more context (inherited features, quality checks)
                # and needs room to experiment.
                if iteration >= 3:
                    self.arch_engineer.max_tool_iterations = 15
                else:
                    self.arch_engineer.max_tool_iterations = 10

                if self.context.custom_preprocessor is not None:
                    if self.config.verbose:
                        print(
                            "      [SKIP FE] Custom preprocessor provided — skipping LLM FE agent"
                        )
                else:
                    self.arch_engineer.run(arch_fe_task)
                # Convert columns_to_drop into a snippet to ensure it runs during inference
                if "columns_to_drop" in arch_config and isinstance(
                    arch_config["columns_to_drop"], list
                ):
                    drop_cols = arch_config.pop("columns_to_drop")
                    if drop_cols:
                        drop_code = (
                            f"df = df.drop(columns={drop_cols}, errors='ignore')"
                        )
                        self.context.arch_feature_snippets[arch_name].append(drop_code)

                if self.config.verbose:
                    n_arch_snippets = len(
                        self.context.arch_feature_snippets.get(arch_name, [])
                    )
                    print(f"      Arch FE: {n_arch_snippets} snippets created")

                # --- 2. Combine base + arch snippets → preprocessor ---
                # If the user provided a custom_preprocessor, always use it.
                # LLM-generated snippets should not silently replace a
                # deterministic, user-provided preprocessor.
                combined_snippets = list(self.context.feature_code_snippets) + list(
                    self.context.arch_feature_snippets.get(arch_name, [])
                )
                if self.context.custom_preprocessor is not None:
                    preprocessor = self.context.custom_preprocessor
                elif combined_snippets:
                    preprocessor = AIFeaturePreprocessor(combined_snippets)
                else:
                    preprocessor = None

                # --- 3. Build + tune model ---
                config = dict(arch_config)
                if not use_xgboost:
                    config["ml_model"] = ml_model

                result = self._build_single_arch(
                    config, arch_name, use_xgboost, preprocessor=preprocessor
                )

                if not result["success"]:
                    error_msg = result.get("error", "unknown")
                    if self.config.verbose:
                        print(f"      FAILED: {error_msg}")

                    # Record the error for this architecture to provide feedback
                    self.context.arch_errors[arch_name] = str(error_msg)

                    # Record in run history
                    self.context.run_history.append(
                        {
                            "success": False,
                            "error": str(error_msg),
                            "architecture": arch_name,
                            "iteration": iteration + 1,
                            "config": result.get("config_used", config),
                            "snippets": list(
                                self.context.arch_feature_snippets.get(arch_name, [])
                            ),
                        }
                    )
                    continue  # Move to next iteration instead of breaking

                if self.config.verbose:
                    print(f"      Metrics: {result['metrics']}")

                # --- 4. Extract feature importances and error analysis ---
                self._extract_feature_importances(arch_name, result)
                self._extract_error_analysis(arch_name, result)

                # --- 5. Track best exploration result for this architecture ---
                is_arch_best = (best_arch_result is None) or self._compare_results(
                    result, best_arch_result
                )
                if is_arch_best:
                    best_arch_result = result
                    best_arch_config = dict(config)
                    best_arch_snippets = list(combined_snippets)
                    arch_best_pipeline = result.get("pipeline")
                    if self.config.verbose:
                        print(
                            f"      [Best exploration] {arch_name} with {result['metrics']}"
                        )

                # --- 6. Update global best pipeline ---
                is_better = self._is_result_better(result)
                if is_better:
                    self.context.best_pipeline = result["pipeline"]
                    self.context.best_metrics = result["metrics"]

                    # Store the complete FE code for the best architecture so it can be exported
                    arch_snips = self.context.arch_feature_snippets.get(arch_name, [])
                    all_snippets = list(self.context.feature_code_snippets) + list(
                        arch_snips
                    )
                    if all_snippets:
                        self.context.feature_engineering_code = "\n\n".join(
                            all_snippets
                        )

                    if self.config.verbose:
                        print(
                            f"      [Best overall] {arch_name} "
                            f"with {result['metrics']}"
                        )

                # Record in run history
                self.context.run_history.append(
                    {
                        "success": True,
                        "metrics": result["metrics"],
                        "config": result["config_used"],
                        "architecture": arch_name,
                        "iteration": iteration + 1,
                        "snippets": list(
                            self.context.arch_feature_snippets.get(arch_name, [])
                        ),
                    }
                )

                # --- 7. Evaluator + critique for next iteration ---
                if iteration < iters - 1:
                    suggestions = self._evaluate_for_arch(
                        arch_name,
                        arch_info["name"],
                        result,
                        iteration=iteration,
                        total_iterations=iters,
                    )
                    arch_config.update(suggestions)
                    arch_config = self._enforce_config_constraints(
                        arch_config, plan, arch_name, override_max_runtime
                    )

            # --- N+1 Refinement Iteration (100% tuning, best FE snippets, no FE agent) ---
            if best_arch_result is not None and iters > 1:
                # Check timeout before refinement
                budget_ok = True
                if (
                    self.config.global_tuning_budget
                    and self.config.global_tuning_budget > 0
                ):
                    elapsed = time.time() - getattr(
                        self.context, "pipeline_start_time", time.time()
                    )
                    if elapsed > self.config.global_tuning_budget * 0.7:
                        budget_ok = False
                        if self.config.verbose:
                            print(
                                f"    [SKIP REFINEMENT] Budget nearly exhausted ({elapsed:.0f}s)."
                            )

                if budget_ok:
                    if self.config.verbose:
                        print(
                            "    Refinement (100% tuning on best exploration config):"
                        )
                        print(f"      Tuning rounds: {base_rounds} (full budget)")

                    # Rebuild preprocessor from the best exploration snippets
                    refinement_preprocessor = (
                        AIFeaturePreprocessor(best_arch_snippets)
                        if best_arch_snippets
                        else None
                    )

                    # Use best config but with full tuning rounds
                    refinement_config = (
                        dict(best_arch_config) if best_arch_config is not None else {}
                    )
                    refinement_config["tuning_rounds"] = base_rounds

                    # Re-instantiate a fresh model for refinement
                    refinement_ml_model = arch_info["factory"](problem)
                    if refinement_ml_model is not None and hasattr(
                        refinement_ml_model, "scoring"
                    ):
                        if arch_config.get("regression_eval_metric"):
                            from bluecast.ai.metrics import get_regression_metric_config

                            metric_config = get_regression_metric_config(
                                str(arch_config.get("regression_eval_metric", "mae"))
                            )
                            refinement_ml_model.scoring = metric_config[
                                "sklearn_scoring"
                            ]
                        clf_metric = arch_config.get("classification_eval_metric")
                        if clf_metric == "balanced_accuracy":
                            refinement_ml_model.scoring = "balanced_accuracy"
                        elif clf_metric == "log_loss":
                            refinement_ml_model.scoring = "neg_log_loss"

                    if not use_xgboost:
                        refinement_config["ml_model"] = refinement_ml_model

                    refinement_result = self._build_single_arch(
                        refinement_config,
                        arch_name,
                        use_xgboost,
                        preprocessor=refinement_preprocessor,
                    )

                    if refinement_result["success"]:
                        if self.config.verbose:
                            print(
                                f"      Refinement metrics: {refinement_result['metrics']}"
                            )

                        # Only keep refinement if it beats the best exploration result
                        refinement_is_better = self._compare_results(
                            refinement_result, best_arch_result
                        )
                        if refinement_is_better:
                            arch_best_pipeline = refinement_result.get("pipeline")
                            if self.config.verbose:
                                print(
                                    "      [Refinement IMPROVED] Keeping refined model."
                                )

                            # Update global best if refinement is overall best
                            if self._is_result_better(refinement_result):
                                self.context.best_pipeline = refinement_result[
                                    "pipeline"
                                ]
                                self.context.best_metrics = refinement_result["metrics"]
                                if best_arch_snippets:
                                    self.context.feature_engineering_code = "\n\n".join(
                                        best_arch_snippets
                                    )

                            self.context.run_history.append(
                                {
                                    "success": True,
                                    "metrics": refinement_result["metrics"],
                                    "config": refinement_result["config_used"],
                                    "architecture": arch_name,
                                    "iteration": "refinement",
                                }
                            )
                        else:
                            if self.config.verbose:
                                print(
                                    "      [Refinement DISCARDED] Exploration result was better."
                                )
                    else:
                        if self.config.verbose:
                            print(
                                f"      Refinement FAILED: {refinement_result.get('error', 'unknown')}"
                            )

            if arch_best_pipeline is not None:
                self.context.best_pipelines.append(arch_best_pipeline)

            self._save_checkpoint(f"build_arch_{arch_name}")

    def _enforce_config_constraints(
        self,
        config: dict,
        original_plan: dict,
        arch_name: str = "",
        override_max_runtime: Optional[int] = None,
    ) -> dict:
        """Enforce strict bounds on tuning rounds and runtime to prevent runaway LLM configs."""
        max_rounds = original_plan.get("tuning_rounds")
        if max_rounds is None:
            max_rounds = 200

        max_runtime = (
            override_max_runtime
            if override_max_runtime
            else original_plan.get("tuning_max_runtime")
        )
        if max_runtime is None:
            max_runtime = 1800

        if "tuning_rounds" in config:
            val = config.get("tuning_rounds")
            if val is None:
                val = max_rounds
            if isinstance(val, int) and isinstance(max_rounds, int):
                config["tuning_rounds"] = min(val, max_rounds)
            else:
                config["tuning_rounds"] = max_rounds

        if "tuning_max_runtime" in config:
            val = config.get("tuning_max_runtime")
            if val is None:
                val = max_runtime
            if isinstance(val, int) and isinstance(max_runtime, int):
                config["tuning_max_runtime"] = min(val, max_runtime)
            else:
                config["tuning_max_runtime"] = max_runtime

        if "nn_max_iter" in config:
            current = config.get("nn_max_iter", 200)
            config["nn_max_iter"] = min(current, 1000)

        if "enable_feature_selection" in config:
            config["enable_feature_selection"] = False

        if arch_name in ["linear", "randomforest", "mlp", "so1dcnn"]:
            config["cat_encoding_via_ml_algorithm"] = False

        return config

    def _build_arch_config(
        self, plan: dict, arch_name: str, override_max_runtime: Optional[int] = None
    ) -> dict:
        """Build a base pipeline config for a specific architecture."""
        class_problem = plan.get(
            "class_problem", self.context.class_problem or "binary"
        )
        config = {
            "class_problem": class_problem,
            "use_cv": plan.get("use_cv", True),
            "ensemble_strategy": plan.get("ensemble_strategy", "hill_climbing"),
            "n_folds": plan.get("n_folds", 5),
            "n_repeats": plan.get("n_repeats", 1),
            "tuning_rounds": plan.get("tuning_rounds", 50),
            "tuning_max_runtime": (
                override_max_runtime
                if override_max_runtime
                else plan.get("tuning_max_runtime", 120)
            ),
            "autotune_on_device": self.config.autotune_on_device,
        }

        # Propagate regression eval metric so MAE (or other metrics) are
        # used instead of defaulting to RMSE in tool_build_and_run_pipeline.
        # Only propagate for regression tasks to prevent classification
        # pipelines from being poisoned with regression objectives.
        if class_problem == "regression" and plan.get("regression_eval_metric"):
            config["regression_eval_metric"] = plan["regression_eval_metric"]

        # Propagate classification eval metric (e.g. balanced_accuracy)
        if class_problem != "regression" and plan.get("classification_eval_metric"):
            config["classification_eval_metric"] = plan["classification_eval_metric"]

        # Linear models don't benefit from gradient boosting tuning
        if arch_name == "linear":
            config["tuning_rounds"] = 1
            config["tuning_max_runtime"] = 30

        if arch_name in ["linear", "randomforest", "mlp", "so1dcnn"]:
            config["cat_encoding_via_ml_algorithm"] = False

        return config

    def _build_single_arch(
        self,
        config: dict,
        arch_name: str,
        use_xgboost: bool,
        preprocessor=None,
    ) -> dict:
        """Build a single architecture pipeline.

        :param preprocessor: If provided, use this preprocessor instead of
            constructing one from shared ``context.feature_code_snippets``.
        """
        from bluecast.ai.tools import tool_build_and_run_pipeline

        if preprocessor is None:
            if self.context.custom_preprocessor is not None:
                preprocessor = self.context.custom_preprocessor
            elif self.context.feature_code_snippets:
                from bluecast.ai.fe_preprocessor import AIFeaturePreprocessor

                preprocessor = AIFeaturePreprocessor(
                    list(self.context.feature_code_snippets)
                )
            else:
                preprocessor = None

        ml_model = config.pop("ml_model", None)

        if use_xgboost:
            from bluecast.ml_modelling.xgboost import XgboostModel
            from bluecast.ml_modelling.xgboost_regression import XgboostModelRegression

            problem = config.get(
                "class_problem", self.context.class_problem or "binary"
            )
            if problem == "regression":
                conf_xgboost = None
                conf_params_xgboost = None
                if config.get("regression_eval_metric"):
                    from bluecast.ai.metrics import get_regression_metric_config
                    from bluecast.config.training_config import (
                        XgboostRegressionFinalParamConfig,
                        XgboostTuneParamsRegressionConfig,
                    )

                    metric_config = get_regression_metric_config(
                        str(config.get("regression_eval_metric", "mae"))
                    )
                    conf_xgboost = XgboostTuneParamsRegressionConfig(
                        xgboost_eval_metric=metric_config.get(
                            "catboost_loss", "RMSE"
                        ).lower(),  # xgboost eval metrics are usually lower case of catboost ones or specific strings
                        xgboost_objective=metric_config["xgboost_loss"],
                    )
                    conf_params_xgboost = XgboostRegressionFinalParamConfig()

                    # Ensure we pass the right eval metric
                    if metric_config["xgboost_loss"] == "reg:absoluteerror":
                        eval_m = "mae"
                    elif metric_config["xgboost_loss"] == "reg:squarederror":
                        eval_m = "rmse"
                    else:
                        eval_m = metric_config.get("catboost_loss", "rmse").lower()

                    conf_params_xgboost.params["eval_metric"] = eval_m
                    conf_params_xgboost.params["objective"] = metric_config[
                        "xgboost_loss"
                    ]

                ml_model = XgboostModelRegression(
                    class_problem="regression",
                    conf_xgboost=conf_xgboost,
                    conf_params_xgboost=conf_params_xgboost,
                )
            else:
                ml_model = XgboostModel(class_problem=problem)

        return tool_build_and_run_pipeline(
            self.context.df_train,
            self.context.target_col,
            config,
            custom_preprocessor=preprocessor,
            ml_model=ml_model,
        )

    def _create_arch_fe_task(
        self,
        arch_name: str,
        arch_display_name: str,
        iteration: int,
        total_iterations: int = 1,
        inherited_snippets: Optional[List[str]] = None,
    ) -> str:
        """Build a task string for the architecture-specific FE agent."""
        data_summary = self.context.get_data_summary()

        # --- Inherited features info ---
        inherited_info = ""
        if inherited_snippets:
            inherited_info = (
                "\n\n--- INHERITED FEATURES (already applied, do NOT recreate) ---\n"
                f"The following {len(inherited_snippets)} feature snippet(s) from the best "
                "previous iteration are ALREADY applied to the DataFrame. "
                "Do NOT call create_feature for these — they are pre-loaded.\n"
                "```python\n" + "\n# ---\n".join(inherited_snippets) + "\n```\n"
                "Focus ONLY on creating NEW features that complement these.\n"
            )

        # --- Iteration-specific strategy ---
        if iteration == 0:
            strategy = (
                "STRATEGY: This is the FIRST iteration — establish a baseline.\n"
                "Create 1-3 simple, high-signal features (e.g., missing value "
                "indicators, basic imputation, and strongly correlated group-level aggregations).\n"
                "The goal is speed: get a baseline score quickly so later iterations "
                "can compare against it."
            )
        elif iteration < total_iterations - 1:
            n_inherited = len(inherited_snippets) if inherited_snippets else 0
            strategy = (
                f"STRATEGY: This is iteration {iteration + 1} of {total_iterations} — improve on the best so far.\n"
                f"The {n_inherited} best features from previous iterations are ALREADY applied (see above).\n"
                f"Create 2-4 NEW features on top of them. Focus on interaction features and "
                f"ratios between the top predictors identified in the previous iteration.\n"
                f"Use feature importances below to decide which columns to combine.\n"
                f"If some inherited features have near-zero importance, consider using "
                f"drop_collinear_features or l1_feature_selection to prune them."
            )
        else:
            n_inherited = len(inherited_snippets) if inherited_snippets else 0
            strategy = (
                f"STRATEGY: This is the FINAL iteration ({iteration + 1} of {total_iterations}) — maximize performance.\n"
                f"The {n_inherited} best features from previous iterations are ALREADY applied (see above).\n"
                f"Create 3-5 NEW advanced features: group-level aggregations, polynomial features, "
                f"binned features. Focus specifically on the error analysis rows where the "
                f"model struggles most.\n"
                f"Consider pruning low-importance inherited features via drop_collinear_features."
            )

        # Include previous iteration feedback if available
        feedback = ""
        importances = self.context.arch_feature_importances.get(arch_name)
        if importances and iteration > 0:
            sorted_feats = sorted(
                importances.items(), key=lambda x: abs(x[1]), reverse=True
            )
            top_5 = sorted_feats[:5]
            feedback += (
                "\n\nPrevious iteration results are available. "
                "Top features by importance: "
                + ", ".join(f"{f}={v:.4f}" for f, v in top_5)
                + "\nUse this to guide your feature engineering — create "
                "more features similar to the top ones and avoid "
                "creating features similar to low-importance ones."
            )

        error_analysis = self.context.arch_error_analysis.get(arch_name)
        if error_analysis and iteration > 0:
            feedback += (
                f"\n\nERROR ANALYSIS FROM PREVIOUS ITERATION (Out-of-Fold):\n"
                f"The model struggles most with the following unseen rows (highest OOF residuals/loss):\n"
                f"{error_analysis}\n"
                f"Analyze these specific rows. What feature is missing that would help the model predict them correctly?"
            )

        # Include full iteration history for this architecture
        arch_runs = [
            r for r in self.context.run_history if r.get("architecture") == arch_name
        ]
        metrics_info = ""
        if arch_runs:
            metrics_info = "\n\n--- CUMULATIVE ITERATION HISTORY ---\n"
            for r in arch_runs:
                iter_idx = r.get("iteration", "?")
                snippets = r.get("snippets", [])
                snip_text = (
                    "\n".join(snippets) if snippets else "No features generated."
                )
                if r.get("success"):
                    metrics_info += f"Iteration {iter_idx}: Achieved Metrics: {r.get('metrics', 'N/A')}\nSnippets Used:\n```python\n{snip_text}\n```\n\n"
                else:
                    metrics_info += f"Iteration {iter_idx}: FAILED with error: {r.get('error', 'unknown error')}\nSnippets Used:\n```python\n{snip_text}\n```\n\n"
            metrics_info += "Use this history to see what worked and what failed. Do not repeat failed experiments. Build upon the successful ones.\n"

        # Check for persistent errors in context
        error_feedback = ""
        last_error = self.context.arch_errors.get(arch_name)
        if last_error:
            error_feedback = (
                f"\n\nCRITICAL: The last build for this architecture FAILED. "
                f"Error: {last_error}\n"
                f"Please analyze if your proposed features or configuration "
                f"caused this (e.g. infinity values, nulls, or incompatible "
                f"categorical encoding) and adjust your strategy to fix it."
            )

        # Include imputation recommendations from DataAnalyst
        imputation_ctx = ""
        if self.context.imputation_recommendations:
            imputation_ctx = (
                f"\n\nIMPUTATION GUIDANCE FROM DATA ANALYST:\n"
                f"{self.context.imputation_recommendations}\n"
                f"Use the recommended imputation strategies above when handling "
                f"missing/sentinel values in your features."
            )

        return (
            f"Create features specifically for the **{arch_display_name}** "
            f"model (iteration {iteration + 1} of {total_iterations}).\n\n"
            f"{strategy}\n\n"
            f"Dataset:\n{data_summary}\n"
            f"{inherited_info}{metrics_info}{feedback}{error_feedback}{imputation_ctx}"
        )

    def _extract_feature_importances(
        self,
        arch_name: str,
        result: dict,
    ) -> None:
        """Extract feature importances from the trained pipeline."""
        pipeline = result.get("pipeline")
        if pipeline is None:
            return

        try:
            # BlueCast pipelines store feature importances in different places
            importances: dict = {}

            # Try to get SHAP-based or model-based feature importances
            if hasattr(pipeline, "feature_importances"):
                raw = pipeline.feature_importances
                if isinstance(raw, dict):
                    importances = raw
                elif hasattr(raw, "items"):
                    importances = dict(raw.items())

            # For CV pipelines, try to get from the first trained model
            if not importances and hasattr(pipeline, "bluecast_models"):
                models = pipeline.bluecast_models
                if models and len(models) > 0:
                    first_model = models[0]
                    if hasattr(first_model, "feature_importances"):
                        raw = first_model.feature_importances
                        if isinstance(raw, dict):
                            importances = raw

            if importances:
                self.context.arch_feature_importances[arch_name] = importances
                if self.config.verbose:
                    top_3 = sorted(
                        importances.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:3]
                    print(
                        "      Feature importances: top-3 = "
                        + ", ".join(f"{k}={v:.4f}" for k, v in top_3)
                    )
        except Exception as e:
            logger.debug(f"Could not extract feature importances: {e}")

    def _extract_error_analysis(self, arch_name: str, result: dict) -> None:
        """Run predictions on the OOF data to find rows with highest residuals."""
        pipeline = result.get("pipeline")
        if pipeline is None:
            return

        try:
            # For CV pipelines, the OOF predictions are stored on the inner auto_pipeline
            inner_pipeline = getattr(pipeline, "auto_pipeline", pipeline)

            if not hasattr(inner_pipeline, "oof_predictions_") or not hasattr(
                inner_pipeline, "oof_valid_mask_"
            ):
                return

            y_preds = inner_pipeline.oof_predictions_
            mask = inner_pipeline.oof_valid_mask_

            # Reconstruct the feature set for the OOF rows to provide full context to the LLM
            # Ensure df_train is aligned
            df_train = self.context.get_working_df()
            if len(mask) == len(df_train):
                df_res = df_train[mask].copy()
                df_res["_prediction"] = y_preds

                from bluecast.ai.tools import tool_inspect_residuals

                task_type = self.context.class_problem or "regression"
                target_col = self.context.target_col

                residuals_str = tool_inspect_residuals(
                    df_res, target_col, "_prediction", task_type, n_rows=10
                )
                if (
                    "Residual analysis failed" not in residuals_str
                    and "Target or prediction column not found" not in residuals_str
                ):
                    self.context.arch_error_analysis[arch_name] = residuals_str
                    if self.config.verbose:
                        print(
                            f"      Extracted OOF error analysis for {arch_name} (top 10 residuals)."
                        )
            else:
                logger.debug(
                    "OOF mask length does not match training data length; skipping error analysis."
                )

        except Exception as e:
            logger.debug(f"Could not extract error analysis: {e}")

    def _is_result_better(self, result: dict) -> bool:
        """Check if a result is better than the current best."""
        if self.context.best_metrics is None:
            return True

        new_m = result["metrics"]
        old_m = self.context.best_metrics
        eval_metrics = [
            "roc_auc",
            "oof_mean",
            "r2_score",
            "mae",
            "rmse",
            "mse",
            "mean_absolute_error",
            "mean_squared_error",
            "median_absolute_error",
            "mean_squared_log_error",
        ]
        error_metrics = [
            "oof_mean",
            "mae",
            "rmse",
            "mse",
            "mean_absolute_error",
            "mean_squared_error",
            "median_absolute_error",
            "mean_squared_log_error",
        ]
        for key in eval_metrics:
            if key in new_m and key in old_m:
                if key in error_metrics:
                    return abs(new_m[key]) < abs(old_m[key])
                return new_m[key] > old_m[key]
        if not old_m:
            return True
        return False

    def _compare_results(self, new_result: dict, old_result: dict) -> bool:
        """Compare two result dicts directly. Returns True if new_result is better."""
        new_m = new_result.get("metrics", {})
        old_m = old_result.get("metrics", {})
        if not old_m:
            return True
        eval_metrics = [
            "roc_auc",
            "oof_mean",
            "r2_score",
            "mae",
            "rmse",
            "mse",
            "mean_absolute_error",
            "mean_squared_error",
            "median_absolute_error",
            "mean_squared_log_error",
        ]
        error_metrics = [
            "oof_mean",
            "mae",
            "rmse",
            "mse",
            "mean_absolute_error",
            "mean_squared_error",
            "median_absolute_error",
            "mean_squared_log_error",
        ]
        for key in eval_metrics:
            if key in new_m and key in old_m:
                if key in error_metrics:
                    return abs(new_m[key]) < abs(old_m[key])
                return new_m[key] > old_m[key]
        return False

    def _evaluate_for_arch(
        self,
        arch_name: str,
        arch_display_name: str,
        result: dict,
        iteration: int = 0,
        total_iterations: int = 1,
    ) -> dict:
        """Ask the Evaluator for architecture-specific improvements.

        Provides feature importance data and arch-specific FE guidance
        so the evaluator can recommend both model config and FE changes.
        """
        extra_info = ""
        if arch_name == "linear":
            extra_info = (
                "\nWARNING: Linear models strictly require rigorous "
                "missing value imputation, categorical encoding, and "
                "feature scaling to perform well."
            )
        elif arch_name in ("histgb", "xgboost"):
            extra_info = (
                "\nWARNING: This architecture strictly requires categorical "
                "features to be numerically encoded and missing values "
                "to be imputed."
            )

        # Include feature importance data
        importance_info = ""
        importances = self.context.arch_feature_importances.get(arch_name)
        if importances:
            sorted_feats = sorted(
                importances.items(), key=lambda x: abs(x[1]), reverse=True
            )
            importance_info = "\n\nFeature importances:\n"
            for feat, imp in sorted_feats[:15]:
                importance_info += f"  {feat}: {imp:.4f}\n"
            if len(sorted_feats) > 15:
                importance_info += f"  ... ({len(sorted_feats) - 15} more)\n"

        # Include error info if any
        error_info = ""
        last_error = self.context.arch_errors.get(arch_name)
        if last_error:
            error_info = (
                f"\n\nWARNING: The last model build FAILED with this error:\n"
                f"'''\n{last_error}\n'''\n"
                f"Diagnose the cause (e.g. data types, tuning limits, or "
                f"specific features) and suggest a fix."
            )

        # Include data profile so the Evaluator knows column cardinalities
        data_profile = self.context.data_profile
        profile_info = ""
        if data_profile:
            profile_info = f"\n\nINITIAL DATA ANALYSIS:\n{data_profile}\n"

        # Iteration context for strategic recommendations
        remaining = total_iterations - iteration - 1
        iteration_context = (
            f"\n\nIteration context: This is iteration {iteration + 1} of {total_iterations} "
            f"({remaining} iteration(s) remaining after this).\n"
        )
        if iteration == 0:
            iteration_context += (
                "This was the BASELINE iteration with minimal features and tuning. "
                "Focus your recommendations on the most impactful improvements "
                "for the next iteration. Be conservative — suggest moderate "
                "tuning budget increases, not maximum values."
            )
        elif remaining > 1:
            iteration_context += (
                "There are several iterations remaining. Suggest targeted "
                "improvements — don't try to fix everything at once."
            )
        else:
            iteration_context += (
                "This is the SECOND-TO-LAST iteration. Suggest aggressive "
                "improvements: maximum tuning rounds, expanded search spaces, "
                "and advanced feature engineering for the final run."
            )

        # Include the actual FE snippets used so the evaluator can make
        # targeted FE recommendations (e.g., "drop the PCA feature")
        snippet_info = ""
        arch_snippets = self.context.arch_feature_snippets.get(arch_name, [])
        if arch_snippets:
            snippet_info = (
                "\n\nFEATURE ENGINEERING SNIPPETS USED THIS ITERATION:\n"
                "```python\n" + "\n# ---\n".join(arch_snippets) + "\n```\n"
            )

        convergence_info = result.get("convergence_info", {})
        convergence_context = ""
        if convergence_info:
            convergence_context = (
                f"\n\nNN CONVERGENCE DIAGNOSTICS:\n"
                f"  max_iter used: {convergence_info.get('nn_max_iter_used', 'N/A')}\n"
                f"  Trials completed: {convergence_info.get('trials_completed', 'N/A')}\n"
                f"  Best trial score: {convergence_info.get('best_trial_score', 'N/A')}\n"
                f"\nIf the model did not converge (budget_exhausted=True or "
                f"epochs_run == max_iter), you may increase nn_max_iter by up to 2×. "
                f'Set it via: {{"nn_max_iter": <new_value>}} in your JSON response. '
                f"Maximum allowed: 1000.\n"
            )

        arch_context = (
            f"You are evaluating the **{arch_display_name}** architecture track.\n"
            f"Current metrics: {result['metrics'] if result.get('success') else 'N/A'}\n\n"
            f"Available levers for this architecture:\n"
            f"- enable_feature_selection: true/false (recursive feature elimination)\n"
            f"- tuning_rounds: integer (hyperparameter tuning iterations)\n"
            f"- n_folds / n_repeats: cross-validation settings\n"
            f"- ensemble_strategy: mean / stacking / hill_climbing\n"
            f"- columns_to_drop: list of column names to exclude\n"
            f"- nn_max_iter: integer (max training epochs for PyTorch models, current: {result.get('config_used', {}).get('nn_max_iter', 200)})\n"
            f"- Architecture specific bounds: e.g. rf_max_depth_max, rf_estimators_max, catboost_depth_max, histgb_depth_max, etc.\n"
            f"{extra_info}{importance_info}{snippet_info}{error_info}{profile_info}{iteration_context}{convergence_context}\n\n"
            f"Based on the feature importances and metrics, suggest specific "
            f"improvements as a JSON dict. Consider recommending:\n"
            f"- Which features to drop (low importance or high risk)\n"
            f"- LEAKAGE CHECK: Review the initial data analysis above. If any generated features group by categorical columns that have >50 unique values (high cardinality), recommend dropping them immediately! Grouping by granular IDs causes severe target leakage on validation folds.\n"
            f"- Whether to enable recursive feature selection\n"
            f"- Tuning parameter adjustments. If `tuning_score` is significantly better (lower error or higher metric) than `oof_mean`, the model is overfitting — decrease `max_depth_max` or increase regularization. If both are poor, try increasing `tuning_rounds` or expanding the search space bounds.\n"  # noqa: E501
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
            "ensemble_strategy": plan.get("ensemble_strategy", "hill_climbing"),
            "n_folds": plan.get("n_folds", 5),
            "n_repeats": plan.get("n_repeats", 1),
            "tuning_rounds": plan.get("tuning_rounds", 50),
            "tuning_max_runtime": plan.get("tuning_max_runtime", 120),
        }

        if iteration > 0:
            config_hints["tuning_rounds"] = min(
                config_hints["tuning_rounds"] * (iteration + 1), 500
            )
        if iteration >= 2 and config_hints["ensemble_strategy"] == "stacking":
            config_hints["ensemble_strategy"] = "hill_climbing"

        return (
            f"Build and run a BlueCast pipeline (iteration {iteration + 1}).\n"
            f"Recommended configuration: {json.dumps(config_hints)}\n"
            f"Call build_and_run_pipeline with these parameters."
        )

    def _assemble_result(self) -> BlueCastAIResult:
        from bluecast.ensemble.hill_climbing import (
            HillClimbingEnsemble,
            _mae_regression_metric,
        )

        hc_ensemble = None
        valid_pipelines = []

        if len(self.context.best_pipelines) > 1:
            oof_list = []
            valid_masks = []
            for p in self.context.best_pipelines:
                if hasattr(p, "oof_predictions_") and hasattr(p, "oof_valid_mask_"):
                    oof_list.append(p.oof_predictions_)
                    valid_masks.append(p.oof_valid_mask_)
                    valid_pipelines.append(p)

            if len(oof_list) > 1 and self.context.df_train is not None:
                # Find rows where all architectures successfully predicted
                common_valid_mask = np.all(np.column_stack(valid_masks), axis=1)

                # Extract true targets
                y_true = self.context.df_train[self.context.target_col].values[
                    common_valid_mask
                ]

                # Mask OOF predictions
                filtered_oof_list = [oof[common_valid_mask] for oof in oof_list]

                is_classification = self.context.class_problem != "regression"
                eval_metric = _mae_regression_metric if not is_classification else None

                # --- NEW FILTERING LOGIC ---
                if not is_classification and eval_metric == _mae_regression_metric:
                    # Calculate negative MAE for each architecture
                    arch_scores = [
                        eval_metric(y_true, oof) for oof in filtered_oof_list
                    ]
                    best_mae = -max(arch_scores)  # convert back to positive MAE

                    final_oof_list = []
                    final_pipelines = []
                    for i, score in enumerate(arch_scores):
                        mae = -score
                        if mae <= 1.5 * best_mae:
                            final_oof_list.append(filtered_oof_list[i])
                            final_pipelines.append(valid_pipelines[i])
                        elif self.config.verbose:
                            print(
                                f"    [SKIP ENSEMBLE] Architecture {i} OOF MAE {mae:.2f} is more than 2x worse than best {best_mae:.2f}. Excluding."
                            )

                    filtered_oof_list = final_oof_list
                    valid_pipelines = final_pipelines
                # ---------------------------

                if len(filtered_oof_list) > 0:
                    hc_ensemble = HillClimbingEnsemble(
                        is_classification=is_classification,
                        eval_metric=eval_metric,
                        blending_method="probability",  # Use raw predictions for Regression
                        weight_min=0.0,
                        tolerance=1e-4,
                    )

                    # Fit global ensemble
                    model_names = [f"arch_{i}" for i in range(len(filtered_oof_list))]
                    hc_ensemble.fit(filtered_oof_list, y_true, model_names)
        else:
            valid_pipelines = self.context.best_pipelines

        result = BlueCastAIResult(
            pipeline=self.context.best_pipeline,
            pipelines=valid_pipelines,
            hill_climbing_ensemble=hc_ensemble,
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
