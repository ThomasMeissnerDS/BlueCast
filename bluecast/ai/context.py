"""Shared context that accumulates knowledge across agents."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AgentLogEntry:
    """A single structured log entry from an agent interaction."""

    timestamp: float
    agent: str
    event_type: str  # "task", "tool_call", "tool_result", "response", "error"
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"[{ts}] [{self.agent}] ({self.event_type}) {self.content[:200]}"


@dataclass
class SharedContext:
    """Mutable state shared between all agents during a BlueCastAI run.

    Each agent reads from and writes to this context, building up a
    progressively richer understanding of the data and pipeline.
    """

    # Inputs
    df_train: Optional[pd.DataFrame] = None
    target_col: str = ""
    user_prompt: str = ""
    mode: str = "balanced"
    context_file_contents: List[str] = field(default_factory=list)

    # Sampling
    df_sample: Optional[pd.DataFrame] = None
    was_sampled: bool = False
    original_shape: Optional[tuple] = None

    # execution metadata
    prompt_tokens: int = 0
    completion_tokens: int = 0
    callbacks: List[Callable] = field(default_factory=list)

    # Detected by DataAnalyst
    class_problem: Optional[str] = None
    data_profile: Optional[Dict[str, Any]] = None
    data_warnings: List[str] = field(default_factory=list)

    # Created by FeatureEngineer
    feature_engineering_code: Optional[str] = None
    engineered_df: Optional[pd.DataFrame] = None

    # Created by PipelineBuilder
    pipeline_config: Optional[Dict[str, Any]] = None
    pipeline_code: Optional[str] = None

    # Created by Researcher
    web_research: Optional[str] = None

    # Created by Reporter
    report_markdown: Optional[str] = None

    # Run history from Evaluator
    run_history: List[Dict[str, Any]] = field(default_factory=list)
    best_metrics: Optional[Dict[str, float]] = None
    best_pipeline: Optional[Any] = None

    # Structured execution log
    structured_log: List[AgentLogEntry] = field(default_factory=list)

    # Checkpoint tracking
    completed_steps: List[str] = field(default_factory=list)
    current_step: Optional[str] = None

    @property
    def agent_log(self) -> List[str]:
        """Backward-compatible flat log as list of strings."""
        return [str(entry) for entry in self.structured_log]

    def log(
        self,
        agent_name: str,
        message: str,
        event_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = AgentLogEntry(
            timestamp=time.time(),
            agent=agent_name,
            event_type=event_type,
            content=message,
            metadata=metadata,
        )
        self.structured_log.append(entry)

        for cb in self.callbacks:
            try:
                cb(entry)
            except Exception as e:
                logger.warning(f"Callback failed for agent {agent_name}: {e}")

    def get_working_df(self) -> pd.DataFrame:
        """Return the appropriate DataFrame for agent work (sampled if large)."""
        if self.df_sample is not None:
            return self.df_sample
        if self.df_train is not None:
            return self.df_train
        raise ValueError("No data loaded.")

    def get_full_df(self) -> pd.DataFrame:
        """Return the full training DataFrame (never sampled)."""
        if self.df_train is None:
            raise ValueError("No data loaded.")
        return self.df_train

    def get_data_summary(self) -> str:
        """Produce a concise text summary of the dataset for LLM consumption."""
        if self.df_train is None:
            return "No data loaded."

        df = self.get_working_df()
        lines = []

        if self.was_sampled and self.original_shape is not None:
            lines.append(
                f"NOTE: Working on a stratified sample of {df.shape[0]} rows "
                f"(original: {self.original_shape[0]} rows x {self.original_shape[1]} cols). "
                f"Full data will be used for final model training."
            )

        lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        lines.append(f"Target column: '{self.target_col}'")

        if self.target_col in df.columns:
            target = df[self.target_col]
            n_unique = target.nunique()
            if n_unique <= 20:
                dist = target.value_counts().to_dict()
                lines.append(f"Target distribution ({n_unique} classes): {dist}")
            else:
                lines.append(
                    f"Target: continuous, mean={target.mean():.4f}, "
                    f"std={target.std():.4f}, min={target.min():.4f}, max={target.max():.4f}"
                )

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        lines.append(
            f"Numerical columns ({len(num_cols)}): "
            f"{num_cols[:15]}{'...' if len(num_cols) > 15 else ''}"
        )
        lines.append(
            f"Categorical columns ({len(cat_cols)}): "
            f"{cat_cols[:15]}{'...' if len(cat_cols) > 15 else ''}"
        )

        null_pct = df.isnull().mean()
        cols_with_nulls = null_pct[null_pct > 0]
        if len(cols_with_nulls) > 0:
            lines.append(f"Columns with nulls: {dict(cols_with_nulls.round(3))}")
        else:
            lines.append("No missing values.")

        lines.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
        lines.append(f"\nColumn dtypes:\n{df.dtypes.to_string()}")

        if num_cols:
            summary_cols = num_cols[:30]
            lines.append(
                f"\nNumeric summary:\n{df[summary_cols].describe().round(3).to_string()}"
            )

        return "\n".join(lines)
