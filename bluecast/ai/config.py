"""Configuration for the BlueCastAI multi-agent system."""

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional


@dataclass
class AIConfig:
    """Configuration for BlueCastAI.

    :param api_key: API key for the LLM provider.
    :param provider: LLM provider to use.
    :param model: Provider-specific model name. Pass the exact string the provider
        expects (e.g. 'gemini-2.5-pro', 'gpt-4o-mini', 'claude-opus-4-20250514').
        Defaults: gemini -> 'gemini-2.5-flash', openai -> 'gpt-4o',
        anthropic -> 'claude-sonnet-4-20250514', vertexai -> 'gemini-2.5-flash'.
    :param mode: Controls speed vs thoroughness trade-off.
        'fast' = skip FE, 1 iteration, basic config.
        'balanced' = targeted FE, 2-3 iterations.
        'precise' = full FE, ensemble, 5+ iterations, web search.
        'ultimate' = trains multiple architectures (CatBoost, XGBoost,
            linear, HistGradientBoosting) with per-arch iterative
            improvement, then selects the best pipeline.
    :param max_iterations: Maximum build-evaluate-improve cycles.
    :param enable_web_search: Whether the Researcher agent can search the web.
    :param verbose: Whether to print progress to stdout.
    :param temperature: LLM temperature (0.0 = deterministic, 1.0 = creative).
    :param context_files: Paths to files providing domain context.
    :param max_rows_for_agents: Maximum rows to use for data analysis and feature
        engineering agents. If the dataset exceeds this, a stratified sample is used.
        The full dataset is always used for final pipeline training.
    :param max_columns_for_agents: Maximum columns to profile in detail. Columns
        beyond this are summarized but not individually analyzed.
    :param checkpoint_dir: Directory to save/resume checkpoints. If None, no
        checkpoints are saved.
    :param llm_sleep_time: Pause in seconds before each LLM call to avoid API rate limits.
    :param max_tokens_budget: Hard limit on total LLM tokens used. 0 means unlimited.
    :param callbacks: List of callable functions to trigger during agent execution.
    :param project_id: Optional GCP Project ID (only used when provider='vertexai').
    :param location: Optional GCP Location/Region (only used when provider='vertexai').
    :param global_tuning_budget: Optional global time budget for hyperparameter tuning in seconds.
        If provided, the orchestrator divides this evenly across all folds, iterations, and architectures.
    :param critique_max_rounds: Optional override for the number of critique-refine rounds.
        If None, mode-based defaults are used (fast=0, balanced=1, precise=2, ultimate=5).
        Set to 0 to disable critique entirely.
    :param architectures_to_run: Optional list of architecture names to run in ultimate mode.
        If provided, only these architectures will be evaluated. Defaults to None (run all).
    """

    api_key: str = ""
    provider: Literal["gemini", "openai", "anthropic", "vertexai"] = "gemini"
    model: Optional[str] = None
    mode: Literal["fast", "balanced", "precise", "ultimate"] = "balanced"
    max_iterations: int = 3
    enable_web_search: bool = False
    verbose: bool = True
    temperature: float = 0.2
    context_files: List[str] = field(default_factory=list)
    max_rows_for_agents: int = 50_000
    max_columns_for_agents: int = 200
    checkpoint_dir: Optional[str] = None
    llm_sleep_time: float = 0.0
    max_tokens_budget: int = 0
    callbacks: List[Callable] = field(default_factory=list)
    project_id: Optional[str] = None
    location: Optional[str] = None
    critique_max_rounds: Optional[int] = None
    global_tuning_budget: Optional[int] = None
    autotune_on_device: Literal["cpu", "gpu", "auto"] = "cpu"
    architectures_to_run: Optional[List[str]] = None

    def get_model_name(self) -> str:
        if self.model:
            return self.model
        defaults = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "vertexai": "gemini-2.5-flash",
        }
        return defaults.get(self.provider, "gemini-2.5-flash")

    def get_max_iterations(self) -> int:
        if self.max_iterations > 0:
            return self.max_iterations
        mode_defaults = {"fast": 1, "balanced": 3, "precise": 5, "ultimate": 1}
        return mode_defaults.get(self.mode, 3)
