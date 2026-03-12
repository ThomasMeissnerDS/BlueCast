"""Configuration for the BlueCastAI multi-agent system."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class AIConfig:
    """Configuration for BlueCastAI.

    :param api_key: API key for the LLM provider.
    :param provider: LLM provider to use.
    :param model: Provider-specific model name. If None, uses a sensible default.
    :param mode: Controls speed vs thoroughness trade-off.
        'fast' = skip FE, 1 iteration, basic config.
        'balanced' = targeted FE, 2-3 iterations.
        'precise' = full FE, ensemble, 5+ iterations, web search.
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
    """

    api_key: str = ""
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    model: Optional[str] = None
    mode: Literal["fast", "balanced", "precise"] = "balanced"
    max_iterations: int = 3
    enable_web_search: bool = False
    verbose: bool = True
    temperature: float = 0.2
    context_files: List[str] = field(default_factory=list)
    max_rows_for_agents: int = 50_000
    max_columns_for_agents: int = 200
    checkpoint_dir: Optional[str] = None

    def get_model_name(self) -> str:
        if self.model:
            return self.model
        defaults = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
        }
        return defaults.get(self.provider, "gemini-2.5-flash")

    def get_max_iterations(self) -> int:
        if self.max_iterations > 0:
            return self.max_iterations
        mode_defaults = {"fast": 1, "balanced": 3, "precise": 5}
        return mode_defaults.get(self.mode, 3)
