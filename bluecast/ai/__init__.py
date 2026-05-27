"""
BlueCastAI: Multi-agent LLM-powered AutoML.

Optional module -- install dependencies with:
    pip install bluecast[ai]           # all providers
    pip install bluecast[ai-gemini]    # Google Gemini only
    pip install bluecast[ai-openai]    # OpenAI only
    pip install bluecast[ai-anthropic] # Anthropic Claude only

Usage::

    from bluecast.ai import BlueCastAI

    ai = BlueCastAI(api_key="...", provider="gemini")
    result = ai.run(df_train, target_col="target",
                    prompt="Build a precise binary classifier")
    result.predict(df_test)
    result.save_code("pipeline.py")
"""

import logging
from typing import List, Literal, Optional

import pandas as pd

from bluecast.ai.config import AIConfig
from bluecast.ai.result import BlueCastAIResult

logger = logging.getLogger(__name__)


def _create_provider(config: AIConfig):
    """Factory to create the right LLM provider based on config."""
    model = config.get_model_name()

    if config.provider == "gemini":
        from bluecast.ai.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=config.api_key,
            model=model,
            temperature=config.temperature,
            delay_in_seconds=config.llm_sleep_time,
        )
    elif config.provider == "openai":
        from bluecast.ai.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=config.api_key,
            model=model,
            temperature=config.temperature,
            delay_in_seconds=config.llm_sleep_time,
        )
    elif config.provider == "anthropic":
        from bluecast.ai.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=config.api_key,
            model=model,
            temperature=config.temperature,
            delay_in_seconds=config.llm_sleep_time,
        )
    elif config.provider == "vertexai":
        from bluecast.ai.providers.vertexai_provider import VertexAIProvider

        return VertexAIProvider(
            api_key=config.api_key,
            model=model,
            temperature=config.temperature,
            delay_in_seconds=config.llm_sleep_time,
            project_id=config.project_id,
            location=config.location,
        )
    else:
        raise ValueError(
            f"Unknown provider: {config.provider}. Use 'gemini', 'openai', 'anthropic', or 'vertexai'."
        )


class BlueCastAI:
    """Multi-agent LLM-powered AutoML for BlueCast.

    Provide an API key, a dataset, and a natural language prompt.
    BlueCastAI will analyze the data, engineer features, build a pipeline,
    evaluate it, and iteratively improve it -- all guided by LLM agents.

    :param api_key: API key for the LLM provider.
    :param provider: LLM provider: 'gemini', 'openai', 'anthropic', or 'vertexai'.
    :param model: Provider-specific model name. Pass the exact string the provider
        expects (e.g. 'gemini-2.5-pro', 'gpt-4o-mini', 'claude-sonnet-4-20250514').
        Defaults per provider when not specified:
        gemini/vertexai -> 'gemini-2.5-flash', openai -> 'gpt-4o', anthropic -> 'claude-sonnet-4-20250514'.
    :param enable_web_search: Whether agents can search the web for techniques.
    :param verbose: Whether to print progress to stdout.
    :param temperature: LLM temperature (0.0 = deterministic, 1.0 = creative).
    :param checkpoint_dir: Directory for saving checkpoints. If a run crashes,
        the next call to .run() with the same checkpoint_dir resumes from where
        it left off. Set to None to disable checkpointing.
    :param llm_sleep_time: Time in seconds to sleep prior to an LLM provider call
        to prevent hitting rate limits.
    :param project_id: Optional Project ID when using Vertex AI (GCP).
    :param location: Optional Region/Location when using Vertex AI (GCP).
    :param global_tuning_budget: Optional global time budget for hyperparameter tuning in seconds.
        If provided, the orchestrator divides this evenly across all folds, iterations, and architectures.
    :param critique_max_rounds: Optional override for the number of critique-refine rounds per agent.
        If None, mode-based defaults are used (fast=0, balanced=1, precise=2, ultimate=5).
        Set to 0 to disable critique entirely.
    :param autotune_on_device: Whether to autotune on CPU or GPU. Choose any of ["gpu", "cpu", "auto"].
        Defaults to "cpu".
    :param architectures_to_run: Optional list of architecture names to run in ultimate mode.
        If provided, only these architectures will be evaluated. Defaults to None (run all).

    Usage::

        from bluecast.ai import BlueCastAI

        ai = BlueCastAI(api_key="your-key", provider="gemini")
        result = ai.run(
            df_train,
            target_col="target",
            prompt="Build a high-precision binary classifier with hill climbing ensemble",
            mode="precise",
        )

        # Use the trained pipeline
        predictions = result.predict(df_test)

        # Export reproducible code
        result.save_code("my_pipeline.py")

        # View what happened
        result.show_report()
    """

    def __init__(
        self,
        api_key: str,
        provider: Literal["gemini", "openai", "anthropic", "vertexai"] = "gemini",
        model: Optional[str] = None,
        enable_web_search: bool = False,
        verbose: bool = True,
        temperature: float = 0.2,
        checkpoint_dir: Optional[str] = None,
        llm_sleep_time: float = 0.0,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        global_tuning_budget: Optional[int] = None,
        critique_max_rounds: Optional[int] = None,
        autotune_on_device: Literal["cpu", "gpu", "auto"] = "auto",
        architectures_to_run: Optional[List[str]] = None,
    ):
        self.config = AIConfig(
            api_key=api_key,
            provider=provider,
            model=model,
            enable_web_search=enable_web_search,
            verbose=verbose,
            temperature=temperature,
            checkpoint_dir=checkpoint_dir,
            llm_sleep_time=llm_sleep_time,
            project_id=project_id,
            location=location,
            global_tuning_budget=global_tuning_budget,
            critique_max_rounds=critique_max_rounds,
            autotune_on_device=autotune_on_device,
            architectures_to_run=architectures_to_run,
        )
        self._llm = _create_provider(self.config)

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        prompt: str = "Build a good model",
        context_files: Optional[List[str]] = None,
        mode: Literal["fast", "balanced", "precise", "ultimate"] = "balanced",
        max_iterations: int = 0,
        custom_preprocessor=None,
    ) -> BlueCastAIResult:
        """Run the multi-agent pipeline on the dataset.

        :param df: Training DataFrame including the target column.
        :param target_col: Name of the target column.
        :param prompt: Natural language instructions for what to build.
            Examples:
            - "Build a fast baseline model"
            - "Build a precise binary classifier with stacking ensemble"
            - "Maximize ROC AUC using hill climbing and feature engineering"
        :param context_files: Optional list of file paths containing domain knowledge.
            Supports .pdf, .docx, .csv, .tsv, .txt, .md, .rst files.
            PDF requires ``pip install pypdf``; docx requires ``pip install python-docx``.
        :param mode: Speed vs thoroughness trade-off:
            'fast' = skip FE, 1 iteration (~2 min),
            'balanced' = targeted FE, critic on data analysis, 3 iterations (~10 min),
            'precise' = full FE with critic, ensemble, 5+ iterations (~30 min),
            'ultimate' = multi-architecture (CatBoost, XGBoost, Linear, HistGB)
                with per-arch iteration and critique (~60 min).
        :param max_iterations: Override the number of build-evaluate-improve cycles.
            If 0, uses the mode default.
        :returns: BlueCastAIResult with trained pipeline, code, metrics, and logs.
        """
        self.config.mode = mode
        self.config.context_files = context_files or []
        if max_iterations > 0:
            self.config.max_iterations = max_iterations
            if mode == "ultimate":
                self.config.ultimate_iterations_per_arch = max_iterations

        from bluecast.ai.orchestrator import Orchestrator

        orchestrator = Orchestrator(
            llm=self._llm,
            config=self.config,
            df=df,
            target_col=target_col,
            prompt=prompt,
            custom_preprocessor=custom_preprocessor,
        )
        return orchestrator.run()
