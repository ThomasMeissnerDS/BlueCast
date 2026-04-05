"""Critique system: adversarial review loop for agent outputs.

A CritiqueLoop wraps any agent's output with a critic LLM that
challenges quality, completeness, and adherence to the chosen mode.
The agent then refines its work based on the critique. This repeats
for up to ``max_rounds`` or until the critic approves.
"""

import logging

from bluecast.ai.context import SharedContext
from bluecast.ai.providers.base import BaseLLMProvider, Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode-aware critique prompts
# ---------------------------------------------------------------------------

ANALYST_CRITIQUE_PROMPT = """You are a senior data science reviewer critiquing a data analysis.

Mode: {mode}

Review the analysis below and challenge it on:
1. **Completeness** — Did the analyst check: uniqueness/cardinality, outliers,
   temporal patterns, group-level differences, leakage, correlations?
2. **Depth** — Are the findings specific enough to act on? Vague findings like
   "some nulls exist" are not acceptable. Which columns? How many? Impact?
3. **Oddities** — Did the analyst look for surprises in the data? Unusual
   distributions, constant columns, extreme skew, impossible values?
4. **Actionability** — Are the recommendations concrete enough for the
   feature engineer and model builder to act on?

If mode is 'fast', accept a lighter analysis. For 'precise' or 'ultimate',
demand thorough coverage of ALL aspects.

Respond with:
- APPROVED if the analysis is sufficient for the mode
- NEEDS_IMPROVEMENT: [specific gaps] if not

Be constructive. List exactly what is missing."""

FEATURE_ENGINEER_CRITIQUE_PROMPT = """You are a senior ML engineer reviewing feature engineering work.

Mode: {mode}

Review the feature engineering below and challenge it on:
1. **Relevance** — Are the features actually useful for the problem type?
2. **Text handling** — If text columns exist, were they handled (e.g., TF-IDF)?
3. **Missed opportunities** — Are there obvious features not created?
   (ratios, interactions, group aggregations, temporal features)
4. **Quality** — Could any feature cause leakage? Are names descriptive?
5. **Preprocessing** — Were high-cardinality categoricals grouped?

Respond with:
- APPROVED if the feature engineering is solid
- NEEDS_IMPROVEMENT: [specific gaps] if not"""

EVALUATOR_CRITIQUE_PROMPT = """You are a meta-evaluator reviewing model evaluation feedback.

Mode: {mode}

Review the evaluation and improvement suggestions below:
1. **Specificity** — Are suggestions concrete? "Try more tuning" is not acceptable.
   "Increase tuning_rounds from 50 to 150" is acceptable.
2. **Bottleneck** — Do the suggestions address the actual bottleneck?
3. **Mode adherence** — For 'ultimate' mode, did the evaluator consider
   architecture-specific advice?
4. **Diminishing returns** — Are suggestions likely to improve meaningfully?

Respond with:
- APPROVED if the evaluation is actionable
- NEEDS_IMPROVEMENT: [specific gaps] if not"""

PIPELINE_BUILDER_CRITIQUE_PROMPT = """You are a senior ML engineer reviewing pipeline configuration.

Mode: {mode}

Review the pipeline configuration below:
1. **Appropriateness** — Is the configuration suitable for the problem type and mode?
2. **Feature selection** — Should feature selection be enabled for this dataset?
3. **Ensemble strategy** — Is the ensemble strategy appropriate?
4. **Tuning** — Are tuning rounds and runtime reasonable?

Respond with:
- APPROVED if the configuration is sensible
- NEEDS_IMPROVEMENT: [specific gaps] if not"""

ARCH_FEATURE_ENGINEER_CRITIQUE_PROMPT = """You are a senior ML engineer reviewing architecture-specific feature engineering.

Mode: {mode}

Review the feature engineering below, keeping in mind this is for a SPECIFIC model architecture:
1. **Architecture fit** — Are the features appropriate for this model type?
   (e.g., CatBoost shouldn't have manual category encoding, linear models need scaling)
2. **Relevance** — Are the features likely to improve this specific model's performance?
3. **Missed opportunities** — Based on feature importances (if available), are there
   obvious interactions or transformations not explored?
4. **Quality** — Could any feature cause leakage? Are stateful transforms avoided?
5. **Diversity** — Do the features add different types of signal (ratios, interactions,
   binning, aggregations)?

Respond with:
- APPROVED if the feature engineering is solid for this architecture
- NEEDS_IMPROVEMENT: [specific gaps] if not"""

CRITIQUE_PROMPTS = {
    "DataAnalyst": ANALYST_CRITIQUE_PROMPT,
    "FeatureEngineer": FEATURE_ENGINEER_CRITIQUE_PROMPT,
    "ArchFeatureEngineer": ARCH_FEATURE_ENGINEER_CRITIQUE_PROMPT,
    "Evaluator": EVALUATOR_CRITIQUE_PROMPT,
    "PipelineBuilder": PIPELINE_BUILDER_CRITIQUE_PROMPT,
}


class CritiqueLoop:
    """Adversarial review loop for any agent's output.

    1. Agent produces initial output
    2. Critic LLM reviews output, raises objections
    3. Agent responds / refines based on critique
    4. Repeat up to max_rounds or until critic approves

    :param llm: The LLM provider to use for the critic.
    :param context: Shared context for logging.
    :param max_rounds: Maximum critique-refine cycles.
    :param verbose: Whether to print progress.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        context: SharedContext,
        max_rounds: int = 2,
        verbose: bool = True,
    ):
        self.llm = llm
        self.context = context
        self.max_rounds = max_rounds
        self.verbose = verbose

    def run_with_critique(
        self,
        agent,
        agent_task: str,
        mode: str = "balanced",
    ) -> str:
        """Run the agent, then critique, then refine.

        :param agent: The BaseAgent instance to run.
        :param agent_task: The task string to pass to the agent.
        :param mode: The current BlueCastAI mode (for mode-aware critique).
        :returns: The final (possibly refined) agent output.
        """
        # Snapshot FE state before the first run so that critique rounds
        # do not triply-accumulate snippets.
        is_fe_agent = agent.name == "FeatureEngineer"
        if is_fe_agent:
            pre_snippets = list(self.context.feature_code_snippets)
            pre_code = self.context.feature_engineering_code
            pre_df = (
                self.context.engineered_df.copy()
                if self.context.engineered_df is not None
                else None
            )

        # Step 1: Agent produces initial output
        initial_output = agent.run(agent_task)

        if self.max_rounds <= 0:
            return initial_output

        # Get the critique prompt for this agent type
        agent_name = agent.name
        critique_template = CRITIQUE_PROMPTS.get(agent_name)
        if critique_template is None:
            # No critique prompt defined for this agent
            return initial_output

        critique_prompt = critique_template.format(mode=mode)

        current_output = initial_output

        for round_num in range(self.max_rounds):
            if self.verbose:
                print(
                    f"    [Critique] Round {round_num + 1}/{self.max_rounds} "
                    f"for {agent_name}..."
                )

            # Step 2: Critic reviews
            critique = self._get_critique(critique_prompt, current_output)

            self.context.log(
                f"Critic:{agent_name}",
                critique[:500],
                event_type="critique",
                metadata={
                    "round": round_num + 1,
                    "agent": agent_name,
                },
            )

            # Check if approved
            if self._is_approved(critique):
                if self.verbose:
                    print(f"    [Critique] {agent_name} work APPROVED.")
                break

            if self.verbose:
                print(f"    [Critique] Requesting refinement from {agent_name}...")

            # Reset FE state before refinement so that the new round
            # replaces (not appends to) the previous round's snippets.
            if is_fe_agent:
                self.context.feature_code_snippets = list(pre_snippets)
                self.context.feature_engineering_code = pre_code
                self.context.engineered_df = (
                    pre_df.copy() if pre_df is not None else None
                )

            # Step 3: Agent refines based on critique
            refinement_task = (
                f"A critique of your previous work was received:\n\n"
                f"{critique}\n\n"
                f"Please address these concerns. Use your tools again if needed "
                f"to fill any gaps in your analysis. Provide an improved, "
                f"comprehensive response."
            )
            current_output = agent.run(refinement_task)

            self.context.log(
                agent_name,
                f"Refined after critique round {round_num + 1}",
                event_type="refinement",
                metadata={"round": round_num + 1},
            )

        return current_output

    def _get_critique(self, critique_prompt: str, agent_output: str) -> str:
        """Get a critique of the agent's output."""
        messages = [
            Message(role="system", content=critique_prompt),
            Message(
                role="user",
                content=f"Here is the work to review:\n\n{agent_output[:4000]}",
            ),
        ]
        response = self.llm.chat(messages)

        if response and response.usage:
            self.context.prompt_tokens += response.usage.get("prompt_tokens", 0)
            self.context.completion_tokens += response.usage.get("completion_tokens", 0)

        return response.text if response else "APPROVED"

    @staticmethod
    def _is_approved(critique: str) -> bool:
        """Check if the critique indicates approval."""
        critique_upper = critique.upper()
        return (
            "APPROVED" in critique_upper and "NEEDS_IMPROVEMENT" not in critique_upper
        )
