"""Tests for bluecast.ai.critique — adversarial review loop."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from bluecast.ai.agents.base import BaseAgent
from bluecast.ai.agents.feature_engineer import FeatureEngineerAgent
from bluecast.ai.context import SharedContext
from bluecast.ai.critique import CritiqueLoop
from bluecast.tests.test_ai_mock_provider import (
    MockLLMProvider,
    make_text_response,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "num1": rng.normal(0, 1, 50),
            "target": rng.choice([0, 1], 50),
        }
    )


@pytest.fixture
def context(sample_df):
    return SharedContext(
        df_train=sample_df,
        target_col="target",
        mode="balanced",
    )


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


class MockAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "DataAnalyst"

    def system_prompt(self) -> str:
        return ""

    def get_tools(self):
        return []

    def run(self, task: str) -> str:
        # We'll use a mocked run method in tests
        return ""


class TestCritiqueLoop:
    def test_approved_first_round(self, context, mock_llm):
        mock_agent = MockAgent(mock_llm, context)
        mock_agent.run = MagicMock(return_value="Initial analysis")

        # Critic approves immediately
        mock_llm.enqueue_response(make_text_response("APPROVED, looks good."))

        loop = CritiqueLoop(mock_llm, context, max_rounds=2, verbose=False)
        result = loop.run_with_critique(mock_agent, "Do analysis", mode="fast")

        assert result == "Initial analysis"
        assert mock_agent.run.call_count == 1
        assert mock_llm.call_count == 1

    def test_needs_improvement_then_approved(self, context, mock_llm):
        mock_agent = MockAgent(mock_llm, context)
        mock_agent.run = MagicMock(side_effect=["Initial analysis", "Refined analysis"])

        # Critic rejects first, approves second
        mock_llm.enqueue_responses(
            [
                make_text_response("NEEDS_IMPROVEMENT: missing X"),
                make_text_response("APPROVED."),
            ]
        )

        loop = CritiqueLoop(mock_llm, context, max_rounds=2, verbose=False)
        result = loop.run_with_critique(mock_agent, "Do analysis", mode="fast")

        assert result == "Refined analysis"
        assert mock_agent.run.call_count == 2
        assert mock_llm.call_count == 2

    def test_max_rounds_exhausted(self, context, mock_llm):
        mock_agent = MockAgent(mock_llm, context)
        mock_agent.run = MagicMock(side_effect=["Initial", "Refined 1", "Refined 2"])

        # Critic keeps rejecting
        mock_llm.enqueue_responses(
            [
                make_text_response("NEEDS_IMPROVEMENT"),
                make_text_response("NEEDS_IMPROVEMENT"),
                make_text_response("NEEDS_IMPROVEMENT"),
            ]
        )

        loop = CritiqueLoop(mock_llm, context, max_rounds=2, verbose=False)
        result = loop.run_with_critique(mock_agent, "Do analysis", mode="fast")

        # Returns the output of the final run
        assert result == "Refined 2"
        assert mock_agent.run.call_count == 3  # 1 initial + 2 refinement

    def test_zero_rounds(self, context, mock_llm):
        mock_agent = MockAgent(mock_llm, context)
        mock_agent.run = MagicMock(return_value="Initial analysis")

        loop = CritiqueLoop(mock_llm, context, max_rounds=0, verbose=False)
        result = loop.run_with_critique(mock_agent, "Do analysis", mode="fast")

        assert result == "Initial analysis"
        assert mock_agent.run.call_count == 1
        assert mock_llm.call_count == 0  # Critic never called

    def test_is_approved(self):
        assert CritiqueLoop._is_approved("APPROVED") is True
        assert CritiqueLoop._is_approved("This is APPROVED.") is True
        assert CritiqueLoop._is_approved("APPROVED but NEEDS_IMPROVEMENT") is False
        assert CritiqueLoop._is_approved("NEEDS_IMPROVEMENT: fix it") is False

    def test_summarise_tool_results(self, context, mock_llm):
        context.log("Agent", "Called tool A", event_type="tool_call")
        context.log("Agent", "Tool A returned X", event_type="tool_result")
        context.log("Agent", "Just talking", event_type="task")

        loop = CritiqueLoop(mock_llm, context)
        summary = loop._summarise_recent_tool_results()
        assert "Called tool A" in summary
        assert "Tool A returned X" in summary
        assert "Just talking" not in summary

    def test_fe_state_reset(self, context, mock_llm, sample_df):
        """Test that FeatureEngineer state is snapshot and reset during critique."""
        agent = FeatureEngineerAgent(mock_llm, context)

        # Simulate initial run adding a snippet
        def mock_initial_run(task):
            if "critique" not in task:
                context.feature_code_snippets.append("df['initial'] = 1")
                return "Did initial"
            else:
                context.feature_code_snippets.append("df['refined'] = 2")
                return "Did refinement"

        agent.run = MagicMock(side_effect=mock_initial_run)

        # Pre-existing snippet
        context.feature_code_snippets.append("df['pre'] = 0")

        # Critic rejects once
        mock_llm.enqueue_responses(
            [
                make_text_response("NEEDS_IMPROVEMENT"),
                make_text_response("APPROVED"),
            ]
        )

        loop = CritiqueLoop(mock_llm, context, max_rounds=2, verbose=False)
        loop.run_with_critique(agent, "Do FE", mode="fast")

        # Final snippets should be pre + refined (initial is wiped out by reset)
        assert "df['pre'] = 0" in context.feature_code_snippets
        assert "df['refined'] = 2" in context.feature_code_snippets
        assert "df['initial'] = 1" not in context.feature_code_snippets
