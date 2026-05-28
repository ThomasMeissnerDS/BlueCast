"""Reusable MockLLMProvider for testing the BlueCastAI module without API calls."""

from collections import deque
from typing import Deque, Dict, List, Optional

from bluecast.ai.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


class MockLLMProvider(BaseLLMProvider):
    """A fully controllable mock LLM provider for unit tests.

    Supports scripted responses via a queue. If the queue is empty,
    returns a default text response. Tracks call count and messages
    for test assertions.

    Usage::

        mock = MockLLMProvider()
        # Queue a simple text response
        mock.enqueue_response(LLMResponse(text="Hello"))
        # Queue a tool call
        mock.enqueue_response(LLMResponse(
            tool_calls=[ToolCall(id="1", name="describe_data", arguments={})]
        ))
    """

    def __init__(
        self,
        responses: Optional[List[LLMResponse]] = None,
        default_text: str = "Mock LLM response.",
    ):
        super().__init__(api_key="mock-key", model="mock-model")
        self._response_queue: Deque[LLMResponse] = deque(responses or [])
        self._default_text = default_text
        self.call_count = 0
        self.call_history: List[Dict] = []

    def enqueue_response(self, response: LLMResponse) -> None:
        """Add a response to the end of the queue."""
        self._response_queue.append(response)

    def enqueue_responses(self, responses: List[LLMResponse]) -> None:
        """Add multiple responses to the queue."""
        self._response_queue.extend(responses)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.call_history.append(
            {"messages": messages, "tools": tools, "call_number": self.call_count}
        )

        if self._response_queue:
            return self._response_queue.popleft()

        return LLMResponse(
            text=self._default_text,
            tool_calls=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


def make_text_response(text: str, usage: Optional[dict] = None) -> LLMResponse:
    """Helper to create a simple text LLMResponse."""
    return LLMResponse(
        text=text,
        tool_calls=[],
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 5},
    )


def make_tool_response(
    tool_name: str,
    arguments: Optional[dict] = None,
    text: str = "",
    call_id: str = "call_1",
) -> LLMResponse:
    """Helper to create an LLMResponse with a single tool call."""
    return LLMResponse(
        text=text,
        tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=arguments or {})],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


def make_planner_response(
    class_problem: str = "binary",
    needs_fe: bool = True,
    max_iterations: int = 1,
    regression_eval_metric: Optional[str] = None,
) -> LLMResponse:
    """Helper to create a planner JSON response."""
    import json

    plan = {
        "class_problem": class_problem,
        "needs_feature_engineering": needs_fe,
        "feature_engineering_hints": [],
        "needs_web_research": False,
        "research_queries": [],
        "ensemble_strategy": "stacking",
        "use_cv": True,
        "n_folds": 3,
        "n_repeats": 1,
        "tuning_rounds": 5,
        "tuning_max_runtime": 10,
        "max_iterations": max_iterations,
        "reasoning": "Test plan.",
    }
    if regression_eval_metric:
        plan["regression_eval_metric"] = regression_eval_metric
    return make_text_response(json.dumps(plan))
