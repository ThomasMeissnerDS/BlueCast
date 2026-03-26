"""Base agent class with tool-use loop and structured logging."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from bluecast.ai.context import SharedContext
from bluecast.ai.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10


class BaseAgent(ABC):
    """Base class for all BlueCastAI agents.

    Each agent has a system prompt, access to specific tools, and runs
    an LLM tool-use loop until the LLM produces a final text response.
    All inputs and outputs are logged to the shared context for
    traceability and reporting.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        context: SharedContext,
        verbose: bool = True,
    ):
        self.llm = llm
        self.context = context
        self.verbose = verbose
        self._tool_implementations: Dict[str, Callable] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        pass

    def register_tool_impl(self, name: str, func: Callable) -> None:
        self._tool_implementations[name] = func

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        impl = self._tool_implementations.get(tool_name)
        if impl is None:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            result = impl(**arguments)
            if isinstance(result, dict):
                res_str = json.dumps(result, indent=2, default=str)
            else:
                res_str = str(result)
            if len(res_str) > 5000:
                return res_str[:5000] + "\n...[Output truncated due to length]..."
            return res_str
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return f"Error executing '{tool_name}': {e}"

    def run(self, task: str) -> str:
        """Execute the agent's task with a tool-use loop.

        All interactions (task, tool calls, tool results, final response) are
        logged to the shared context with structured metadata.

        :param task: The task description for this agent.
        :returns: The agent's final text response.
        """
        if self.verbose:
            print(f"  [{self.name}] Starting: {task[:80]}...")

        self.context.log(
            self.name,
            task[:500],
            event_type="task",
            metadata={"full_task_length": len(task)},
        )

        messages = [
            Message(role="system", content=self.system_prompt()),
            Message(role="user", content=task),
        ]

        tools = self.get_tools()
        response: Optional[LLMResponse] = None
        previous_tool_calls: List[ToolCall] = []

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = self.llm.chat(messages, tools=tools if tools else None)
            except Exception as e:
                error_msg = (
                    f"LLM API call failed: {type(e).__name__}: " f"{str(e)[:200]}"
                )
                logger.warning(f"[{self.name}] {error_msg}")
                if self.verbose:
                    print(f"  [{self.name}] ⚠️ {error_msg}")
                self.context.log(
                    self.name,
                    error_msg,
                    event_type="error",
                )
                # Return a fallback so the pipeline can continue
                final = (
                    f"Agent {self.name} encountered an API error and could "
                    f"not complete this step. Error: {error_msg}"
                )
                if self.verbose:
                    print(f"  [{self.name}] Done (with errors).")
                return final

            if response and response.usage:
                self.context.prompt_tokens += response.usage.get("prompt_tokens", 0)
                self.context.completion_tokens += response.usage.get(
                    "completion_tokens", 0
                )

            if response.has_tool_calls:
                tool_results_text = []

                current_calls_signature = [
                    (tc.name, json.dumps(tc.arguments, sort_keys=True))
                    for tc in response.tool_calls
                ]
                previous_calls_signature = [
                    (tc.name, json.dumps(tc.arguments, sort_keys=True))
                    for tc in previous_tool_calls
                ]
                is_looping = (
                    current_calls_signature == previous_calls_signature
                ) and len(current_calls_signature) > 0
                previous_tool_calls = response.tool_calls

                for tc in response.tool_calls:
                    if self.verbose:
                        print(f"    [{self.name}] Calling tool: {tc.name}")

                    self.context.log(
                        self.name,
                        f"{tc.name}({json.dumps(tc.arguments, default=str)[:300]})",
                        event_type="tool_call",
                        metadata={"tool": tc.name, "arguments": tc.arguments},
                    )

                    if is_looping:
                        result = "ERROR: You just tried this exact tool call and it failed or looped. You MUST try a completely different approach or exit tools."
                        if self.verbose:
                            print(
                                f"    [{self.name}] Intercepted infinite iteration loop."
                            )
                    else:
                        result = self.execute_tool(tc.name, tc.arguments)

                    tool_results_text.append(f"Result of {tc.name}: {result[:3000]}")

                    self.context.log(
                        self.name,
                        result[:500],
                        event_type="tool_result",
                        metadata={"tool": tc.name, "result_length": len(result)},
                    )

                messages.append(
                    Message(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )
                for tc, result_text in zip(response.tool_calls, tool_results_text):
                    messages.append(
                        Message(
                            role="tool_result",
                            content=result_text,
                            tool_call_id=tc.id,
                        )
                    )
            else:
                if self.verbose:
                    print(f"  [{self.name}] Done.")

                self.context.log(
                    self.name,
                    response.text[:500],
                    event_type="response",
                    metadata={
                        "response_length": len(response.text),
                        "usage": response.usage,
                    },
                )
                return response.text

        final_text = response.text if response else "Agent reached max tool iterations."
        self.context.log(
            self.name,
            final_text[:300],
            event_type="error",
            metadata={"reason": "max_iterations_reached"},
        )
        return final_text
