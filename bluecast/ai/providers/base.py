"""Base class for LLM providers with unified message and tool-calling interface."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant", "tool_result"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None


@dataclass
class ToolDefinition:
    """Definition of a tool the LLM can call."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters
    function: Optional[Callable] = None


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]
    raw_tool_call: Any = None


@dataclass
class LLMResponse:
    """Response from an LLM, possibly containing tool calls."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers.

    Subclasses must implement `chat()` which handles both regular chat
    and tool-calling scenarios.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        delay_in_seconds: float = 0.0,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.delay_in_seconds = delay_in_seconds

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> LLMResponse:
        """Send messages to the LLM, optionally with tool definitions.

        :param messages: Conversation history.
        :param tools: Available tools the LLM can call.
        :returns: LLMResponse with text and/or tool calls.
        """
        ...

    def simple_chat(self, system_prompt: str, user_message: str) -> str:
        """Convenience method for a single-turn chat without tools."""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]
        response = self.chat(messages)
        return response.text if response else ""
