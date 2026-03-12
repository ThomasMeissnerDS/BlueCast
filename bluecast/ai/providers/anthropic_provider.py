"""Anthropic (Claude) LLM provider."""

import logging
from typing import List, Optional

from bluecast.ai.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider using the anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.2,
    ):
        super().__init__(api_key, model, temperature)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for the Anthropic provider. "
                "Install with: pip install 'bluecast[ai-anthropic]' or pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Extract system prompt and convert messages to Anthropic format."""
        system_prompt = ""
        converted = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                converted.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,  # type: ignore[dict-item]
                            }
                        )
                converted.append({"role": "assistant", "content": content_blocks})  # type: ignore[dict-item]
            elif msg.role == "tool_result":
                converted.append(
                    {  # type: ignore[dict-item]
                        "role": "user",
                        "content": [  # type: ignore[dict-item]
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content,
                            }
                        ],
                    }
                )
        return system_prompt, converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> LLMResponse:
        system_prompt, anthropic_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": anthropic_messages,
            "temperature": self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self._client.messages.create(**kwargs)

        text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
