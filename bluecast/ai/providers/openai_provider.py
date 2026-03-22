"""OpenAI (ChatGPT / GPT-4) LLM provider."""

import json
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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider using the openai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        delay_in_seconds: float = 0.0,
    ):
        super().__init__(api_key, model, temperature, delay_in_seconds)
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for the OpenAI provider. "
                "Install with: pip install 'bluecast[ai-openai]' or pip install openai"
            )
        self._client = openai.OpenAI(api_key=api_key)

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        converted = []
        for msg in messages:
            if msg.role == "tool_result":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                oai_tool_calls = []
                for tc in msg.tool_calls:
                    oai_tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    )
                converted.append(
                    {  # type: ignore[dict-item]
                        "role": "assistant",
                        "content": msg.content or None,  # type: ignore[dict-item]
                        "tool_calls": oai_tool_calls,  # type: ignore[dict-item]
                    }
                )
            else:
                converted.append({"role": msg.role, "content": msg.content})
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> LLMResponse:
        if self.delay_in_seconds > 0:
            import time

            time.sleep(self.delay_in_seconds)

        oai_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        text = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
