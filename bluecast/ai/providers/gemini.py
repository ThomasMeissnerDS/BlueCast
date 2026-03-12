"""Google Gemini LLM provider."""

import json
import logging
import uuid
from typing import List, Optional

from bluecast.ai.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider using the google-generativeai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        super().__init__(api_key, model, temperature)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for the Gemini provider. "
                "Install with: pip install 'bluecast[ai-gemini]' or pip install google-generativeai"
            )
        genai.configure(api_key=api_key)
        self._genai = genai

    def _convert_tools(self, tools: List[ToolDefinition]) -> list:
        """Convert tool definitions to Gemini function declarations."""
        declarations = []
        for tool in tools:
            params = tool.parameters.copy()
            params.pop("additionalProperties", None)
            declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": params,
            })
        return declarations

    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Gemini format, extracting system instruction."""
        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append({
                            "function_call": {
                                "name": tc.name,
                                "args": tc.arguments,
                            }
                        })
                contents.append({"role": "model", "parts": parts})
            elif msg.role == "tool_result":
                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": msg.tool_call_id or "",
                            "response": {"result": msg.content},
                        }
                    }],
                })
        return system_instruction, contents

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> LLMResponse:
        system_instruction, contents = self._convert_messages(messages)

        model_kwargs = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = self._genai.GenerativeModel(self.model, **model_kwargs)

        gen_config = self._genai.GenerationConfig(temperature=self.temperature)

        call_kwargs = {"generation_config": gen_config}
        if tools:
            gemini_tools = self._convert_tools(tools)
            call_kwargs["tools"] = [{"function_declarations": gemini_tools}]

        response = model.generate_content(contents, **call_kwargs)

        text = ""
        tool_calls = []

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCall(
                            id=fc.name,
                            name=fc.name,
                            arguments=args,
                        )
                    )

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            }

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
