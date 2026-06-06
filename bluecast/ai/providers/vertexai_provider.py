"""Google Cloud Vertex AI LLM provider using the google-genai SDK."""

import logging
from typing import Any, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None

from bluecast.ai.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class VertexAIProvider(BaseLLMProvider):
    """Google Cloud Vertex AI provider using the google-genai SDK."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        delay_in_seconds: float = 0.0,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[Any] = None,
    ):
        super().__init__(api_key, model, temperature, delay_in_seconds)

        if genai is None:
            raise ImportError(
                "google-genai is required for the Vertex AI provider. "
                "Install with: pip install google-genai"
            )

        # In Kaggle or GCP, calling Client(vertexai=True) without project/location
        # will use default credentials and project/location if configured.
        from typing import Any

        client_kwargs: dict[str, Any] = {"vertexai": True}
        if project_id:
            client_kwargs["project"] = project_id
        if location:
            client_kwargs["location"] = location
        if credentials:
            if isinstance(credentials, str):
                from google.oauth2.credentials import Credentials

                credentials = Credentials(token=credentials)
            elif isinstance(credentials, tuple) and len(credentials) == 2:
                credentials = credentials[0]
            client_kwargs["credentials"] = credentials

        self._client = genai.Client(**client_kwargs)

    def _convert_tools(self, tools: List[ToolDefinition]) -> list:
        """Convert tool definitions to Vertex AI function declarations."""
        if not types:
            return []

        declarations = []
        for tool in tools:
            params = tool.parameters.copy()
            params.pop("additionalProperties", None)

            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=params,
                )
            )

        if not declarations:
            return []

        return [types.Tool(function_declarations=declarations)]

    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Vertex AI format, extracting system instruction."""
        if not types:
            return None, []

        system_instruction = None
        contents = []

        # Buffer to group consecutive tool responses into a single Content block
        tool_response_parts: list = []

        def flush_tool_responses():
            if tool_response_parts:
                contents.append(
                    types.Content(role="user", parts=tool_response_parts.copy())
                )
                tool_response_parts.clear()

        for msg in messages:
            # If we hit a non-tool message, flush any pending tool responses first
            if msg.role != "tool_result":
                flush_tool_responses()

            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                part = types.Part.from_text(text=msg.content or "")
                contents.append(types.Content(role="user", parts=[part]))
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if getattr(tc, "raw_tool_call", None) is not None:
                            parts.append(tc.raw_tool_call)
                        else:
                            parts.append(
                                types.Part.from_function_call(
                                    name=tc.name, args=tc.arguments
                                )
                            )
                if parts:  # Only append if there are actually parts
                    contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool_result":
                # Ensure the response is always a dictionary
                response_dict = (
                    msg.content
                    if isinstance(msg.content, dict)
                    else {"result": str(msg.content)}
                )

                part = types.Part.from_function_response(
                    name=msg.tool_call_id or "", response=response_dict
                )
                # Append to buffer instead of directly to contents
                tool_response_parts.append(part)

        # Flush any remaining tool responses at the very end of the message history
        flush_tool_responses()

        return system_instruction, contents

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> LLMResponse:
        if self.delay_in_seconds > 0:
            import time

            time.sleep(self.delay_in_seconds)

        system_instruction, contents = self._convert_messages(messages)

        from typing import Any

        config_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            vertex_tools = self._convert_tools(tools)
            if vertex_tools:
                config_kwargs["tools"] = vertex_tools

        gen_config = types.GenerateContentConfig(**config_kwargs)

        max_retries = 5
        base_delay = 2.0

        # Circuit breaker: if auth is already known broken, fail fast
        if getattr(self, "_auth_broken", False):
            raise RuntimeError(
                "Vertex AI authentication is unavailable. "
                "The GCE metadata server could not be reached. "
                "Please restart the kernel or check your credentials."
            )

        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=gen_config
                )
                break
            except Exception as e:
                import random
                import time

                error_str = str(e)

                # Detect auth/credential errors — do NOT retry these
                is_auth_error = any(
                    keyword in error_str
                    for keyword in [
                        "metadata.google.internal",
                        "RefreshError",
                        "AuthMetadataPlugin",
                        "credentials",
                        "Could not automatically determine",
                    ]
                )

                if is_auth_error:
                    self._auth_broken = True  # type: ignore[attr-defined]
                    logger.warning(
                        f"Authentication error (not retrying): {error_str[:200]}"
                    )
                    raise

                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"API Error (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        text = ""
        tool_calls = []

        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if (
                    hasattr(candidate, "content")
                    and candidate.content
                    and hasattr(candidate.content, "parts")
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text += part.text
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            args = dict(fc.args) if getattr(fc, "args", None) else {}
                            tool_calls.append(
                                ToolCall(
                                    id=fc.name,
                                    name=fc.name,
                                    arguments=args,
                                    raw_tool_call=part,
                                )
                            )

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
            }

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
