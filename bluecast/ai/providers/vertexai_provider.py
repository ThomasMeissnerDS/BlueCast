"""Google Cloud Vertex AI LLM provider."""

import logging
from typing import List, Optional

try:
    # PRE-IMPORT WORKAROUND: In Kaggle notebooks, kaggle_gcp.py intercepts google.cloud imports.
    # If vertexai is imported first, it triggers a circular import on aiplatform.init().
    # Importing google.cloud.storage first forces Kaggle to patch aiplatform safely.
    import google.cloud.storage  # noqa: F401
    import vertexai
    import vertexai.generative_models as generative_models
except ImportError:
    vertexai = None  # type: ignore
    generative_models = None

from bluecast.ai.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class VertexAIProvider(BaseLLMProvider):
    """Google Cloud Vertex AI provider using the google-cloud-aiplatform SDK."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        delay_in_seconds: float = 0.0,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ):
        super().__init__(api_key, model, temperature, delay_in_seconds)

        if vertexai is None:
            raise ImportError(
                "google-cloud-aiplatform is required for the Vertex AI provider. "
                "Install with: pip install google-cloud-aiplatform"
            )

        # In Kaggle or GCP, calling init() without args will use default credentials
        if project_id or location:
            vertexai.init(project=project_id, location=location)
        else:
            vertexai.init()

        self._vertexai = vertexai
        self._generative_models = generative_models

    def _convert_tools(self, tools: List[ToolDefinition]) -> list:
        """Convert tool definitions to Vertex AI function declarations."""
        if not self._generative_models:
            return []

        declarations = []
        for tool in tools:
            params = tool.parameters.copy()
            params.pop("additionalProperties", None)

            declarations.append(
                self._generative_models.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=params,
                )
            )

        if not declarations:
            return []

        return [self._generative_models.Tool(function_declarations=declarations)]

    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Vertex AI format, extracting system instruction."""
        if not self._generative_models:
            return None, []

        system_instruction = None
        contents = []

        # Buffer to group consecutive tool responses into a single Content block
        tool_response_parts: list = []

        def flush_tool_responses():
            if tool_response_parts:
                contents.append(
                    self._generative_models.Content(
                        role="user", parts=tool_response_parts.copy()
                    )
                )
                tool_response_parts.clear()

        for msg in messages:
            # If we hit a non-tool message, flush any pending tool responses first
            if msg.role != "tool_result":
                flush_tool_responses()

            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                part = self._generative_models.Part.from_text(msg.content or "")
                contents.append(
                    self._generative_models.Content(role="user", parts=[part])
                )
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append(self._generative_models.Part.from_text(msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if getattr(tc, "raw_tool_call", None) is not None:
                            parts.append(tc.raw_tool_call)
                        else:
                            parts.append(
                                self._generative_models.Part.from_function_call(
                                    name=tc.name, args=tc.arguments
                                )
                            )
                if parts:  # Only append if there are actually parts
                    contents.append(
                        self._generative_models.Content(role="model", parts=parts)
                    )
            elif msg.role == "tool_result":
                # Ensure the response is always a dictionary
                response_dict = (
                    msg.content
                    if isinstance(msg.content, dict)
                    else {"result": str(msg.content)}
                )

                part = self._generative_models.Part.from_function_response(
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

        model_kwargs = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = self._generative_models.GenerativeModel(self.model, **model_kwargs)

        gen_config = self._generative_models.GenerationConfig(
            temperature=self.temperature
        )

        call_kwargs = {"generation_config": gen_config}
        if tools:
            vertex_tools = self._convert_tools(tools)
            if vertex_tools:
                call_kwargs["tools"] = vertex_tools

        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = model.generate_content(contents, **call_kwargs)
                break
            except Exception as e:
                import random
                import time

                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"API Error (attempt {attempt + 1}): {e}. Retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        text = ""
        tool_calls = []

        for candidate in response.candidates:
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
