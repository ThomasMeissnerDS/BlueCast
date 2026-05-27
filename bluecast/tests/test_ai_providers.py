"""Tests for bluecast.ai.providers — all LLM provider implementations."""

import pytest
from unittest.mock import MagicMock, patch

from bluecast.ai.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


# ---------------------------------------------------------------------------
# BaseLLMProvider
# ---------------------------------------------------------------------------


class TestBaseLLMProvider:
    def test_simple_chat(self):
        """Test the convenience simple_chat method."""

        class MinimalProvider(BaseLLMProvider):
            def chat(self, messages, tools=None):
                return LLMResponse(
                    text="response text",
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                )

        provider = MinimalProvider(api_key="test", model="test-model")
        result = provider.simple_chat("hello")
        assert result == "response text"

    def test_simple_chat_none_response(self):
        class NoneProvider(BaseLLMProvider):
            def chat(self, messages, tools=None):
                return None

        provider = NoneProvider(api_key="test", model="test-model")
        result = provider.simple_chat("hello")
        assert result == ""


# ---------------------------------------------------------------------------
# GeminiProvider (mocked SDK)
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    @patch("bluecast.ai.providers.gemini.genai")
    def test_init(self, mock_genai):
        from bluecast.ai.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        assert provider.api_key == "test-key"

    @patch("bluecast.ai.providers.gemini.genai")
    def test_convert_tools(self, mock_genai):
        from bluecast.ai.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="test", model="gemini-2.5-flash")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test",
                parameters={
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "An arg"}
                    },
                },
            )
        ]
        result = provider._convert_tools(tools)
        assert result is not None

    @patch("bluecast.ai.providers.gemini.genai")
    def test_chat_text_response(self, mock_genai):
        from bluecast.ai.providers.gemini import GeminiProvider

        # Mock the model
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Hello from Gemini"
        mock_response.candidates[0].content.parts[0].function_call = None
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_model_instance.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model_instance

        provider = GeminiProvider(api_key="test", model="gemini-2.5-flash")
        messages = [Message(role="user", content="Hello")]
        result = provider.chat(messages)
        assert result is not None
        assert result.text == "Hello from Gemini"

    @patch("bluecast.ai.providers.gemini.genai")
    def test_convert_messages(self, mock_genai):
        from bluecast.ai.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="test", model="gemini-2.5-flash")
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
        ]
        result = provider._convert_messages(messages)
        # System message is extracted separately
        assert isinstance(result, (list, tuple))


# ---------------------------------------------------------------------------
# VertexAIProvider (mocked SDK)
# ---------------------------------------------------------------------------


class TestVertexAIProvider:
    @patch("bluecast.ai.providers.vertexai_provider.vertexai")
    @patch("bluecast.ai.providers.vertexai_provider.GenerativeModel")
    def test_init(self, mock_gm, mock_vertexai):
        from bluecast.ai.providers.vertexai_provider import VertexAIProvider

        provider = VertexAIProvider(
            api_key="test",
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        assert provider.project == "test-project"

    @patch("bluecast.ai.providers.vertexai_provider.vertexai")
    @patch("bluecast.ai.providers.vertexai_provider.GenerativeModel")
    def test_convert_tools(self, mock_gm, mock_vertexai):
        from bluecast.ai.providers.vertexai_provider import VertexAIProvider

        provider = VertexAIProvider(
            api_key="test",
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test",
                parameters={
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "An arg"}
                    },
                },
            )
        ]
        result = provider._convert_tools(tools)
        assert result is not None

    @patch("bluecast.ai.providers.vertexai_provider.vertexai")
    @patch("bluecast.ai.providers.vertexai_provider.GenerativeModel")
    def test_convert_messages(self, mock_gm, mock_vertexai):
        from bluecast.ai.providers.vertexai_provider import VertexAIProvider

        provider = VertexAIProvider(
            api_key="test",
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content="Tool result",
                tool_call_id="call_1",
                tool_name="test_tool",
            ),
        ]
        result = provider._convert_messages(messages)
        assert isinstance(result, (list, tuple))

    @patch("bluecast.ai.providers.vertexai_provider.vertexai")
    @patch("bluecast.ai.providers.vertexai_provider.GenerativeModel")
    def test_chat_text_response(self, mock_gm, mock_vertexai):
        from bluecast.ai.providers.vertexai_provider import VertexAIProvider

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "VertexAI response"
        mock_part.function_call = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_model.generate_content.return_value = mock_response
        mock_gm.return_value = mock_model

        provider = VertexAIProvider(
            api_key="test",
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        messages = [Message(role="user", content="Hello")]
        result = provider.chat(messages)
        assert result is not None


# ---------------------------------------------------------------------------
# OpenAIProvider (mocked SDK)
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    @patch("bluecast.ai.providers.openai_provider.openai")
    def test_init(self, mock_openai):
        from bluecast.ai.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert provider.model == "gpt-4o"

    @patch("bluecast.ai.providers.openai_provider.openai")
    def test_convert_messages(self, mock_openai):
        from bluecast.ai.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = provider._convert_messages(messages)
        assert isinstance(result, list)
        assert result[0]["role"] == "system"

    @patch("bluecast.ai.providers.openai_provider.openai")
    def test_convert_messages_with_tool_result(self, mock_openai):
        from bluecast.ai.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        messages = [
            Message(role="user", content="Hello"),
            Message(
                role="tool",
                content="Tool result",
                tool_call_id="call_1",
                tool_name="test_tool",
            ),
        ]
        result = provider._convert_messages(messages)
        tool_msg = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msg) >= 1

    @patch("bluecast.ai.providers.openai_provider.openai")
    def test_chat_text_response(self, mock_openai):
        from bluecast.ai.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI response"
        mock_choice.message.tool_calls = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        messages = [Message(role="user", content="Hello")]
        result = provider.chat(messages)
        assert result is not None
        assert result.text == "OpenAI response"


# ---------------------------------------------------------------------------
# AnthropicProvider (mocked SDK)
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    @patch("bluecast.ai.providers.anthropic_provider.anthropic")
    def test_init(self, mock_anthropic):
        from bluecast.ai.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key="test-key", model="claude-sonnet-4-20250514"
        )
        assert provider.model == "claude-sonnet-4-20250514"

    @patch("bluecast.ai.providers.anthropic_provider.anthropic")
    def test_convert_messages(self, mock_anthropic):
        from bluecast.ai.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Hello"),
        ]
        result = provider._convert_messages(messages)
        assert isinstance(result, (list, tuple))

    @patch("bluecast.ai.providers.anthropic_provider.anthropic")
    def test_chat_text_response(self, mock_anthropic):
        from bluecast.ai.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "Anthropic response"
        mock_response.content = [mock_content_block]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        messages = [Message(role="user", content="Hello")]
        result = provider.chat(messages)
        assert result is not None
        assert result.text == "Anthropic response"

    @patch("bluecast.ai.providers.anthropic_provider.anthropic")
    def test_convert_tools(self, mock_anthropic):
        from bluecast.ai.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-20250514")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test",
                parameters={
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "An arg"}
                    },
                },
            )
        ]
        result = provider._convert_tools(tools)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_with_tool(self):
        msg = Message(
            role="tool",
            content="result",
            tool_call_id="call_1",
            tool_name="test_tool",
        )
        assert msg.tool_call_id == "call_1"

    def test_tool_call(self):
        tc = ToolCall(id="1", name="tool", arguments={"key": "val"})
        assert tc.name == "tool"
        assert tc.arguments["key"] == "val"

    def test_llm_response(self):
        resp = LLMResponse(
            text="Hello",
            tool_calls=[ToolCall(id="1", name="tool", arguments={})],
            usage={"prompt_tokens": 10},
        )
        assert resp.text == "Hello"
        assert len(resp.tool_calls) == 1
        assert resp.usage["prompt_tokens"] == 10

    def test_tool_definition(self):
        td = ToolDefinition(
            name="test",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert td.name == "test"
        assert td.description == "A test tool"
