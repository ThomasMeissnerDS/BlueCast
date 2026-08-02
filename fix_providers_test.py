import re

with open("bluecast/tests/test_ai_providers.py", "r") as f:
    content = f.read()

# Add sys.modules mock at top
mock_code = """
import sys
from unittest.mock import MagicMock

sys.modules["google.generativeai"] = MagicMock()
sys.modules["vertexai"] = MagicMock()
sys.modules["vertexai.generative_models"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
"""
content = content.replace(
    "from unittest.mock import MagicMock, patch",
    mock_code + "\nfrom unittest.mock import MagicMock, patch",
)

# Remove mock arguments from function definitions
content = re.sub(
    r"def test_([a-zA-Z0-9_]+)\(self, mock_[a-zA-Z0-9_]+(, mock_[a-zA-Z0-9_]+)*\):",
    r"def test_\1(self):",
    content,
)

# Fix simple_chat
content = content.replace(
    'provider.simple_chat("hello")', 'provider.simple_chat("sys", "hello")'
)

# Fix Message instantiation
content = content.replace('tool_name="test_tool",', 'name="test_tool",')

with open("bluecast/tests/test_ai_providers.py", "w") as f:
    f.write(content)
