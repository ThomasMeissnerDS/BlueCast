import sys

with open("bluecast/tests/test_ai_providers.py", "r") as f:
    content = f.read()

# Add mock variable assignments back to the tests
content = content.replace("    def test_chat_text_response(self):\n        from bluecast.ai.providers.gemini", 
                          "    def test_chat_text_response(self):\n        from bluecast.ai.providers.gemini\n        import sys\n        mock_genai = sys.modules['google.generativeai']")
content = content.replace("    def test_chat_text_response(self):\n        from bluecast.ai.providers.vertexai_provider", 
                          "    def test_chat_text_response(self):\n        from bluecast.ai.providers.vertexai_provider\n        import sys\n        mock_gm = sys.modules['vertexai.generative_models.GenerativeModel'] if 'vertexai.generative_models.GenerativeModel' in sys.modules else MagicMock()\n        mock_gm = sys.modules['vertexai.generative_models']")
# wait, actually let's just do it directly using a Python script that replaces all instances
