import re

with open("bluecast/tests/test_ai_providers.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if "def test_chat_text_response(self):" in line:
        if "GeminiProvider" in "".join(lines):
            pass # wait, I need to know which class I am in
