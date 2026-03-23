import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel("gemini-3.0-flash", tools=[{"function_declarations": [{"name": "fake_tool", "description": "Fake tool about weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}]}])

messages = [{"role": "user", "parts": [{"text": "What is the weather in Paris?"}]}]
response = model.generate_content(messages)

print(response)
print("PARTS length:", len(response.candidates[0].content.parts))
for idx, p in enumerate(response.candidates[0].content.parts):
    print(f"Part {idx}:")
    if hasattr(p, "function_call"):
        print("Function call:", p.function_call)
    if hasattr(p, "text"):
        print("Text:", p.text)
    if hasattr(p, "executable_code"):
        print("Exe Code:", p.executable_code)
    try:
        print("dir:", dir(p))
    except (Exception,):
        pass
    try:
        if getattr(p, "thought"):
            print("Thought:", getattr(p, "thought"))
    except (Exception,):
        pass
    try:
        if getattr(p, "_pb"):
            print("PB:", p._pb)
    except (Exception,):
        pass
