from google.genai import types
import json

parameters = {
    "type": "object",
    "properties": {
        "location": {"type": "string"}
    },
    "required": ["location"]
}

try:
    fd = types.FunctionDeclaration(
        name="my_func",
        description="desc",
        parameters=parameters
    )
    t = types.Tool(function_declarations=[fd])
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)

