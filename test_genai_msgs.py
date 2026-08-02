from google.genai import types

try:
    part1 = types.Part.from_text(text="hello")
    content1 = types.Content(role="user", parts=[part1])

    fc = types.FunctionCall(name="my_func", args={"a": 1})
    part2 = types.Part.from_function_call(name="my_func", args={"a": 1})
    # or part2 = types.Part(function_call=fc)

    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
