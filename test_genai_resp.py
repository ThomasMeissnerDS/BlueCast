from google.genai import types

try:
    # FunctionResponse in google-genai
    response_dict = {"result": "success"}
    part = types.Part.from_function_response(name="my_func", response=response_dict)
    # or part = types.Part(function_response=types.FunctionResponse(name="my_func", response=response_dict))
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
