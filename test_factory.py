from bluecast.ai.architectures import get_architectures_for_problem
archs = get_architectures_for_problem("binary")
print(list(archs.keys()))
print("Creating models")
for k, v in archs.items():
    print(f"Creating {k}")
    m = v["factory"]("binary")
print("Done")
