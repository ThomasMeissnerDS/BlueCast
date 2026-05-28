import re

with open("bluecast/tests/test_architectures.py", "r") as f:
    text = f.read()

# Replace X.drop(columns=["cat1"]) with X[[col for col in X.columns if col != "cat1"]]
text = text.replace('X_num = X.drop(columns=["cat1"])', 'X_num = X[[col for col in X.columns if col != "cat1"]]')
text = text.replace('X_num = X.drop(columns=["cat2", "cat3"])', 'X_num = X[[col for col in X.columns if col not in ["cat2", "cat3"]]]')
text = text.replace('X_num = X.drop(columns=["cat4", "cat5"])', 'X_num = X[[col for col in X.columns if col not in ["cat4", "cat5"]]]')

with open("bluecast/tests/test_architectures.py", "w") as f:
    f.write(text)
