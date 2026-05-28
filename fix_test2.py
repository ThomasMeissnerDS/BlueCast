import re

with open("bluecast/tests/test_architectures.py", "r") as f:
    text = f.read()

# Replace X[[col...]] with pd.DataFrame({col: X[col].values for col in X.columns if col != ...})
text = text.replace('X_num = X[[col for col in X.columns if col != "cat1"]]', 'X_num = __import__("pandas").DataFrame({col: X[col].values for col in X.columns if col != "cat1"})')
text = text.replace('X_num = X[[col for col in X.columns if col not in ["cat2", "cat3"]]]', 'X_num = __import__("pandas").DataFrame({col: X[col].values for col in X.columns if col not in ["cat2", "cat3"]})')
text = text.replace('X_num = X[[col for col in X.columns if col not in ["cat4", "cat5"]]]', 'X_num = __import__("pandas").DataFrame({col: X[col].values for col in X.columns if col not in ["cat4", "cat5"]})')

with open("bluecast/tests/test_architectures.py", "w") as f:
    f.write(text)
