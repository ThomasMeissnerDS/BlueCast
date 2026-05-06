import pandas as pd
import numpy as np
from bluecast.blueprints.custom_model_recipes import RegularizedRegressionModel
from bluecast.ai.architectures import MLPRegressionModel

# Fake data
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 5), columns=[f"col_{i}" for i in range(5)])
# Log-normal distribution
y = pd.Series(np.exp(np.random.randn(100) * 0.5 + 5))

print("Testing RegularizedRegressionModel...")
reg = RegularizedRegressionModel(scoring="neg_mean_absolute_error", cv_folds=2)
# Set fewer tuning rounds for test
reg.conf_tuning = {"tuning_rounds": 2, "tuning_max_runtime": 10}
reg.fit(X, X, y, y)
preds = reg.predict(X)
print("Regularized predictions mean:", preds.mean())
print("Poly selected:", getattr(reg, 'poly_transformer', None) is not None)
print("Target transformer:", type(reg.target_scaler).__name__)

print("\nTesting MLPRegressionModel...")
mlp = MLPRegressionModel(scoring="neg_mean_absolute_error", cv_folds=2)
mlp.conf_tuning = {"tuning_rounds": 2, "tuning_max_runtime": 10}
mlp.fit(X, X, y, y)
preds2 = mlp.predict(X)
print("MLP predictions mean:", preds2.mean())
print("Poly selected:", getattr(mlp, 'poly_transformer', None) is not None)
print("Target transformer:", type(mlp.target_scaler).__name__)
