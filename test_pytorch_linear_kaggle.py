import os
import warnings
import pandas as pd

from custom_preprocessor import PerfectFitPreprocessor
from bluecast.blueprints.custom_model_recipes import RegularizedRegressionModel

warnings.filterwarnings("ignore")

def main():
    train_path = "data/dataset.csv"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not find {train_path}.")
        
    print("Loading data...")
    train = pd.read_csv(train_path, index_col="id")
    
    target_col = "target"
    y = train.pop(target_col)
    X = train
    
    print("Running Custom Preprocessor...")
    preprocessor = PerfectFitPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(X, y)
    
    print("Initializing RegularizedRegressionModel (PyTorch)...")
    model = RegularizedRegressionModel(scoring="neg_mean_absolute_error")
    
    # Configure tuning rounds (similar to AI exploration phase)
    model.conf_tuning = {
        "tuning_rounds": 50,  # Give it enough rounds to explore batch/LR/dropout combinations
        "tuning_max_runtime": 600
    }
    
    print("Starting Optuna Hyperparameter Tuning & Cross-Validation...")
    # The fit method runs autotune which inherently performs 5-fold CV to find the best params.
    # We pass a dummy test set as the tuning loop strictly evaluates OOF scores on x_train.
    model.fit(X_processed, X_processed.head(10), y_processed, y_processed.head(10))
    
    print("\n========== RAW MODEL OOF PERFORMANCE ==========\n")
    print(f"Best OOF CV Score (Neg MAE): {model.best_tuning_score_:.4f}")
    print(f"Equivalent MAE: {abs(model.best_tuning_score_):.4f}")
    print(f"Best Configuration Type: {model.best_model_type}")

if __name__ == "__main__":
    main()
