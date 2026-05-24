import os
import warnings
import pandas as pd

from custom_preprocessor import PerfectFitPreprocessor
from bluecast.ai.architectures import SO1DCNNRegressionModel

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
    
    print("Initializing SO1DCNNRegressionModel (PyTorch)...")
    model = SO1DCNNRegressionModel(scoring="neg_mean_absolute_error")
    
    # Configure tuning rounds — give it enough budget to explore
    # sign_size, cha_input, cha_hidden, K, dropout, learning_rate, batch_size
    model.conf_tuning = {
        "tuning_rounds": 30,
        "tuning_max_runtime": 600
    }
    
    print("Starting Optuna Hyperparameter Tuning & Cross-Validation...")
    model.fit(X_processed, X_processed.head(10), y_processed, y_processed.head(10))
    
    print("\n========== RAW MODEL OOF PERFORMANCE ==========\n")
    print(f"Best OOF CV Score (Neg MAE): {model.best_tuning_score_:.4f}")
    print(f"Equivalent MAE: {abs(model.best_tuning_score_):.4f}")

if __name__ == "__main__":
    main()
