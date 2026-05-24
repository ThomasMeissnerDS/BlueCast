import pandas as pd
from sklearn.model_selection import train_test_split
from bluecast.ai import BlueCastAI
from custom_preprocessor import PerfectFitPreprocessor
import sys

def main():
    train_full = pd.read_csv("data/dataset.csv", index_col="id")
    
    # Take a 10% sample to make it fast
    train_full = train_full.sample(frac=0.3, random_state=42)
    
    train, test = train_test_split(train_full, test_size=0.2, random_state=42)
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    ai = BlueCastAI(
        api_key="",
        provider="vertexai",
        project_id="bluecastai-kaggle",
        model="gemini-3.1-pro-preview",
        location="global",
        global_tuning_budget=120,
    )
    
    ai.config.mode = "ultimate"
    ai.config.architectures_to_run = ["catboost", "linear", "mlp"]
    
    result = ai.run(
        df=train,
        target_col="target",
        prompt="""
      We use BlueCastAI inside a Kaggle competition: this is a regression task for the Kaggle 'The Perfect Fit' competition.

      The evaluation metric is Mean Absolute Error (MAE).
      Please build a highly precise ensemble to minimize the MAE.
      If possible use MAE also as the loss during hyperparameter tuning.

      We have provided a custom preprocessor that handles all feature engineering.
      Therefore, set needs_feature_engineering=False in your plan.
      """,
        custom_preprocessor=PerfectFitPreprocessor()
    )
    
    print("OOF Ensemble info:")
    for arch in result.pipelines:
        print("Pipeline:", type(arch).__name__)
        
    print("Hill Climbing Ensemble info:")
    if getattr(result, "hill_climbing_ensemble", None):
        print(result.hill_climbing_ensemble.get_selected_model_info())
    else:
        print("No Hill Climbing Ensemble formed.")
        
    # Predict on test
    y_test = test.pop("target")
    preds = result.predict(test)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae}")

if __name__ == "__main__":
    main()
