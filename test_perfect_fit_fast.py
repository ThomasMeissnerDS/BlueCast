import os
import warnings

import pandas as pd

from bluecast.ai import BlueCastAI

warnings.filterwarnings("ignore")


def main():
    # 1. LOAD DATA
    # Assumes data is extracted to a 'data/' directory as per the instructions above.
    train_path = "data/dataset.csv"
    test_path = "data/test.csv"
    sub_path = "data/sample_submission.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Could not find {train_path}. Please follow the Kaggle download instructions in the docstring."
        )

    print("Loading data...")
    train = pd.read_csv(train_path, index_col="id")
    test = pd.read_csv(test_path, index_col="id")
    sub_fl = pd.read_csv(sub_path, index_col="id")

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # 2. RUN BLUECAST AI
    print("Initializing BlueCast AI...")
    ai = BlueCastAI(
        api_key="",
        provider="vertexai",
        project_id="bluecastai-kaggle",  # <-- Replace with your local GCP Project ID if different
        model="gemini-3.1-pro-preview",  # Or gemini-3.0-flash-preview depending on GCP availability
        location="global",
        global_tuning_budget=54000,
        # temperature=0.2,
        # architectures_to_run=["linear"] # catboost, xgboost, linear, histgb, mlp, randomforest
    )

    print("Starting pipeline execution...")
    result = ai.run(
        df=train,
        target_col="target",
        prompt="""
      We use BlueCastAI inside a Kaggle competition: this is a regression task for the Kaggle 'The Perfect Fit' competition.

      The evaluation metric is Mean Absolute Error (MAE).
      Please perform extensive feature engineering and build a highly precise ensemble to minimize the MAE.
      If possible use MAE also as the loss during hyperparameter tuning.

      The pipeline must apply the same feature engineering to the unseen test/submission data to prevent any schema mismatches during inference!
      The dataset to execute inference on is loaded as test into the global context.


    In a nutshell:
    * use 8 iterations
    * use up to 5 hypertuning rounds for each model
    * make sure we do not have schema mismatches between train and unseen data (inference)
    * use MAE inside ml algorithm tunings and also for OOF evaluation
    * save out of fold predictions to folder 'data/output/'
        """,
        mode="ultimate",
        max_iterations=1,
    )

    # 3. INSPECT RESULTS & EXPORT
    print("\n========== AI EXECUTION COMPLETE ==========\n")
    print(f"Final Metrics: {result.metrics}\n")

    print("--- AI Generated Report ---")
    result.show_report()

    os.makedirs("output", exist_ok=True)
    result.save_code("output/generated_ai_pipeline.py")
    print("\nSaved reproducible pipeline code to output/generated_ai_pipeline.py")

    result.save_log("output/agent_execution_log.json")

    # 4. PREDICT & SUBMIT
    print("\nGenerating predictions on the test set...")
    preds = result.predict(test)

    if not isinstance(preds, pd.DataFrame):
        pd.DataFrame(preds).to_csv("output/predictions.csv", index=True)

    # Handle potential tuple return (predictions, probabilities/intervals)
    if isinstance(preds, tuple):
        final_preds = preds[0]
    else:
        final_preds = preds

    sub_fl["target"] = final_preds.values
    sub_fl.to_csv("output/submission.csv", index=True)
    print("Submission created successfully at output/submission.csv!")
    print(sub_fl.head())


if __name__ == "__main__":
    main()
