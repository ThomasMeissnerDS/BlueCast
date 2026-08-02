import os
import warnings

import numpy as np
import pandas as pd

from bluecast.ai import BlueCastAI
from bluecast.config.training_config import TrainingConfig

warnings.filterwarnings("ignore")


def main():
    # 1. LOAD DATA
    # Assumes data is extracted to a 'data/' directory.
    # Download from Kaggle: kaggle competitions download -c playground-series-s6e6
    train_path = "data/stellar_train.csv"
    test_path = "data/stellar_test.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Could not find {train_path}. Please download the playground-series-s6e6 dataset and place it in the 'data/' folder."
        )

    print("Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Save test IDs for submission
    test_ids = df_test["id"]
    df_train = df_train.drop(columns=["id"])
    df_test_features = df_test.drop(columns=["id"])

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    # optional: Force big batch sizes to speed up training
    custom_train = TrainingConfig(
        autotune_on_device="gpu",
        #bluecast_cv_train_n_model=(5, 3),
        conf_tuning={"nn_batch_size_choices": [512, 1024, 2048]},
        calculate_shap_values=False,
        plot_hyperparameter_tuning_overview=False,
    )

    # 2. RUN BLUECAST AI
    print("Initializing BlueCast AI...")
    # Locally, Vertex AI automatically picks up Application Default Credentials 
    # if you have run `gcloud auth application-default login`.
    ai = BlueCastAI(
        api_key="",
        provider="vertexai",
        project_id="bluecastai-kaggle",  # <-- Replace with your local GCP Project ID
        model="gemini-3.1-pro-preview",
        location="global",
        global_tuning_budget=108000,  # 10 hours
        # architectures_to_run=[
        #    "catboost",
        #    "histgb",
        #    "xgboost",
        # ],
        conf_training=custom_train,
        enable_web_search=True,
        max_rows_for_agents=50_000,
    )

    # 3. Define the prompt tailored for ultimate mode
    prompt = """
Build a robust multi-class classification model to predict stellar classes (GALAXY, STAR, QSO).
The evaluation metric for the Kaggle competition is 'balanced_accuracy'. Optimize for the highest possible overall accuracy.
DO NOT use 'spectral_type' or 'galaxy_population' as they are synthetic target leakages that will ruin the test set predictions. Drop them immediately.
Use cross-validation via an ensemble strategy like hill_climbing or stacking.
Perform extensive feature engineering focusing on astronomical data features like u, g, r, i, z, and redshift. Do NOT drop these core features.
Ensure the model performs well across all three classes.

In a nutshell:
    * use 2 iterations
    * use up to 200 hypertuning rounds for each model
    * make sure we do not have schema mismatches between train and unseen data (inference)
"""


    print("Starting pipeline execution...")
    # 4. Run BlueCastAI in 'precise' mode
    result = ai.run(
        df_train,
        target_col="class",
        prompt=prompt,
        mode="ultimate",  # Runs the full AI orchestration loop
        max_iterations=2,
    )

    # INSPECT RESULTS & EXPORT
    print("\n========== AI EXECUTION COMPLETE ==========\n")
    print(f"Final Metrics: {result.metrics}\n")

    print("--- AI Generated Report ---")
    result.show_report()

    os.makedirs("output", exist_ok=True)
    result.save_code("output/generated_stellar_pipeline.py")
    print("\nSaved reproducible pipeline code to output/generated_stellar_pipeline.py")

    # 5. Predict on the test set
    print("\nGenerating predictions on test set...")
    # We use return_original_labels=True to request the string labels instead of integers
    preds = result.pipeline.predict(df_test_features, return_original_labels=True)

    # BlueCastAuto predict returns a tuple: (y_probs, y_classes)
    if isinstance(preds, tuple):
        y_classes = preds[1]
    else:
        y_classes = preds

    # Fallback: Depending on if the AI chose CV or Single Model, predictions might still be numerical.
    # We ensure safe reverse mapping back to strings using the model's TargetLabelEncoder:
    if np.issubdtype(np.array(y_classes).dtype, np.number):
        try:
            # Check if single model backend was used
            encoder = result.pipeline.inner_model.target_label_encoder
        except AttributeError:
            # Check if CV model backend was used
            encoder = result.pipeline.inner_model.bluecast_models[
                0
            ].target_label_encoder

        if encoder:
            y_classes = encoder.label_encoder_reverse_transform(
                pd.Series(y_classes)
            ).values

    # 6. Create Kaggle Submission
    submission = pd.DataFrame({"id": test_ids, "class": y_classes})

    submission_path = "output/submission_stellar.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}! Ready to submit to Kaggle.")


if __name__ == "__main__":
    main()
