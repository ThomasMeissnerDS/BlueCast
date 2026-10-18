import pandas as pd
import numpy as np
from bluecast.ai import BlueCastAI
from bluecast.config.training_config import TrainingConfig

# 1. Credentials
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()

# 2. Data Loading
# Note: Adjust path if necessary for your Kaggle environment
df_train = pd.read_csv('/kaggle/input/playground-series-s6e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s6e8/test.csv')

test_ids = df_test['id']
df_train = df_train.drop(columns=['id'])
df_test_features = df_test.drop(columns=['id'])

# 3. Training Config (Activating our new features!)
custom_train = TrainingConfig(
    autotune_on_device="gpu",
    bluecast_cv_train_n_model=(1, 1),
    # Enable our new elite Data Science additions!
    enable_autofe=True, 
    advanced_target_encoding=True,
    use_focal_loss_for_imbalance=True,
    # Make sure we use a good amount of early stopping and eval rounds
    early_stopping_rounds=30,
)

# 4. Initialize BlueCastAI
ai = BlueCastAI(
    api_key="",
    provider="vertexai",
    credentials=user_credential,
    project_id="bluecastai-kaggle",  # Replace with your GCP project ID
    model="gemini-3.1-pro-preview",
    location="global",
    global_tuning_budget=36000,      # 10 hours total budget
    architectures_to_run=[
        "xgboost", # Only training a single model as requested!
    ],
    conf_training=custom_train,
    enable_web_search=True,
    max_rows_for_agents=50_000,
)

# 5. Define Prompt tailored for Binary Classification & ROC AUC
prompt = """
Build a highly accurate binary classification model to predict Electric Vehicle Interest (Will_Buy_EV).
The evaluation metric for this Kaggle competition is ROC AUC. Optimize all tuning decisions for area under the ROC curve.
Perform effective feature engineering on user demographics, financial statistics, and categorical features to identify patterns of electric vehicle adoption.
Since we are aiming for the ultimate single-model performance, focus on extracting maximum signal before training the XGBoost model.

Requirements:
- Run exactly 1 iteration.
- Run up to 250 hypertuning rounds for the model to ensure maximum single-model optimization.
- Prevent schema mismatches between training and test sets during inference.
- Utilize the new AutoFE and Bayesian Target Encoding capabilities internally by not dropping critical raw features.
"""

# 6. Execute BlueCastAI Pipeline
result = ai.run(
    df_train,
    target_col="Will_Buy_EV",
    prompt=prompt,
    mode="ultimate",
    max_iterations=1,
)

result.show_report()

# 7. Generate Test Probabilities
# predict_proba handles prediction and returns probabilities for the positive class
y_probs = result.predict_proba(df_test_features)

# 8. Create Kaggle Submission
submission = pd.DataFrame({
    'id': test_ids,
    'Will_Buy_EV': y_probs
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully as submission.csv!")
