import matplotlib
import pandas as pd
import plotly.io as pio

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig
from bluecast.preprocessing.feature_types import FeatureTypeDetector

# Make sure no pop up windows pause the code execution
matplotlib.use("Agg")
pio.renderers.default = "svg"


def competition_pipeline():
    print("Starting competition pipeline.")

    target = "class"
    class_problem = "binary"
    debug = False

    folder = "/home/thomas/Schreibtisch/Data Science/Preprocessing lib test/automl_competition"
    train = pd.read_csv(f"{folder}/train.csv")
    # train_original = pd.read_csv("original.csv")
    test = pd.read_csv(f"{folder}/test.csv")
    submission = pd.read_csv(f"{folder}/sample_submission.csv")

    if debug:
        train = train.sample(5000, random_state=500)

    train = train.drop("id", axis=1)
    test = test.drop("id", axis=1)

    # detect cat columns
    ignore_cols = []

    feat_type_detector = FeatureTypeDetector()
    _ = feat_type_detector.fit_transform_feature_types(
        train.drop(ignore_cols + [target], axis=1)
    )
    _ = feat_type_detector.transform_feature_types(
        test.drop(ignore_cols, axis=1), ignore_cols
    )
    print(f"Cat columns are: {feat_type_detector.cat_columns}")
    print(f"Num columns are: {feat_type_detector.num_columns}")
    print(f"Date columns are: {feat_type_detector.date_columns}")

    # Iterate through each categorical column
    for col in feat_type_detector.cat_columns:
        # Get unique values for the column in both train and test dataframes
        unique_train = set(train[col].unique())
        unique_test = set(test[col].unique())

        # Find values in test that are not in train
        diff_values = unique_test - unique_train
        print(
            f"For column {col}, the values {diff_values} are in test but not in train."
        )

        # Define a function to replace values not in unique_train with None
        def replace_diff_values(x, diff_values=diff_values):
            return None if x in diff_values else x

        # Apply the replacement function to the column in train dataframe
        train[col] = train[col].apply(replace_diff_values)

    # Model training

    train_config = TrainingConfig()
    if debug:
        train_config.autotune_model = False
        train_config.calculate_shap_values = False
    else:
        train_config.autotune_model = True
        train_config.hypertuning_cv_folds = 1
        train_config.hypertuning_cv_repeats = 1
        train_config.cardinality_threshold_for_onehot_encoding = 3
        train_config.hyperparameter_tuning_rounds = 50
        train_config.hyperparameter_tuning_max_runtime_secs = 60 * 60 * 3
        # train_config.sample_data_during_tuning = True
        train_config.enable_grid_search_fine_tuning = False
        train_config.calculate_shap_values = False
        train_config.show_detailed_tuning_logs = True
        train_config.train_size = 0.85
        train_config.autotune_on_device = "gpu"
        # train_config.infrequent_categories_threshold = 10
        train_config.bluecast_cv_train_n_model = (5, 1)
        train_config.cat_encoding_via_ml_algorithm = True
        train_config.out_of_fold_dataset_store_path = "/home/thomas/Schreibtisch/Data Science/Preprocessing lib test/automl_competition/"

    automl = BlueCastCV(
        class_problem=class_problem,
        conf_training=train_config,
        # single_fold_eval_metric_func=cew
    )

    automl.conf_xgboost.max_bin_max = 2500

    if isinstance(automl, BlueCast):
        train = train.sample(frac=1.0, random_state=500)
        df_unseen = train.sample(frac=0.1, random_state=500)
        df_unseed_target = df_unseen.pop(target)
        train = train.drop(df_unseen.index)

        automl.fit_eval(
            train, df_eval=df_unseen, target_eval=df_unseed_target, target_col=target
        )
    else:
        automl.fit_eval(
            train,
            target_col=target,  # df_eval=df_unseen, target_eval=df_unseed_target,
        )

    if isinstance(automl, BlueCast):
        y_probs, y_classes = automl.predict(test, return_original_labels=True)
    else:
        y_probs, y_classes = automl.predict(test)

    submission[target] = y_classes
    print(submission)
    print(submission[target].value_counts())

    submission.to_csv(
        "automl_grandprix_bluecastcv_xgboost_repeatedkfold_submission.csv", index=False
    )


if __name__ == "__main__":
    competition_pipeline()
