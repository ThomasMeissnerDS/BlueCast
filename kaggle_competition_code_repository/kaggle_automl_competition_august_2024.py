import matplotlib
import pandas as pd
import plotly.io as pio

from bluecast.blueprints.cast import BlueCast

# from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig

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

    train = train.drop("id", axis=1)
    test = test.drop("id", axis=1)

    # Model training

    train_config = TrainingConfig()
    if debug:
        train_config.autotune_model = False
        train_config.calculate_shap_values = False
    else:
        train_config.autotune_model = False
        train_config.hypertuning_cv_folds = 5
        train_config.hypertuning_cv_repeats = 3
        train_config.hyperparameter_tuning_rounds = 25
        train_config.hyperparameter_tuning_max_runtime_secs = 60 * 60 * 10
        # train_config.sample_data_during_tuning = True
        train_config.enable_grid_search_fine_tuning = False
        train_config.calculate_shap_values = False
        train_config.show_detailed_tuning_logs = True
        train_config.train_size = 0.85
        train_config.autotune_on_device = "gpu"
        # train_config.infrequent_categories_threshold = 10
        train_config.cat_encoding_via_ml_algorithm = False
        train_config.out_of_fold_dataset_store_path = "/home/thomas/Schreibtisch/Data Science/Preprocessing lib test/automl_competition/"

    automl = BlueCast(
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

    y_probs, y_classes = automl.predict(test, return_original_labels=True)

    print(submission)

    submission.to_csv(
        "automl_grandprix_bluecastcv_xgboost_repeatedkfold_submission.csv", index=False
    )


if __name__ == "__main__":
    competition_pipeline()
