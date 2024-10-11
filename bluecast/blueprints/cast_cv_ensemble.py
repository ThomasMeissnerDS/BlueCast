import logging
from copy import deepcopy
from functools import partial
from typing import Any, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.orchestration import climb_hill
from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.xgboost import XgboostModel
from bluecast.preprocessing.custom import CustomPreprocessing


class BlueCastCVEnsemble:
    def __init__(
        self,
        class_problem: Literal["binary", "multiclass"] = "binary",
        cat_columns: Optional[List[Union[str, float, int]]] = None,
        stratifier: Optional[Any] = None,
        conf_training: Optional[TrainingConfig] = None,
        conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
        conf_params_xgboost: Optional[XgboostFinalParamConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        custom_in_fold_preprocessor: Optional[CustomPreprocessing] = None,
        custom_last_mile_computation: Optional[CustomPreprocessing] = None,
        custom_preprocessor: Optional[CustomPreprocessing] = None,
        custom_feature_selector: Optional[CustomPreprocessing] = None,
        ml_model: Optional[Union[XgboostModel, Any]] = None,
        single_fold_eval_metric_func: Optional[Any] = None,
        n_ensembles: int = 5,  # Number of BlueCastCV instances to create
        out_of_fold_dataset_store_path=None,
    ):
        # Store parameters
        self.class_problem = class_problem
        self.cat_columns = cat_columns
        self.stratifier = stratifier
        self.conf_training = conf_training or TrainingConfig()
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost
        self.experiment_tracker = experiment_tracker or ExperimentTracker()
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.custom_preprocessor = custom_preprocessor
        self.custom_feature_selector = custom_feature_selector
        self.custom_last_mile_computation = custom_last_mile_computation
        self.ml_model = ml_model
        self.single_fold_eval_metric_func = single_fold_eval_metric_func
        self.n_ensembles = n_ensembles

        if not self.single_fold_eval_metric_func:
            self.single_fold_eval_metric_func = ClassificationEvalWrapper()

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsConfig()

        if isinstance(out_of_fold_dataset_store_path, str):
            self.conf_training.out_of_fold_dataset_store_path = (
                out_of_fold_dataset_store_path
            )

        # List to hold BlueCastCV instances
        self.bluecast_cv_instances: List[BlueCastCV] = []

        # Check that conf_training has out_of_fold_dataset_store_path set
        if not self.conf_training.out_of_fold_dataset_store_path:
            raise ValueError(
                "out_of_fold_dataset_store_path must be set in TrainingConfig"
            )

        # If stratifier is not provided, create one with fixed random_state
        if not self.stratifier:
            self.stratifier = RepeatedStratifiedKFold(
                n_splits=self.conf_training.bluecast_cv_train_n_model[0],
                n_repeats=self.conf_training.bluecast_cv_train_n_model[1],
                random_state=self.conf_training.global_random_state,
            )

    def fit_eval(self, df: pd.DataFrame, target_col: str) -> None:
        """Fit multiple BlueCastCV instances on the same data splits.

        :param df: Pandas DataFrame that includes the target column.
        :param target_col: String indicating the name of the target column.
        """
        for i in range(self.n_ensembles):
            # Create a deep copy of conf_training
            conf_training_copy = deepcopy(self.conf_training)

            # Set the random seed for this instance
            conf_training_copy.global_random_state = (
                self.conf_training.global_random_state
                + i * conf_training_copy.increase_random_state_in_bluecast_cv_by * 100
            )
            logging.info(
                f"Global random state is {conf_training_copy.global_random_state}"
            )

            # Instantiate BlueCastCV with the shared stratifier
            bluecast_cv = BlueCastCV(
                class_problem=self.class_problem,
                cat_columns=self.cat_columns,
                stratifier=self.stratifier,
                conf_training=conf_training_copy,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=self.conf_params_xgboost,
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=self.ml_model,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
            )

            # Call fit_eval on bluecast_cv
            bluecast_cv.fit_eval(df, target_col)

            # Append the instance to the list
            self.bluecast_cv_instances.append(bluecast_cv)

    def collect_oof_data(self) -> pd.DataFrame:
        """Collect the out-of-fold predictions from all BlueCast models in all BlueCastCV instances.

        :return: DataFrame containing the oof predictions and true labels.
        """
        oof_preds_list = []
        for bluecast_cv in self.bluecast_cv_instances:
            # For each BlueCast model in bluecast_cv
            for bluecast_model in bluecast_cv.bluecast_models:
                # Construct the path to the oof data file
                if isinstance(
                    bluecast_model.conf_training.out_of_fold_dataset_store_path,
                    str,
                ):
                    path = bluecast_model.conf_training.out_of_fold_dataset_store_path
                else:
                    raise ValueError(
                        "out_of_fold_dataset_store_path has not been configured in TrainingConfig"
                    )
                random_state = bluecast_model.conf_training.global_random_state
                # Read the oof data
                print(
                    "Read from path:" + " " + path + f"oof_data_{random_state}.parquet"
                )
                oof_dataset = pd.read_parquet(path + f"oof_data_{random_state}.parquet")

                # Extract the true labels and the predictions
                target_col = bluecast_model.target_column
                y_true = oof_dataset[target_col]
                # For binary classification, predictions_class_1 as the predicted probabilities of the positive class
                if self.class_problem == "binary":
                    y_pred = oof_dataset["predictions_class_1"]
                    model_name = f"model_{random_state}"
                    df = pd.DataFrame(
                        {
                            "y_true": y_true,
                            model_name: y_pred,
                        }
                    ).reset_index(drop=True)
                    oof_preds_list.append(df)
                else:
                    # Handle multiclass classification
                    # You need to adjust this part to handle multiclass predictions
                    pass

                bluecast_model.conf_training.global_random_state -= (
                    bluecast_model.conf_training.increase_random_state_in_bluecast_cv_by
                )

        # Now, concatenate all DataFrames
        # Start with the first DataFrame to get 'y_true'
        y_true = oof_preds_list[0]["y_true"].reset_index(drop=True)
        oof_preds_df = pd.DataFrame({"y_true": y_true})
        for df in oof_preds_list:
            model_col = [col for col in df.columns if col != "y_true"][0]
            oof_preds_df[model_col] = df[model_col].reset_index(drop=True)

        oof_preds_df = oof_preds_df.reset_index(drop=True)
        return oof_preds_df

    def collect_test_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collect the test predictions from all BlueCast models in all BlueCastCV instances.

        :param df: DataFrame with test data (without target column).
        :return: DataFrame containing the test predictions from each model.
        """
        test_preds_list = []
        for bluecast_cv in self.bluecast_cv_instances:
            # For each BlueCast model in bluecast_cv
            for idx, bluecast_model in enumerate(bluecast_cv.bluecast_models):
                # Make predictions
                y_pred_probs, _ = bluecast_model.predict(df)
                # For binary classification, y_pred_probs is an array of probabilities
                if self.class_problem == "binary":
                    model_name = f"model_{bluecast_model.conf_training.global_random_state + (idx + 1) * bluecast_model.conf_training.increase_random_state_in_bluecast_cv_by}"
                    df_pred = pd.DataFrame({model_name: y_pred_probs}).reset_index(
                        drop=True
                    )
                    test_preds_list.append(df_pred)
                else:
                    # Handle multiclass classification
                    # You need to adjust this part to handle multiclass predictions
                    pass

        # Concatenate all test predictions
        test_preds_df = pd.concat(test_preds_list, axis=1).reset_index(drop=True)

        return test_preds_df

    def blend_models_with_hill_climbing(self, test_df: pd.DataFrame) -> np.ndarray:
        """Blend the models using hill climbing.

        :param test_df: DataFrame with test data (without target column).
        :return: Numpy array with the blended test predictions.
        """
        # Collect oof data
        oof_preds_df = self.collect_oof_data()
        # Collect test predictions
        test_preds_df = self.collect_test_predictions(test_df)

        # Get the true labels
        target_col = self.bluecast_cv_instances[0].bluecast_models[0].target_column
        y_true = oof_preds_df["y_true"]

        # Prepare the train DataFrame
        train = pd.DataFrame({target_col: y_true})

        # Prepare oof_pred_df (drop 'y_true' column)
        oof_pred_df = oof_preds_df.drop(columns=["y_true"])

        # Define the eval_metric
        eval_metric = partial(roc_auc_score)

        # Decide on the objective: 'maximize' for AUC
        objective = "maximize"

        # Call climb_hill
        test_preds, oof_preds = climb_hill(
            train=train,
            oof_pred_df=oof_pred_df,
            test_pred_df=test_preds_df,
            target=target_col,
            objective=objective,
            eval_metric=eval_metric,
            negative_weights=False,
            precision=0.01,
            plot_hill=True,
            plot_hist=False,
            return_oof_preds=True,
        )

        return test_preds
