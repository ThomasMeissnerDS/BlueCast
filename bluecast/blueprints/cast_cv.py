import logging
from copy import deepcopy
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import (
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostTuneParamsConfig,
)
from bluecast.conformal_prediction.conformal_prediction import (
    ConformalPredictionWrapper,
)
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.ml_modelling.xgboost import XgboostModel
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.feature_selection import BoostaRootaWrapper


class BlueCastCV:
    """Wrapper to train and predict multiple BlueCast instances.

    A custom splitter can be provided.

    :param :class_problem: Takes a string containing the class problem type. Either "binary" or "multiclass".
    :param :target_column: Takes a string containing the name of the target column.
    :param :cat_columns: Takes a list of strings containing the names of the categorical columns. If not provided,
        BlueCast will infer these automatically.
    :param :date_columns: Takes a list of strings containing the names of the date columns. If not provided,
        BlueCast will infer these automatically.
    :param :time_split_column: Takes a string containing the name of the time split column. If not provided,
        BlueCast will not split the data by time or order, but do a random split instead.
    :param :ml_model: Takes an instance of a XgboostModel class. If not provided, BlueCast will instantiate one.
        This is an API to pass any model class. Inherit the baseclass from ml_modelling.base_model.BaseModel.
    :param custom_in_fold_preprocessor: Takes an instance of a CustomPreprocessing class. Allows users to eeecute
        preprocessing after the train test split within cv folds. This will be executed only if precise_cv_tuning in
        the conf_Training is True. Custom ML models need to implement this themselves. This step is only useful when
        the proprocessing step has a high chance of overfitting otherwise (i.e: oversampling techniques).
    :param custom_preprocessor: Takes an instance of a CustomPreprocessing class. Allows users to inject custom
        preprocessing steps which take place right after the train test spit.
    :param custom_last_mile_computation: Takes an instance of a CustomPreprocessing class. Allows users to inject custom
        preprocessing steps which take place right before the model training.
    :param experiment_tracker: Takes an instance of an ExperimentTracker class. If not provided this will be initialized
        automatically.
    :param single_fold_eval_metric_func: Takes a function which calculates the evaluation metric for a single fold.
           Default is matthews_corrcoef. This function is used to calculate the evaluation metric for each fold during
           hyperparameter tuning when hyperparameter_tuning_rounds = 1 (default). Lower must be better.
    """

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
        custom_feature_selector: Optional[
            Union[BoostaRootaWrapper, CustomPreprocessing]
        ] = None,
        ml_model: Optional[Union[XgboostModel, Any]] = None,
        single_fold_eval_metric_func: Optional[ClassificationEvalWrapper] = None,
    ):
        self.class_problem = class_problem
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost
        self.custom_in_fold_preprocessor = custom_in_fold_preprocessor
        self.custom_preprocessor = custom_preprocessor
        self.custom_feature_selector = custom_feature_selector
        self.custom_last_mile_computation = custom_last_mile_computation
        self.bluecast_models: List[BlueCast] = []
        self.stratifier = stratifier
        self.ml_model = ml_model
        self.single_fold_eval_metric_func = single_fold_eval_metric_func
        self.conformal_prediction_wrapper: Optional[ConformalPredictionWrapper] = None

        if not cat_columns:
            self.cat_columns = []
        else:
            self.cat_columns = cat_columns

        if experiment_tracker:
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_tracker = ExperimentTracker()

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostFinalParamConfig()

        self.conf_training: TrainingConfig = conf_training or TrainingConfig()

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsConfig()

        if not self.single_fold_eval_metric_func:
            self.single_fold_eval_metric_func = ClassificationEvalWrapper()

    def prepare_data(
        self, df: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.reset_index(drop=True)
        y = df[target]
        X = df.drop(target, axis=1)
        return X, y

    def show_oof_scores(self, metric: str = "matthews") -> Tuple[float, float]:
        """
        Show out of fold scores.

        When calling BlueCastCVRegression's fit_eval function multiple BlueCastRegression
        instances are called and each of them predicts on unseen/oof data.

        This function collects these scores and return mean and average of them.

        :param metric: String indicating which metric shall be returned.
        :return: Tuple with (mean, std) of oof scores
        """
        all_metrics = []
        for bluecast_instance in self.bluecast_models:
            if bluecast_instance.eval_metrics:
                score = bluecast_instance.eval_metrics.get(metric)
                all_metrics.append(score)

        score_mean = np.asarray(all_metrics).mean()
        score_std = np.asarray(all_metrics).std()
        logging.info(
            f"The mean out of fold {metric} score is {score_mean} with an std of {score_std}"
        )

        return score_mean, score_std

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """Fit multiple BlueCast instances on different data splits.

        Input df is expected the target column."""
        X, y = self.prepare_data(df, target_col)

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        if not self.stratifier:
            self.stratifier = RepeatedStratifiedKFold(
                n_splits=self.conf_training.bluecast_cv_train_n_model[0],
                n_repeats=self.conf_training.bluecast_cv_train_n_model[1],
                random_state=self.conf_training.global_random_state,
            )

        for fn, (trn_idx, val_idx) in enumerate(self.stratifier.split(X, y)):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
            x_train = pd.concat([X_train, X_val], ignore_index=True)
            y_train = pd.concat([y_train, y_val], ignore_index=True)

            X_train = x_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            X_train[target_col] = y_train.values

            self.conf_training.global_random_state += (
                self.conf_training.increase_random_state_in_bluecast_cv_by
            )
            logging.info(
                f"Start fitting model number {fn} with random seed {self.conf_training.global_random_state}"
            )

            automl = BlueCast(
                class_problem=self.class_problem,
                cat_columns=self.cat_columns,
                conf_training=self.conf_training,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=deepcopy(self.conf_params_xgboost),
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=self.ml_model,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
            )
            automl.fit(X_train, target_col=target_col)
            self.bluecast_models.append(automl)

            # overwrite experiment tracker to pass it into next iteration
            self.experiment_tracker = automl.experiment_tracker

    def fit_eval(self, df: pd.DataFrame, target_col: str) -> Tuple[float, float]:
        """Fit multiple BlueCast instances on different data splits.

        Input df is expected the target column. Evaluation is executed on out-of-fold dataset.
        in each split.
        :param df: Pandas DataFrame that includes the target column
        :param target_col: String indicating the name of the target column
        :returns Tuple of (oof_mean, oof_std) with scores on unseen data during eval
        """
        X, y = self.prepare_data(df, target_col)

        if not self.conf_training:
            self.conf_training = TrainingConfig()

        if not self.stratifier:
            self.stratifier = RepeatedStratifiedKFold(
                n_splits=self.conf_training.bluecast_cv_train_n_model[0],
                n_repeats=self.conf_training.bluecast_cv_train_n_model[1],
                random_state=self.conf_training.global_random_state,
            )

        for fn, (trn_idx, val_idx) in enumerate(self.stratifier.split(X, y)):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            X_train.loc[:, target_col] = y_train

            self.conf_training.global_random_state += (
                self.conf_training.increase_random_state_in_bluecast_cv_by
            )
            logging.info(
                f"Start fitting model number {fn} with random seed {self.conf_training.global_random_state}"
            )

            automl = BlueCast(
                class_problem=self.class_problem,
                cat_columns=self.cat_columns,
                conf_training=self.conf_training,
                conf_xgboost=self.conf_xgboost,
                conf_params_xgboost=deepcopy(self.conf_params_xgboost),
                experiment_tracker=self.experiment_tracker,
                custom_in_fold_preprocessor=self.custom_in_fold_preprocessor,
                custom_preprocessor=self.custom_preprocessor,
                custom_feature_selector=self.custom_feature_selector,
                custom_last_mile_computation=self.custom_last_mile_computation,
                ml_model=self.ml_model,
                single_fold_eval_metric_func=self.single_fold_eval_metric_func,
            )
            automl.fit_eval(X_train, X_val, y_val, target_col=target_col)
            self.bluecast_models.append(automl)

            # overwrite experiment tracker to pass it into next iteration
            self.experiment_tracker = automl.experiment_tracker

        oof_mean, oof_std = self.show_oof_scores()
        return oof_mean, oof_std

    def predict(
        self,
        df: pd.DataFrame,
        return_sub_models_preds: bool = False,
        save_shap_values: bool = False,
    ) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        """Predict on unseen data using multiple trained BlueCast instances.

        :param df: Pandas DataFrame with unseen data
        :param return_sub_models_preds: If true will return a DataFrame with the predictions of each model for each class
            stored in separate columns.
        :param save_shap_values: If True, calculates and saves shap values, so they can be used to plot
            waterfall plots for selected rows o demand.
        """
        result_df = pd.DataFrame()  # Create an empty DataFrame to store the results
        or_cols = df.columns
        prob_cols: list[str] = []
        class_cols: list[str] = []
        for fn, pipeline in enumerate(self.bluecast_models):
            y_probs, y_classes = pipeline.predict(
                df.loc[:, or_cols], save_shap_values=save_shap_values
            )
            if self.class_problem == "multiclass":
                proba_cols = [
                    f"class_{col}_proba_model_{fn}" for col in range(y_probs.shape[1])
                ]
                result_df[proba_cols] = y_probs
                result_df[f"classes_{fn}"] = y_classes
                for col in proba_cols:
                    prob_cols.append(col)
                class_cols.append(f"classes_{fn}")

            else:
                result_df[f"proba_{fn}"] = y_probs
                result_df[f"classes_{fn}"] = y_classes
                prob_cols.append(f"proba_{fn}")
                class_cols.append(f"classes_{fn}")

        if self.class_problem == "multiclass":
            if return_sub_models_preds:
                return result_df.loc[:, prob_cols], result_df.loc[:, class_cols]
            else:
                classes = result_df.loc[:, class_cols].mode(axis=1)[0].astype(int)

                if self.bluecast_models[0].feat_type_detector:
                    if (
                        self.bluecast_models[0].target_label_encoder
                        and self.bluecast_models[0].feat_type_detector
                    ):
                        classes = self.bluecast_models[
                            0
                        ].target_label_encoder.label_encoder_reverse_transform(classes)

                mean_class_proba_cols = []
                for col_idx in range(y_probs.shape[1]):
                    class_proba_cols = [
                        f"class_{col_idx}_proba_model_{fn}"
                        for fn, pipeline in enumerate(self.bluecast_models)
                    ]
                    result_df[f"mean_proba_class_{col_idx}"] = result_df.loc[
                        :, class_proba_cols
                    ].mean(axis=1)
                    mean_class_proba_cols.append(f"mean_proba_class_{col_idx}")

                return (
                    result_df.loc[:, mean_class_proba_cols],
                    classes,
                )
        else:
            if return_sub_models_preds:
                return result_df.loc[:, prob_cols], result_df.loc[:, class_cols]
            else:
                if self.conf_params_xgboost:
                    classification_threshold = (
                        self.conf_params_xgboost.classification_threshold
                    )
                else:
                    classification_threshold = 0.5

                y_probs = result_df.loc[:, prob_cols].mean(axis=1)
                y_classes = (
                    result_df.loc[:, prob_cols].mean(axis=1) > classification_threshold
                ).astype(int)

                if (
                    self.bluecast_models[0].feat_type_detector
                    and self.bluecast_models[0].target_label_encoder
                ):
                    y_classes = self.bluecast_models[
                        0
                    ].target_label_encoder.label_encoder_reverse_transform(y_classes)

                return y_probs, y_classes

    def predict_proba(
        self,
        df: pd.DataFrame,
        return_sub_models_preds: bool = False,
        save_shap_values: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """Predict on unseen data using multiple trained BlueCast instances.

        :param df: Pandas DataFrame with unseen data
        :param return_sub_models_preds: If true will return a DataFrame with the predictions of each model for each class
            stored in separate columns.
        :param save_shap_values: If True, calculates and saves shap values, so they can be used to plot
            waterfall plots for selected rows o demand.
        """
        result_df = pd.DataFrame()  # Create an empty DataFrame for storing results
        or_cols = df.columns
        prob_cols: list[str] = []
        for fn, pipeline in enumerate(self.bluecast_models):
            y_probs, _y_classes = pipeline.predict(
                df.loc[:, or_cols], save_shap_values=save_shap_values
            )
            if self.class_problem == "multiclass":
                proba_cols = [
                    f"class_{col}_proba_model_{fn}" for col in range(y_probs.shape[1])
                ]
                result_df[proba_cols] = y_probs
                for col in proba_cols:
                    prob_cols.append(col)

            else:
                result_df[f"proba_{fn}"] = y_probs
                prob_cols.append(f"proba_{fn}")

        if self.class_problem == "multiclass":
            if return_sub_models_preds:
                return result_df.loc[:, prob_cols]
            else:
                # TODO: Take mean by class instead of overall
                mean_class_proba_cols = []
                for col_idx in range(y_probs.shape[1]):
                    class_proba_cols = [
                        f"class_{col_idx}_proba_model_{fn}"
                        for fn, pipeline in enumerate(self.bluecast_models)
                    ]
                    result_df[f"mean_proba_class_{col_idx}"] = result_df.loc[
                        :, class_proba_cols
                    ].mean(axis=1)
                    mean_class_proba_cols.append(f"mean_proba_class_{col_idx}")
                return result_df.loc[:, mean_class_proba_cols]
        else:
            if return_sub_models_preds:
                return result_df.loc[:, prob_cols]
            else:
                return result_df.loc[:, prob_cols].mean(axis=1)

    def calibrate(
        self, x_calibration: pd.DataFrame, y_calibration: pd.Series, **kwargs
    ) -> None:
        """Calibrate the model.

        Via this function the nonconformity measures are taken and used to predict calibrated sets via the
        predict_sets function, or to return p-values of a row for being the class via the predict_p_values function.
        This calibrates the blended prediction of all sub models.
        :param: x_calibration: Pandas DataFrame without target column, that has not been seen by the model during
            training.
        :param y_calibration: Pandas Series holding the target value, hat has not been seen by the model during
            training.
        """
        if isinstance(y_calibration, np.ndarray):
            y_calibration = pd.Series(y_calibration)

        if self.bluecast_models[0].target_label_encoder:
            x_calibration[self.bluecast_models[0].target_column] = y_calibration
            x_calibration = self.bluecast_models[
                0
            ].target_label_encoder.transform_target_labels(
                x_calibration, self.bluecast_models[0].target_column
            )
            y_calibration = x_calibration.pop(self.bluecast_models[0].target_column)

        self.conformal_prediction_wrapper = ConformalPredictionWrapper(self, **kwargs)
        self.conformal_prediction_wrapper.calibrate(x_calibration, y_calibration)

    def predict_p_values(self, df: pd.DataFrame) -> np.ndarray:
        """Create p-values for each class.

        The p_values indicate the probability of being the correct class.
        :param df: Pandas DataFrame holding unseen data
        :returns: Numpy array where each column holds p-values of a row being the class. If string labels were passed
            each column maps the index of target_label_encoder.target_label_mapping stored in this class.
        """
        if self.conformal_prediction_wrapper:
            pred_interval = self.conformal_prediction_wrapper.predict_interval(df)
            return pred_interval
        else:
            raise ValueError(
                """This instance has not been calibrated yet. Make use of calibrate to fit the
            ConformalPredictionWrapper."""
            )

    def predict_sets(self, df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """Create prediction sets based on a certain confidence level.

        Conformal prediction guarantees, that the correct label is present in the prediction sets with a probability of
        1 - alpha.
        :param df: Pandas DataFrame holding unseen data
        :param alpha: Float indicating the desired confidence level.
        :returns a Pandas DataFrame with a column called 'prediction_set' holding a nested set with predicted classes.
        """
        if self.conformal_prediction_wrapper:
            pred_sets = self.conformal_prediction_wrapper.predict_sets(df, alpha)
            # transform numerical values back to original strings for the end user
            if self.bluecast_models[0].target_label_encoder:
                reverse_mapping = {
                    value: key
                    for key, value in self.bluecast_models[
                        0
                    ].target_label_encoder.target_label_mapping.items()
                }
                string_pred_sets = []
                for numerical_set in pred_sets:
                    # Convert numerical labels to string labels
                    string_set = {reverse_mapping[label] for label in numerical_set}
                    string_pred_sets.append(string_set)
                return pd.DataFrame({"prediction_set": string_pred_sets})
            else:
                string_pred_sets = []
                for numerical_set in pred_sets:
                    # Convert numerical labels to string labels
                    string_set = {label for label in numerical_set}
                    string_pred_sets.append(string_set)
                return pd.DataFrame({"prediction_set": string_pred_sets})
        else:
            raise ValueError(
                """This instance has not been calibrated yet. Make use of calibrate to fit the
            ConformalPredictionWrapper."""
            )
