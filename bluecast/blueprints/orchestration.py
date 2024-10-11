"""Module containing model orchestration tools."""

import logging
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
from colorama import Fore, Style

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.monitoring.data_monitoring import DataDrift


class ModelMatchMaker:
    """
    Matching the incoming data with the best model based on the adversarial validation score.
    """

    def __init__(self):
        self.bluecast_instances = []
        self.training_datasets = []

    def append_model_and_dataset(
        self,
        bluecast_instance: Union[
            BlueCast, BlueCastRegression, BlueCastCV, BlueCastCVRegression
        ],
        df: pd.DataFrame,
    ):
        """
        Append the model and the dataset to the matchmaker.

        :param bluecast_instance: The BlueCast instance to append.
        :param df: The dataset to append.
        """
        self.bluecast_instances.append(bluecast_instance)
        self.training_datasets.append(df)

    def find_best_match(
        self,
        df: pd.DataFrame,
        use_cols: List[Union[int, float, str]],
        cat_columns: Optional[List],
        delta: float,
    ) -> Tuple[
        Optional[Union[BlueCast, BlueCastRegression, BlueCastCV, BlueCastCVRegression]],
        Optional[pd.DataFrame],
    ]:
        """
        Find the best match based on the adversarial validation score.
        :param df: Dataset to match.
        :param use_cols: Columns to use for the adversarial validation. Numerical columns are allowed only.
        :param delta: Maximum delta for the adversarial validation score to be away from 0.5. If no dataset reaches this
         delta, (None, None) is returned.
        :param cat_columns: (Optional) List with names of categorical columns.
        :return: If a match is found, the BlueCast instance and the dataset are returned. Otherwise, (None, None) is
            returned.
        """
        best_score = np.inf
        best_idx = None

        for idx in range(len(self.bluecast_instances)):
            data_drift_checker = DataDrift()
            auc_score = data_drift_checker.adversarial_validation(
                self.training_datasets[idx].loc[:, use_cols],
                df.loc[:, use_cols],
                cat_columns,
            )
            score_delta = np.abs(auc_score - 0.5)
            if score_delta < best_score and np.abs(auc_score - 0.5) <= delta:
                print(
                    f"Found best match using idx {idx} with AUC score of {auc_score} and delta of {score_delta}"
                )
                best_score = score_delta
                best_idx = idx
                print(f"Best idx: {best_idx}, {self.bluecast_instances[best_idx]}")

        if best_idx:
            return self.bluecast_instances[best_idx], self.training_datasets[best_idx]
        else:
            print("No training dataset has reached the threshold criterium.")
            return None, None


def climb_hill(
    train: pd.DataFrame,
    oof_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    target: str,
    objective: str,  # ["minimize", "maximize"]
    eval_metric: Callable,
    negative_weights: bool = False,
    precision: float = 0.01,
    plot_hill: bool = True,
    plot_hist: bool = False,
    return_oof_preds: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform hill climbing to ensemble model predictions.

    This function combines predictions from multiple models using hill climbing to optimize an evaluation metric.
    It iteratively adds models to an ensemble to improve the cross-validation score on the training data.

    :param train: Pandas DataFrame containing the training data with the target column.
    :param oof_pred_df: Pandas DataFrame containing out-of-fold predictions from individual models.
    :param test_pred_df: Pandas DataFrame containing test predictions from individual models.
    :param target: String indicating the name of the target column in the training data.
    :param objective: String indicating whether to 'minimize' or 'maximize' the evaluation metric.
    :param eval_metric: A callable evaluation metric function that accepts true labels and predictions.
    :param negative_weights: Boolean indicating whether to consider negative weights during hill climbing.
    :param precision: Float indicating the precision step size for weight increments during hill climbing.
    :param plot_hill: Boolean indicating whether to plot the hill climbing progress.
    :param plot_hist: Boolean indicating whether to plot the histogram of final test predictions.
    :param return_oof_preds: Boolean indicating whether to return out-of-fold predictions along with test predictions.
    :return: Numpy array of blended test predictions, and optionally out-of-fold predictions if return_oof_preds is True.
    """
    bold_text = Style.BRIGHT
    yellow_text = bold_text + Fore.YELLOW
    green_text = bold_text + Fore.GREEN
    red_text = bold_text + Fore.RED
    reset_text = Style.RESET_ALL

    stop = False
    scores = {}
    iteration = 0

    oof_df = oof_pred_df.copy()
    test_preds_df = test_pred_df.copy()

    # Compute CV scores on the train data
    for col in oof_df.columns:
        scores[col] = eval_metric(train[target], oof_df[col])

    # Sort CV scores
    if objective == "minimize":
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=False))
    elif objective == "maximize":
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    else:
        raise ValueError(
            "Please provide a valid hill climbing objective ('minimize' or 'maximize')."
        )

    logging.info(
        f"{yellow_text}Models to be ensembled ({len(scores)} total):{reset_text}"
    )
    max_model_len = max(len(model_name) for model_name in scores.keys())

    # Display models with their associated metric score
    for idx, (model_name, score) in enumerate(scores.items()):
        model_padding = " " * (max_model_len - len(model_name))
        score_str = f"{score:.5f}".rjust(2)
        if idx == 0:
            logging.info(
                f"{green_text}{model_name}:{model_padding} {score_str} (best solo model){reset_text}"
            )
        else:
            logging.info(
                f"{bold_text}{model_name}:{model_padding} {score_str}{reset_text}"
            )

    oof_df = oof_df.loc[:, list(scores.keys())]
    test_preds_df = test_preds_df.loc[:, list(scores.keys())]

    current_best_ensemble = oof_df.iloc[:, 0].copy()
    current_best_test_preds = test_preds_df.iloc[:, 0].copy()
    models_to_consider = oof_df.iloc[:, 1:].copy()
    test_models_to_consider = test_preds_df.iloc[:, 1:].copy()
    history = [eval_metric(train[target], current_best_ensemble)]

    if precision > 0:
        if negative_weights:
            weight_range = np.arange(-0.5, 0.5 + precision, precision)
        else:
            weight_range = np.arange(precision, 0.5 + precision, precision)
    else:
        raise ValueError("Precision must be a positive number.")

    decimal_length = len(str(precision).split(".")[1]) if "." in str(precision) else 0
    eval_metric_name = (
        eval_metric.__name__
        if hasattr(eval_metric, "__name__")
        else eval_metric.func.__name__
    )

    logging.info(
        f"{yellow_text}[Data preparation completed successfully] - [Initiating hill climbing]{reset_text}"
    )

    # Hill climbing
    while not stop:
        iteration += 1
        potential_new_best_cv_score = eval_metric(train[target], current_best_ensemble)
        best_model_name = None
        best_weight = None

        for model_name in models_to_consider.columns:
            for weight in weight_range:
                potential_ensemble = (
                    1 - weight
                ) * current_best_ensemble + weight * models_to_consider[model_name]
                cv_score = eval_metric(train[target], potential_ensemble)

                if objective == "minimize" and cv_score < potential_new_best_cv_score:
                    potential_new_best_cv_score = cv_score
                    best_model_name = model_name
                    best_weight = weight
                elif objective == "maximize" and cv_score > potential_new_best_cv_score:
                    potential_new_best_cv_score = cv_score
                    best_model_name = model_name
                    best_weight = weight

        if best_model_name is not None:
            current_best_ensemble = (
                1 - best_weight
            ) * current_best_ensemble + best_weight * models_to_consider[
                best_model_name
            ]
            current_best_test_preds = (
                1 - best_weight
            ) * current_best_test_preds + best_weight * test_models_to_consider[
                best_model_name
            ]
            models_to_consider.drop(columns=best_model_name, inplace=True)
            test_models_to_consider.drop(columns=best_model_name, inplace=True)

            if models_to_consider.shape[1] == 0:
                stop = True

            weight_str = f"{best_weight:.{decimal_length}f}"
            color = green_text if best_weight > 0 else red_text
            logging.info(
                f"{color}Iteration: {iteration} | Model added: {best_model_name} | "
                f"Best weight: {weight_str} | Best {eval_metric_name}: {potential_new_best_cv_score:.5f}{reset_text}"
            )
            history.append(potential_new_best_cv_score)
        else:
            stop = True

    if plot_hill:
        fig = px.line(
            x=np.arange(len(history)) + 1,
            y=history,
            color_discrete_sequence=["#009933"],
            text=np.round(history, 5),
            labels={"x": "Number of Models", "y": "CV"},
            title=f"Cross Validation {eval_metric_name} vs. Number of Models with Hill Climbing",
            template="plotly_dark",
        )

        # Ensure integer x-axis ticks
        x_tickvals = list(range(1, len(history) + 1))
        fig.update_traces(
            textposition="top center",
            hovertemplate="Number of Models: %{x}<br>CV: %{y}",
            textfont={"size": 10},
            marker=dict(
                size=10, color="#33cc33", line=dict(width=1.5, color="#FFFFFF")
            ),
        )
        fig.update_layout(
            autosize=False,
            width=900,
            height=500,
            xaxis={"tickmode": "array", "tickvals": x_tickvals},
            xaxis_title="Number of models",
            yaxis_title=eval_metric_name,
        )
        fig.show()

    if plot_hist:
        fig = px.histogram(
            x=current_best_test_preds,
            marginal="box",
            color_discrete_sequence=["#33cc33"],
            title=f"Histogram of Final Test Predictions: {target}",
            template="plotly_dark",
        )
        fig.update_layout(
            autosize=False,
            width=900,
            height=500,
            xaxis_title=target,
        )
        fig.update_traces(
            hovertemplate="Test Prediction: %{x}<br>Count: %{y}",
        )
        fig.show()

    if return_oof_preds:
        return current_best_test_preds.values, current_best_ensemble.values
    else:
        return current_best_test_preds.values
