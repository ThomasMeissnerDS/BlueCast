"""Module for evaluation metrics.

This is called as part of the fit_eval function.
"""

import logging
import warnings
from typing import Any, Dict, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error as root_mean_squared_error

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def plot_lift_chart(y_probs: np.array, y_true: np.array, num_bins: int = 20) -> None:
    # Check if the length of the predicted probabilities matches the length of actual outcomes
    if len(y_probs) != len(y_true):
        raise ValueError(
            "The length of predicted probabilities and actual outcomes must be the same."
        )

    # Create a DataFrame for easy sorting
    data = pd.DataFrame({"Predicted Probability": y_probs, "Actual Outcome": y_true})
    data["Bucket"] = pd.qcut(data["Predicted Probability"], num_bins, duplicates="drop")

    # Calculate the lift values for each bucket
    lift_values = data.groupby("Bucket").agg({"Actual Outcome": ["count", "sum"]})
    lift_values.columns = ["Total Count", "Positive Count"]
    lift_values["Negative Count"] = (
        lift_values["Total Count"] - lift_values["Positive Count"]
    )
    lift_values["Bucket Lift"] = (
        lift_values["Positive Count"] / lift_values["Total Count"]
    ).cumsum()

    # Calculate the baseline lift (if predictions were random)
    random_lift = data["Actual Outcome"].sum() / len(data)

    # Create the lift chart
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, num_bins + 1),
        lift_values["Bucket Lift"],
        marker="o",
        label="Model Lift",
    )
    plt.axhline(y=random_lift, color="red", linestyle="--", label="Random Lift")
    plt.xlabel("Bucket")
    plt.ylabel("Lift")
    plt.title("Lift Chart")
    plt.legend()
    plt.grid()
    plt.show()


def plot_roc_auc(
    y_true: np.array, predicted_probabilities: np.array, title="ROC Curve"
) -> None:
    """
    Plot the ROC curve and calculate the AUC (Area Under the Curve).

    :param y_true: True labels (0 or 1) for the binary classification problem.
    :param predicted_probabilities: Predicted probabilities for the positive class.
    :param title: Title for the ROC curve plot.
    """

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, predicted_probabilities)

    # Calculate AUC
    auc = roc_auc_score(y_true, predicted_probabilities)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_probability_distribution(
    probs: np.ndarray, y_classes: np.ndarray, opacity: float = 0.5
):
    """
    Plots the probability distribution of each class in an individual color.

    :param probs: A 2D array of shape (n_samples, n_classes) containing probability distributions.
    :param y_classes: An array of shape (n_samples,) containing the class labels for each sample.
    :param opacity: Opacity level for the plots. Default is 0.5.
    """
    assert (
        probs.shape[0] == y_classes.shape[0]
    ), "probs and y_classes must have the same number of samples"

    # Ensure probs is a 2D array
    if probs.ndim == 1:
        probs = np.column_stack((probs, 1 - probs))
    elif probs.ndim == 2 and probs.shape[1] == 1:
        probs = np.column_stack((probs[:, 0], 1 - probs[:, 0]))
    colors = plt.get_cmap("tab10")  # Get a colormap

    for class_idx in range(probs.shape[1]):
        class_probs = probs[:, class_idx]
        plt.hist(
            class_probs,
            bins=30,
            alpha=opacity,
            color=colors(class_idx),
            label=f"Class {class_idx}",
        )

    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Probability Distribution by Class")
    plt.show()


def balanced_log_loss(y_true, y_pred):
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.reshape(-1)
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.reshape(-1)

    assert ((y_true == 0) | (y_true == 1)).all()
    assert len(y_true) == len(y_pred)
    assert y_pred.ndim == 1

    eps = 1e-15
    y_pred = y_pred.clip(eps, 1 - eps)
    l0 = -np.log(1 - y_pred[y_true == 0])
    l1 = -np.log(y_pred[y_true != 0])
    return (l0.mean() + l1.mean()) / 2


def eval_classifier(
    y_true: np.ndarray, y_probs: np.ndarray, y_classes: np.ndarray
) -> Dict[str, Any]:
    try:
        matthews = matthews_corrcoef(y_true, y_classes)
    except Exception:
        matthews = 0

    logging.info(f"The Matthew correlation is {matthews}")
    accuracy = accuracy_score(y_true, y_classes)
    logging.info(f"The accuracy is {accuracy}")
    recall = recall_score(y_true, y_classes, average="weighted")
    logging.info(f"The recall is {recall}")
    f1_score_macro = f1_score(y_true, y_classes, average="macro", zero_division=0)
    logging.info(f"The macro F1 score is {f1_score_macro}")
    f1_score_micro = f1_score(y_true, y_classes, average="micro", zero_division=0)
    logging.info(f"The micro F1 score is {f1_score_micro}")
    f1_score_weighted = f1_score(y_true, y_classes, average="weighted", zero_division=0)
    logging.info(f"The weighted F1 score is {f1_score_weighted}")

    if len(y_probs.shape) == 1:
        bll = balanced_log_loss(y_true, y_probs)
        logging.info(f"The balanced logloss is {bll}")
    else:
        bll = 99
        logging.info("Skip balanced logloss as number of classes is less than 2.")

    if len(y_probs.shape) == 1:
        roc_auc = roc_auc_score(y_true, y_probs)
    else:
        roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
    logging.info(f"The ROC auc score is {roc_auc}")
    logloss = log_loss(y_true, y_probs)
    logging.info(f"The log loss score is {logloss}")

    full_classification_report = classification_report(y_true, y_classes)
    logging.info(full_classification_report)

    if len(y_probs.shape) == 1:
        plot_roc_auc(y_true, y_probs)
        try:
            plot_lift_chart(y_probs, y_true)
        except ValueError:
            warnings.warn(
                """Failed to create lift chart. This indicates an issue with the classifier.
            Check if there is any varance in the predicted probabilities.""",
                stacklevel=2,
            )
        plot_probability_distribution(y_probs, y_classes, opacity=0.5)
    else:
        logging.info(f"Skip ROC AUC curve as number of classes is {y_probs.shape[1]}.")

    evaluation_scores = {
        "matthews": matthews,
        "accuracy": accuracy,
        "recall": recall,
        "f1_score_macro": f1_score_macro,
        "f1_score_micro": f1_score_micro,
        "f1_score_weighted": f1_score_weighted,
        "log_loss": logloss,
        "balanced_logloss": bll,
        "roc_auc": roc_auc,
        "classfication_report": full_classification_report,
        "confusion_matrix": confusion_matrix(y_true, y_classes),
    }
    return evaluation_scores


def mean_squared_error_diff_sklearn_versions(y_true, y_preds):
    try:
        mean_squared_error_score = mean_squared_error(y_true, y_preds)
        print(f"The MSE score is {mean_squared_error_score}")
    except Exception:
        mean_squared_error_score = mean_squared_error(y_true, y_preds, squared=True)
        print(f"The MSE score is {mean_squared_error_score}")
    return mean_squared_error_score


def root_mean_squared_error_diff_sklearn_versions(y_true, y_preds):
    try:
        root_mean_squared_error_score = root_mean_squared_error(y_true, y_preds)
    except Exception:
        root_mean_squared_error_score = mean_squared_error(
            y_true, y_preds, squared=False
        )
    print(f"The RMSE score is {root_mean_squared_error_score}")
    return root_mean_squared_error_score


def eval_regressor(y_true: np.ndarray, y_preds: np.ndarray) -> Dict[str, Any]:
    r2 = r2_score(y_true, y_preds)
    print(f"The R2 score is {r2}")
    mean_absolute_error_score = mean_absolute_error(y_true, y_preds)
    print(f"The MAE score is {mean_absolute_error_score}")
    median_absolute_error_score = median_absolute_error(y_true, y_preds)
    print(f"The Median absolute error score is {median_absolute_error_score}")

    mean_squared_error_score = mean_squared_error_diff_sklearn_versions(y_true, y_preds)
    root_mean_squared_error_score = root_mean_squared_error_diff_sklearn_versions(
        y_true, y_preds
    )

    evaluation_scores = {
        "mae": mean_absolute_error_score,
        "r2_score": r2,
        "MSE": mean_squared_error_score,
        "RMSE": root_mean_squared_error_score,
        "median_absolute_error": median_absolute_error_score,
    }
    return evaluation_scores


class ClassificationEvalWrapper:
    """
    Wrapper to evaluate classification metrics.

    :param metric_func: Function object to calculate the metric.
    :param higher_is_better: Boolean indicating if higher metric values are better.
    :param eval_against: String indicating if the metric should be evaluated against probabilities or classes. Can be
        'probas_all_classes', 'probas_target_class' or 'classes'. For 'probas_all_classes', the metric is calculated
        against the predicted probabilities for all classes. For 'probas_target_class', the metric is calculated against
        the predicted probabilities for the best class. For 'classes', the metric is calculated against the predicted
        classes. This parameter decides how the predictions arrive in the metric function.
    :return: Float value of the metric score.
    """

    def __init__(
        self,
        higher_is_better: bool = True,
        eval_against: Literal[
            "probas_all_classes", "probas_target_class", "classes"
        ] = "classes",
        metric_func=matthews_corrcoef,
        metric_name: str = "Matthews Correlation Coefficient",
        **metric_func_args,
    ):
        self.higher_is_better = higher_is_better
        self.eval_against = eval_against
        self.metric_func = metric_func
        self.metric_name = metric_name
        self.metric_func_args = metric_func_args

        if eval_against not in ["probas_all_classes", "probas_target_class", "classes"]:
            raise ValueError(
                f"Argument eval_against must be one of ['probas_all_classes', 'probas_best_class', 'classes']. However {self.eval_against} has been provided."
            )

    def classification_eval_func_wrapper(
        self,
        y_true: Union[np.ndarray, pd.Series, list],
        y_probs: Union[np.ndarray, pd.Series, list],
    ) -> Union[float, int]:
        """
        Wrapper function to evaluate classification metrics.

        :param y_true: Numpy array of true labels.
        :param y_probs: NumPy array of predicted probabilities.
        :return: Float value of the metric score.
        """
        if not isinstance(y_true, list):
            y_true = y_true.tolist()

        if not isinstance(y_probs, list) and self.eval_against != "probas_target_class":
            y_probs = y_probs.tolist()

        if self.eval_against == "probas_all_classes":
            metric_score = self.metric_func(y_true, y_probs, **self.metric_func_args)
        elif self.eval_against == "probas_target_class":
            y_probs_best_class = np.asarray([line[1] for line in y_probs])
            metric_score = self.metric_func(
                y_true, y_probs_best_class, **self.metric_func_args
            )
        elif self.eval_against == "classes":
            y_classes = np.asarray([np.argmax(line) for line in y_probs])
            metric_score = self.metric_func(y_true, y_classes, **self.metric_func_args)
        else:
            raise ValueError(
                f"Unknown value for eval_against: {self.eval_against}. Possible values are 'probas' or 'classes'"
            )

        if self.higher_is_better:
            return -metric_score
        else:
            return metric_score


class RegressionEvalWrapper:
    """
    Wrapper to evaluate regression metrics.

    :param metric_func: Function object to calculate the metric.
    :param higher_is_better: Boolean indicating if higher metric values are better.
    :return: Float value of the metric score.
    """

    def __init__(
        self,
        higher_is_better: bool = False,
        metric_func=None,
        metric_name: str = "Mean squared error",
        **metric_func_args,
    ):
        self.higher_is_better = higher_is_better
        if metric_func:
            self.metric_func = metric_func
        else:
            try:
                self.metric_func = root_mean_squared_error
            except Exception:
                self.metric_func = mean_squared_error
                self.metric_func_args = {"squared": True}
        self.metric_name = metric_name
        self.metric_func_args = metric_func_args

    def regression_eval_func_wrapper(
        self,
        y_true: Union[np.ndarray, pd.Series, list],
        y_hat: Union[np.ndarray, pd.Series, list],
    ) -> Union[float, int]:
        """
        Wrapper function to evaluate classification metrics.

        :param y_true: Numpy array of true targets.
        :param y_hat: NumPy array of predicted targets.
        :return: Float value of the metric score.
        """
        if not isinstance(y_true, list):
            y_true = y_true.tolist()

        if not isinstance(y_hat, list):
            y_hat = y_hat.tolist()

        metric_score = self.metric_func(y_true, y_hat, **self.metric_func_args)

        if self.higher_is_better:
            return -metric_score
        else:
            return metric_score
