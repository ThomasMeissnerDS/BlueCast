"""Module for evaluation metrics.

This is called as part of the fit_eval function.
"""

import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from bluecast.general_utils.general_utils import logger


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
    y_probs: np.array, num_bins: int = 20, title: str = "Probability Distribution"
) -> None:
    """
    Plot the distribution of predicted probabilities as a histogram using Matplotlib.

    Parameters:
    :param y_probs: NumPy array of predicted probabilities.
    :param num_bins: Number of bins for the histogram (default is 20).
    :param title: Title for the plot (default is "Probability Distribution").
    """
    # Create a histogram of the probabilities
    plt.hist(y_probs, bins=num_bins, edgecolor="k", alpha=0.7)

    # Set plot labels and title
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title(title)

    # Display the plot
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

    logger(f"The Matthew correlation is {matthews}")
    accuracy = accuracy_score(y_true, y_classes)
    logger(f"The accuracy is {accuracy}")
    recall = recall_score(y_true, y_classes, average="weighted")
    logger(f"The recall is {recall}")
    f1_score_macro = f1_score(y_true, y_classes, average="macro", zero_division=0)
    logger(f"The macro F1 score is {f1_score_macro}")
    f1_score_micro = f1_score(y_true, y_classes, average="micro", zero_division=0)
    logger(f"The micro F1 score is {f1_score_micro}")
    f1_score_weighted = f1_score(y_true, y_classes, average="weighted", zero_division=0)
    logger(f"The weighted F1 score is {f1_score_weighted}")

    if len(y_probs.shape) == 1:
        bll = balanced_log_loss(y_true, y_probs)
        logger(f"The balanced logloss is {bll}")
    else:
        bll = 99
        logger("Skip balanced logloss as number of classes is less than 2.")

    if len(y_probs.shape) == 1:
        roc_auc = roc_auc_score(y_true, y_probs)
    else:
        roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
    logger(f"The ROC auc score is {roc_auc}")
    logloss = log_loss(y_true, y_probs)
    logger(f"The log loss score is {logloss}")

    full_classification_report = classification_report(y_true, y_classes)
    logger(full_classification_report)

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
        plot_probability_distribution(y_probs)
    else:
        logger(f"Skip ROC AUC curve as number of classes is {y_probs.shape[1]}.")

    evaluation_scores = {
        "matthews": matthews,
        "accuracy": accuracy,
        "recall": recall,
        "f1_score_macro": f1_score_macro,
        "f1_score_micro": f1_score_micro,
        "f1_score_weighted": f1_score_weighted,
        "log_loss": log_loss,
        "balanced_logloss": bll,
        "roc_auc": roc_auc,
        "classfication_report": full_classification_report,
        "confusion_matrix": confusion_matrix(y_true, y_classes),
    }
    return evaluation_scores


def eval_regressor(y_true: np.ndarray, y_preds: np.ndarray) -> Dict[str, Any]:
    r2 = r2_score(y_true, y_preds)
    print(f"The R2 score is {r2}")
    mean_absolute_error_score = mean_absolute_error(y_true, y_preds)
    print(f"The MAE score is {mean_absolute_error_score}")
    median_absolute_error_score = median_absolute_error(y_true, y_preds)
    print(f"The Median absolute error score is {median_absolute_error_score}")
    mean_squared_error_score = mean_squared_error(y_true, y_preds, squared=True)
    print(f"The MSE score is {mean_squared_error_score}")
    root_mean_squared_error_score = mean_squared_error(y_true, y_preds, squared=False)
    print(f"The RMSE score is {root_mean_squared_error_score}")

    evaluation_scores = {
        "mae": mean_absolute_error_score,
        "r2_score": r2,
        "MSE": mean_squared_error_score,
        "RMSE": root_mean_squared_error_score,
        "median_absolute_error": median_absolute_error_score,
    }
    return evaluation_scores
