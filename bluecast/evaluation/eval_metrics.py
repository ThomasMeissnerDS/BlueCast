"""Module for evaluation metrics.

This is called as part of the fit_eval function.
"""
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    log_loss,
)


from bluecast.general_utils.general_utils import logger


def balanced_log_loss(y_true, y_pred):
    nc = np.bincount(y_true)
    return log_loss(y_true, y_pred, sample_weight=1/nc[y_true], eps=1e-15)


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
    bll = balanced_log_loss(y_true, y_probs)
    logger(f"The balanced logloss is {bll}")
    roc_auc = roc_auc_score(y_true, y_probs)
    logger(f"The ROC auc score is {roc_auc}")
    logloss = log_loss(y_true, y_probs)
    logger(f"The log loss score is {logloss}")

    full_classification_report = classification_report(y_true, y_classes)
    logger(full_classification_report)

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
