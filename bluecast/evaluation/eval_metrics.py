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
)

from bluecast.general_utils.general_utils import logger


def balanced_log_loss(y_true, y_pred):
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
    bll = balanced_log_loss(y_true, y_probs)
    logger(f"The balanced logloss is {bll}")

    full_classification_report = classification_report(y_true, y_classes)
    logger(full_classification_report)

    evaluation_scores = {
        "matthews": matthews,
        "accuracy": accuracy,
        "recall": recall,
        "f1_score_macro": f1_score_macro,
        "f1_score_micro": f1_score_micro,
        "f1_score_weighted": f1_score_weighted,
        "balanced_logloss": bll,
        "classfication_report": full_classification_report,
        "confusion_matrix": confusion_matrix(y_true, y_classes),
    }
    return evaluation_scores
