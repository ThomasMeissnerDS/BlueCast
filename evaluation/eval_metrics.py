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

from preprocessing.general_utils import logger


def eval_classifier(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    try:
        matthews = matthews_corrcoef(y_true, y_pred)
    except Exception:
        matthews = 0

    print(f"The Matthew correlation is {matthews}")
    logger(f"The Matthew correlation is {matthews}")
    print("-------------------")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"The accuracy is {accuracy}")
    recall = recall_score(y_true, y_pred, average="weighted")
    print(f"The recall is {recall}")
    f1_score_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"The macro F1 score is {f1_score_macro}")
    logger(f"The macro F1 score is {f1_score_macro}")
    f1_score_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"The micro F1 score is {f1_score_micro}")
    logger(f"The micro F1 score is {f1_score_micro}")
    f1_score_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"The weighted F1 score is {f1_score_weighted}")
    logger(f"The weighted F1 score is {f1_score_weighted}")

    full_classification_report = classification_report(y_true, y_pred)
    print(full_classification_report)

    logger(f"The classification report is {full_classification_report}")

    evaluation_scores = {
        "matthews": matthews,
        "accuracy": accuracy,
        "recall": recall,
        "f1_score_macro": f1_score_macro,
        "f1_score_micro": f1_score_micro,
        "f1_score_weighted": f1_score_weighted,
        "classfication_report": full_classification_report,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return evaluation_scores
