from typing import Tuple, Union

import numpy as np
import pandas as pd


def convert_to_numpy(
    y_true: pd.Series, y_hat: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(y_hat, pd.Series):
        y_hat = y_hat.values
    elif isinstance(y_hat, pd.DataFrame):  # pd..core.frame.DataFrame
        y_hat = y_hat.values
    else:
        pass

    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    elif isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    else:
        pass
    return y_true, y_hat


def hinge_loss(
    y_true: pd.Series, y_hat: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> np.ndarray:
    """
    Calculate Hinge loss per row.

    To compute the nonconformity score, take the probability score of the true class and
    subtract it from 1.
    :param y_true: True labels
    :param y_hat: Predicted probabilities
    :return: Hinge loss per row
    """
    hinge_losses = []
    if len(y_hat.shape) == 1:  # if a binary classifier only gives proba of target class
        y_hat = np.asarray([1 - y_hat, y_hat]).T

    y_true, y_hat = convert_to_numpy(y_true, y_hat)

    for true_class, preds_arr in zip(y_true, y_hat):
        hinge_losses.append(1 - preds_arr[true_class])
    return np.asarray(hinge_losses)


def margin_nonconformity_measure(
    y_true: pd.Series, y_hat: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> np.ndarray:
    """
    Calculate margin nonconformity score per row.

    The margin nonconformity measure is defined as the difference between the predicted probability of
    the most likely incorrect class label and the predicted probability of the true label.
    :param y_true: True labels
    :param y_hat: Predicted probabilities
    :return: Margin nonconformity score loss per row
    """
    mnm_losses = []
    if len(y_hat.shape) == 1:  # if a binary classifier only gives proba of target class
        y_hat = np.asarray([1 - y_hat, y_hat]).T

    y_true, y_hat = convert_to_numpy(y_true, y_hat)

    for true_class, preds_arr in zip(y_true, y_hat):
        prob_y_true = preds_arr[true_class]

        # remove true label proba
        probas_y_false = np.delete(preds_arr, true_class)
        prob_y_most_likely_false = np.max(probas_y_false)
        mnm_losses.append(prob_y_true - prob_y_most_likely_false)
    return np.asarray(mnm_losses)


def brier_score(
    y_true: pd.Series, y_hat: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> np.ndarray:
    """
    Calculate Brier score per row.

    It calculates the squared difference between the predicted probabilities and the actual binary results.
    The score’s values can range from 0 (perfect accuracy) to 1 (complete inaccuracy).
    :param y_true: True labels
    :param y_hat: Predicted probabilities
    :return: Brier score loss per row
    """
    if len(y_hat.shape) == 1:  # if a binary classifier only gives proba of target class
        y_hat = np.asarray([1 - y_hat, y_hat]).T

    y_true, y_hat = convert_to_numpy(y_true, y_hat)

    brier_losses = []
    for true_class, preds_arr in zip(y_true, y_hat):
        brier_losses.append((1 - preds_arr[true_class]) ** 2)
    return np.asarray(brier_losses)
