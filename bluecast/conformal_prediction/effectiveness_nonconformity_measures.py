from typing import Dict, Union

import numpy as np
import pandas as pd


def convert_expected_effectiveness_nonconformity_input_types(
    y_hat: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    if isinstance(y_hat, pd.Series):
        y_hat = y_hat.values
        return y_hat
    elif isinstance(y_hat, np.ndarray):
        return y_hat
    elif isinstance(y_hat, pd.DataFrame):
        y_hat = y_hat.values
        return y_hat
    else:
        raise ValueError(
            f"Please provide a numpy array or Pandas Series rather than a {type(y_hat)}."
        )


def one_c(y_hat: Union[np.ndarray, pd.Series]):
    """
    Calculate proportion of singleton sets among all prediction sets.
    :param y_hat: Predicted probabilities of shape (n_samples, 1) where each row is a set of classes.
    """
    y_hat = convert_expected_effectiveness_nonconformity_input_types(y_hat)

    singleton_counts = [len(ps[0]) == 1 for ps in y_hat]
    num_singletons = sum(singleton_counts)
    return num_singletons / y_hat.shape[0]


def avg_c(y_hat: Union[np.ndarray, pd.Series]):
    """
    Calculate the average number of labels in all prediction sets.
    :param y_hat: Predicted probabilities of shape (n_samples, 1) where each row is a set of classes.
    """
    y_hat = convert_expected_effectiveness_nonconformity_input_types(y_hat)
    set_counts = [len(ps[0]) for ps in y_hat]
    return np.mean(np.asarray(set_counts))


def prediction_interval_spans(
    prediction_intervals: pd.DataFrame, alphas
) -> Dict[float, float]:
    """
    Calculate the mean span or width prediction intervals.

    This checks the distance between low and high band for each alpha.
    :param prediction_intervals: Predicted bands according to provided confidence levels.
    :param alphas: List of alphas indicating which confidence levels to check
    """
    interval_spans = {}
    for alpha in alphas:
        interval_spans[alpha] = np.mean(
            (
                prediction_intervals[f"{1-alpha}_high"]
                - prediction_intervals[f"{alpha}_low"]
            )
        )

    return interval_spans
