from typing import Union

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
