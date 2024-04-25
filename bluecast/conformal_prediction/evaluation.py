from typing import Union

import numpy as np
import pandas as pd

from bluecast.conformal_prediction.effectiveness_nonconformity_measures import (
    convert_expected_effectiveness_nonconformity_input_types,
)


def prediction_set_coverage(
    y_true: Union[np.ndarray, pd.Series], y_hat: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate the percentyge of prediction sets that include the true label.

    This check can be used to validate that the model covers the true labels
    according to the alpha set during the prediction process.
    :param y_true: Ground truth labels..
    :param y_hat: Predicted probabilities of shape (n_samples, 1) where each row is a set of classes.
    """
    y_hat = convert_expected_effectiveness_nonconformity_input_types(y_hat)
    set_with_corr_label = [str(label) in str(ps) for label, ps in zip(y_true, y_hat)]
    return np.mean(np.asarray(set_with_corr_label))
