from typing import Dict, List, Union

import numpy as np
import pandas as pd

from bluecast.conformal_prediction.effectiveness_nonconformity_measures import (
    convert_expected_effectiveness_nonconformity_input_types,
)


def prediction_set_coverage(
    y_true: Union[np.ndarray, pd.Series], prediction_sets: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate the percentyge of prediction sets that include the true label.

    This check can be used to validate that the model covers the true labels
    according to the alpha set during the prediction process.
    :param y_true: Ground truth labels..
    :param prediction_sets: Predicted probabilities of shape (n_samples, 1) where each row is a set of classes.
    """
    y_hat = convert_expected_effectiveness_nonconformity_input_types(prediction_sets)
    set_with_corr_label = [str(label) in str(ps) for label, ps in zip(y_true, y_hat)]
    return np.mean(np.asarray(set_with_corr_label))


def prediction_interval_coverage(
    y_true: Union[np.ndarray, pd.Series],
    prediction_intervals: pd.DataFrame,
    alphas: List[float],
) -> Dict[float, float]:
    """
    Calculate the percentage of prediction intervals that cover the true value.

    This check can be used to validate that the model covers the true values
    according to the alpha set during the prediction process.
    :param y_true: Ground truth labels.
    :param prediction_intervals: DataFrame with predicted bands according to provided confidence levels. This must
        contain columns of format f"{alpha}_low" and f"{1-alpha}_high" for each format.
    :param alphas: List of alphas indicating which confidence levels to check
    """
    coverages = {}
    for alpha in alphas:
        coverages[alpha] = np.mean(
            np.where(
                (
                    (prediction_intervals[f"{alpha}_low"] <= y_true)
                    & (prediction_intervals[f"{1-alpha}_high"] >= y_true)
                ),
                1,
                0,
            )
        )

    return coverages
