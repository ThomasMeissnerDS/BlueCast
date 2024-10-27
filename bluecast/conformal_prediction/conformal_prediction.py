from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bluecast.conformal_prediction.base_classes import (
    ConformalPredictionWrapperBaseClass,
)
from bluecast.conformal_prediction.nonconformity_measures import hinge_loss


class ConformalPredictionWrapper(ConformalPredictionWrapperBaseClass):
    """
    Calibrate a model instance given a calibration set.

    :param model: An already fitted model instance of any type
    :param nonconformity_measure_scorer: A function object to calculate nonconformity scores with args
        y_calibration, preds
    """

    def __init__(
        self,
        model: Any,
        nonconformity_measure_scorer: Callable = hinge_loss,
        random_seed: int = 20,
    ):
        self.model = model
        self.nonconformity_measure_scorer = nonconformity_measure_scorer
        self.nonconformity_scores: List[float] = []
        self.random_seed = random_seed
        self.random_generator = np.random.default_rng(self.random_seed)

    def plot_non_conformity_scores(self, nonconformity_scores: List[float]) -> None:
        """
        Plot the distribution of nonconformity scores.
        :param nonconformity_scores: List of nonconformity scores
        """
        calib_conformal_vals = np.sort(nonconformity_scores)
        plt.plot(calib_conformal_vals)
        plt.grid(True)
        plt.ylabel("Conformity value")
        plt.title("Distribution of non-conformity values")

    def calibrate(self, x_calibration: pd.DataFrame, y_calibration: pd.Series):
        """
        Calibrate a model instance given a calibration set.

        :param x_calibration: Calibration set features. Must be unseen data for the model
        :param y_calibration: Calibration set labels or values
        """
        preds = self.model.predict_proba(x_calibration)
        self.nonconformity_scores = self.nonconformity_measure_scorer(
            y_calibration, preds
        )
        self.plot_non_conformity_scores(self.nonconformity_scores)
        return self.nonconformity_scores

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict_interval(self, x: pd.DataFrame) -> np.ndarray:
        # Get predictions and ensure preds is a numpy array
        preds = self.model.predict_proba(x)
        if len(preds.shape) == 1:  # binary classifier giving only target class proba
            preds = np.asarray([1 - preds, preds]).T
        elif isinstance(preds, pd.DataFrame):
            preds = preds.values

        # Prepare for p-values calculation
        n_samples = len(self.nonconformity_scores) + 1  # denominator for p-value calc
        random_values = self.random_generator.random(len(preds) * preds.shape[1])

        # Vectorize p-value calculations
        p_values = np.empty_like(preds, dtype=float)

        for i, pred in enumerate(preds):
            # Compute the nonconformity measure for each class in the current row
            nonconformity_measures = np.array(
                [
                    self.nonconformity_measure_scorer(np.array([1]), np.array([p]))
                    for p in pred
                ]
            )

            # Vectorize the p-value calculations per row
            for j, score in enumerate(nonconformity_measures):
                # Get counts for nonconformity score comparisons
                greater_equal_count = np.sum(self.nonconformity_scores >= score)
                equal_count = np.sum(self.nonconformity_scores == score)

                # Compute p-value with random tie handling
                p_values[i, j] = (
                    greater_equal_count
                    + random_values[i * len(pred) + j] * equal_count
                    + 1
                ) / n_samples

        return p_values

    def predict_sets(self, x: pd.DataFrame, alpha: float = 0.05) -> np.ndarray:
        credible_intervals = self.predict_interval(x)

        # Create the list of lists (rows x classes) where each entry is 1 (in set) or 0 (not in set)
        prediction_matrix = [
            [1 if credible_interval >= alpha else 0 for credible_interval in row]
            for row in credible_intervals
        ]

        # Convert the list of lists to a numpy array
        return np.array(prediction_matrix)
