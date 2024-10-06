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
        preds = self.model.predict_proba(x)
        if (
            len(preds.shape) == 1
        ):  # if a binary classifier only gives proba of target class
            preds = np.asarray([1 - preds, preds]).T

        if isinstance(preds, pd.DataFrame):
            preds = preds.values

        # calculate p-values (credibility intervals) for each label
        p_values = []
        # loop through all rows
        for _, pred in enumerate(preds):
            p_values_each_class = []
            # within each row loop through each class score
            for _class_idx, pred_j in enumerate(pred):
                p_value = (
                    np.sum(
                        self.nonconformity_scores
                        >= self.nonconformity_measure_scorer(
                            np.asarray([1]), np.asarray([pred_j])
                        )
                    )
                    # handle ties
                    + self.random_generator.random()
                    * np.sum(
                        self.nonconformity_scores
                        == self.nonconformity_measure_scorer(
                            np.asarray([1]), np.asarray([pred_j])
                        )
                    )
                    + 1
                ) / (len(self.nonconformity_scores) + 1)
                p_values_each_class.append(p_value)
            p_values.append(p_values_each_class)

        return np.asarray(p_values)

    def predict_sets(self, x: pd.DataFrame, alpha: float = 0.05) -> np.ndarray:
        credible_intervals = self.predict_interval(x)

        # Create the list of lists (rows x classes) where each entry is 1 (in set) or 0 (not in set)
        prediction_matrix = [
            [1 if credible_interval >= alpha else 0 for credible_interval in row]
            for row in credible_intervals
        ]

        # Convert the list of lists to a numpy array
        return np.array(prediction_matrix)
