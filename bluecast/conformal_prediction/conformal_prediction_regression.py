from typing import Any, Callable, List

import numpy as np
import pandas as pd

from bluecast.conformal_prediction.base_classes import (
    ConformalPredictionWrapperBaseClass,
)
from bluecast.conformal_prediction.nonconformity_measures_regression import (
    absolute_error,
)


class ConformalPredictionRegressionWrapper(ConformalPredictionWrapperBaseClass):
    """
    Calibrate a model instance given a calibration set.

    :param model: An already fitted model instance of any type
    :param nonconformity_measure_scorer: A function object to calculate nonconformity scores with args
        y_calibration, preds
    """

    def __init__(
        self, model: Any, nonconformity_measure_scorer: Callable = absolute_error
    ):
        self.model = model
        self.nonconformity_measure_scorer = nonconformity_measure_scorer
        self.nonconformity_scores: np.ndarray = np.empty((0, 0))
        self.quantiles: List[float] = []

    def calibrate(self, x_calibration, y_calibration):
        """
        Calibrate a model instance given a calibration set.

        :param x_calibration: Calibration set features. Must be unseen data for the model
        :param y_calibration: Calibration set labels or values
        """
        preds = self.model.predict(x_calibration)
        self.nonconformity_scores = self.nonconformity_measure_scorer(
            y_calibration, preds
        )
        return self.nonconformity_scores

    def predict(self, x):
        return self.model.predict(x)

    def _calculate_intervals(
        self, y_hat: np.ndarray, quantiles: List[float], alphas: List[float]
    ) -> pd.DataFrame:
        """
        Add lower and upper prediction bands for every quantile in quantiles.
        """

        prediction_bands = np.zeros((len(y_hat), 2, len(quantiles)))

        lower_band_cols = []
        higher_band_cols = []
        for i, q in enumerate(quantiles):
            prediction_bands[:, :, i] = np.stack([y_hat - q, y_hat + q], axis=1)
            lower_band_cols.append(f"{alphas[i]}_low")
            higher_band_cols.append(f"{1-alphas[i]}_high")

        lower_preds = pd.DataFrame(prediction_bands[:, 0, :], columns=lower_band_cols)
        upper_preds = pd.DataFrame(prediction_bands[:, 1, :], columns=higher_band_cols)
        all_preds = pd.concat(
            [
                lower_preds,
                upper_preds.reindex(upper_preds.columns.to_list()[::-1], axis=1),
            ],
            axis=1,
        )
        return all_preds

    def predict_interval(self, x: pd.DataFrame, alphas: List[float]) -> pd.DataFrame:
        preds = self.model.predict(x)
        self.quantiles = [
            np.nanquantile(self.nonconformity_scores, 1.0 - alpha, method="higher")
            for alpha in alphas
        ]

        prediction_bands = self._calculate_intervals(preds, self.quantiles, alphas)
        return prediction_bands
