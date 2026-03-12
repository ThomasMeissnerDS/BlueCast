import logging
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bluecast.conformal_prediction.base_classes import (
    ConformalPredictionWrapperBaseClass,
)
from bluecast.conformal_prediction.nonconformity_measures_regression import (
    absolute_error,
)


class ConformalPredictionRegressionWrapper(ConformalPredictionWrapperBaseClass):
    """Conformal prediction wrapper for regression with optional group-conditional intervals.

    :param model: An already fitted model instance of any type
    :param nonconformity_measure_scorer: A function object to calculate nonconformity scores with args
        y_calibration, preds
    :param min_group_size: Minimum number of calibration samples required for a group to get
        its own conditional interval. Groups below this threshold fall back to global scores.
    """

    def __init__(
        self,
        model: Any,
        nonconformity_measure_scorer: Callable = absolute_error,
        min_group_size: int = 30,
    ):
        self.model = model
        self.nonconformity_measure_scorer = nonconformity_measure_scorer
        self.nonconformity_scores: np.ndarray = np.empty((0, 0))
        self.nonconformity_scores_by_group: Optional[Dict[Any, np.ndarray]] = None
        self.group_columns: Optional[List[str]] = None
        self.min_group_size = min_group_size
        self.quantiles: List[float] = []

    def plot_non_conformity_scores(self, nonconformity_scores: np.ndarray) -> None:
        """Plot the distribution of nonconformity scores."""
        calib_conformal_vals = np.sort(nonconformity_scores)
        plt.plot(calib_conformal_vals)
        plt.grid(True)
        plt.ylabel("Conformity value")
        plt.title("Distribution of non-conformity values")

    def _get_group_key(self, row: pd.Series) -> tuple:
        """Extract group key from a row based on group_columns."""
        if self.group_columns is None:
            return ()
        return tuple(row[col] for col in self.group_columns)

    def _get_group_keys_for_df(self, df: pd.DataFrame) -> pd.Series:
        """Get group keys for all rows in a DataFrame."""
        if self.group_columns is None or len(self.group_columns) == 0:
            return pd.Series([() for _ in range(len(df))], index=df.index)
        if len(self.group_columns) == 1:
            return df[self.group_columns[0]].apply(lambda x: (x,))
        return df[self.group_columns].apply(tuple, axis=1)

    def calibrate(
        self,
        x_calibration: pd.DataFrame,
        y_calibration: pd.Series,
        group_columns: Optional[List[str]] = None,
    ):
        """Calibrate a model instance given a calibration set.

        :param x_calibration: Calibration set features. Must be unseen data for the model
        :param y_calibration: Calibration set labels or values
        :param group_columns: Optional list of column names for group-conditional calibration.
            When provided, separate nonconformity score distributions are maintained per group,
            yielding group-specific prediction interval widths.
        """
        preds = self.model.predict(x_calibration)
        self.nonconformity_scores = self.nonconformity_measure_scorer(
            y_calibration, preds
        )

        self.group_columns = group_columns
        if group_columns is not None and len(group_columns) > 0:
            self.nonconformity_scores_by_group = {}
            group_keys = self._get_group_keys_for_df(x_calibration)

            for group_key in group_keys.unique():
                mask = group_keys == group_key
                group_scores = self.nonconformity_scores[mask.values]

                if len(group_scores) >= self.min_group_size:
                    self.nonconformity_scores_by_group[group_key] = group_scores
                else:
                    logging.info(
                        f"Group {group_key} has {len(group_scores)} samples "
                        f"(< {self.min_group_size}), using global scores."
                    )

            logging.info(
                f"Group-conditional calibration: {len(self.nonconformity_scores_by_group)} "
                f"groups with sufficient samples out of {len(group_keys.unique())} total."
            )

        return self.nonconformity_scores

    def predict(self, x):
        return self.model.predict(x)

    def _get_scores_for_group(self, group_key: tuple) -> np.ndarray:
        """Get nonconformity scores for a group, falling back to global if needed."""
        if (
            self.nonconformity_scores_by_group is not None
            and group_key in self.nonconformity_scores_by_group
        ):
            return self.nonconformity_scores_by_group[group_key]
        return self.nonconformity_scores

    def _calculate_intervals(
        self, y_hat: np.ndarray, quantiles: List[float], alphas: List[float]
    ) -> pd.DataFrame:
        """Add lower and upper prediction bands for every quantile in quantiles."""
        prediction_bands = np.zeros((len(y_hat), 2, len(quantiles)))

        lower_band_cols = []
        higher_band_cols = []
        for i, q in enumerate(quantiles):
            if isinstance(q, np.ndarray):
                prediction_bands[:, 0, i] = y_hat - q
                prediction_bands[:, 1, i] = y_hat + q
            else:
                prediction_bands[:, :, i] = np.stack([y_hat - q, y_hat + q], axis=1)
            lower_band_cols.append(f"{alphas[i]}_low")
            higher_band_cols.append(f"{1 - alphas[i]}_high")

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

    def predict_interval(
        self,
        x: pd.DataFrame,
        alphas: List[float],
        group_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Predict intervals, optionally conditioned on groups.

        :param x: Features for prediction.
        :param alphas: List of significance levels (e.g. [0.05, 0.1]).
        :param group_columns: Column names to use for group-conditional intervals.
            Must match the columns used during calibration. If None, uses the columns
            from calibration (if any).
        """
        preds = self.model.predict(x)

        effective_group_cols = group_columns or self.group_columns

        if (
            effective_group_cols is not None
            and self.nonconformity_scores_by_group is not None
        ):
            quantiles_per_alpha = []
            for alpha in alphas:
                per_sample_quantiles = np.zeros(len(x))
                group_keys = self._get_group_keys_for_df(x)

                for i, group_key in enumerate(group_keys):
                    scores = self._get_scores_for_group(group_key)
                    per_sample_quantiles[i] = np.nanquantile(
                        scores, 1.0 - alpha, method="higher"
                    )
                quantiles_per_alpha.append(per_sample_quantiles)

            self.quantiles = quantiles_per_alpha
            prediction_bands = self._calculate_intervals(
                preds, quantiles_per_alpha, alphas
            )
        else:
            self.quantiles = [
                np.nanquantile(self.nonconformity_scores, 1.0 - alpha, method="higher")
                for alpha in alphas
            ]
            prediction_bands = self._calculate_intervals(preds, self.quantiles, alphas)

        return prediction_bands
