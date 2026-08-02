import logging
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bluecast.conformal_prediction.base_classes import (
    ConformalPredictionWrapperBaseClass,
)
from bluecast.conformal_prediction.nonconformity_measures import hinge_loss


class ConformalPredictionWrapper(ConformalPredictionWrapperBaseClass):
    """Conformal prediction wrapper for classification with optional group-conditional sets.

    :param model: An already fitted model instance of any type
    :param nonconformity_measure_scorer: A function object to calculate nonconformity scores with args
        y_calibration, preds
    :param random_seed: Random seed for tie-breaking in p-value computation
    :param min_group_size: Minimum calibration samples per group for conditional prediction sets
    """

    def __init__(
        self,
        model: Any,
        nonconformity_measure_scorer: Callable = hinge_loss,
        random_seed: int = 20,
        min_group_size: int = 30,
    ):
        self.model = model
        self.nonconformity_measure_scorer = nonconformity_measure_scorer
        self.nonconformity_scores: List[float] = []
        self.nonconformity_scores_by_group: Optional[Dict[Any, List[float]]] = None
        self.group_columns: Optional[List[str]] = None
        self.min_group_size = min_group_size
        self.random_seed = random_seed
        self.random_generator = np.random.default_rng(self.random_seed)

    def plot_non_conformity_scores(self, nonconformity_scores: List[float]) -> None:
        """Plot the distribution of nonconformity scores."""
        calib_conformal_vals = np.sort(nonconformity_scores)
        plt.plot(calib_conformal_vals)
        plt.grid(True)
        plt.ylabel("Conformity value")
        plt.title("Distribution of non-conformity values")

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
        :param group_columns: Optional list of column names for group-conditional calibration
        """
        preds = self.model.predict_proba(x_calibration)
        self.nonconformity_scores = self.nonconformity_measure_scorer(
            y_calibration, preds
        )
        self.plot_non_conformity_scores(self.nonconformity_scores)

        self.group_columns = group_columns
        if group_columns is not None and len(group_columns) > 0:
            self.nonconformity_scores_by_group = {}
            group_keys = self._get_group_keys_for_df(x_calibration)
            scores_array = np.array(self.nonconformity_scores)

            for group_key in group_keys.unique():
                mask = group_keys == group_key
                group_scores = scores_array[mask.values].tolist()

                if len(group_scores) >= self.min_group_size:
                    self.nonconformity_scores_by_group[group_key] = group_scores
                else:
                    logging.info(
                        f"Group {group_key} has {len(group_scores)} samples "
                        f"(< {self.min_group_size}), using global scores."
                    )

            logging.info(
                f"Group-conditional calibration: {len(self.nonconformity_scores_by_group)} "
                f"groups with sufficient samples."
            )

        return self.nonconformity_scores

    def _get_scores_for_group(self, group_key: tuple) -> List[float]:
        """Get nonconformity scores for a group, falling back to global if needed."""
        if (
            self.nonconformity_scores_by_group is not None
            and group_key in self.nonconformity_scores_by_group
        ):
            return self.nonconformity_scores_by_group[group_key]
        return self.nonconformity_scores

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict_interval(
        self,
        x: pd.DataFrame,
        group_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute p-values for each class, optionally conditioned on groups.

        :param x: Features for prediction.
        :param group_columns: Column names for group-conditional p-values.
        """
        preds = self.model.predict_proba(x)
        if len(preds.shape) == 1:
            preds = np.asarray([1 - preds, preds]).T
        elif isinstance(preds, pd.DataFrame):
            preds = preds.values

        effective_group_cols = group_columns or self.group_columns
        use_groups = (
            effective_group_cols is not None
            and self.nonconformity_scores_by_group is not None
        )

        if use_groups:
            group_keys = self._get_group_keys_for_df(x)

        n_total_random = len(preds) * preds.shape[1]
        random_values = self.random_generator.random(n_total_random)

        p_values = np.empty_like(preds, dtype=float)

        for i, pred in enumerate(preds):
            if use_groups:
                scores = np.array(self._get_scores_for_group(group_keys.iloc[i]))
            else:
                scores = np.array(self.nonconformity_scores)

            n_samples = len(scores) + 1

            nonconformity_measures = np.array(
                [
                    self.nonconformity_measure_scorer(np.array([1]), np.array([p]))
                    for p in pred
                ]
            )

            for j, score in enumerate(nonconformity_measures):
                greater_equal_count = np.sum(scores >= score)
                equal_count = np.sum(scores == score)

                p_values[i, j] = (
                    greater_equal_count
                    + random_values[i * len(pred) + j] * equal_count
                    + 1
                ) / n_samples

        return p_values

    def predict_sets(
        self,
        x: pd.DataFrame,
        alpha: float = 0.05,
        group_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Create prediction sets based on a confidence level.

        :param x: Features for prediction.
        :param alpha: Significance level.
        :param group_columns: Column names for group-conditional prediction sets.
        """
        credible_intervals = self.predict_interval(x, group_columns=group_columns)

        prediction_matrix = [
            [1 if credible_interval >= alpha else 0 for credible_interval in row]
            for row in credible_intervals
        ]

        return np.array(prediction_matrix)
