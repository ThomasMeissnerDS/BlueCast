"""
Module containing classes and function to monitor data drifts.

This is meant for pipelines on production.
"""
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from bluecast.general_utils.general_utils import logger


class DataDrift:
    """
    Monitor data drift.

    Measures training meta data and compares new data against it.
    This is suitable for batch models and not recommended for online models.
    """

    def __init__(self):
        self.random_generator = np.random.default_rng(25)
        self.kolmogorov_smirnov_flags: Dict[str, bool] = {}
        self.population_stability_index_flags: Dict[str, Any] = {}

    def eval_data_drift_categorical(self):
        pass

    def kolmogorov_smirnov_test(
        self,
        data: pd.DataFrame,
        new_data: pd.DataFrame,
        threshold: float = 0.05,
    ):
        """
        Checks for data drift in new data based on K-S test.

        OThe K-S test is a nonparametric test that compares the cumulative distributions of two data sets.
        Only columns falling under pd.api.types.is_numeric_dtype will be considered.

        :param data: Pandas DataFrame with the original data
        :param new_data: Pandas DataFrame containing new data to compare against
        :param threshold: Threshold for the Kolmogorov-Smirnov test (default is 0.05)
        :return drift_flags: Dictionary containing flags indicating data drift for each column
        """
        logger(
            f"{datetime.utcnow()}: Start checking for data drift via Kolmogorov-Smirnov test."
        )

        for column in new_data.columns:
            # Check for numerical columns
            if pd.api.types.is_numeric_dtype(new_data[column]):
                # Perform Kolmogorov-Smirnov test for numerical columns
                #  test the null hypothesis that two samples were drawn from the same distribution
                ks_stat, p_value = ks_2samp(
                    new_data[column],
                    data[column],
                )

                if p_value < threshold:
                    self.kolmogorov_smirnov_flags[
                        column
                    ] = True  # not drawn from same distribution
                else:
                    self.kolmogorov_smirnov_flags[
                        column
                    ] = False  # drawn from same distribution

    def _calculate_psi(self, expected, actual, buckettype="bins", buckets=10, axis=0):
        def scale_range(input, min_val, max_val):
            input += -(np.min(input))
            input /= np.max(input) / (max_val - min_val)
            input += min_val
            return input

        def sub_psi(e_perc, a_perc):
            a_perc = max(a_perc, 0.0001)
            e_perc = max(e_perc, 0.0001)
            return (e_perc - a_perc) * np.log(e_perc / a_perc)

        def psi(expected_array, actual_array, buckets):
            breakpoints = np.arange(0, buckets + 1) / buckets * 100
            breakpoints = scale_range(
                breakpoints, np.min(expected_array), np.max(expected_array)
            )
            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(
                expected_array
            )
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(
                actual_array
            )
            return np.sum(
                sub_psi(expected_percents[i], actual_percents[i])
                for i in range(len(expected_percents))
            )

        if len(expected.shape) == 1:
            psi_values = np.empty(len(expected))
        else:
            psi_values = np.empty(expected.shape[axis])

        for i in range(len(psi_values)):
            psi_values[i] = psi(expected, actual, buckets)

        return psi_values

    def population_stability_index(self, data: pd.DataFrame, new_data: pd.DataFrame):
        """
        Checks for data drift in new data based on population stability index.

        Interpretation of PSI scores:
        - psi <= 0.1: no change or shift in the distributions of both datasets.
        - psi 0.1 < PSI <0.2: indicates a slight change or shift has occurred.
        - psi > 0.2: indicates a large shift in the distribution has occurred between both datasets

        :param data: Pandas DataFrame with the original data
        :param new_data: Pandas DataFrame containing new data to compare against
        :return drift_flags: Dictionary containing flags indicating data drift for each column
        """
        logger(
            f"{datetime.utcnow()}: Start checking for data drift via population stability index."
        )
        top_feature_list = data.columns
        for column in top_feature_list:
            if pd.api.types.is_numeric_dtype(new_data[column]):
                # Assuming you have a validation and training set
                psi_t = self._calculate_psi(data[column], new_data[column])
                self.population_stability_index_flags[column] = psi_t
