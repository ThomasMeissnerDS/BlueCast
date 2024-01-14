"""
Module containing classes and function to monitor data drifts.

This is meant for pipelines on production.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from bluecast.monitoring.base_classes import BaseClassDataDrift


class DataDrift(BaseClassDataDrift):
    """
    Monitor data drift.

    Measures training meta data and compares new data against it.
    This is suitable for batch models and not recommended for online models.
    """

    def __init__(self):
        self.drift_stats: Dict[str, Any] = {}

    def fit_data_drift(self, data: pd.DataFrame, **params) -> Dict[str, Any]:
        """
        Collects statistical information about a Pandas DataFrame for monitoring data drift.

        :param: data: Pandas DataFrame
        :return drift_stats: Dictionary containing statistics for each column
        """

        for column in data.columns:
            # Calculate mean and standard deviation for numerical columns
            if pd.api.types.is_numeric_dtype(data[column]):
                mean = data[column].mean()
                std_dev = data[column].std()
                self.drift_stats[column] = {"mean": mean, "std_dev": std_dev}
            else:
                # For categorical columns, calculate the frequency of each category
                value_counts = data[column].value_counts(normalize=True).to_dict()
                self.drift_stats[column] = {"value_counts": value_counts}

        return self.drift_stats

    def check_drift(
        self, new_data: pd.DataFrame, threshold: float = 0.05, **params
    ) -> Dict[str, bool]:
        """
        Checks for data drift in new data based on the statistics collected by fit_data_drift.

        :param new_data: Pandas DataFrame
        :param threshold: Threshold for the Kolmogorov-Smirnov test (default is 0.05)
        :return drift_flags: Dictionary containing flags indicating data drift for each column
        """

        drift_flags = {}

        for column in new_data.columns:
            # Check for numerical columns
            if pd.api.types.is_numeric_dtype(new_data[column]):
                # Perform Kolmogorov-Smirnov test for numerical columns
                ks_stat, p_value = ks_2samp(
                    new_data[column],
                    np.random.normal(
                        loc=self.drift_stats[column]["mean"],
                        scale=self.drift_stats[column]["std_dev"],
                        size=len(new_data),
                    ),
                )

                if p_value < threshold:
                    drift_flags[column] = True
                else:
                    drift_flags[column] = False

            else:
                # Check for categorical columns
                value_counts_new = (
                    new_data[column].value_counts(normalize=True).to_dict()
                )
                # Compare the frequency of each category
                if value_counts_new != self.drift_stats[column]["value_counts"]:
                    drift_flags[column] = True
                else:
                    drift_flags[column] = False

        return drift_flags
