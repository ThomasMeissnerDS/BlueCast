from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseClassDataDrift(ABC):
    """
    Monitor data drift.

    Measures training meta data and compares new data against it.
    This is suitable for batch models and not recommended for online models.
    """

    @abstractmethod
    def fit_data_drift(
        self, data: pd.DataFrame, anonymize_categories: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Collects statistical information about a Pandas DataFrame for monitoring data drift.

        :param data: Pandas DataFrame
        :param anonymize_categories: If True, the categorical values will be replaced by integers
        :return drift_stats: Dictionary containing statistics for each column
        """

        pass

    @abstractmethod
    def check_drift(
        self,
        new_data: pd.DataFrame,
        threshold: float = 0.05,
        anonymize_categories: bool = True,
        **kwargs
    ) -> Dict[str, bool]:
        """
        Checks for data drift in new data based on the statistics collected by fit_data_drift.

        :param new_data: Pandas DataFrame
        :param threshold: Threshold for the Kolmogorov-Smirnov test (default is 0.05)
        :param anonymize_categories: If True, the categorical values will be replaced by integers. Must match
            the setting of fit_data_drift.
        :return drift_flags: Dictionary containing flags indicating data drift for each column
        """
        pass
