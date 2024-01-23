from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bluecast.monitoring.data_monitoring import DataDrift


@pytest.fixture
def mock_logger():
    with patch("bluecast.general_utils.general_utils.logger") as mock:
        yield mock


def test_kolmogorov_smirnov_test(mock_logger):
    data_drift = DataDrift()

    # Generate sample data for testing
    data = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]})
    new_data = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [9, 10, 11, 12]})

    # Test Kolmogorov-Smirnov test with no data drift
    data_drift.kolmogorov_smirnov_test(data, new_data, threshold=0.05)
    assert not data_drift.kolmogorov_smirnov_flags["col1"]
    assert data_drift.kolmogorov_smirnov_flags["col2"]


def test_population_stability_index(mock_logger):
    data_drift = DataDrift()

    # Generate sample data for testing
    data = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]})
    new_data = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [9, 10, 11, 13]})

    # Test Population Stability Index with no data drift
    data_drift.population_stability_index(data, new_data)
    print(data_drift.population_stability_index_flags)
    assert not data_drift.population_stability_index_flags["col1"]
    assert data_drift.population_stability_index_flags["col2"]


def test_qqplot_two_samples():
    data_drift = DataDrift()
    # Generate sample data
    x = np.random.randn(100)
    y = np.random.randn(100)

    # Call the function with sample data
    data_drift.qqplot_two_samples(x, y)
    assert True
