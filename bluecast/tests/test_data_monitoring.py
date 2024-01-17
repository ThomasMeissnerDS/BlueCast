import numpy as np
import pandas as pd

from bluecast.monitoring.data_monitoring import DataDrift


def test_data_drift_initialization():
    drift_monitor = DataDrift()
    assert isinstance(drift_monitor, DataDrift)
    assert isinstance(drift_monitor.drift_stats, dict)


def test_fit_data_drift():
    drift_monitor = DataDrift()

    # Create a sample DataFrame for testing
    data = pd.DataFrame(
        {
            "numeric_column": [1, 2, 3, 4, 5],
            "categorical_column": ["A", "B", "A", "B", "A"],
        }
    )

    drift_stats = drift_monitor.fit_data_drift(data)

    # Check if statistics are calculated for each column
    assert "numeric_column" in drift_stats
    assert "categorical_column" in drift_stats

    # Check if mean and std_dev are calculated for numeric columns
    assert "mean" in drift_stats["numeric_column"]
    assert "std_dev" in drift_stats["numeric_column"]

    # Check if value_counts are calculated for categorical columns
    assert "value_counts" in drift_stats["categorical_column"]


def test_check_drift_numeric():
    drift_monitor = DataDrift()

    # Set some initial statistics for testing
    drift_monitor.drift_stats = {"numeric_column": {"mean": 2.5, "std_dev": 1.5}}

    # Create a sample DataFrame with drift
    new_data_drift = pd.DataFrame({"numeric_column": [5, 6, 7, 8, 9]})

    drift_flags = drift_monitor.check_drift(new_data_drift)

    # Check if drift is detected for numeric column
    assert drift_flags["numeric_column"] is False


def generate_random_data(mean, std_dev, size):
    """Generate random data with given mean and standard deviation."""
    random_generator = np.random.default_rng(25)
    return random_generator.normal(loc=mean, scale=std_dev, size=size)


def test_check_drift_numeric_large_array():
    drift_monitor = DataDrift()

    # Set some initial statistics for testing
    initial_mean = 2.5
    initial_std_dev = 1.5
    drift_monitor.drift_stats = {
        "numeric_column": {"mean": initial_mean, "std_dev": initial_std_dev}
    }

    # Create a sample DataFrame with drift
    array_size = 1000
    new_data_drift = pd.DataFrame(
        {
            "numeric_column": generate_random_data(
                initial_mean, initial_std_dev, array_size
            )
        }
    )
    new_data_drift += 5
    print(f"Synthetic mean: {new_data_drift['numeric_column'].mean()}")
    print(f"Synthetic std: {new_data_drift['numeric_column'].std()}")

    drift_flags = drift_monitor.check_drift(new_data_drift)

    # Check if drift is detected for numeric column
    assert drift_flags["numeric_column"] is False


def test_check_drift_categorical():
    drift_monitor = DataDrift()

    # Set some initial statistics for testing
    drift_monitor.drift_stats = {
        "categorical_column": {"value_counts": {"A": 0.6, "B": 0.4}}
    }

    # Create a sample DataFrame with drift
    new_data_drift = pd.DataFrame({"categorical_column": ["A", "A", "B", "B", "B"]})

    drift_flags = drift_monitor.check_drift(new_data_drift)

    # Check if drift is detected for categorical column
    assert drift_flags["categorical_column"] is True
