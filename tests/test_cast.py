from blueprints.cast import BlueCast
from config.training_config import TrainingConfig
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from typing import Tuple


def create_synthetic_dataframe(num_samples=1000) -> pd.DataFrame:
    # Generate synthetic data using make_classification
    x, y = make_classification(n_samples=num_samples, n_features=5, n_informative=5, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)

    # Create a datetime feature
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-12-31')
    datetime_feature = pd.date_range(start=start_date, end=end_date, periods=num_samples)

    # Create categorical features
    categorical_feature_1 = np.random.choice(['A', 'B', 'C'], size=num_samples)
    categorical_feature_2 = np.random.choice(['X', 'Y', 'Z'], size=num_samples)

    # Create a DataFrame
    df = pd.DataFrame({
        'categorical_feature_1': categorical_feature_1,
        'categorical_feature_2': categorical_feature_2,
        'numerical_feature_1': x[:, 0],
        'numerical_feature_2': x[:, 1],
        'numerical_feature_3': x[:, 2],
        'datetime_feature': datetime_feature,
        'target': y
    })

    return df


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(1000)
    df_val = create_synthetic_dataframe(100)
    return df_train, df_val


def test_fit(synthetic_train_test_data):
    """Test that tests the BlueCast class"""
    df_train = synthetic_train_test_data[0]
    df_val = synthetic_train_test_data[1]
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10

    automl = BlueCast(class_problem="binary", target_column="target", conf_training=train_config)
    automl.fit(df_train, target_col="target")
    y_probs, y_classes = automl.predict(df_val)
    assert y_probs.shape[0] == len(df_val.index)
    assert y_classes.shape[0] == len(df_val.index)
    assert y_probs.shape[1] == 2
    assert y_classes.shape[1] == 1
    assert np.min(y_classes) == 0
    assert np.max(y_classes) == 1
