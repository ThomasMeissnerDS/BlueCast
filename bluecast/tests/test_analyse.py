from typing import Tuple

import pandas as pd
import pytest

from bluecast.eda.analyse import (
    bi_variate_plots,
    correlation_heatmap,
    correlation_to_target,
    univariate_plots,
)
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


@pytest.fixture
def synthetic_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = create_synthetic_dataframe(2000, random_state=20)
    df_val = create_synthetic_dataframe(2000, random_state=200)
    return df_train, df_val


def test_univariate_plots(synthetic_train_test_data):
    univariate_plots(
        synthetic_train_test_data[0].loc[
            :, ["numerical_feature_1", "numerical_feature_2", "numerical_feature_3"]
        ],
        "target",
    )
    assert True


def test_bi_variate_plots(synthetic_train_test_data):
    bi_variate_plots(synthetic_train_test_data[0], "target")
    assert True


def test_correlation_heatmap(synthetic_train_test_data):
    correlation_heatmap(synthetic_train_test_data[0])
    assert True


def test_correlation_to_target(synthetic_train_test_data):
    correlation_to_target(synthetic_train_test_data[0], "target")
    assert True
