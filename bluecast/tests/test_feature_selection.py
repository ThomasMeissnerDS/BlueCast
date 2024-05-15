import pandas as pd
import pytest

from bluecast.preprocessing.feature_selection import BoostaRootaWrapper


@pytest.fixture
def synthetic_data():
    df = pd.DataFrame(
        {
            "feature1": [i for i in range(100)],
            "feature2": [i for i in range(100, 200)],
        }
    )
    targets = pd.Series(
        [0 for _i in range(100)] + [1 for _i in range(100)]
    )  # Sample targets
    return df, targets


def test_boostaroota_wrapper(synthetic_data):
    df, targets = synthetic_data
    # Initialize the wrapper
    boosta = BoostaRootaWrapper(class_problem="binary", random_state=45)
    # Fit the model
    trans_df, trans_targets = boosta.fit_transform(df, targets)

    assert trans_targets.equals(targets)
