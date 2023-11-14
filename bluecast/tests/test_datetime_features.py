import pandas as pd
import pytest

from bluecast.preprocessing.datetime_features import date_converter


@pytest.fixture
def sample_dataframe():
    # Create a sample dataframe for testing
    data = {
        "datetime_col": pd.to_datetime(["2021-01-01 10:30:00", "2021-02-15 15:45:00"]),
        "other_col": [1, 2],
    }
    df = pd.DataFrame(data)
    return df


def test_date_converter(sample_dataframe):
    # Define the expected result after applying date_converter
    expected_result = pd.DataFrame(
        {
            "datetime_col_month": [1, 2],
            "datetime_col_day": [1, 15],
            "datetime_col_dayofweek": [4, 0],
            "datetime_col_hour": [10, 15],
            "other_col": [1, 2],
        }
    )

    # Call the date_converter function with the sample dataframe
    result = date_converter(
        sample_dataframe,
        ["datetime_col"],
        date_parts=["month", "day", "dayofweek", "hour"],
    )

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(
        result,
        expected_result,
        check_like=True,
        check_column_type=False,
        check_dtype=False,
    )
