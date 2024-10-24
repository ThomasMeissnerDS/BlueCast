import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

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
    # Call the date_converter function with the sample dataframe
    result = date_converter(
        sample_dataframe,
        ["datetime_col"],
        date_parts=["year", "week_of_year", "month", "day", "dayofweek", "hour"],
    )

    # Create the expected result after applying date_converter
    expected_data = {
        "other_col": [1, 2],
    }

    # Extract date parts
    date_col = sample_dataframe["datetime_col"]
    year = date_col.dt.year.astype(float)
    week_of_year = date_col.dt.isocalendar().week.astype(float)
    month = date_col.dt.month.astype(float)
    day = date_col.dt.day.astype(float)
    dayofweek = date_col.dt.dayofweek.astype(float)
    hour = date_col.dt.hour.astype(float)

    # Add date parts to expected data if there are more than one unique values
    if year.nunique() > 1:
        expected_data["datetime_col_year"] = year
    if week_of_year.nunique() > 1:
        expected_data["datetime_col_week_of_year"] = week_of_year
    if month.nunique() > 1:
        expected_data["datetime_col_month"] = month
        # Add cyclic features for month
        expected_data["datetime_col_month_sin"] = np.sin(2 * np.pi * month / 12)
        expected_data["datetime_col_month_cos"] = np.cos(2 * np.pi * month / 12)
    if week_of_year.nunique() > 1:
        expected_data["datetime_col_week_of_year"] = week_of_year
        expected_data["datetime_col_week_of_year_sin"] = np.sin(
            2 * np.pi * week_of_year / 52
        )
        expected_data["datetime_col_week_of_year_cos"] = np.cos(
            2 * np.pi * week_of_year / 52
        )

    if day.nunique() > 1:
        expected_data["datetime_col_day"] = day
        # Add cyclic features for day
        expected_data["datetime_col_day_sin"] = np.sin(2 * np.pi * day / 31)
        expected_data["datetime_col_day_cos"] = np.cos(2 * np.pi * day / 31)
    if dayofweek.nunique() > 1:
        expected_data["datetime_col_dayofweek"] = dayofweek
        # Add cyclic features for dayofweek
        expected_data["datetime_col_dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
        expected_data["datetime_col_dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    if hour.nunique() > 1:
        expected_data["datetime_col_hour"] = hour
        # Add cyclic features for hour
        expected_data["datetime_col_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        expected_data["datetime_col_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Create expected DataFrame
    expected_result = pd.DataFrame(expected_data)

    # Reorder columns to match the result DataFrame
    expected_result = expected_result[result.columns]

    # Assert that the result matches the expected result
    assert_frame_equal(
        result.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )

    # Test with date_parts=None (which defaults to all parts)
    result_with_default_parts = date_converter(
        sample_dataframe,
        ["datetime_col"],
        date_parts=None,
    )

    # The expected_result remains the same
    assert_frame_equal(
        result_with_default_parts.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )


def test_date_converter_single_unique_value():
    # Sample data where all dates have the same month
    data = {
        "datetime_col": pd.to_datetime(["2021-01-01 10:30:00", "2021-01-15 15:45:00"]),
        "other_col": [1, 2],
    }
    df = pd.DataFrame(data)

    result = date_converter(
        df,
        ["datetime_col"],
        date_parts=["month"],
    )

    # Since there's only one unique month, no 'datetime_col_month' column should be added
    expected_result = pd.DataFrame(
        {
            "other_col": [1, 2],
        }
    )

    assert_frame_equal(
        result.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )
