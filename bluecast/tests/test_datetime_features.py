import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bluecast.preprocessing.datetime_features import DatePartExtractor


@pytest.fixture
def sample_dataframe():
    # Create a sample dataframe for testing
    data = {
        "datetime_col": pd.to_datetime(["2021-01-01 10:30:00", "2021-02-15 15:45:00"]),
        "other_col": [1, 2],
    }
    df = pd.DataFrame(data)
    return df


def test_date_part_extractor(sample_dataframe):
    # Create an instance of DatePartExtractor with specified date parts including 'dayofyear'
    extractor = DatePartExtractor(
        date_columns=["datetime_col"],
        date_parts=[
            "year",
            "dayofyear",
            "week_of_year",
            "month",
            "day",
            "dayofweek",
            "hour",
        ],
    )

    # Call the fit_transform method with the sample dataframe
    result = extractor.fit_transform(sample_dataframe)

    # Create the expected result after applying DatePartExtractor
    expected_data = {
        "other_col": [1, 2],
    }

    # Extract date parts
    date_col = sample_dataframe["datetime_col"]
    year = date_col.dt.year.astype(float)
    dayofyear = date_col.dt.dayofyear.astype(float)
    week_of_year = date_col.dt.isocalendar().week.astype(float)
    month = date_col.dt.month.astype(float)
    day = date_col.dt.day.astype(float)
    dayofweek = date_col.dt.dayofweek.astype(float)
    hour = date_col.dt.hour.astype(float)

    # Add date parts to expected data if there are more than one unique values
    if year.nunique() > 1:
        expected_data["datetime_col_year"] = year
    if dayofyear.nunique() > 1:
        expected_data["datetime_col_dayofyear"] = dayofyear
        expected_data["datetime_col_dayofyear_sin"] = np.sin(
            2 * np.pi * dayofyear / 365
        )
        expected_data["datetime_col_dayofyear_cos"] = np.cos(
            2 * np.pi * dayofyear / 365
        )
    if week_of_year.nunique() > 1:
        expected_data["datetime_col_week_of_year"] = week_of_year
        expected_data["datetime_col_week_of_year_sin"] = np.sin(
            2 * np.pi * week_of_year / 52
        )
        expected_data["datetime_col_week_of_year_cos"] = np.cos(
            2 * np.pi * week_of_year / 52
        )
    if month.nunique() > 1:
        expected_data["datetime_col_month"] = month
        expected_data["datetime_col_month_sin"] = np.sin(2 * np.pi * month / 12)
        expected_data["datetime_col_month_cos"] = np.cos(2 * np.pi * month / 12)
    if day.nunique() > 1:
        expected_data["datetime_col_day"] = day
        expected_data["datetime_col_day_sin"] = np.sin(2 * np.pi * day / 31)
        expected_data["datetime_col_day_cos"] = np.cos(2 * np.pi * day / 31)
    if dayofweek.nunique() > 1:
        expected_data["datetime_col_dayofweek"] = dayofweek
        expected_data["datetime_col_dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
        expected_data["datetime_col_dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    if hour.nunique() > 1:
        expected_data["datetime_col_hour"] = hour
        expected_data["datetime_col_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        expected_data["datetime_col_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Create expected DataFrame
    expected_result = pd.DataFrame(expected_data)

    # Reorder columns to match the result DataFrame
    expected_result = expected_result.reindex(columns=result.columns)

    # Assert that the result matches the expected result
    assert_frame_equal(
        result.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )

    # Test with date_parts=None (which defaults to all parts including 'dayofyear')
    extractor_default = DatePartExtractor(
        date_columns=["datetime_col"],
        date_parts=None,
    )
    result_with_default_parts = extractor_default.fit_transform(sample_dataframe)

    # Create expected data for default date parts
    expected_default_data = {
        "other_col": [1, 2],
    }

    # Add date parts including 'dayofyear'
    if year.nunique() > 1:
        expected_default_data["datetime_col_year"] = year
    if dayofyear.nunique() > 1:
        expected_default_data["datetime_col_dayofyear"] = dayofyear
        expected_default_data["datetime_col_dayofyear_sin"] = np.sin(
            2 * np.pi * dayofyear / 365
        )
        expected_default_data["datetime_col_dayofyear_cos"] = np.cos(
            2 * np.pi * dayofyear / 365
        )
    if week_of_year.nunique() > 1:
        expected_default_data["datetime_col_week_of_year"] = week_of_year
        expected_default_data["datetime_col_week_of_year_sin"] = np.sin(
            2 * np.pi * week_of_year / 52
        )
        expected_default_data["datetime_col_week_of_year_cos"] = np.cos(
            2 * np.pi * week_of_year / 52
        )
    if month.nunique() > 1:
        expected_default_data["datetime_col_month"] = month
        expected_default_data["datetime_col_month_sin"] = np.sin(2 * np.pi * month / 12)
        expected_default_data["datetime_col_month_cos"] = np.cos(2 * np.pi * month / 12)
    if day.nunique() > 1:
        expected_default_data["datetime_col_day"] = day
        expected_default_data["datetime_col_day_sin"] = np.sin(2 * np.pi * day / 31)
        expected_default_data["datetime_col_day_cos"] = np.cos(2 * np.pi * day / 31)
    if dayofweek.nunique() > 1:
        expected_default_data["datetime_col_dayofweek"] = dayofweek
        expected_default_data["datetime_col_dayofweek_sin"] = np.sin(
            2 * np.pi * dayofweek / 7
        )
        expected_default_data["datetime_col_dayofweek_cos"] = np.cos(
            2 * np.pi * dayofweek / 7
        )
    if hour.nunique() > 1:
        expected_default_data["datetime_col_hour"] = hour
        expected_default_data["datetime_col_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        expected_default_data["datetime_col_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Create expected DataFrame for default date parts
    expected_default_result = pd.DataFrame(expected_default_data)

    # Reorder columns to match the result_with_default_parts DataFrame
    expected_default_result = expected_default_result.reindex(
        columns=result_with_default_parts.columns
    )

    # Assert that the result with default parts matches the expected result
    assert_frame_equal(
        result_with_default_parts.reset_index(drop=True),
        expected_default_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )


def test_date_part_extractor_single_unique_value():
    # Sample data where all dates have the same month and dayofyear
    data = {
        "datetime_col": pd.to_datetime(["2021-01-01 10:30:00", "2021-01-01 15:45:00"]),
        "other_col": [1, 2],
    }
    df = pd.DataFrame(data)

    # Create an instance of DatePartExtractor with 'month' and 'dayofyear'
    extractor = DatePartExtractor(
        date_columns=["datetime_col"],
        date_parts=["month", "dayofyear"],
    )
    result = extractor.fit_transform(df)

    # Since there's only one unique month and one unique dayofyear, no related columns should be added
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


def test_date_part_extractor_missing_date_column(sample_dataframe, caplog):
    # Create an instance of DatePartExtractor with a non-existent date column
    extractor = DatePartExtractor(
        date_columns=["non_existent_datetime_col"],
        date_parts=["year", "dayofyear"],
    )

    with caplog.at_level(logging.WARNING):
        # Call fit_transform which should log a warning
        result = extractor.fit_transform(sample_dataframe)

    # Check that a warning was logged for the missing date column
    assert any(
        "Date column non_existent_datetime_col not found in dataframe." in message
        for message in caplog.text.splitlines()
    ), "Expected warning for missing date column was not logged."

    # The result should be the same as the input dataframe since the date column does not exist
    expected_result = sample_dataframe.copy()

    # Assert that the result matches the expected result
    assert_frame_equal(
        result.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )


def test_date_part_extractor_transform_missing_column(sample_dataframe, caplog):
    # Create and fit the extractor with 'year' and 'dayofyear'
    extractor = DatePartExtractor(
        date_columns=["datetime_col"],
        date_parts=["year", "dayofyear"],
    )
    extractor.fit_transform(sample_dataframe)

    # Modify the dataframe by dropping the date column
    transformed_df = sample_dataframe.drop(columns=["datetime_col"])

    with caplog.at_level(logging.WARNING):
        # Attempt to transform the modified dataframe, which lacks the date column
        result = extractor.transform(transformed_df)

    # Check that a warning was logged for the missing date column during transform
    assert any(
        "Date column datetime_col not found in dataframe." in message
        for message in caplog.text.splitlines()
    ), "Expected warning for missing date column during transform was not logged."

    # The result should be the same as the input transformed_df since the date column is missing
    expected_result = transformed_df.copy()

    # Assert that the result matches the expected result
    assert_frame_equal(
        result.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )


def test_date_part_extractor_transform_unfitted_column(sample_dataframe, caplog):
    # Create an extractor with 'year' and 'dayofyear' but do not fit it
    extractor = DatePartExtractor(
        date_columns=["datetime_col"],
        date_parts=["year", "dayofyear"],
    )

    with caplog.at_level(logging.WARNING):
        # Attempt to transform without fitting
        result = extractor.transform(sample_dataframe)

    # Since fit_transform was not called, included_date_parts should be empty
    # A warning should be logged indicating that the column was not processed during fit_transform
    assert any(
        "Date column datetime_col was not processed during fit_transform." in message
        for message in caplog.text.splitlines()
    ), "Expected warning for unfitted date column during transform was not logged."

    # The result should be the same as the input dataframe with the date column dropped
    expected_result = sample_dataframe.drop(columns=["datetime_col"], errors="ignore")

    # Assert that the result matches the expected result
    assert_frame_equal(
        result.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )
