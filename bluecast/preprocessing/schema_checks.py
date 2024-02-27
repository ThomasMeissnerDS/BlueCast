"""Module for DataFrame schema checks."""

from datetime import datetime
from typing import List, Union

import pandas as pd

from bluecast.general_utils.general_utils import logger


class SchemaDetector:
    """Detect and check DataFrame schema."""

    def __init__(self):
        self.train_schema: List[Union[str, float, int]] = []

    def fit(self, df: pd.DataFrame):
        """Store the schema of the train dataset."""
        logger(f"{datetime.utcnow()}: Start fitting DataFrame schema.")
        self.train_schema = df.columns.to_list()
        return self.train_schema

    def transform(self, df: pd.DataFrame):
        """Check if the test dataset has the same schema as the train dataset.

        Will raise an error if schema length does not match and will raise a warning indicating the missing or extra
        columns."""
        logger(
            f"{datetime.utcnow()}: Start checking if DataFrame schema of new data is consistent with previous data."
        )
        if len(df.columns) > len(self.train_schema):
            new_cols = [col for col in df.columns if col not in self.train_schema]
            raise ValueError(
                f"""The number of columns in the test dataset is greater than the number of columns
            in the train dataset. Found the following new columns: {new_cols}."""
            )
        elif len(df.columns) < len(self.train_schema):
            missing_cols = [col for col in self.train_schema if col not in df.columns]
            raise ValueError(
                f"""The number of columns in the test dataset is smaller than the number of columns
            in the train dataset. Missing the following columns: {missing_cols}."""
            )
        return df[self.train_schema]
