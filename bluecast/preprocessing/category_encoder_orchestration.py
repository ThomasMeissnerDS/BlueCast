from typing import List, Union

import pandas as pd


class CategoryEncoderOrchestrator:
    def __init__(
        self,
        target_col: Union[str, float, int],
    ):
        self.to_onehot_encode: List[Union[str, int, float]] = []
        self.to_target_encode: List[Union[str, int, float]] = []
        self.target_col = target_col

    def fit(
        self,
        df: pd.DataFrame,
        cat_columns: List[Union[str, int, float]],
        threshold: int = 5,
    ) -> None:
        """
        Map categorical columns to appropriate encoder.

        Measures the cardinality of each categorical column.
        Assign either onehot or target encoder depending on cardinality threshold.
        :param df: DataFrame containing the categorical columns
        :param cat_columns: List containing the name of categorical columns.
        :param threshold: If cardinality is less or equal the threshold, the column will be assigned  to onehot
            encoding, otherwise target encoding will be assigned.
        """
        for col in cat_columns:
            if col != self.target_col:
                cardinality = df[col].nunique()
                if cardinality <= threshold:
                    self.to_onehot_encode.append(col)
                else:
                    self.to_target_encode.append(col)
