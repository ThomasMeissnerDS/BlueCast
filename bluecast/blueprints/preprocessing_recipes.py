from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer

from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.remove_collinearity import remove_correlated_columns


class PreprocessingForLinearModels(CustomPreprocessing):
    def __init__(self, num_columns: Optional[List]):
        super().__init__()
        self.missing_val_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.scaler = PowerTransformer(method="yeo-johnson")

        if isinstance(num_columns, List):
            self.num_columns = num_columns
        else:
            self.num_columns = []
        self.non_correlated_columns: List[Union[str, float, int]] = []

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:

        df.loc[:, self.num_columns] = df.loc[:, self.num_columns].replace(
            [np.inf, -np.inf], np.nan
        )

        if len(self.num_columns) > 0:
            df.loc[:, self.num_columns] = self.missing_val_imputer.fit_transform(
                df.loc[:, self.num_columns]
            )
            df.loc[:, self.num_columns] = self.scaler.fit_transform(
                df.loc[:, self.num_columns]
            )

        df_non_numerical = df.loc[
            :, [col for col in df.columns.to_list() if col not in self.num_columns]
        ]

        self.non_correlated_columns = remove_correlated_columns(
            df.loc[:, self.num_columns], 0.9
        ).columns.to_list()
        df_numerical = df.loc[:, self.non_correlated_columns]
        self.non_correlated_columns = df_numerical.columns.to_list()

        df = pd.concat([df_numerical, df_non_numerical], axis=1)

        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df.loc[:, self.num_columns] = df.loc[:, self.num_columns].replace(
            [np.inf, -np.inf], np.nan
        )

        if len(self.num_columns) > 0:
            df.loc[:, self.num_columns] = self.missing_val_imputer.transform(
                df.loc[:, self.num_columns]
            )
            df.loc[:, self.num_columns] = self.scaler.transform(
                df.loc[:, self.num_columns]
            )

        df_non_numerical = df.loc[
            :, [col for col in df.columns.to_list() if col not in self.num_columns]
        ]
        df_numerical = df.loc[:, self.non_correlated_columns]
        df = pd.concat([df_numerical, df_non_numerical], axis=1)

        return df, target
