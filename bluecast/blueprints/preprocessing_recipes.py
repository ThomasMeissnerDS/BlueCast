from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.preprocessing.remove_collinearity import remove_correlated_columns


class PreprocessingForLinearModels(CustomPreprocessing):
    def __init__(self):
        super().__init__()
        self.missing_val_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.scaler = StandardScaler()
        self.num_columns = []
        self.non_correlated_columns = []

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if len(self.num_columns) == 0:
            self.num_columns = df.columns.to_list()

        df.loc[:, self.num_columns] = df.loc[:, self.num_columns].replace(
            [np.inf, -np.inf], np.nan
        )

        df.loc[:, self.num_columns] = self.missing_val_imputer.fit_transform(
            df.loc[:, self.num_columns]
        )
        df.loc[:, self.num_columns] = self.scaler.fit_transform(
            df.loc[:, self.num_columns]
        )
        df = remove_correlated_columns(df, 0.9)
        self.non_correlated_columns = df.columns.to_list()

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

        df.loc[:, self.num_columns] = self.missing_val_imputer.transform(
            df.loc[:, self.num_columns]
        )
        df.loc[:, self.num_columns] = self.scaler.transform(df.loc[:, self.num_columns])
        df = df.loc[:, self.non_correlated_columns]

        return df, target
