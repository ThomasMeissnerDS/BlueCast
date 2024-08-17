"""Module containing model orchestration tools."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.monitoring.data_monitoring import DataDrift


class ModelMatchMaker:
    """
    Matching the incoming data with the best model based on the adversarial validation score.
    """

    def __init__(self):
        self.bluecast_instances = []
        self.training_datasets = []

    def append_model_and_dataset(
        self,
        bluecast_instance: Union[
            BlueCast, BlueCastRegression, BlueCastCV, BlueCastCVRegression
        ],
        df: pd.DataFrame,
    ):
        """
        Append the model and the dataset to the matchmaker.

        :param bluecast_instance: The BlueCast instance to append.
        :param df: The dataset to append.
        """
        self.bluecast_instances.append(bluecast_instance)
        self.training_datasets.append(df)

    def find_best_match(
        self,
        df: pd.DataFrame,
        use_cols: List[Union[int, float, str]],
        cat_columns: Optional[List],
        delta: float,
    ) -> Tuple[
        Optional[Union[BlueCast, BlueCastRegression, BlueCastCV, BlueCastCVRegression]],
        Optional[pd.DataFrame],
    ]:
        """
        Find the best match based on the adversarial validation score.
        :param df: Dataset to match.
        :param use_cols: Columns to use for the adversarial validation. Numerical columns are allowed only.
        :param delta: Maximum delta for the adversarial validation score to be away from 0.5. If no dataset reaches this
         delta, (None, None) is returned.
        :param cat_columns: (Optional) List with names of categorical columns.
        :return: If a match is found, the BlueCast instance and the dataset are returned. Otherwise, (None, None) is
            returned.
        """
        best_score = np.inf
        best_idx = None

        for idx in range(len(self.bluecast_instances)):
            data_drift_checker = DataDrift()
            auc_score = data_drift_checker.adversarial_validation(
                self.training_datasets[idx].loc[:, use_cols],
                df.loc[:, use_cols],
                cat_columns,
            )
            score_delta = np.abs(auc_score - 0.5)
            if score_delta < best_score and np.abs(auc_score - 0.5) <= delta:
                print(
                    f"Found best match using idx {idx} with AUC score of {auc_score} and delta of {score_delta}"
                )
                best_score = score_delta
                best_idx = idx
                print(f"Best idx: {best_idx}, {self.bluecast_instances[best_idx]}")

        if best_idx:
            return self.bluecast_instances[best_idx], self.training_datasets[best_idx]
        else:
            print("No training dataset has reached the threshold criterium.")
            return None, None
