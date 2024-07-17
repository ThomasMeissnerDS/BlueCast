"""
Module for error analysis.

This step follows the training step. Ideally
it uses stored out of fold datasets from using the 'fit_eval' methods.
"""

from typing import Literal, Union

import pandas as pd
import polars as pl

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.evaluation.base_classes import ErrorAnalyser


class ErrorAnalyserClassification(ErrorAnalyser):
    def __init__(self, class_problem: Literal["binary", "multiclass"]):
        self.class_problem = class_problem

    def read_data_from_path(
        self,
        bluecast_instance: Union[
            BlueCast, BlueCastRegression, BlueCastCV, BlueCastCVRegression
        ],
    ) -> pl.DataFrame:
        """
        Read and create DataFrame for analyse_errors function

        :param bluecast_instance: Instance of BlueCast that created the data.
        """
        pass

    def analyse_errors(self, df: Union[pd.DataFrame, pl.DataFrame]):
        """
        Analyse errors of predictions on out of fold data.

        :param df: DataFrame holding out of fold data and predictions.
        :return: None
        """
        pass

    def show_leaderboard(self) -> pd.DataFrame:
        pass
