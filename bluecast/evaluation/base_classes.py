"""Base classes for evaluation purposes"""

from abc import ABC, abstractmethod

import pandas as pd


class ErrorAnalyser(ABC):
    """Abstract class to define error analysis."""

    @abstractmethod
    def analyse_errors(self, df) -> None:
        raise NotImplementedError

    @abstractmethod
    def show_leaderboard(self) -> pd.DataFrame:
        raise NotImplementedError
