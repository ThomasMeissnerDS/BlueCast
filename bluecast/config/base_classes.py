from abc import ABC, abstractmethod
from typing import Dict, Literal, Union

import pandas as pd

from bluecast.config.training_config import TrainingConfig


class BaseClassExperimentTracker(ABC):
    """Base class for the experiment tracker.

    Enforces the implementation of the add_results and retrieve_results_as_df methods.
    """

    @abstractmethod
    def add_results(
        self,
        experiment_id: Union[int, str, float],
        experiment_name: Union[int, str, float],
        score_category: Literal["hyperparameter_tuning", "oof_score"],
        training_configs: TrainingConfig,
        model_parameters: Dict[
            Union[str, int, float, None], Union[str, int, float, None]
        ],
        eval_scores: Union[float, int, None],
        metric_used: str,
        metric_higher_is_better: bool,
    ) -> None:
        """
        Add results to the ExperimentTracker class.
        """
        pass

    @abstractmethod
    def retrieve_results_as_df(self) -> pd.DataFrame:
        """
        Retrieve results from the ExperimentTracker class
        """
        pass
