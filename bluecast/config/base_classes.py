from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Union

import pandas as pd

from bluecast.config.training_config import TrainingConfig


class BaseClassExperimentTracker(ABC):
    """Base class for the experiment tracker.

    Enforces the implementation of the add_results and retrieve_results_as_df methods.
    """

    @abstractmethod
    def add_results(
        self,
        experiment_id: int,
        score_category: Literal["simple_train_test_score", "cv_score", "oof_score"],
        training_config: TrainingConfig,
        model_parameters: Dict[Any, Any],
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
