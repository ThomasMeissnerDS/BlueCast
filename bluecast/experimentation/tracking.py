import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Union

import pandas as pd

from bluecast.config.base_classes import BaseClassExperimentTracker
from bluecast.config.training_config import TrainingConfig


class ExperimentTracker(BaseClassExperimentTracker):
    """
    Default implementation of ExperimentTracker used in BlueCast
    and BlueCastCV pipelines. This triggers automatically as long
    as the default Xgboost model is used. For custom ml models
    ueers need to create an own Tracker. The base class from
    bluecast.config.base_classes can be used as an inspiration.
    """

    def __init__(self):
        self.experiment_id: List[int] = []
        self.experiment_name: List[Union[int, str, float]] = []
        self.score_category: List[
            Literal["simple_train_test_score", "cv_score", "oof_score"]
        ] = []
        self.training_configs: List[TrainingConfig] = []
        self.model_parameters: List[Dict[Union[str, int, float, None]]] = []
        self.eval_scores: List[Union[float, int, None]] = []
        self.metric_used: List[str] = []
        self.metric_higher_is_better: List[bool] = []
        self.created_at: List[datetime] = []

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
        Add an individual experiment result into the tracker.

        :param experiment_id: Sequential id. Make sure add an increment.
        :param score_category: Chose one of ["simple_train_test_score", "cv_score", "oof_score"].
            "simple_train_test_score" is the default where a simple train-test split is done. "cv_score" is called
            when cross validation has been enabled in the TrainingConfig.
        :param training_config: TrainingConfig instance from bluecast.config.training_config.
        :param model_parameters: Dictionary with parameters of ml model (i.e. learning rate)
        :param eval_scores: The actual score of the experiment.
        :param metric_used: The name of the eval metric.
        :param metric_higher_is_better: True or False.
        """
        logging.info(f"{datetime.utcnow()}: Start adding results to ExperimentTracker.")
        self.experiment_id.append(experiment_id)
        self.score_category.append(score_category)
        try:
            self.training_configs.append(training_config.model_dump(mode="json"))
        except AttributeError:  # triggers for older Pydantic versions
            self.training_configs.append(training_config.dict())
        self.model_parameters.append(model_parameters)
        self.eval_scores.append(eval_scores)
        self.metric_used.append(metric_used)
        self.metric_higher_is_better.append(metric_higher_is_better)
        self.created_at.append(datetime.utcnow())

    def retrieve_results_as_df(self) -> pd.DataFrame:
        """
        Convert ExperimentTracker information into a Pandas DataFrame.

        In the default implementation this contains TrainingConfig, XgboostConfig, hyperparameters, eval metric
        and score.
        """
        model_parameters_df = pd.DataFrame(self.model_parameters)
        training_df = pd.DataFrame(self.training_configs)

        results_df = pd.DataFrame(
            {
                "experiment_id": self.experiment_id,
                "score_category": self.score_category,
                "eval_scores": self.eval_scores,
                "metric_used": self.metric_used,
                "metric_higher_is_better": self.metric_higher_is_better,
            }
        )
        results_df = results_df.merge(
            training_df, how="left", left_index=True, right_index=True
        )

        results_df = results_df.merge(
            model_parameters_df, how="left", left_index=True, right_index=True
        )
        return results_df

    def get_best_score(self, target_metric: str) -> Union[int, float]:
        """Expects results in the tracker"""

        results_df = pd.DataFrame(
            {
                "experiment_id": self.experiment_id,
                "score_category": self.score_category,
                "eval_scores": self.eval_scores,
                "metric_used": self.metric_used,
                "metric_higher_is_better": self.metric_higher_is_better,
            }
        )
        if results_df.empty:
            raise ValueError("No results have been found in experiment tracker")

        if self.metric_higher_is_better[-1]:
            return results_df.loc[results_df["metric_used"] == target_metric][
                "eval_scores"
            ].max()
        else:
            return results_df.loc[results_df["metric_used"] == target_metric][
                "eval_scores"
            ].min()
