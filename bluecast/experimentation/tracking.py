from datetime import datetime
from typing import Dict, List, Literal, Union

import pandas as pd

from bluecast.config.base_classes import BaseClassExperimentTracker
from bluecast.config.training_config import TrainingConfig


class ExperimentTracker(BaseClassExperimentTracker):
    def __init__(self):
        self.experiment_id: List[Union[int, str, float]] = []
        self.experiment_name: List[Union[int, str, float]] = []
        self.score_category: List[
            Literal["simple_train_test_score", "cv_score", "oof_score"]
        ] = []
        self.training_configs: List[TrainingConfig] = []
        self.model_parameters: List[Dict[Union[str, int, float, None]]] = []
        self.eval_scores: List[Union[float, int, None]] = []
        self.metric_used: List[str] = []  # TODO: Split by metrics in eval_results?
        self.metric_higher_is_better: List[bool] = []
        self.created_at: List[datetime.datetime] = []

    def add_results(
        self,
        experiment_id: Union[int, str, float],
        score_category: Literal["simple_train_test_score", "cv_score", "oof_score"],
        training_config: TrainingConfig,
        model_parameters: Dict[
            Union[str, int, float, None], Union[str, int, float, None]
        ],
        eval_scores: Union[float, int, None],
        metric_used: str,
        metric_higher_is_better: bool,
    ) -> None:
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
