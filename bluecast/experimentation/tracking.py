from typing import Dict, List, Literal, Union

import pandas as pd

from bluecast.config.base_classes import BaseClassExperimentTracker
from bluecast.config.training_config import TrainingConfig


class ExperimentTracker(BaseClassExperimentTracker):
    def __init__(self):
        self.experiment_id: List[Union[int, str, float]] = []
        self.experiment_name: List[Union[int, str, float]] = []
        self.score_category: List[Literal["hyperparameter_tuning", "oof_score"]] = []
        self.training_configs: List[TrainingConfig] = []
        self.model_parameters: List[Dict[Union[str, int, float, None]]] = []
        self.eval_scores: List[Union[float, int, None]] = []
        self.metric_used: List[str] = []  # TODO: Split by metrics in eval_results?
        self.metric_higher_is_better: List[bool] = []

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
        self.experiment_id.append(experiment_id)
        self.experiment_name.append(experiment_name)
        self.score_category.append(score_category)
        self.training_configs.append(training_configs.dump(mode="json"))
        self.model_parameters.append(model_parameters)
        self.eval_scores.append(eval_scores)
        self.metric_used.append(metric_used)
        self.metric_higher_is_better.append(metric_higher_is_better)

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
                # section where we make use of values in the training config
                "experiment_name": [
                    conf.experiment_name for conf in self.training_configs
                ],
            }
        )
        results_df = results_df.merge(
            training_df, how="left", left_index=True, right_index=True
        )

        results_df = results_df.merge(
            model_parameters_df, how="left", left_index=True, right_index=True
        )
        return results_df
