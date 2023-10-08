from typing import List, Literal

import pandas as pd

from bluecast.config.training_config import TrainingConfig


class ExperimentTracker:
    def __init__(self):
        self.experiment_id: List[int] = []
        self.experiment_name: List[str] = []
        self.score_category: List[Literal["hyperparameter_tuning", "oof_score"]] = []
        self.training_configs: List[TrainingConfig] = []
        self.eval_scores: List[float] = []
        self.metric_used: List[str] = []  # TODO: Split by metrics in eval_results?
        self.metric_higher_is_better: List[bool] = []

    def add_results(
        self,
        experiment_id,
        experiment_name,
        score_category,
        training_configs,
        eval_scores,
        metric_used,
        metric_higher_is_better,
    ) -> None:
        self.experiment_id.append(experiment_id)
        self.experiment_name.append(experiment_name)
        self.score_category.append(score_category)
        self.training_configs.append(training_configs)
        self.eval_scores.append(eval_scores)
        self.metric_used.append(metric_used)
        self.metric_higher_is_better.append(metric_higher_is_better)

    def retrieve_results_as_df(self) -> pd.DataFrame:
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
                "shuffle_during_training": [
                    conf.shuffle_during_training for conf in self.training_configs
                ],
                "hyperparameter_tuning_rounds": [
                    conf.hyperparameter_tuning_rounds for conf in self.training_configs
                ],
                "hyperparameter_tuning_max_runtime_secs": [
                    conf.hyperparameter_tuning_max_runtime_secs
                    for conf in self.training_configs
                ],
                "hypertuning_cv_folds": [
                    conf.hypertuning_cv_folds for conf in self.training_configs
                ],
                "global_random_state": [
                    conf.global_random_state for conf in self.training_configs
                ],
                "early_stopping_rounds": [
                    conf.early_stopping_rounds for conf in self.training_configs
                ],
                "autotune_model": [
                    conf.autotune_model for conf in self.training_configs
                ],
                "enable_feature_selection": [
                    conf.enable_feature_selection for conf in self.training_configs
                ],
                "train_size": [conf.train_size for conf in self.training_configs],
                "train_split_stratify": [
                    conf.train_split_stratify for conf in self.training_configs
                ],
                "use_full_data_for_final_model": [
                    conf.use_full_data_for_final_model for conf in self.training_configs
                ],
                "min_features_to_select": [
                    conf.min_features_to_select for conf in self.training_configs
                ],
                "cat_encoding_via_ml_algorithm": [
                    conf.cat_encoding_via_ml_algorithm for conf in self.training_configs
                ],
                "optuna_sampler_n_startup_trials": [
                    conf.optuna_sampler_n_startup_trials
                    for conf in self.training_configs
                ],
            }
        )
        return results_df
