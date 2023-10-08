from typing import List, Literal

class ExperimentTracker:
    def __init__(self):
        self.experiment_id: List[int] = []
        self.experiment_name: List[str] = []
        self.score_category: List[Literal["hyperparameter_tuning", "oof_score"]] = []
        self.nb_cv_folds: List[int] = []
        self.eval_scores: List[float] = []
        self.metric_used: List[str] = []

    def add_results(self, experiment_id, experiment_name, score_category, nb_cv_folds, eval_scores, metric_used) -> None:
        self.experiment_id.append(experiment_id)
        self.experiment_name.append(experiment_name)
        self.score_category.append(score_category)
        self.nb_cv_folds.append(nb_cv_folds)
        self.eval_scores.append(eval_scores)
        self.metric_used.append(metric_used)

    def retrieve_results_as_df(self) -> pd.DataFrame:
        pass
