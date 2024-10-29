"""Module containing model orchestration tools."""

from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import HyperbandPruner
from optuna.samplers import CmaEsSampler
from sklearn.metrics import roc_auc_score

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
        train_on_device: str = "cpu",
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
        :param train_on_device: Device to train the model on. Options are 'cpu' and 'gpu'. (Default is 'cpu')
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
                train_on_device=train_on_device,
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


class OptunaWeights:
    """
    Optimize the weights of multiple models using Optuna.

    :param random_state: Random state for reproducibility.
    :param n_trials: Number of trials to run.
    :param objective: Objective function to optimize.
    :param optimize_direction: Direction to optimize in (either 'maximize' or 'minimize').
    """

    def __init__(
        self,
        random_state,
        n_trials: int = 5000,
        objective: Callable = roc_auc_score,
        optimize_direction: Literal["minimize", "maximize"] = "maximize",
    ):
        self.study = None
        self.weights: List[float] = []
        self.random_state = random_state
        self.n_trials = n_trials
        self.objective = objective
        self.optimize_direction = optimize_direction

    def _objective(
        self, trial: optuna.Trial, y_true: np.ndarray, y_preds: List[np.ndarray]
    ) -> float:
        """
        Objective function for Optuna, defines the weighted prediction and computes the score.

        Parameters
        ----------
        :param trial: Optuna trial object to suggest weight values.
        :param y_true: Array of true target values.
        :param y_preds: List of arrays containing model predictions.
        """
        weights = [
            trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds) - 1)
        ]
        weights.append(1 - sum(weights))  # Ensure weights sum to 1

        weighted_pred = np.average(np.array(y_preds), axis=0, weights=weights)
        auc_cv = self.objective(y_true, weighted_pred)
        return auc_cv

    def fit(self, y_true: np.ndarray, y_preds: List[np.ndarray]) -> None:
        """
        Optimize weights for each model's prediction based on the objective function.

        :param y_true : Array of true target values.
        :param y_preds : List of arrays containing model predictions from different models.
        """
        if len(y_preds) < 2:
            raise ValueError(
                "`y_preds` must contain predictions from at least two models."
            )

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = CmaEsSampler(seed=self.random_state)
        pruner = HyperbandPruner()
        self.study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            study_name="OptunaWeights",
            direction=self.optimize_direction,
        )

        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)

        if isinstance(self.study, optuna.study.Study):
            self.study.optimize(
                objective_partial, n_trials=self.n_trials, show_progress_bar=True
            )
        else:
            raise ValueError(
                "Optuna study has not been created. Please create a GitHub issue."
            )

        weights = [
            self.study.best_params[f"weight{n}"] for n in range(len(y_preds) - 1)
        ]
        weights.append(1 - sum(weights))  # Ensure weights sum to 1
        self.weights = weights

    def predict(self, y_preds: List[np.ndarray]) -> np.ndarray:
        """
        Generate weighted prediction using the optimized weights.

        :param y_preds : List of arrays containing model predictions.
        """
        if len(self.weights) == 0:
            raise ValueError("Model weights have not been optimized. Call `fit` first.")

        weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
        return weighted_pred
