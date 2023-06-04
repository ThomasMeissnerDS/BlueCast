from config.training_config import TrainingConfig, XgboostTuneParamsConfig, XgboostFinalParamConfig
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import class_weight
from typing import Dict, Literal, Optional, Tuple
import xgboost as xgb

from preprocessing.general_utils import check_gpu_support


class XgboostModel:
    def __init__(self, class_problem: Literal["binary", "multiclass"],
                 conf_training: Optional[TrainingConfig] = None,
                 conf_xgboost: Optional[XgboostTuneParamsConfig] = None,
                 conf_params_xgboost: Optional[XgboostFinalParamConfig] = None):
        self.model: Optional[xgb.XGBClassifier] = None
        self.autotune_params: bool = True
        self.class_problem = class_problem
        self.conf_training = conf_training
        self.conf_xgboost = conf_xgboost
        self.conf_params_xgboost = conf_params_xgboost

    def calculate_class_weights(self, y: pd.Series) -> Dict[str, float]:
        classes_weights = class_weight.compute_sample_weight(
            class_weight="balanced", y=y
        )
        return classes_weights

    def check_load_confs(self):
        if not self.conf_training:
            self.conf_training = TrainingConfig()

        if not self.conf_xgboost:
            self.conf_xgboost = XgboostTuneParamsConfig()

        if not self.conf_params_xgboost:
            self.conf_params_xgboost = XgboostFinalParamConfig()

    def fit(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> xgb.Booster:
        self.check_load_confs()
        if self.autotune_params:
            self.autotune(x_train, y_train, x_test, y_test)

        if self.conf_params_xgboost.sample_weight:
            classes_weights = self.calculate_class_weights(y_train)
            d_train = xgb.DMatrix(
                x_train, label=y_train, weight=classes_weights
            )
        else:
            d_train = xgb.DMatrix(x_train, label=y_train)
        d_test = xgb.DMatrix(x_test, label=y_test)
        eval_set = [(d_train, "train"), (d_test, "test")]

        model = xgb.train(
            self.conf_params_xgboost.params,
            d_train,
            num_boost_round=self.conf_params_xgboost.params["steps"],
            early_stopping_rounds=self.conf_training.early_stopping_rounds,
            evals=eval_set,
        )
        self.model = model
        return self.model

    def autotune(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        d_test = xgb.DMatrix(x_test, label=y_test)
        train_on = check_gpu_support()

        self.check_load_confs()

        def objective(trial):
            param = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "verbose": 0,
                "tree_method": train_on,
                "num_class": y_train.nunique(),
                "max_depth": trial.suggest_int("max_depth", self.conf_xgboost.max_depth_min, self.conf_xgboost.max_depth_max),
                "alpha": trial.suggest_loguniform("alpha", self.conf_xgboost.alpha_min, self.conf_xgboost.alpha_max),
                "lambda": trial.suggest_loguniform("lambda", self.conf_xgboost.lambda_min, self.conf_xgboost.lambda_max),
                "num_leaves": trial.suggest_int("num_leaves", self.conf_xgboost.num_leaves_min, self.conf_xgboost.num_leaves_max),
                "subsample": trial.suggest_uniform("subsample", self.conf_xgboost.sub_sample_min, self.conf_xgboost.sub_sample_max),
                "colsample_bytree": trial.suggest_uniform(
                    "colsample_bytree", self.conf_xgboost.col_sample_by_tree_min, self.conf_xgboost.col_sample_by_tree_max
                ),
                "colsample_bylevel": trial.suggest_uniform(
                    "colsample_bylevel", self.conf_xgboost.col_sample_by_level_min, self.conf_xgboost.col_sample_by_level_max
                ),
                "colsample_bynode": trial.suggest_uniform(
                    "colsample_bynode", self.conf_xgboost.col_sample_by_node_min, self.conf_xgboost.col_sample_by_node_max
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", self.conf_xgboost.min_child_samples_min, self.conf_xgboost.min_child_samples_max
                ),
                "eta": self.conf_xgboost.eta,  # 0.001
                "steps": trial.suggest_int("steps", self.conf_xgboost.steps_min, self.conf_xgboost.steps_max),
                "num_parallel_tree": trial.suggest_int(
                    "num_parallel_tree", self.conf_xgboost.num_parallel_tree_min, self.conf_xgboost.num_parallel_tree_max
                ),
            }
            sample_weight = trial.suggest_categorical(
                "sample_weight", [True, False]
            )
            if sample_weight:
                classes_weights = self.calculate_class_weights(y_train)
                d_train = xgb.DMatrix(
                    x_train, label=y_train, weight=classes_weights
                )
            else:
                d_train = xgb.DMatrix(x_train, label=y_train)

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-mlogloss"
            )

            if self.conf_training.hypertuning_cv_folds == 1:
                eval_set = [(d_train, "train"), (d_test, "test")]
                model = xgb.train(
                    param,
                    d_train,
                    num_boost_round=param["steps"],
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    evals=eval_set,
                    callbacks=[pruning_callback],
                )
                preds = model.predict(d_test)
                pred_labels = np.asarray(
                    [np.argmax(line) for line in preds]
                )
                matthew = matthews_corrcoef(y_test, pred_labels) * -1
                return matthew
            else:
                result = xgb.cv(
                    params=param,
                    dtrain=d_train,
                    num_boost_round=param["steps"],
                    early_stopping_rounds=self.conf_training.early_stopping_rounds,
                    nfold=self.conf_training.hypertuning_cv_folds,
                    as_pandas=True,
                    seed=self.conf_training.global_random_state,
                    callbacks=[pruning_callback],
                    shuffle=self.conf_training.shuffle_during_training,
                )

                return result["test-mlogloss-mean"].mean()

        algorithm = "xgboost"
        sampler = optuna.samplers.TPESampler(
            multivariate=True, seed=self.conf_training.global_random_state
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"{algorithm} tuning",
        )

        study.optimize(
            objective,
            n_trials=self.conf_training.hyperparameter_tuning_rounds,
            timeout=self.conf_training.hyperparameter_tuning_max_runtime_secs,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.show()
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()
        except ZeroDivisionError:
            pass

        xgboost_best_param = study.best_trial.params
        self.conf_params_xgboost.params = {
            "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
            "eval_metric": "mlogloss",
            "verbose": 0,
            "tree_method": train_on,  # use GPU for training
            "num_class": y_train.nunique(),
            "max_depth": xgboost_best_param[
                "max_depth"
            ],  # maximum depth of the decision trees being trained
            "alpha": xgboost_best_param["alpha"],
            "lambda": xgboost_best_param["lambda"],
            "num_leaves": xgboost_best_param["num_leaves"],
            "subsample": xgboost_best_param["subsample"],
            "colsample_bytree": xgboost_best_param["colsample_bytree"],
            "colsample_bylevel": xgboost_best_param["colsample_bylevel"],
            "colsample_bynode": xgboost_best_param["colsample_bynode"],
            "min_child_samples": xgboost_best_param["min_child_samples"],
            "eta": xgboost_best_param["eta"],
            "steps": xgboost_best_param["steps"],
            "num_parallel_tree": xgboost_best_param["num_parallel_tree"],
        }
        self.conf_params_xgboost.sample_weight = xgboost_best_param["sample_weight"]

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated based on SHAP values.
        :return: Updates class attributes.
        """
        d_test = xgb.DMatrix(df)
        model = self.model
        partial_probs = model.predict(d_test)
        if self.class_problem == "binary":
            predicted_probs = np.asarray([line[1] for line in partial_probs])
            predicted_classes = (
                    predicted_probs
                    > self.conf_params_xgboost.classification_threshold
            )
        else:
            predicted_probs = partial_probs
            predicted_classes = np.asarray(
                [np.argmax(line) for line in partial_probs]
            )
        return predicted_probs, predicted_classes

