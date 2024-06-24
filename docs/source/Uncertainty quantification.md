# Uncertainty quantification

Over the past years conformal prediction gained increasing attention. It allows to
add uncertainty quantification around every model at the cost of just a bit of
additional computation.

<!-- toc -->

* [Uncertainty quantification](#uncertainty-quantification)
  * [Conformal prediction for classification](#conformal-prediction-for-classification)
  * [Conformal prediction for regression](#conformal-prediction-for-regression)
  * [Conformal prediction for non-BlueCast models](#conformal-prediction-for-non-bluecast-models)

<!-- tocstop -->

## Conformal prediction for classification

In BlueCast we provide a model and architecture agnostic conformal prediction
wrapper that allows the usage of any class that has a `predict` (regression) or
`predict_proba` (classification) method.
For `BlueCast` and `BlueCastRegression` conformal prediction can be used after
the instance has been trained.

```sh
from bluecast.blueprints.cast import BlueCast
from sklearn.model_selection import train_test_split


# we leave it up to the user to split off a calibration set
X_train, X_calibrate, y_train, y_calibrate = train_test_split(
     X, y, test_size=0.33, random_state=42)

automl = BlueCast(
        class_problem="binary",
    )

X_train["target"] = y
automl.fit(X_train, target_col="target")

# make use of calibration
automl.calibrate(X_calibrate, y_calibrate)

# point prediction as usual
y_probs, y_classes = automl.predict(df_val)

# prediction sets given a certain confidence interval alpha
pred_sets = automl.predict_sets(df_val, alpha=0.05)

# p-values for each class being the correct one
pred_intervals = automl.predict_p_values(df_val)
```

For prediction sets BlueCast offers two validation functions to test
the quality of prediction sets:

```sh
from bluecast.conformal_prediction.effectiveness_nonconformity_measures import one_c, avg_c

# return the percentage of sets with one label only (higher is better)
one_c(pred_sets.values)

# return the mean number of labels per prediction set (lower is better)
avg_c(pred_sets.values)
```

Finally we can check if the prediction sets have the credibility as expected
from the alpha. We ask the question: In how much percent of prediction sets
do we find the true class?

```python
from bluecast.conformal_prediction.evaluation import prediction_set_coverage

prediction_set_coverage(y_val, pred_sets) # where y_val has not been used during training or calibration
```

## Conformal prediction for regression

The same is possible for `BlueCastRegression` instances. However
`BlueCastRegression` offers the `predict_interval` function only:

```sh
from bluecast.blueprints.cast_regression import BlueCastRegression
from sklearn.model_selection import train_test_split


# we leave it up to the user to split off a calibration set
X_train, X_calibrate, y_train, y_calibrate = train_test_split(
     X, y, test_size=0.33, random_state=42)

automl = BlueCast(
        class_problem="binary",
    )

X_train["target"] = y
automl.fit(X_train, target_col="target")

# make use of calibration
automl.calibrate(X_calibrate, y_calibrate)

# point prediction as usual
y_hat = automl.predict(df_val)

# prediction sets given a list of confidence intervals
pred_sets = automl.predict_interval(df_val, alphas=[0.01, 0.05, 0.1])
```

Finally we can check if the prediction intervals have the credibility as expected
from the given alphas. We ask the question: In how much percent of prediction bands
do we find the true value?

```python
from bluecast.conformal_prediction.evaluation import prediction_interval_coverage

prediction_interval_coverage(y_val, pred_intervals, alphas=[0.01, 0.05, 0.1])
```

Furthermore we can calculate how broad the prediction intervals are:

```python
from bluecast.conformal_prediction.effectiveness_nonconformity_measures import prediction_interval_spans

prediction_interval_spans(pred_intervals, alphas=[0.01, 0.05, 0.1])
```

This returns the mean band size per alpha.

The variable `pred_intervals` is expected to contain columns of format
f"{alpha}_low" and f"{1-alpha}_high" for each format for both functions.

All of these functions are also available via the ensemble classes
`BlueCastCV` and `BlueCastCVRegression`.

## Conformal prediction for non-BlueCast models

Even though conformal prediction is available via all BlueCast
classes, the library offers standalone wrappers to enjoy conformal
prediction for any class, that has a `predict` or `predict_proba` method.
This can be done even without sklearn models as long as the expected
methods are available and expects one input parameter for the prediction
functions.

```python
from bluecast.conformal_prediction.conformal_prediction import ConformalPredictionWrapper
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
        n_samples=100, n_features=5, random_state=42, n_classes=2
  )
  X_train, X_calibrate, y_train, y_calibrate = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Train a logistic regression model
  model = LogisticRegression(random_state=42)
  model.fit(X_train, y_train)

  # Create a ConformalPredictionWrapper instance
  wrapper = ConformalPredictionWrapper(model)

  # Calibrate the wrapper
  wrapper.calibrate(X_calibrate, y_calibrate)

  wrapper.predict_proba(x_test)  # or predict_sets
```

For regression there is also a wrapper available:

```python
from bluecast.conformal_prediction.conformal_prediction_regression import ConformalPredictionRegressionWrapper
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = make_regression(
        n_samples=100, n_features=5, random_state=42, n_classes=2
  )
  X_train, X_calibrate, y_train, y_calibrate = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Train a logistic regression model
  model = LinearRegression(random_state=42)
  model.fit(X_train, y_train)

  # Create a ConformalPredictionWrapper instance
  wrapper = ConformalPredictionRegressionWrapper(model)

  # Calibrate the wrapper
  wrapper.calibrate(X_calibrate, y_calibrate)

  wrapper.predict_interval(x_test)
```

Some model API's don't have a `predict_proba` method. In example the native LGBM
API does not have a `predict_proba` method, but will predict all class scores,
if the 'metric' parameter is set to 'multi_logloss'. In this case we can just
copy the 'predict' method to 'predict_proba' and use the wrapper as usual.
Here we make an advanced example and pass a custom LGBM model that uses
conformal prediction for pruning and pass it to the BlueCast class.

```python
from bluecast.blueprints.cast import BlueCast
from bluecast.conformal_prediction.evaluation import prediction_set_coverage
from bluecast.conformal_prediction.conformal_prediction import ConformalPredictionWrapper

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from typing import Tuple

from bluecast.ml_modelling.base_classes import (
  PredictedClasses,  # just for linting checks
  PredictedProbas,  # just for linting checks
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

def fillna(df, strategy="zero"):
  if strategy == "zero":
    return df.fillna(0)


class LGBMTuner:
  def __init__(self, random_state=78, tune_with_alpha=0.95):
    self.param = {}
    self.model = None
    self.random_state = random_state
    self.tune_with_alpha = tune_with_alpha

  def fit(
          self,
          x_train: pd.DataFrame,
          x_test: pd.DataFrame,
          y_train: pd.Series,
          y_test: pd.Series,
  ) -> None:
    # some re-arangement, because we do nested cv here
    x_train["target"], x_test["target"] = y_train, y_test
    X = pd.concat([x_train, x_test]).reset_index(drop=True)
    y = X.pop("target")
    y = y.astype(int)
    _, _ = x_train.pop("target"), x_test.pop("target")

    print(X.columns)

    # Define an Optuna objective function for hyperparameter tuning
    def objective(trial):
      param = {
        # TODO: Move to additional folder with pyfile "constants" (use OS absolute path)
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": y.nunique(),
        "num_boost_round": trial.suggest_int("num_boost_round", 50, 1000),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-6, 100, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 100, log=True),
        "linear_lambda": trial.suggest_float("linear_lambda", 1, 100, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float(
          "feature_fraction", 0.4, 1.0
        ),
        "feature_fraction_bynode": trial.suggest_float(
          "feature_fraction_bynode", 0.4, 1.0
        ),
        "bagging_fraction": trial.suggest_float(
          "bagging_fraction", 0.1, 1
        ),
        "min_gain_to_split": trial.suggest_float(
          "min_gain_to_split", 0, 1
        ),
        "learning_rate": trial.suggest_float(
          "learning_rate", 1e-3, 0.1, log=True
        ),
        "verbose": -1,
      }

      #nb_stdevs = trial.suggest_float(
      #        "nb_stdevs", 1.5, 10
      #    )

      stratifier = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=self.random_state,
      )

      fold_losses = []
      coverages = []
      for fn, (trn_idx, val_idx) in enumerate(stratifier.split(X, y.astype(int))):
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        #X_train = fillna(X_train, strategy="zero")

        # remove outliers
        #X_train, y_train = remove_outliers(X_train, y_train, nb_stdevs)

        dtrain = lgb.Dataset(X_train, label=y_train)

        gbm = lgb.train(param, dtrain)
        gbm.predict_proba = gbm.predict

        X_val = fillna(X_val, strategy="zero")
        preds = gbm.predict(X_val)
        predicted_classes = np.asarray([np.argmax(line) for line in preds])
        matthews_loss = accuracy_score(y_val, predicted_classes)

        fold_losses.append(matthews_loss)

        X_val, X_calibrate, y_val, y_calibrate = train_test_split(
          X_val, y_val, test_size=0.50, random_state=42)

        # Create a ConformalPredictionWrapper instance
        wrapper = ConformalPredictionWrapper(gbm)

        # Calibrate the wrapper
        wrapper.calibrate(X_calibrate, y_calibrate)

        pred_sets = wrapper.predict_sets(X_val, alpha=self.tune_with_alpha)

        coverage = prediction_set_coverage(y_val, pd.DataFrame({"prediction_set": pred_sets}))
        if coverage < 1-self.tune_with_alpha-0.05:
          print(f"Prune trial, because of missing coverage. Achieved coverage of {coverage} %. Accuracy is {matthews_loss}")
          return -1, 0
        else:
          # print(f"Achieved {coverage} % of singleton sets. Matthews is {matthews_loss}")
          coverages.append(coverage)

      score = np.mean(np.asarray(fold_losses))
      coverage = np.mean(np.asarray(coverages))
      print(f"Achieved {coverage} % coverage. Accuracy is {score}")

      return score, coverage

    # Create an Optuna study and optimize hyperparameters
    sampler = optuna.samplers.TPESampler(
      multivariate=True, seed=self.random_state
    )
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
    study.optimize(
      objective,
      n_trials=500,
      timeout=60 * 60 * 3,
      gc_after_trial=True,
      show_progress_bar=True
    )

    # Get the best hyperparameters
    #nb_stdevs = study.best_trial.params["nb_stdevs"]
    #del study.best_trial.params["nb_stdevs"]

    self.param = max(study.best_trials, key=lambda t: t.values[0]).params
    self.param["objective"] = "multiclass",
    self.param["metric"] = "multi_logloss",
    self.param["num_class"] = y.nunique(),

    print("++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++")
    print(f"Best params are: {self.param}")


    #X = fillna(X, strategy="zero")
    # remove outliers
    #X, y = remove_outliers(X, y, nb_stdevs)

    dtrain = lgb.Dataset(X, label=y)
    self.model = lgb.train(self.param, dtrain)

  def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
    predicted_probas = self.model.predict(df)
    predicted_classes = np.asarray([np.argmax(line) for line in predicted_probas])
    return predicted_probas, predicted_classes

  def predict_proba(self, df: pd.DataFrame) -> PredictedProbas:
    predicted_probas = self.model.predict(df)
    return predicted_probas

custom_model = LGBMTuner(tune_with_alpha=0.05)

automl = BlueCast(
  class_problem="multiclass",
  ml_model=custom_model,
)


```
