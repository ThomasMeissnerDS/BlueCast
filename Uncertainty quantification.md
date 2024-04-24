# Uncertainty quantification

Over the past years conformal prediction gained increasing attention. It allows to
add uncertainty quantification around every model at the cost of just a bit of
additional computation.

<!-- toc -->

* [Uncertainty quantification](#uncertainty-quantification)
  * [Conformal prediction for classification](#conformal-prediction-for-classification)
  * [Conformal prediction for regression](#conformal-prediction-for-regression)

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

All of these functions are also available via the ensemble classes
`BlueCastCV` and `BlueCastCVRegression`.
