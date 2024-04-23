# Advanced usage

<!-- toc -->

* [Advanced usage](#advanced-usage)
  * [Enable cross-validation](#enable-cross-validation)
  * [Enable even more overfitting-robust cross-validation](#enable-even-more-overfitting-robust-cross-validation)
  * [Gaining extra performance](#gaining-extra-performance)
  * [Use multi-model blended pipeline](#use-multi-model-blended-pipeline)
  * [Categorical encoding](#categorical-encoding)

<!-- tocstop -->

## Enable cross-validation

While the default behaviour of BlueCast is to use a simple
train-test-split, cross-validation can be enabled easily:

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig


# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hypertuning_cv_folds = 5 # default is 1

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

This will use Xgboost's inbuilt cross validation routine which allows BlueCast
to execute early pruning on not promising hyperparameter sets. This way BlueCast
can test many more hyperparameters than usual cross validation.

## Enable even more overfitting-robust cross-validation

There might be situations where a preprocessing step has a high risk of overfitting
and  needs even more careful evaluation (i.e. oversampling techniques). For such
scenarios BlueCast offers a solution as well.

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig


# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hypertuning_cv_folds = 5 # default is 1
train_config.precise_cv_tuning = True # this enables the better routine

# this only makes sense if we have an overfitting risky step
custom_preprocessor = MyCustomPreprocessing() # see section Custom Preprocessing for details

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
        custom_in_fold_preprocessor=custom_preprocessor # this happens during each fold
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

The custom in fold preprocessing takes place within the cross validation and
executes the step on each fold. The evaluation metric is special here:
Instead of calculating matthews correlation coefficient reversed only,
it applied increasingly random noise to the eval dataset to find an even
more robust hyperparameter set.

This is much more robust, but does not offer
early pruning and is much slower. BlueCastCV supports this as well.

Please note that this is an experimental feature.

## Gaining extra performance

By default BlueCast uses Optuna's Bayesian hyperparameter optimization,
however Bayesian methods give an estimate and do not necessarly find
the ideal spot, thus BlueCast has an optional GridSearch setting
that allows BlueCast to refine some of the parameters Optuna has found.
This can be enabled by setting `enable_grid_search_fine_tuning` to True.
This fine-tuning step uses a different random seed than the autotuning
routine (seed from the settings + 1000).

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig


# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hypertuning_cv_folds = 5 # default is 1
train_config.enable_grid_search_fine_tuning = True # default is False
train_config.gridsearch_tuning_max_runtime_secs = 3600 # max runtime in secs
train_config.gridsearch_nb_parameters_per_grid = 5 # increasing this means X^3 trials atm

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

This comes with a tradeoff of longer runtime. This behaviour can be further
controlled with two parameters:

* `gridsearch_nb_parameters_per_grid`: Decides how
  many steps the grid shall have per parameter
* `gridsearch_tuning_max_runtime_secs`: Sets the maximum time in seconds
  the tuning shall run. This will finish the latest trial nd will exceed
  this limit though.

## Use multi-model blended pipeline

By default, BlueCast trains a single model. However, it is possible to
train multiple models with one call for extra robustness. `BlueCastCV`
has a `fit` and a `fit_eval` method. The `fit_eval` method trains the
models, but also provides out-of-fold validation. Also `BlueCastCV`
allows to pass custom configurations.

```sh
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig

# Pass the custom configs to the BlueCast class
automl = BlueCastCV(
        class_problem="binary",
        #conf_training=train_config,
        #conf_xgboost=xgboost_param_config,
        #custom_preprocessor=custom_preprocessor, # this takes place right after test_train_split
        #custom_last_mile_computation=custom_last_mile_computation, # last step before model training/prediction
        #custom_feature_selector=custom_feature_selector,
    )

# this class has a train method:
# automl.fit(df_train, target_col="target")

automl.fit_eval(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

Also here a variant for regression is available:

```sh
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig

# Pass the custom configs to the BlueCast class
automl = BlueCastCVRegression(
        class_problem="regression",
        #conf_training=train_config,
        #conf_xgboost=xgboost_param_config,
        #custom_preprocessor=custom_preprocessor, # this takes place right after test_train_split
        #custom_last_mile_computation=custom_last_mile_computation, # last step before model training/prediction
        #custom_feature_selector=custom_feature_selector,
    )

# this class has a train method:
# automl.fit(df_train, target_col="target")

automl.fit_eval(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

## Categorical encoding

By default, BlueCast uses onehot and target encoding. An orchestrator measures the
columns' cardinality and routes each categorical column to onehot or target encoding.
Onehot encoding is applied when the cardinality is less or equal
`cardinality_threshold_for_onehot_encoding` from the training config (5 by default).

This behaviour can be changed in the TrainingConfig by setting `cat_encoding_via_ml_algorithm`
to True. This will change the expectations of `custom_last_mile_computation` though.
If `cat_encoding_via_ml_algorithm` is set to False, `custom_last_mile_computation`
will receive numerical features only as target encoding will apply before. If `cat_encoding_via_ml_algorithm`
is True (default setting) `custom_last_mile_computation` will receive categorical
features as well, because Xgboost's or a custom model's inbuilt categorical encoding
will be used.
