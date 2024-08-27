# Advanced usage

<!-- toc -->

* [Advanced usage](#advanced-usage)
  * [Enable cross-validation](#enable-cross-validation)
  * [Enable even more overfitting-robust cross-validation](#enable-even-more-overfitting-robust-cross-validation)
  * [Gaining extra performance](#gaining-extra-performance)
  * [Use multi-model blended pipeline](#use-multi-model-blended-pipeline)
  * [Categorical encoding](#categorical-encoding)
  * [Training speed and performance](#training-speed-and-performance)

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

The same can also be achieved without the need of a custom training config.
In any way BlueCast will initialize a config during class instantiation.
These settings can be adjusted afterwards as well.

```sh
from bluecast.blueprints.cast import BlueCast
# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
    )
automl.train_config.hypertuning_cv_folds = 5

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

This will use Xgboost's inbuilt cross validation routine which allows BlueCast
to execute early pruning on not promising hyperparameter sets. This way BlueCast
can test many more hyperparameters than usual cross validation.
For regression problems Xgboost's inbuilt cross validation routine is also used,
however BlueCast uses stratification to ensure that the folds are balanced.

## Repeated cross-validation

BlueCast supports repeated cross-validation as well. This can be enabled by
setting `hypertuning_cv_repeats` to a value greater than 1. This will repeat
the cross-validation routine n times and will return the average performance
of all runs.

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
executes the step on each fold.

This is more robust, but does not offer
early pruning and is much (!) slower. BlueCastCV supports this as well.

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

## Optimizing the random seed

As Optuna uses sampling to find the best hyperparameters, the random seed
might have a significant impact on the results. BlueCast offers a way to
optimize the random seed by running the tuning process multiple times
with different seeds and then selecting the best hyperparameters.

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig


# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.autotune_n_random_seeds = 5 # default is 1 (train_config.autotune_model must be True)

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

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

By default `BlueCastCV` trains five models. This can be adjusted in the
training config via `bluecast_cv_train_n_model`, which expects a tuple like (5, 3)
to train 5 models via 3 repeats (so 15 models in total).

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
y_hat = automl.predict(df_val)
```

Be aware that `fit_eval` is meant to be used for prototyping and
not for final model training as the model will have less training data
available. For final model training use `fit` instead.

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

## Passing custom eval metrics

BlueCast allows to pass eval metrics for single fold training and for the final
model training. The eval metrics can be passed to the `BlueCast` class during
instantiation or afterwards.
Classification and regression classes have different requirements for the eval
metrics.

### Passing eval metrics for classification

This feature is available since version 1.4.0. For classification tasks
the eval metrics must be passed as a `ClassificationEvalWrapper` object.
This is required as classification prediction outputs can arrive in different shapes.

```sh
from sklearn.metrics import accuracy_score, log_loss, matthews_corrcoef
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig

wrapper = ClassificationEvalWrapper(
        eval_against="probas_target_class", metric_func=log_loss, higher_is_better=False
    )

# Pass the custom configs to the BlueCast class
automl = BlueCastCV(
        class_problem="binary",
        single_fold_eval_metric_func=wrapper.
    )

# this class has a train method:
# automl.fit(df_train, target_col="target")

automl.fit_eval(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

### Passing eval metrics for regression

This feature is available since version 1.4.0. For regression a similar
API is available.

```sh
from sklearn.metrics import mean_absolute_percentage_error
from bluecast.evaluation.eval_metrics import RegressionEvalWrapper
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig

wrapper = RegressionEvalWrapper(
                higher_is_better=False,
                metric_func=mean_squared_error,
                metric_name="Mean squared error",
                **{"squared": False}, # here args to the eval metric can be passed
            )

# Pass the custom configs to the BlueCast class
automl = BlueCastCVRegression(
        class_problem="regression",
        single_fold_eval_metric_func=wrapper.
    )

# this class has a train method:
# automl.fit(df_train, target_col="target")

automl.fit_eval(df_train, target_col="target")
y_hat = automl.predict(df_val)
```

## Training speed and performance

BlueCast offers various options to train models. The combinations of settings will
have a significant impact on the training speed and performance. The following
overview shall help making the right decisions:

### Hardware

BlueCast will automatically detect if a GPU is available and will use it for
Xgboost training. For large datasets this can speed up training significantly.
Since version 1.4.0 the training config has a parameter `autotune_on_device`,
which allows to specify the hardware to use for the hyperparameter tuning
manually. Available options are `cpu`, `gpu` and `auto` (default).

### Number of models to train

Use `BlueCastCV` instead of `BlueCast` for more robust hyperparameter tuning ->
`BlueCast` trains a single model while `BlueCastCV` trains five models. All
the training config settings will apply to all models (more info see above).

### Hyperparameter tuning

BlueCast offers various settings to adjust training speed and performance:

* default: simple train-test split -> fast training, but might overfit
* disable `autotune_model`: This will disable the hyperparameter tuning and
  will use the default hyperparameters as defined in `XgboostFinalParamConfig`
  and `XgboostRegressionFinalParamConfig`. This will speed up the training
  significantly, but will decrease the performance. Default parameters
  can be adjusted and passed to the `BlueCast` class during instantiation or
  afterwards.
* increase `hypertuning_cv_folds`: more folds -> slower training, but less
  overfitting (5 or 10 are good values usually)
* decrease `hyperparameter_tuning_rounds`: fewer rounds -> faster training, but
  less optimal hyperparameters
* enable `enable_feature_selection`: For datasets with a big number of features
  this can speed up training significantly, but will decrease performance.
  Custom feature selection methods can be passed to optimize speed and performance.
* enable `sample_data_during_tuning`: This will sample the data during the tuning
  process. This can speed up the tuning process significantly, but might decrease
  the performance. This is especially useful for large datasets where the tuning
  process takes too long for each trial and thus the tuning process is not able
  to test many hyperparameters (at least 15 rounds).
* enable `enable_grid_search_fine_tuning`: This will enable another fine-tuning step
  after the Bayesian optimization. This will require two or three times of the
  original tuning time additionally. Marginal performance increase possible.
* enable `precise_cv_tuning`: This will enable a more robust cross-validation routine
  that will apply random noise to the eval dataset to find even more robust
  hyperparameters. This is an experimental feature and will slow down the training
  massively as it runs without parallelism and without trial pruning.
* Adjust `early_stopping_rounds` (if None, no early stopping happens). By default,
  early stopping is enabled and will stop the final training if the eval metric does
  not improve for the given number of rounds. Early stopping returns the best
  model, not the one after stopping. If `early_stopping_rounds`
  is set, `use_full_data_for_final_model` must be set to `False` to prevent
  overfitting.
* Adjust `autotune_n_random_seeds` to optimize the random seed. This will run the
  tuning process multiple times with different seeds (for Optuna only) and then
  select the best hyperparameters. This can be useful if the random seed has a
  significant impact on the results.

### Recommended settings

A summary of recommended settings for different scenarios:

* Prototype training or debugging:
  * use `BlueCast`
  * `autotune_model = False`

* Fast training, but might overfit:
  * use `BlueCast`
  * `hypertuning_cv_folds = 1` (default)
  * `hyperparameter_tuning_rounds = 20`
  * `early_stopping_rounds = 20` (default)

* Balanced training:
  * use `BlueCast`
  * `hypertuning_cv_folds = 5`
  * `hyperparameter_tuning_rounds = 20`
  * `enable_feature_selection = True`
  * `early_stopping_rounds = 20` (default)

* Robust training:
  * use `BlueCastCV`
  * `hypertuning_cv_folds = 10`
  * `hyperparameter_tuning_rounds = 100`
  * `early_stopping_rounds = 20` (default)

* Robust training with fine-tuning:
  * use `BlueCastCV`
  * `hypertuning_cv_folds = 10`
  * `hyperparameter_tuning_rounds = 200` (default)
  * `enable_grid_search_fine_tuning = True`
  * `early_stopping_rounds = 20` (default)
