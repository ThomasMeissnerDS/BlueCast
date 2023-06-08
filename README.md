# BlueCast

A lightweight and fast auto-ml library. This is the successor of the
e2eml automl library. While e2eml tried to cover many model
architectures and a lot of different preprocessing options,
BlueCast focuses on a few model architectures (on default Xgboost
only) and a few preprocessing options (only what is
needed for Xgboost). This allows for a much faster development
cycle and a much more stable codebase while also having as few dependencies
as possible for the library.

<!-- toc -->

* [Installation for end users](#installation-for-end-users)
* [Installation for developers](#installation-for-developers)
* [General usage](#general-usage)
  * [Basic usage](#basic-usage)
  * [Advanced usage](#advanced-usage)
* [Convenience features](#convenience-features)
* [Code quality](#code-quality)
* [Meta](#meta)

<!-- tocstop -->

## Installation for end users

From PyPI:

```sh
pip install bluecast
```

Using a fresh environment with Python 3.9 or higher is recommended. We consciously
do not support Python 3.8 or lower to prevent the usage of outdated Python versions
and issues connected to it.

## Installation for developers

* Clone the repository:
* Create a new conda environment with Python 3.9 or higher
* run `pip install poetry` to install poetry as dependency manager
* run `poetry install` to install all dependencies

## General usage

### Basic usage

The module blueprints contains the main functionality of the library. The main
entry point is the `Blueprint` class. It already includes needed preprocessing
(including some convenience functionality like feature type detection)
and model hyperparameter tuning.

```sh
from bluecast.blueprints.cast import BlueCast

automl = BlueCast(
        class_problem="binary",
        target_column="target"
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

### Advanced usage

Despite e2eml, BlueCast allows easy customization. Users can adjust the
configuration and just pass it to the `BlueCast` class. Here is an example:

```sh
from bluecast.blueprints.cast import BlueCast

# Create a custom tuning config and adjust hyperparameter search space
xgboost_param_config = XgboostTuneParamsConfig()
xgboost_param_config.steps_max = 100
xgboost_param_config.num_leaves_max = 16
# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hyperparameter_tuning_rounds = 10
train_config.autotune_model = False # we want to run just normal training, no hyperparameter tuning
# We could even just overwrite the final Xgboost params using the XgboostFinalParamConfig class

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        target_column="target"
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,

    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

## Convenience features

Despite being a lightweight library, BlueCast also includes some convenience
with the following features:

* automatic feature type detection and casting
* automatic DataFrame schema detection: checks if unseen data has new or missing columns
* categorical feature encoding
* datetime feature encoding
* automated GPU availability check and usage for Xgboost
a fit_eval method to fit a model and evaluate it on a validation set
to mimic production environment reality
* functions to save and load a trained pipeline
* shapley values

The fit_eval method can be used like this:

```sh
from bluecast.blueprints.cast import BlueCast

automl = BlueCast(
        class_problem="binary",
        target_column="target"
    )

automl.fit_eval(df_train, df_eval, y_eval, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

It is important to note that df_train contains the target column while
df_eval does not. The target column is passed separately as y_eval.

## Code quality

To ensure code quality, we use the following tools:

* various pre-commit libraries
* strong type hinting in the code base
* unit tests using Pytest

For contributors, it is expected that all pre-commit and unit tests pass.
For new features it is expected that unit tests are added.

## Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)
