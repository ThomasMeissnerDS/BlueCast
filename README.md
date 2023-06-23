# BlueCast

[![codecov](https://codecov.io/gh/ThomasMeissnerDS/BlueCast/branch/main/graph/badge.svg?token=XRIS04O097)](https://codecov.io/gh/ThomasMeissnerDS/BlueCast)
[![Codecov workflow](https://github.com/ThomasMeissnerDS/BlueCast/actions/workflows/workflow.yaml/badge.svg)](https://github.com/ThomasMeissnerDS/BlueCast/actions/workflows/workflow.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![Documentation Status](https://readthedocs.org/projects/bluecast/badge/?version=latest)](https://bluecast.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/bluecast.svg)](https://pypi.python.org/pypi/bluecast/)
![Known Vulnerabilities](https://snyk.io/test/github/ThomasMeissnerDS/BlueCast/main/badge.svg)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A lightweight and fast auto-ml library. This is the successor of the
e2eml automl library. While e2eml tried to cover many model
architectures and a lot of different preprocessing options,
BlueCast focuses on a few model architectures (on default Xgboost
only) and a few preprocessing options (only what is
needed for Xgboost). This allows for a much faster development
cycle and a much more stable codebase while also having as few dependencies
as possible for the library. Despite being lightweight in its core BlueCast
offers high customization options for advanced users.

<!-- toc -->

* [Installation](#installation)
  * [Installation for end users](#installation-for-end-users)
  * [Installation for developers](#installation-for-developers)
* [General usage](#general-usage)
  * [Basic usage](#basic-usage)
  * [Advanced usage](#advanced-usage)
    * [Custom training configuration](#custom--training-configuration)
    * [Custom preprocessing](#custom-preprocessing)
    * [Custom feature selection](#custom-feature-selection)
    * [Custom ML model](#custom-ml-model)
* [Convenience features](#convenience-features)
* [Code quality](#code-quality)
* [Documentation](#documentation)
* [How to contribute](#how-to-contribute)
* [Meta](#meta)

<!-- tocstop -->

## Installation

### Installation for end users

From PyPI:

```sh
pip install bluecast
```

Using a fresh environment with Python 3.9 or higher is recommended. We consciously
do not support Python 3.8 or lower to prevent the usage of outdated Python versions
and issues connected to it.

### Installation for developers

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

#### Custom  training configuration

Despite e2eml, BlueCast allows easy customization. Users can adjust the
configuration and just pass it to the `BlueCast` class. Here is an example:

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig

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

#### Custom preprocessing

The `BlueCast` class also allows for custom preprocessing. This is done by
an abstract class that can be inherited and passed into the `BlueCast` class.
BlueCast provides two entry points to inject custom preprocessing. The
attribute `custom_preprocessor` is called right after the train_test_split.
The attribute `custom_last_mile_computation` will be called before the model
training or prediction starts (when only numerical features are present anymore)
and allows users to execute last computations (i.e. sub sampling or final calculations).

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.preprocessing.custom import CustomPreprocessing

# Create a custom tuning config and adjust hyperparameter search space
xgboost_param_config = XgboostTuneParamsConfig()
xgboost_param_config.steps_max = 100
xgboost_param_config.num_leaves_max = 16
# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hyperparameter_tuning_rounds = 10
train_config.autotune_model = False # we want to run just normal training, no hyperparameter tuning
# We could even just overwrite the final Xgboost params using the XgboostFinalParamConfig class

class MyCustomPreprocessing(CustomPreprocessing):
    def __init__(self):
        self.trained_patterns = {}

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        cat_df = df[['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ']].copy()

        zscores = Zscores()
        zscores.fit_all(df, ['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'])
        df = zscores.transform_all(df, ['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'])
        self.trained_patterns["zscores"] = zscores

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        imp_mean.fit(df.loc[:, num_columns])
        df = imp_mean.transform(df.loc[:, num_columns])
        self.trained_patterns["imputation"] = imp_mean

        df = pd.DataFrame(df, columns=num_columns).merge(cat_df, left_index=True, right_index=True, how="left")

        df = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha'], axis=1)

        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        cat_df = df[['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ']].copy()

        df = self.trained_patterns["zscores"].transform_all(df, ['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'])

        imp_mean = self.trained_patterns["imputation"]
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        df.loc[:, num_columns] = df.loc[:, num_columns].replace([np.inf, -np.inf], np.nan)
        df = imp_mean.transform(df.loc[:, num_columns])

        df = pd.DataFrame(df, columns=num_columns).merge(cat_df, left_index=True, right_index=True, how="left")

        df = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha'], axis=1)

        return df, target

# add custom last mile computation
class MyCustomLastMilePreprocessing(CustomPreprocessing):
    def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df / 2
        df["custom_col"] = 5
        return df

    # Please note: The base class enforces that the fit_transform method is implemented
    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.custom_function(df)
        df = df.head(1000)
        target = target.head(1000)
        return df, target

    # Please note: The base class enforces that the fit_transform method is implemented
    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.custom_function(df)
        if not predicton_mode and isinstance(target, pd.Series):
            df = df.head(100)
            target = target.head(100)
        return df, targe

custom_last_mile_computation = MyCustomLastMilePreprocessing()
custom_preprocessor = MyCustomPreprocessing()

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        target_column="target"
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
        custom_preprocessor=custom_preprocessor, # this takes place right after test_train_split
        custom_last_mile_computation=custom_last_mile_computation, # last step before model training/prediction
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

#### Custom feature selection

As of version 0.3 BlueCast added automated feature selection. Also this
step can be customized. An instance of `RFECV` is expected for `selection_strategy`.
Otherwise the pipeline will fail. To disable feature selection set `execute_selection`
to `False`. To surpass the `RFECV` limitation a custom feature selection algorithm
can also be passed as part of a custom last mile computation.

```sh
from bluecast.config.training_config import FeatureSelectionConfig
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

# add custom feature selection
custom_feat_sel = FeatureSelectionConfig()
# custom_feat_sel.execute_selection = False
custom_feat_sel.selection_strategy = RFECV(
    estimator=xgb.XGBClassifier(),
    step=1,
    cv=StratifiedKFold(10, random_state=0, shuffle=True),
    min_features_to_select=1,
    scoring=make_scorer(matthews_corrcoef),
    n_jobs=1,
)

# Create an instance of the BlueCast class with the custom model
bluecast = BlueCast(
    class_problem="binary",
    target_column="target",
    conf_feature_selection=custom_feat_sel,

# Create some sample data for testing
x_train = pd.DataFrame(
    {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}
)
y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
x_test = pd.DataFrame(
    {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}

x_train["target"] = y_trai
# Fit the BlueCast model using the custom model
bluecast.fit(x_train, "target"
# Predict on the test data using the custom model
predicted_probas, predicted_classes = bluecast.predict(x_test)
```

#### Custom ML model

For some users it might just be convenient to use the BlueCast class to
enjoy convenience features (details see below), but use a custom ML model.
This is possible by passing a custom model to the BlueCast class. The needed properties
are defined via the BaseClassMlModel class. Here is an example:

```sh
from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    PredictedClasses,  # just for linting checks
    PredictedProbas,  # just for linting checks
)


class CustomModel(BaseClassMlModel):
    def __init__(self):
        self.model = None

    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        predicted_probas = self.model.predict_proba(df)
        predicted_classes = self.model.predict(df)
        return predicted_probas, predicted_classes

custom_model = CustomModel()

# Create an instance of the BlueCast class with the custom model
bluecast = BlueCast(
    class_problem="binary",
    target_column="target",
    ml_model=custom_model,

# Create some sample data for testing
x_train = pd.DataFrame(
    {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}
)
y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
x_test = pd.DataFrame(
    {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}

x_train["target"] = y_trai
# Fit the BlueCast model using the custom model
bluecast.fit(x_train, "target"
# Predict on the test data using the custom model
predicted_probas, predicted_classes = bluecast.predict(x_test)
```

Please note that custom ML models require user defined hyperparameter tuning. Pre-defined
configurations are not available for custom models.

## Convenience features

Despite being a lightweight library, BlueCast also includes some convenience
with the following features:

* automatic feature type detection and casting
* automatic DataFrame schema detection: checks if unseen data has new or
  missing columns
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

## Documentation

Documentation is provided via [Read the Docs](https://bluecast.readthedocs.io/en/latest/)

## How to contribute

Contributions are welcome. Please follow the following steps:

* Create a new branch from develop branch
* Add your feature or fix
* Add unit tests for new features
* Run pre-commit checks and unit tests (using Pytest)
* Adjust the `docs/source/index.md` file
* Copy paste the content of the `docs/source/index.md` file into the
  `README.md` file
* Push your changes and create a pull request

If library or dev dependencies have to be changed, adjust the pyproject.toml.
For readthedocs it is also requited to update the
`docs/srtd_requirements.txt` file. Simply run:

```sh
poetry export --with dev -f requirements.txt --output docs/rtd_requirements.txt
```

If readthedocs will be able to create the documentation can be tested via:

```sh
poetry run sphinx-autobuild docs/source docs/build/html
```

This will show a localhost link containing the documentation.

## Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)
