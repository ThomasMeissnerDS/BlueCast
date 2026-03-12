# BlueCast

<!-- markdownlint-disable MD013 MD004 MD022 MD032 MD051 -->
[![codecov](https://codecov.io/gh/ThomasMeissnerDS/BlueCast/branch/main/graph/badge.svg?token=XRIS04O097)](https://codecov.io/gh/ThomasMeissnerDS/BlueCast)
[![Codecov workflow](https://github.com/ThomasMeissnerDS/BlueCast/actions/workflows/workflow.yaml/badge.svg)](https://github.com/ThomasMeissnerDS/BlueCast/actions/workflows/workflow.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![Documentation Status](https://readthedocs.org/projects/bluecast/badge/?version=latest)](https://bluecast.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/bluecast.svg)](https://pypi.python.org/pypi/bluecast/)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

![BlueCast](docs/source/bluecast_dragon_logo_5.jpeg)

A lightweight and fast auto-ml library, that helps data scientists
tackling real world problems from EDA to model explainability
and even uncertainty quantification.
BlueCast focuses on a few model architectures (CatBoost by default)
and minimal preprocessing. This allows for a much faster development
cycle and a much more stable codebase while also having as few
dependencies as possible for the library. Despite being lightweight
in its core BlueCast offers high customization options for advanced
users. Find the full documentation
[on Read the Docs](https://bluecast.readthedocs.io/en/latest/).

Here you can see our test coverage in more detail:

[![Codecov sunburst](https://codecov.io/gh/ThomasMeissnerDS/BlueCast/graphs/sunburst.svg?token=XRIS04O097)](https://codecov.io/gh/ThomasMeissnerDS/BlueCast/graphs/sunburst.svg?token=XRIS04O097)

<!-- toc -->

* [Philosophy](#philosophy)
* [What BlueCast has to offer](#what-bluecast-has-to-offer)
  * [Basic usage](#basic-usage)
  * [Recent Major Improvements](#recent-major-improvements)
    * [v3.0 Breaking Changes](#v30-breaking-changes)
    * [v2.0 Improvements](#v20-improvements)
  * [Convenience features](#convenience-features)
  * [Example scripts](#example-scripts)
  * [Kaggle competition results](#kaggle-competition-results)
* [About the code](#about-the-code)
  * [Code quality](#code-quality)
  * [Documentation](#documentation)
  * [How to contribute](#how-to-contribute)
  * [Supports us](#supports-us)
  * [Meta](#meta)

<!-- tocstop -->

## Philosophy

There are plenty of excellent automl solutions available.
With BlueCast we don't follow the usual path ("Give me your data, we return the
best model ensemble out of X algorithms"), but have the real world data
scientist in mind. Our philosophy can be summarized as such:

* automl should not be a black box
* automl shall be a help rather than a replacement
* automl shall not be a closed system
* automl should be easy to install
* explainability over another after comma digit in precision
* real world value over pure performance

We support our users with an end-to-end toolkit, allowing fast and rich EDA,
modelling at highest convenience, explainability, evaluation and even
uncertainty quantification.

## What BlueCast has to offer

### Basic usage

```sh
from bluecast.blueprints.unified import BlueCastAuto
from bluecast.ensemble.ensemble_config import EnsembleConfig

# Binary classification - single model
automl = BlueCastAuto(class_problem="binary")
automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)

# Regression with cross-validation and stacking ensemble
automl = BlueCastAuto(
    class_problem="regression",
    use_cross_validation=True,
    ensemble_config=EnsembleConfig(ensemble_strategy="stacking"),
)
automl.fit(df_train, target_col="target")
y_preds = automl.predict(df_val)
```

The original per-class imports still work for users who prefer them:

```sh
from bluecast.blueprints.cast import BlueCast

automl = BlueCast(class_problem="binary")
automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

### Recent Major Improvements

#### v3.0 Breaking Changes

BlueCast v3.0 introduces several API improvements that
require code changes when upgrading from v2.x:

- **Renamed parameters**: `predicton_mode` is now correctly
  spelled `prediction_mode` across the entire API
- **Renamed blueprint attributes**: `conf_xgboost` is now
  `conf_tuning`, and `conf_params_xgboost` is now
  `conf_params` in all blueprint classes
- **CatBoost as default**: CatBoost is the default model
  backend (since v2.x, now reflected in naming)
- **Fixed `classification_report` key**: The misspelled
  `classfication_report` dict key has been removed from
  `eval_classifier()` return values
- **Proper logging**: Library no longer overrides application
  logging configuration via `logging.basicConfig`
- **Instance-isolated configs**: `*FinalParamConfig` classes
  now use instance-level `params` dicts instead of
  shared class-level dicts

#### v2.0 Improvements

- **DuckDB-Powered Experiment Tracking**: Persistent storage
  with structured data management and SQL-based analytics
- **Enhanced Error Analysis**: Plotly-powered interactive
  visualizations with DuckDB backend for both classification
  and regression
- **Advanced Statistics**: Automatic R-squared, correlation,
  heteroscedasticity detection, and comprehensive metrics

### Convenience features

Despite being a lightweight library, BlueCast also includes some convenience
with the following features:

* **Enhanced Experiment Tracking**: DuckDB-powered experiment tracking with persistent storage, separate tables for hyperparameter tuning and model evaluation results
* **Advanced Error Analysis**: Comprehensive error analysis with DuckDB backend and interactive Plotly visualizations for both classification and regression problems
* **Rich EDA Library**: Comprehensive library of EDA functions to visualize and understand the data
* **Uncertainty Quantification**: Inbuilt uncertainty quantification framework using conformal prediction
* **Intelligent Hyperparameter Tuning**: Advanced hyperparameter optimization with extensive customization options
* **Automatic Feature Engineering**:
  - Automatic feature type detection and casting
  - Categorical feature encoding (target encoding or
    natively in CatBoost/XGBoost)
  - Datetime feature encoding
  - Automatic DataFrame schema detection for production
    consistency
* **Production-Ready Features**:
  - Automated GPU availability check and usage
  - fit_eval method to mimic production environment reality
  - Functions to save and load trained pipelines
  - Comprehensive model evaluation and monitoring capabilities
* **Explainability & Insights**:
  - SHAP values for feature importance
  - ROC AUC curves & lift charts
  - Enhanced statistical insights with error distributions and residual analysis
  - Interactive visualizations for better model understanding
* **Quality Assurance**: Built-in warnings for potential misconfigurations

The fit_eval method can be used like this:

```sh
from bluecast.blueprints.cast import BlueCast

automl = BlueCast(
        class_problem="binary",
    )

automl.fit_eval(df_train, df_eval, y_eval, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

It is important to note that df_train contains the target column while
df_eval does not. The target column is passed separately as y_eval.

### Example scripts

The [examples/](examples/) directory contains self-contained scripts
using synthetic data that demonstrate BlueCast's full feature set:

| Script | Topics |
| ------ | ------ |
| [00_full_showcase.py](examples/00_full_showcase.py) | **End-to-end walkthrough of all features** |
| [01_quick_start.py](examples/01_quick_start.py) | Binary, multiclass, regression, `fit_eval` |
| [02_cross_validation_and_ensembles.py](examples/02_cross_validation_and_ensembles.py) | `BlueCastCV`, mean blending, stacking, hill climbing |
| [03_conformal_prediction.py](examples/03_conformal_prediction.py) | Uncertainty quantification, group-conditional intervals |
| [04_linear_models.py](examples/04_linear_models.py) | Logistic/Ridge/Lasso regression, configurable preprocessing |
| [05_unified_interface.py](examples/05_unified_interface.py) | `BlueCastAuto` single entry point for all problem types |
| [06_advanced_customization.py](examples/06_advanced_customization.py) | Custom preprocessing, XGBoost, drift monitoring, experiment tracking, save/load |
| [07_fairness.py](examples/07_fairness.py) | Fairness auditing, demographic parity, equalized odds, conformal fairness |
| [08_eda.py](examples/08_eda.py) | Univariate/bivariate plots, PCA, t-SNE, correlations, data quality, leakage detection |
| [09_bluecast_ai.py](examples/09_bluecast_ai.py) | Multi-agent LLM-powered AutoML (requires API key) |
| [10_serving.py](examples/10_serving.py) | Deploy models as REST APIs, export Dockerfile |

### Kaggle competition results

Even though BlueCast has been designed to be a lightweight
automl framework, it still offers the possibilities to
reach very good performance. We tested BlueCast in Kaggle
competitions to showcase the libraries capabilities
feature- and performance-wise.

* ICR top 20% finish with over 6000 participants ([notebook](https://www.kaggle.com/code/thomasmeiner/icr-bluecast-automl-almost-bronze-ranks))
* An advanced example covering lots of functionalities ([notebook](https://www.kaggle.com/code/thomasmeiner/ps3e23-automl-eda-outlier-detection/notebook))
* PS3E23: Predict software defects top 12% finish ([notebook](https://www.kaggle.com/code/thomasmeiner/ps3e23-automl-eda-outlier-detection?scriptVersionId=145650820))
* PS3E25: Predict hardness of steel via regression ([notebook](https://www.kaggle.com/code/thomasmeiner/ps3e25-bluecast-automl?scriptVersionId=153347618))
* PS4E1: Bank churn top 13% finish ([notebook](https://www.kaggle.com/code/thomasmeiner/ps4e1-eda-feature-engineering-modelling?scriptVersionId=158121062))
* A comprehensive guide about BlueCast showing many capabilities ([notebook](https://www.kaggle.com/code/thomasmeiner/ps4e3-bluecast-a-comprehensive-overview))
* BlueCast using a custom Catboost model for quantile regression
and adding conformal prediction ([notebook](https://www.kaggle.com/code/thomasmeiner/bluecast-has-conformal-prediction))
* 26th place in the Kaggle 24h "AutoMl" GrandPrix July 2024 blitz competition ([notebook](https://www.kaggle.com/code/thomasmeiner/automl-grand-prix-bluecast-26th-place-solution))

Please note that some Kaggle notebooks ran older versions of BlueCast and
might not be compatible with the most recent version anymore.

## About the code

### Code quality

To ensure code quality, we use the following tools:

* various pre-commit libraries
* strong type hinting in the code base
* unit tests using Pytest

For contributors, it is expected that all pre-commit and unit tests pass.
For new features it is expected that unit tests are added.

### Documentation

Documentation is provided via [Read the Docs](https://bluecast.readthedocs.io/en/latest/)
On GitHub we offer multiple ReadMes to cover all aspects of working
with BlueCast, covering:

* [Installation](docs/source/Installation.md)
* [EDA](docs/source/EDA.md)
* [Basic usage](docs/source/Basic%20usage.md)
* [Customize training settings](docs/source/Customize%20training%20settings.md)
* [Feature engineering](docs/source/Feature%20engineering.md)
* [Customizing configurations and objects](docs/source/Customizing%20configurations%20and%20objects.md)
* [Model evaluation](docs/source/Model%20evaluation.md)
* [Error analysis](docs/source/Error%20analysis.md)
* [Model explainability (XAI)](docs/source/Model%20explainability%20(XAI).md)
* [Uncertainty quantification](docs/source/Uncertainty%20quantification.md)
* [Monitoring](docs/source/Monitoring.md)

### How to contribute

Contributions are welcome. Please follow the following steps:

* Get in touch with me (i.e. via LinkedIn) if longer contribution is of interest
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

### Supports us

Being a small open source project we rely on the community. Please
consider giving us a GitHb star and spread the word. Also your feedback
will help the project evolving.

### Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)
