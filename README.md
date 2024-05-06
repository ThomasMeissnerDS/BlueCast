# BlueCast

[![codecov](https://codecov.io/gh/ThomasMeissnerDS/BlueCast/branch/main/graph/badge.svg?token=XRIS04O097)](https://codecov.io/gh/ThomasMeissnerDS/BlueCast)
[![Codecov workflow](https://github.com/ThomasMeissnerDS/BlueCast/actions/workflows/workflow.yaml/badge.svg)](https://github.com/ThomasMeissnerDS/BlueCast/actions/workflows/workflow.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![Documentation Status](https://readthedocs.org/projects/bluecast/badge/?version=latest)](https://bluecast.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/bluecast.svg)](https://pypi.python.org/pypi/bluecast/)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

![BlueCast](docs/source/bluecast_dragon_logo_5.jpeg)

A lightweight and fast auto-ml library, that helps data scientists
tackling real world problems from EDA to model explainability
and even uncertainty quantification.
BlueCast focuses on a few model architectures (on default Xgboost
only) and a few preprocessing options (only what is
needed for Xgboost). This allows for a much faster development
cycle and a much more stable codebase while also having as few dependencies
as possible for the library. Despite being lightweight in its core BlueCast
offers high customization options for advanced users. Find
the full documentation [here](https://bluecast.readthedocs.io/en/latest/).

Here you can see our test coverage in more detail:

[![Codecov sunburst](https://codecov.io/gh/ThomasMeissnerDS/BlueCast/graphs/sunburst.svg?token=XRIS04O097)](https://codecov.io/gh/ThomasMeissnerDS/BlueCast/graphs/sunburst.svg?token=XRIS04O097)

<!-- toc -->

* [Philosophy](#philosophy)
* [What BlueCast has to offer](#what-bluecast-has-to-offer)
  * [Basic usage](#basic-usage)
  * [Convenience features](#convenience-features)
  * [Kaggle competition results and example notebooks](#kaggle-competition-results-and-example-notebooks)
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
from bluecast.blueprints.cast import BlueCast

automl = BlueCast(
        class_problem="binary",
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)

# from version 0.95 also predict_proba is directly available (also for BlueCastCV)
y_probs = automl.predict_proba(df_val)
```

### Convenience features

Despite being a lightweight library, BlueCast also includes some convenience
with the following features:

* rich library of EDA functions to visualize and understand the data
* plenty of customization options via an open API
* inbuilt uncertainty quantification framework (conformal prediction)
* hyperparameter tuning (with lots of customization available)
* automatic feature type detection and casting
* automatic DataFrame schema detection: checks if unseen data has new or
  missing columns
* categorical feature encoding (target encoding or directly in Xgboost)
* datetime feature encoding
* automated GPU availability check and usage for Xgboost
  a fit_eval method to fit a model and evaluate it on a validation set
  to mimic production environment reality
* functions to save and load a trained pipeline
* shapley values
* ROC AUC curve & lift chart
* warnings for potential misconfigurations

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

### Kaggle competition results and example notebooks

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
* BlueCast using a custom Catboost model for quantile regression ([notebook](https://www.kaggle.com/code/thomasmeiner/birth-weight-with-bluecast-catboost))

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
* [Customizing configurations and objects](docs/source/Customizing%20configurations%20and%20objects.md)
* [Model evaluation](docs/source/Model%20evaluation.md)
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
