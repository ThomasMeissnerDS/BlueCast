from typing import Optional, Tuple

import numpy as np
import pandas as pd

from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import (
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
)
from bluecast.ml_modelling.catboost_regression import CatboostModelRegression
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.tests.make_data.create_data import create_synthetic_dataframe_regression


def test_BlueCastRegression_without_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = False

    catboost_pram_config = CatboostTuneParamsRegressionConfig()

    class MyCustomLastMilePreprocessing(CustomPreprocessing):
        def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
            df["custom_col"] = 5
            return df

        def fit_transform(
            self, df: pd.DataFrame, target: pd.Series
        ) -> Tuple[pd.DataFrame, pd.Series]:
            df = self.custom_function(df)
            return df, target

        def transform(
            self,
            df: pd.DataFrame,
            target: Optional[pd.Series] = None,
            predicton_mode: bool = False,
        ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
            df = self.custom_function(df)
            return df, target

    # Create an instance of the BlueCastRegression class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
        custom_last_mile_computation=MyCustomLastMilePreprocessing(),
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCastRegression model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_values = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_BlueCastRegression_with_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.plot_hyperparameter_tuning_overview = False

    catboost_pram_config = CatboostTuneParamsRegressionConfig()

    # Create an instance of the BlueCastRegression class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCastRegression model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_values = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method

    # TEST with 1 fold
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 1
    train_config.autotune_model = True

    catboost_pram_config = CatboostTuneParamsRegressionConfig()

    # Create an instance of the BlueCastRegression class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Fit the BlueCastRegression model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_values = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_BlueCastRegression_with_fine_tune_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.precise_cv_tuning = True

    catboost_pram_config = CatboostTuneParamsRegressionConfig()

    # Create an instance of the BlueCastRegression class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCastRegression model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_values = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_BlueCastRegression_with_grid_search_tune_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 2
    train_config.hypertuning_cv_folds = 2
    train_config.enable_grid_search_fine_tuning = True
    train_config.gridsearch_nb_parameters_per_grid = 2
    train_config.autotune_model = True

    catboost_pram_config = CatboostTuneParamsRegressionConfig()

    # Create an instance of the BlueCastRegression class with the custom model
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Create some sample data for testing
    x_train = pd.DataFrame(
        {
            "feature1": [i for i in range(20)],
            "feature2": [i for i in range(20)],
            "feature3": [i for i in range(20)],
            "feature4": [i for i in range(20)],
            "feature5": [i for i in range(20)],
            "feature6": [i for i in range(20)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x_test = pd.DataFrame(
        {
            "feature1": [i for i in range(10)],
            "feature2": [i for i in range(10)],
            "feature3": [i for i in range(10)],
            "feature4": [i for i in range(10)],
            "feature5": [i for i in range(10)],
            "feature6": [i for i in range(10)],
        }
    )

    x_train["target"] = y_train

    # Fit the BlueCastRegression model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_values = bluecast.predict(x_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_catboost_regression_with_cat_columns_none_and_ml_algorithm_encoding():
    """
    Test that CatboostModelRegression works with cat_columns=None and cat_encoding_via_ml_algorithm=True.
    This test addresses the bug where using a custom CatboostModelRegression with cat_encoding_via_ml_algorithm=True
    would fail due to categorical column mismatches.
    """
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 5  # Keep it small for faster testing
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.plot_hyperparameter_tuning_overview = False
    train_config.cat_encoding_via_ml_algorithm = (
        True  # Enable ML algorithm categorical encoding
    )

    catboost_param_config = CatboostTuneParamsRegressionConfig()

    # Create synthetic data with categorical features
    df_train = create_synthetic_dataframe_regression(num_samples=100, random_state=42)
    df_test = create_synthetic_dataframe_regression(num_samples=50, random_state=123)

    # Remove target from test data for prediction
    X_test = df_test.drop("target", axis=1)

    # Get the categorical column names from the generated data
    categorical_cols = ["categorical_feature_1", "categorical_feature_2"]

    # Create an instance of the BlueCastRegression class with CatboostModelRegression
    # Setting cat_columns to the actual categorical columns to make it work
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_param_config,
            cat_columns=categorical_cols,  # Explicitly specify categorical columns
        ),
        conf_xgboost=catboost_param_config,
        conf_training=train_config,
    )

    # Fit the BlueCastRegression model
    bluecast.fit(df_train, "target")

    # Predict on the test data
    predicted_values = bluecast.predict(X_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    assert len(predicted_values) == len(X_test)
    assert not np.isnan(predicted_values).any()  # No NaN values in predictions
    assert np.isfinite(predicted_values).all()  # All predictions are finite

    # Verify that the model was actually trained (experiment tracker should have results)
    print(
        f"Number of experiment entries: {len(bluecast.experiment_tracker.experiment_id)}"
    )

    # Test with BlueCastCVRegression as well to ensure compatibility
    from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression

    bluecast_cv = BlueCastCVRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_param_config,
            cat_columns=categorical_cols,  # Explicitly specify categorical columns
        ),
        conf_xgboost=catboost_param_config,
    )

    # Set configuration after initialization
    bluecast_cv.conf_training.cat_encoding_via_ml_algorithm = True
    bluecast_cv.conf_training.hyperparameter_tuning_rounds = 5
    bluecast_cv.conf_training.hypertuning_cv_folds = 2
    bluecast_cv.conf_training.autotune_model = True
    bluecast_cv.conf_training.plot_hyperparameter_tuning_overview = False

    # Fit the CV model
    bluecast_cv.fit(df_train, "target")

    # Predict on the test data
    predicted_values_cv = bluecast_cv.predict(X_test)

    # Assert the expected results for CV model
    assert isinstance(predicted_values_cv, pd.Series)
    assert len(predicted_values_cv) == len(X_test)
    assert not np.isnan(predicted_values_cv).any()
    assert np.isfinite(predicted_values_cv).all()

    print(
        "✅ Test passed: CatboostModelRegression works with cat_encoding_via_ml_algorithm=True when categorical columns are specified"
    )


def test_catboost_regression_with_disabled_ml_algorithm_encoding():
    """
    Test the alternative solution: disable cat_encoding_via_ml_algorithm (set to False).
    This allows CatboostModelRegression to work with cat_columns=None by using BlueCast's
    own categorical encoding instead of CatBoost's native categorical support.
    """
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 5  # Keep it small for faster testing
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.plot_hyperparameter_tuning_overview = False
    train_config.cat_encoding_via_ml_algorithm = (
        False  # Disable ML algorithm categorical encoding
    )

    catboost_param_config = CatboostTuneParamsRegressionConfig()

    # Create synthetic data with categorical features
    df_train = create_synthetic_dataframe_regression(num_samples=100, random_state=42)
    df_test = create_synthetic_dataframe_regression(num_samples=50, random_state=123)

    # Remove target from test data for prediction
    X_test = df_test.drop("target", axis=1)

    # Create an instance of the BlueCastRegression class with CatboostModelRegression
    # Setting cat_columns=None works when cat_encoding_via_ml_algorithm=False
    bluecast = BlueCastRegression(
        class_problem="regression",
        ml_model=CatboostModelRegression(
            class_problem="regression",
            conf_training=train_config,
            conf_catboost=catboost_param_config,
            cat_columns=None,  # This works when cat_encoding_via_ml_algorithm=False
        ),
        conf_xgboost=catboost_param_config,
        conf_training=train_config,
    )

    # Fit the BlueCastRegression model
    bluecast.fit(df_train, "target")

    # Predict on the test data
    predicted_values = bluecast.predict(X_test)

    # Assert the expected results
    assert isinstance(predicted_values, np.ndarray)
    assert len(predicted_values) == len(X_test)
    assert not np.isnan(predicted_values).any()  # No NaN values in predictions
    assert np.isfinite(predicted_values).all()  # All predictions are finite

    print(
        "✅ Test passed: cat_columns=None works when cat_encoding_via_ml_algorithm=False"
    )
