import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import CatboostTuneParamsConfig, TrainingConfig, CatboostFinalParamConfig
from bluecast.ml_modelling.catboost import CatboostModel
from bluecast.preprocessing.custom import CustomPreprocessing


def test_catboost_predict_proba_exceptions():
    """Test that predict_proba raises proper exceptions for various error conditions."""
    
    # Test with no conf_catboost
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.conf_catboost = None
    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})
    
    with pytest.raises(ValueError, match="conf_catboost or conf_training is None"):
        catboost_model.predict_proba(dummy_df)
    
    # Test with no conf_training
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.conf_training = None
    
    with pytest.raises(ValueError, match="conf_catboost or conf_training is None"):
        catboost_model.predict_proba(dummy_df)
    
    # Test with no model
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.model = None
    
    with pytest.raises(Exception, match="No trained CatBoost model found."):
        catboost_model.predict_proba(dummy_df)
    
    # Test with no conf_params_catboost (but with a model)
    catboost_model = CatboostModel(class_problem="binary")
    # Create a mock model to avoid the "No trained CatBoost model found" exception
    from catboost import CatBoostClassifier
    catboost_model.model = CatBoostClassifier()
    catboost_model.conf_params_catboost = None
    
    with pytest.raises(Exception, match="No CatBoost model configuration found."):
        catboost_model.predict_proba(dummy_df)


def test_catboost_predict_exceptions():
    """Test that predict raises proper exceptions for various error conditions."""
    
    # Test with no conf_catboost
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.conf_catboost = None
    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})
    
    with pytest.raises(ValueError, match="conf_catboost or conf_training is None"):
        catboost_model.predict(dummy_df)
    
    # Test with no model
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.model = None
    
    with pytest.raises(Exception, match="No trained CatBoost model found."):
        catboost_model.predict(dummy_df)
    
    # Test with no conf_params_catboost (but with a model)
    catboost_model = CatboostModel(class_problem="binary")
    # Create a mock model to avoid the "No trained CatBoost model found" exception
    from catboost import CatBoostClassifier
    catboost_model.model = CatBoostClassifier()
    catboost_model.conf_params_catboost = None
    
    with pytest.raises(Exception, match="No CatBoost model configuration found."):
        catboost_model.predict(dummy_df)


def test_catboost_predict_proba_binary_with_trained_model():
    """Test predict_proba method for binary classification with a properly trained model."""
    
    # Create simple training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Test predict_proba
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })
    
    probs = catboost_model.predict_proba(test_df)
    
    # Assertions
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 3
    assert all(0 <= prob <= 1 for prob in probs)  # Probabilities should be between 0 and 1


def test_catboost_predict_proba_multiclass_with_trained_model():
    """Test predict_proba method for multiclass classification with a properly trained model."""
    
    # Create simple training data for multiclass
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    })
    y_train = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    X_test = pd.DataFrame({
        'feature1': [13, 14, 15],
        'feature2': [1.3, 1.4, 1.5]
    })
    y_test = pd.Series([0, 1, 2])
    
    # Create and configure CatboostModel for multiclass
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="multiclass",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Test predict_proba
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })
    
    probs = catboost_model.predict_proba(test_df)
    
    # Assertions for multiclass
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (3, 3)  # 3 samples, 3 classes
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)  # Each row should sum to 1


def test_catboost_predict_with_trained_model_binary():
    """Test predict method for binary classification with threshold logic."""
    
    # Create simple training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Test predict
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })
    
    probs, classes = catboost_model.predict(test_df)
    
    # Assertions
    assert isinstance(probs, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert len(probs) == 3
    assert len(classes) == 3
    assert all(cls in [0, 1] for cls in classes)  # Binary classes
    assert all(0 <= prob <= 1 for prob in probs)


def test_catboost_predict_with_trained_model_multiclass():
    """Test predict method for multiclass classification with argmax logic."""
    
    # Create simple training data for multiclass
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    })
    y_train = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    X_test = pd.DataFrame({
        'feature1': [13, 14, 15],
        'feature2': [1.3, 1.4, 1.5]
    })
    y_test = pd.Series([0, 1, 2])
    
    # Create and configure CatboostModel for multiclass
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="multiclass",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Test predict
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })
    
    probs, classes = catboost_model.predict(test_df)
    
    # Assertions for multiclass
    assert isinstance(probs, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert probs.shape == (3, 3)  # 3 samples, 3 classes
    assert len(classes) == 3
    assert all(cls in [0, 1, 2] for cls in classes)  # Multiclass classes


def test_catboost_predict_proba_with_custom_preprocessor():
    """Test predict_proba method with custom in-fold preprocessor."""
    
    class TestCustomPreprocessor(CustomPreprocessing):
        def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["custom_feature"] = df["feature1"] * 2
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
    
    # Create simple training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel with custom preprocessor
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config,
        custom_in_fold_preprocessor=TestCustomPreprocessor()
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Test predict_proba with custom preprocessor
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })
    
    probs = catboost_model.predict_proba(test_df)
    
    # Assertions
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 3
    assert all(0 <= prob <= 1 for prob in probs)


def test_catboost_predict_with_custom_preprocessor():
    """Test predict method with custom in-fold preprocessor."""
    
    class TestCustomPreprocessor(CustomPreprocessing):
        def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["custom_feature"] = df["feature1"] * 2
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
    
    # Create simple training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel with custom preprocessor
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config,
        custom_in_fold_preprocessor=TestCustomPreprocessor()
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Test predict with custom preprocessor
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })
    
    probs, classes = catboost_model.predict(test_df)
    
    # Assertions
    assert isinstance(probs, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert len(probs) == 3
    assert len(classes) == 3


def test_catboost_predict_with_categorical_columns():
    """Test predict and predict_proba methods with categorical columns."""
    
    # Create training data with categorical features
    X_train = pd.DataFrame({
        'numeric_feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'cat_feature': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'numeric_feature': [11, 12, 13],
        'cat_feature': ['A', 'B', 'A']
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel with categorical columns
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config,
        cat_columns=['cat_feature']
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Test data with categorical features
    test_df = pd.DataFrame({
        'numeric_feature': [1, 2, 3],
        'cat_feature': ['A', 'B', 'A']
    })
    
    # Test predict_proba
    probs = catboost_model.predict_proba(test_df)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 3
    
    # Test predict
    probs, classes = catboost_model.predict(test_df)
    assert isinstance(probs, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert len(probs) == 3
    assert len(classes) == 3


def test_catboost_train_single_fold_model():
    """Test the train_single_fold_model method."""
    from catboost import Pool
    from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
    from bluecast.experimentation.tracking import ExperimentTracker
    
    # Create training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config,
        experiment_tracker=ExperimentTracker(),
        single_fold_eval_metric_func=ClassificationEvalWrapper()
    )
    
    # Create pools
    train_pool = Pool(X_train, label=y_train)
    test_pool = Pool(X_test, label=y_test)
    
    # Test parameters
    params = {
        'iterations': 10,
        'depth': 3,
        'learning_rate': 0.1,
        'random_seed': 42,
        'logging_level': 'Silent'
    }
    
    # Test train_single_fold_model
    score = catboost_model.train_single_fold_model(train_pool, test_pool, y_test, params)
    
    # Assertions
    assert isinstance(score, (float, int))
    assert score >= 0  # Score should be non-negative
    assert len(catboost_model.experiment_tracker.experiment_id) > 0  # Should have logged experiment


def test_catboost_fit_with_use_full_data_for_final_model():
    """Test fit method with use_full_data_for_final_model enabled."""
    
    # Create training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel with use_full_data_for_final_model
    train_config = TrainingConfig()
    train_config.autotune_model = False
    train_config.use_full_data_for_final_model = True
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    model = catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Assertions
    assert model is not None
    assert catboost_model.model is not None
    assert hasattr(catboost_model.model, 'predict')
    assert hasattr(catboost_model.model, 'predict_proba')


def test_catboost_fit_with_empty_test_data():
    """Test fit method with empty test data."""
    
    # Create training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame()  # Empty test data
    y_test = pd.Series(dtype=y_train.dtype)  # Empty test targets
    
    # Create and configure CatboostModel
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    model = catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Assertions
    assert model is not None
    assert catboost_model.model is not None


def test_catboost_fit_with_early_stopping():
    """Test fit method with early stopping enabled."""
    
    # Create training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel with early stopping
    train_config = TrainingConfig()
    train_config.autotune_model = False
    train_config.enable_early_stopping = True
    train_config.early_stopping_rounds = 5
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    model = catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Assertions
    assert model is not None
    assert catboost_model.model is not None


def test_catboost_predict_with_classification_threshold():
    """Test predict method with custom classification threshold."""
    
    # Create training data
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3]
    })
    y_test = pd.Series([0, 1, 0])
    
    # Create and configure CatboostModel
    train_config = TrainingConfig()
    train_config.autotune_model = False
    catboost_config = CatboostTuneParamsConfig()
    
    catboost_model = CatboostModel(
        class_problem="binary",
        conf_training=train_config,
        conf_catboost=catboost_config
    )
    
    # Fit the model
    catboost_model.fit(X_train, X_test, y_train, y_test)
    
    # Set custom classification threshold
    catboost_model.conf_params_catboost.classification_threshold = 0.3
    
    # Test predict with custom threshold
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })
    
    probs, classes = catboost_model.predict(test_df)
    
    # Assertions
    assert isinstance(probs, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert len(probs) == 3
    assert len(classes) == 3
    assert all(cls in [0, 1] for cls in classes)


# Fixed tests for proper exception handling (not warnings)
def test_catboost_no_catboost_config_fixed():
    """Test that predict_proba raises ValueError when conf_catboost is None."""
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.conf_catboost = None
    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.raises(ValueError, match="conf_catboost or conf_training is None"):
        catboost_model.predict_proba(dummy_df)


def test_catboost_no_conf_training_fixed():
    """Test that predict_proba raises ValueError when conf_training is None."""
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.conf_training = None
    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.raises(ValueError, match="conf_catboost or conf_training is None"):
        catboost_model.predict_proba(dummy_df)


def test_catboost_no_model_fixed():
    """Test that predict_proba raises Exception when model is None."""
    catboost_model = CatboostModel(class_problem="binary")
    catboost_model.model = None
    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.raises(Exception, match="No trained CatBoost model found."):
        catboost_model.predict_proba(dummy_df)


def test_catboost_no_conf_params_catboost_fixed():
    """Test that predict_proba raises Exception when conf_params_catboost is None."""
    catboost_model = CatboostModel(class_problem="binary")
    # Create a mock model to avoid the "No trained CatBoost model found" exception
    from catboost import CatBoostClassifier
    catboost_model.model = CatBoostClassifier()
    catboost_model.conf_params_catboost = None
    dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9]})

    with pytest.raises(Exception, match="No CatBoost model configuration found."):
        catboost_model.predict_proba(dummy_df)


def test_bluecast_without_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = False

    catboost_pram_config = CatboostTuneParamsConfig()

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

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
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

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_bluecast_with_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.plot_hyperparameter_tuning_overview = False

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
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

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method

    # TEST with 1 fold
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 1
    train_config.autotune_model = True

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
            conf_training=train_config,
            conf_catboost=catboost_pram_config,
        ),
        conf_xgboost=catboost_pram_config,
        conf_training=train_config,
    )

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_bluecast_with_fine_tune_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.precise_cv_tuning = True

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
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

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method


def test_bluecast_with_grid_search_tune_hyperparam_tuning():
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 2
    train_config.hypertuning_cv_folds = 2
    train_config.autotune_model = True
    train_config.enable_grid_search_fine_tuning = True
    train_config.gridsearch_nb_parameters_per_grid = 2

    catboost_pram_config = CatboostTuneParamsConfig()

    # Create an instance of the BlueCast class with the custom model
    bluecast = BlueCast(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
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

    # Fit the BlueCast model using the custom model
    bluecast.fit(x_train, "target")

    # Predict on the test data using the custom model
    predicted_probas, predicted_classes = bluecast.predict(x_test)
    _ = bluecast.predict_proba(x_test)

    # Assert the expected results
    assert isinstance(predicted_probas, np.ndarray)
    assert isinstance(predicted_classes, np.ndarray)
    print(bluecast.experiment_tracker.experiment_id)
    assert (
        len(bluecast.experiment_tracker.experiment_id) == 0
    )  # due to custom model and fit method
