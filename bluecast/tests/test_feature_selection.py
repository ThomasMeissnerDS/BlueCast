import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from bluecast.preprocessing.feature_selection import BoostARoota, BoostaRootaWrapper


def test_boostaroota_wrapper_binary():
    df = pd.DataFrame(
        {
            "feature1": [i for i in range(100)],
            "feature2": [i for i in range(100, 200)],
        }
    )
    targets = pd.Series([0 for _i in range(50)] + [1 for _i in range(50)])
    # Initialize the wrapper
    boosta = BoostaRootaWrapper(class_problem="binary", random_state=45)
    # Fit the model
    trans_df, trans_targets = boosta.fit_transform(df, targets)

    assert trans_targets.equals(targets)


def test_boostaroota_wrapper_multiclass():
    df = pd.DataFrame(
        {
            "feature1": [i for i in range(300)],
            "feature2": [i for i in range(100, 400)],
        }
    )
    targets = pd.Series(
        [0 for _i in range(100)] + [1 for _i in range(100)] + [2 for _i in range(100)]
    )
    # Initialize the wrapper
    boosta = BoostaRootaWrapper(class_problem="multiclass", random_state=45)
    # Fit the model
    trans_df, trans_targets = boosta.fit_transform(df, targets)

    assert trans_targets.equals(targets)


def test_boostaroota_wrapper_regression():
    df = pd.DataFrame(
        {
            "feature1": [i for i in range(200)],
            "feature2": [i for i in range(100, 300)],
        }
    )
    targets = pd.Series([0 for _i in range(100)] + [1 for _i in range(100)])
    # Initialize the wrapper
    boosta = BoostaRootaWrapper(class_problem="regression", random_state=45)
    # Fit the model
    trans_df, trans_targets = boosta.fit_transform(df, targets)

    assert trans_targets.equals(targets)


@pytest.fixture
def synthetic_data():
    np.random.seed(0)
    df = pd.DataFrame(
        np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)]
    )
    targets_binary = pd.Series(np.random.randint(0, 2, size=100))
    targets_multiclass = pd.Series(np.random.randint(0, 3, size=100))
    targets_regression = pd.Series(np.random.randn(100))
    return df, targets_binary, targets_multiclass, targets_regression


def test_boostaroota_wrapper_initialization():
    with pytest.raises(ValueError):
        _ = BoostaRootaWrapper(class_problem="unknown", random_state=42)


def test_boostaroota_wrapper_fit_transform_binary(synthetic_data):
    df, targets_binary, _, _ = synthetic_data
    wrapper = BoostaRootaWrapper(class_problem="binary", random_state=42)
    df_transformed, targets_transformed = wrapper.fit_transform(df, targets_binary)
    assert not df_transformed.empty


def test_boostaroota_wrapper_fit_transform_multiclass(synthetic_data):
    df, _, targets_multiclass, _ = synthetic_data
    wrapper = BoostaRootaWrapper(class_problem="multiclass", random_state=42)
    df_transformed, targets_transformed = wrapper.fit_transform(df, targets_multiclass)
    assert not df_transformed.empty


def test_boostaroota_wrapper_fit_transform_regression(synthetic_data):
    df, _, _, targets_regression = synthetic_data
    wrapper = BoostaRootaWrapper(class_problem="regression", random_state=42)
    df_transformed, targets_transformed = wrapper.fit_transform(df, targets_regression)
    assert not df_transformed.empty


def test_boostaroota_wrapper_transform(synthetic_data):
    df, targets_binary, _, _ = synthetic_data
    wrapper = BoostaRootaWrapper(class_problem="binary", random_state=42)
    wrapper.fit_transform(df, targets_binary)
    df_transformed, _ = wrapper.transform(df)
    assert not df_transformed.empty


def test_boostaroota_initialization_errors():
    with pytest.raises(ValueError):
        _ = BoostARoota(metric=None, clf=None)

    with pytest.raises(ValueError):
        _ = BoostARoota(metric="mlogloss", clf=None, cutoff=-1)

    with pytest.raises(ValueError):
        _ = BoostARoota(metric="mlogloss", clf=None, iters=0)

    with pytest.raises(ValueError):
        _ = BoostARoota(metric="mlogloss", clf=None, delta=1.5)


def test_boostaroota_warnings():
    with pytest.warns(UserWarning):
        _ = BoostARoota(metric="mlogloss", clf=XGBClassifier())

    with pytest.warns(UserWarning):
        _ = BoostARoota(metric="mlogloss", clf=None, delta=0.01)

    with pytest.warns(UserWarning):
        _ = BoostARoota(metric="mlogloss", clf=None, max_rounds=0)


def test_boostaroota_fit_transform(synthetic_data):
    df, targets_binary, _, _ = synthetic_data
    clf = XGBClassifier()
    br = BoostARoota(clf=clf)
    br.fit(df, targets_binary)
    transformed_df = br.transform(df)
    assert not transformed_df.empty

    transformed_df_fit_transform = br.fit_transform(df, targets_binary)
    assert not transformed_df_fit_transform.empty


def test_boostaroota_transform_without_fit(synthetic_data):
    df, _, _, _ = synthetic_data
    br = BoostARoota(clf=XGBClassifier())
    with pytest.raises(ValueError):
        _ = br.transform(df)
