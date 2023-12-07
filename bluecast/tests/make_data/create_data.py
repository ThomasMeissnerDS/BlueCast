import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def create_synthetic_dataframe(
    num_samples=1000, random_state: int = 20
) -> pd.DataFrame:
    # Generate synthetic data using make_classification
    x, y = make_classification(
        n_samples=num_samples,
        n_features=20,
        n_informative=20,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    # Create a datetime feature
    start_date = pd.to_datetime("2022-01-01")
    end_date = pd.to_datetime("2022-12-31")
    datetime_feature = pd.date_range(
        start=start_date, end=end_date, periods=num_samples
    )

    # Create categorical features
    categorical_feature_1 = np.random.choice(["A", "B", "C"], size=num_samples)
    categorical_feature_2 = np.random.choice(["X", "Y", "Z"], size=num_samples)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "categorical_feature_1": categorical_feature_1,
            "categorical_feature_2": categorical_feature_2,
            "numerical_feature_1": x[:, 0],
            "numerical_feature_2": x[:, 1],
            "numerical_feature_3": x[:, 2],
            "datetime_feature": datetime_feature,
            "target": y,
        }
    )

    return df


def create_synthetic_multiclass_dataframe(
    num_samples=1000, random_state: int = 20
) -> pd.DataFrame:
    # Generate synthetic data using make_classification
    x, y = make_classification(
        n_samples=num_samples,
        n_features=20,
        n_informative=20,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state,
        n_classes=5,
    )

    # Create a datetime feature
    start_date = pd.to_datetime("2022-01-01")
    end_date = pd.to_datetime("2022-12-31")
    datetime_feature = pd.date_range(
        start=start_date, end=end_date, periods=num_samples
    )

    # Create categorical features
    categorical_feature_1 = np.random.choice(["A", "B", "C"], size=num_samples)
    categorical_feature_2 = np.random.choice(["X", "Y", "Z"], size=num_samples)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "categorical_feature_1": categorical_feature_1,
            "categorical_feature_2": categorical_feature_2,
            "numerical_feature_1": x[:, 0],
            "numerical_feature_2": x[:, 1],
            "numerical_feature_3": x[:, 2],
            "datetime_feature": datetime_feature,
            "target": y,
        }
    )

    return df


def create_synthetic_dataframe_regression(
    num_samples=1000, random_state: int = 20
) -> pd.DataFrame:
    # Generate synthetic data using make_classification
    x, y = make_regression(
        n_samples=num_samples,
        n_features=20,
        n_informative=20,
        random_state=random_state,
    )

    # Create a datetime feature
    start_date = pd.to_datetime("2022-01-01")
    end_date = pd.to_datetime("2022-12-31")
    datetime_feature = pd.date_range(
        start=start_date, end=end_date, periods=num_samples
    )

    # Create categorical features
    categorical_feature_1 = np.random.choice(["A", "B", "C"], size=num_samples)
    categorical_feature_2 = np.random.choice(["X", "Y", "Z"], size=num_samples)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "categorical_feature_1": categorical_feature_1,
            "categorical_feature_2": categorical_feature_2,
            "numerical_feature_1": x[:, 0],
            "numerical_feature_2": x[:, 1],
            "numerical_feature_3": x[:, 2],
            "datetime_feature": datetime_feature,
            "target": y,
        }
    )

    return df
