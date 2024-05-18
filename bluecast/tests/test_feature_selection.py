import pandas as pd

from bluecast.preprocessing.feature_selection import BoostaRootaWrapper


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
