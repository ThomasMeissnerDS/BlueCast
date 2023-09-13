import pandas as pd

from bluecast.preprocessing.target_encoding import MultiClassTargetEncoder


def test_multi_class_target_encoder():
    # Create a sample dataframe and series
    X = pd.DataFrame({"cat1": ["A", "B", "A", "B"], "cat2": ["X", "Y", "X", "Y"]})
    y = pd.Series(["C", "A", "B", "A"])

    # Create an instance of MultiClassTargetEncoder
    encoder = MultiClassTargetEncoder(cat_columns=["cat1", "cat2"], target_col="target")

    # Test fit_target_encode_multiclass
    encoded_x = encoder.fit_target_encode_multiclass(X, y)
    assert isinstance(encoded_x, pd.DataFrame)

    # Test transform_target_encode_multiclass
    transformed_x = encoder.transform_target_encode_multiclass(X)
    assert isinstance(transformed_x, pd.DataFrame)
