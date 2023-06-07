import pandas as pd

from bluecast.preprocessing.train_test_split import (
    train_test_split_cross,
    train_test_split_time,
)


# Test train_test_split_cross function
def test_train_test_split_cross():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    target_col = "B"
    train_size = 0.8
    random_state = 100

    x_train, x_test, y_train, y_test = train_test_split_cross(
        df, target_col, train_size, random_state
    )

    # Assertions
    assert len(x_train) == 4
    assert len(x_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1


# Test train_test_split_time function
def test_train_test_split_time():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    target_col = "B"
    split_by_col = "A"
    train_size = 0.8

    x_train, x_test, y_train, y_test = train_test_split_time(
        df, target_col, split_by_col, train_size
    )

    # Assertions
    assert len(x_train) == 4
    assert len(x_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1
