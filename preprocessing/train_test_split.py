import pandas as pd
from sklearn import model_selection


def train_test_split_cross(
        df: pd.DataFrame, target_col: str, train_size=0.80, random_state: int = 100, stratify_col: str = None
):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        df,
        df[target_col],
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_col
    )
    return x_train, x_test, y_train, y_test


def train_test_split_time(df: pd.DataFrame, target_col: str, split_by_col: str, train_size: float = 0.80):
    length = len(df.index)
    train_length = int(length * train_size)
    test_length = length - train_length
    if not split_by_col:
        df = df.sort_index()
    elif split_by_col:
        df = df.sort_values(by=[split_by_col])
    else:
        pass
    x_train = df.head(train_length)
    x_test = df.tail(test_length)
    y_train = x_train[target_col]
    y_test = x_test[target_col]
    return x_train, x_test, y_train, y_test
