import logging

import pandas as pd


def remove_correlated_columns(df: pd.DataFrame, threshold: float = 0.9):
    """
    Remove collinear columns from a given DataFrame.

    :param df: Pandas DataFrame holding all columns.
    :param threshold: Float indicating the correlation threshold. If the correlation is above
        or equal this value, one of the columns will be dropped.
    :return: DataFrame with reduced number of columns.
    """
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (
                (corr_matrix.iloc[i, j] >= threshold)
                and (corr_matrix.columns[j] not in col_corr)
                and (corr_matrix.columns[i] not in col_corr)
            ):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in df.columns:
                    del df[colname]  # deleting the column from the df
    logging.info(f"Removed the following collinear columns: {col_corr}")
    return df
