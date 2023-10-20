from typing import Union

import pandas as pd


def detect_leakage_via_correlation(
    data: pd.DataFrame, target_column: Union[str, float, int], threshold: float = 0.9
) -> bool:
    """
    Detect data leakage by checking for high correlations between the target column
    and other columns in the DataFrame.

    :param data: The DataFrame containing the data (numerical columns only for features)
    :param target_column: The name of the target column to check for correlations.
    :param threshold: The correlation threshold. If the absolute correlation value is greater than
      or equal to this threshold, it will be considered as a potential data leakage.
    :returns: True if data leakage is detected, False if not.
    """
    if target_column not in data.columns:
        raise ValueError(
            f"The target column '{target_column}' is not found in the DataFrame."
        )

    correlations = data.corr()[target_column].abs()
    potential_leakage = correlations[correlations >= threshold].index.tolist()

    # Exclude the target column itself from potential leakage
    potential_leakage.remove(target_column)

    if len(potential_leakage) > 0:
        print("Potential data leakage detected. High correlation with columns:")
        print(potential_leakage)
        return True
    else:
        print("No data leakage detected.")
        return False
