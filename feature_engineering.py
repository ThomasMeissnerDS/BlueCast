import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame, x5_median: float = 9.8) -> pd.DataFrame:
    """
    Deterministic feature engineering for The Perfect Fit competition.
    """
    df = df.copy()

    # Sentinel handling for x5
    df['x5_is_missing'] = (df['x5'] == 999.0).astype(int)
    df['x5_imputed'] = df['x5'].replace(999.0, x5_median)
    df['x5_log'] = np.log1p(df['x5_imputed'])

    # City binary encoding
    if 'City' in df.columns:
        df['City_encoded'] = (df['City'] == 'Albacete').astype(int)
        df.drop(columns=['City'], inplace=True)
    elif 'City_encoded' not in df.columns:
        # Fallback if already dropped or missing
        df['City_encoded'] = 0
        
    if 'Country' in df.columns:
        df.drop(columns=['Country'], inplace=True)

    # Interaction features
    df['x4_x_x8'] = df['x4'] * df['x8']
    df['x4_x_City'] = df['x4'] * df['City_encoded']
    df['x5_imputed_x_City'] = df['x5_imputed'] * df['City_encoded']
    df['x6_x_x7'] = df['x6'] * df['x7']
    x7_safe = np.where(df['x7'] >= 0, np.maximum(df['x7'], 0.01), np.minimum(df['x7'], -0.01))
    df['x6_div_x7_safe'] = df['x6'] / x7_safe

    # Polynomial features
    df['x4_sq'] = df['x4'] ** 2
    df['x8_sq'] = df['x8'] ** 2
    df['x5_imputed_sq'] = df['x5_imputed'] ** 2

    # Aggregation-derived features
    df['x9_minus_4x4_5'] = df['x9'] - (4 * df['x4'] + 5)
    df['sum_positive'] = df['x9'] + df['x10'] + df['x11']
    df['x6_plus_x7'] = df['x6'] + df['x7']
    df['x6_minus_x7'] = df['x6'] - df['x7']

    # Safe ratio features
    df['x10_div_x11_safe'] = df['x10'] / df['x11'].clip(lower=0.01)
    df['x4_div_x5_imputed'] = df['x4'] / df['x5_imputed'].clip(lower=0.01)

    return df
