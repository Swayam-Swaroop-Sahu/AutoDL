"""
common_utils.py
----------------
Utility functions for preprocessing modules.
"""

import numpy as np
import pandas as pd

def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Handles missing values automatically based on data type.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    strategy : str, optional
        Strategy for missing value handling ('auto', 'mean', 'median', 'mode', 'drop')

    Returns
    -------
    pandas.DataFrame
    """
    df_copy = df.copy()

    if strategy == "drop":
        df_copy.dropna(inplace=True)
    elif strategy in ["mean", "median", "mode"]:
        for col in df_copy.columns:
            if df_copy[col].dtype in [np.float64, np.int64]:
                if strategy == "mean":
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                elif strategy == "median":
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                else:
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            else:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    elif strategy == "auto":
        for col in df_copy.columns:
            if df_copy[col].dtype in [np.float64, np.int64]:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            else:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)

    return df_copy
