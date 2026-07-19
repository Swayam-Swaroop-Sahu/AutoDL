# src/preprocessing/common_utils.py

import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Handles missing values using:
    - mean / median for numeric
    - mode for categorical
    - auto mode picks mean for numbers & mode for object types
    """

    df = df.copy()

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            if strategy in ["auto", "mean"]:
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Automatically detects a target column.
    Prioritizes: ["target", "label", "class", "y"]
    Else returns the last column.
    """

    preferred = ["target", "label", "class", "y"]
    for col in preferred:
        if col in df.columns:
            return col

    return df.columns[-1]
