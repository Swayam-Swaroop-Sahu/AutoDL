"""
tabular_preprocess.py
----------------------
Handles preprocessing for structured tabular datasets.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from explainDL.preprocessing.common_utils import handle_missing_values

def preprocess_tabular_data(df: pd.DataFrame, target_col: str = None):
    """
    Cleans and preprocesses tabular data automatically.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    target_col : str, optional
        Target column for supervised learning (if known)

    Returns
    -------
    tuple (X_train, X_test, y_train, y_test)
    """
    df = handle_missing_values(df, strategy="auto")

    # Identify target column (if not specified)
    if target_col is None:
        target_col = df.columns[-1]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Normalize numeric features
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
