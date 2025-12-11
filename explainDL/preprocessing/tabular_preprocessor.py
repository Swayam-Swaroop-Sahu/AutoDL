# explainDL/preprocessing/tabular_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class TabularPreprocessor:
    """
    Handles:
    - Feature column detection
    - Encoding categorical features
    - Scaling numeric data
    - Encoding target labels
    - Safe transform of new data (unseen categories handled)
    """

    def __init__(self):
        self.scaler = None
        self.label_encoders = {}          # For categorical columns
        self.target_encoder = None        # For output labels
        self.feature_columns = None       # Columns used for training

    # ----------------------------------------------------------------------
    # TRAINING MODE
    # ----------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, target_col=None):
        df = df.copy()

        # Auto-detect target column
        if target_col is None:
            target_col = df.columns[-1]

        y_raw = df[target_col].astype(str)
        X = df.drop(columns=[target_col]).copy()

        self.feature_columns = list(X.columns)

        # Encode categorical columns
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        # Encode target labels
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(y_raw)

        # Scale numeric features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_columns
        )

        return X_scaled, y

    # ----------------------------------------------------------------------
    # PREDICTION MODE
    # ----------------------------------------------------------------------
    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # Keep only training columns
        df = df[self.feature_columns]

        # Encode categorical with unseen category handling
        for col, le in self.label_encoders.items():
            if col in df:
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_
                    else -1  # Unseen category
                )

        # Scale numeric cols
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_columns
        )

        return df_scaled
