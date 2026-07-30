# src/preprocessing/tabular_preprocessor.py

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

    BUGFIX Phase 1e item 6: categorical encoding now uses
    `LabelEncoder` + per-column fill with `UNKNOWN_TOKEN` for unseen
    categories during predict, instead of the old `-1` numeric sentinel.
    Train-side columns are deduplicated by suffix (`_1`, `_2`, ...) before
    encoding (BUGFIX Phase 1e item 8).
    """

    UNKNOWN_TOKEN = "__UNKNOWN__"
    DUPLICATE_SUFFIX_START = 1  # _1, _2, ...

    def __init__(self):
        self.scaler = None
        self.label_encoders = {}          # For categorical columns
        self.target_encoder = None        # For output labels
        self.feature_columns = None       # Columns used for training
        self.numeric_fill_values = {}

    # ----------------------------------------------------------------------
    # TRAINING MODE
    # ----------------------------------------------------------------------
    @staticmethod
    def _deduplicate_columns(df: pd.DataFrame) -> tuple:
        """BUGFIX Phase 1e item 8: rename duplicate columns by appending `_N`.

        Returns (df_deduped, duplicate_report_list). duplicate_report_list contains
        (original_name, new_name) tuples for any renamed columns.
        """
        seen = {}
        new_cols = []
        report = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_name = f"{col}_{seen[col]}"
                while new_name in seen or new_name in df.columns:
                    seen[col] += 1
                    new_name = f"{col}_{seen[col]}"
                report.append((col, new_name))
                new_cols.append(new_name)
            else:
                seen[col] = 0
                new_cols.append(col)
        if report:
            from src.utils.logger import get_logger
            get_logger(__name__).warning(
                "BUGFIX Phase 1e item 8: deduplicated %d duplicate column(s): %s",
                len(report), report,
            )
        df_out = df.copy()
        df_out.columns = new_cols
        return df_out, report

    def fit_transform(self, df: pd.DataFrame, target_col=None):
        df = df.copy()

        # BUGFIX Phase 1e item 8: deduplicate columns BEFORE preprocessing
        df, _dup_report = self._deduplicate_columns(df)

        # Validate input
        if df.empty:
            raise ValueError("Cannot preprocess empty DataFrame. Please provide data with at least one row.")

        if df.shape[0] < 2:
            raise ValueError(f"DataFrame has too few rows ({df.shape[0]}). Minimum 2 rows required.")

        if df.shape[1] < 2:
            raise ValueError(f"DataFrame has too few columns ({df.shape[1]}). Minimum 2 columns required (1 feature + 1 target).")

        # Auto-detect target column
        if target_col is None:
            target_col = df.columns[-1]

        # Validate target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}")

        if df[target_col].isna().any():
            raise ValueError(f"Target column '{target_col}' contains missing values. Please label or remove those rows.")
        y_raw = df[target_col].astype(str)

        # Check for sufficient unique labels
        unique_labels = y_raw.dropna().unique()
        if len(unique_labels) < 2:
            raise ValueError(f"Target column has only {len(unique_labels)} unique label(s). Minimum 2 different labels required for classification.")

        X = df.drop(columns=[target_col]).copy()

        # Validate we have features
        if X.empty or X.shape[1] == 0:
            raise ValueError("No feature columns available after removing target column. Please ensure the dataset has at least one feature column.")

        self.feature_columns = list(X.columns)

        # Encode categorical columns
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                # BUGFIX Phase 1e item 6: also fit on UNKNOWN_TOKEN so unseen
                # categories at predict time can be mapped deterministically.
                categories = X[col].fillna(self.UNKNOWN_TOKEN).astype(str).tolist() + [self.UNKNOWN_TOKEN]
                le = LabelEncoder()
                le.fit(categories)
                X[col] = le.transform(X[col].fillna(self.UNKNOWN_TOKEN).astype(str))
                self.label_encoders[col] = le
            else:
                fill_value = X[col].median()
                if pd.isna(fill_value):
                    raise ValueError(f"Feature column '{col}' contains only missing values.")
                self.numeric_fill_values[col] = float(fill_value)
                X[col] = X[col].fillna(fill_value)

        # Encode target labels
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(y_raw)

        # Scale numeric features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_columns
        )

        # BUGFIX Phase 1e item 14: leakage detection — flag features that are
        # essentially duplicates of the target (correlation |r|>0.95 for numeric
        # or Cramér's V > 0.95 for categorical).
        from src.preprocessing.leakage import detect_leakage
        leakage_report = detect_leakage(X, y_raw, numeric_threshold=0.95,
                                        categorical_threshold=0.95)

        return X_scaled, y

    # ----------------------------------------------------------------------
    # PREDICTION MODE
    # ----------------------------------------------------------------------
    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # BUGFIX Phase 1e item 8: deduplicate columns at predict too
        df, _ = self._deduplicate_columns(df)

        missing_columns = sorted(set(self.feature_columns) - set(df.columns))
        if missing_columns:
            raise KeyError(f"Missing required feature columns: {missing_columns}")
        df = df[self.feature_columns]

        # BUGFIX Phase 1e item 6: unseen categories map to UNKNOWN_TOKEN, not -1.
        for col, le in self.label_encoders.items():
            if col in df:
                values = df[col].fillna(self.UNKNOWN_TOKEN).astype(str)
                unknown_idx = int(np.where(le.classes_ == self.UNKNOWN_TOKEN)[0][0]) \
                    if self.UNKNOWN_TOKEN in le.classes_ else -1
                df[col] = values.apply(
                    lambda x: int(le.transform([x])[0]) if x in le.classes_ else unknown_idx
                )

        for col, fill_value in self.numeric_fill_values.items():
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)

        # Scale numeric cols
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_columns
        )

        return df_scaled
