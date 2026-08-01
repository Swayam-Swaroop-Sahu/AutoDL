"""Input validation functions for AutoDL (F1 trust layer).

Every function raises AutoDLInputError on failure. Error messages follow:
  What went wrong -> Why -> What to do

Coverage:
  - File-level: exists, non-empty, is a regular file
  - DataFrame-level: non-empty, min rows, target column (2–50 classes),
    all-null columns, prediction column mismatch
"""

from __future__ import annotations

import os
from typing import List

import pandas as pd

from src.core.exceptions import AutoDLInputError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# File-level validation
# ---------------------------------------------------------------------------
def validate_file_exists(path: str) -> None:
    """Check the path points to a regular, non-empty file."""
    if not os.path.exists(path):
        raise AutoDLInputError(
            f"Uploaded file not found at '{path}'. "
            "Why: the path may have been deleted or moved between upload and processing. "
            "What to do: re-upload the file and try again."
        )
    if not os.path.isfile(path):
        raise AutoDLInputError(
            f"'{path}' is not a regular file. "
            "Why: the path is a folder or a special device. "
            "What to do: upload a file (.csv, .xlsx, .txt, or .zip)."
        )
    if os.path.getsize(path) == 0:
        raise AutoDLInputError(
            f"The uploaded file '{os.path.basename(path)}' is empty (0 bytes). "
            "Why: the file may have been saved with no content or the upload was interrupted. "
            "What to do: check your file on disk and re-upload a non-empty file."
        )


# ---------------------------------------------------------------------------
# DataFrame-level checks
# ---------------------------------------------------------------------------
def validate_non_empty(df: pd.DataFrame, name: str = "dataset") -> None:
    """Raise if the DataFrame is None or has zero rows."""
    if df is None:
        raise AutoDLInputError(
            f"The {name} could not be loaded (returned None). "
            "Why: the file may be corrupted or in an unsupported format. "
            "What to do: verify your file opens correctly in another program and re-upload."
        )
    if df.empty:
        raise AutoDLInputError(
            f"The {name} is empty (0 rows). "
            "Why: the uploaded file contains a header but no data rows. "
            "What to do: add at least 10 rows of labelled data and re-upload."
        )


def validate_min_rows(df: pd.DataFrame, n: int = 10, name: str = "dataset") -> None:
    """Raise if the DataFrame has fewer than `n` rows."""
    if df.shape[0] < n:
        raise AutoDLInputError(
            f"The {name} has only {df.shape[0]} row(s). Minimum is {n}. "
            f"Why: AutoDL needs enough examples to create a reliable validation split. "
            f"What to do: add at least {n} rows of labelled data and re-upload."
        )


MIN_TARGET_CLASSES = 2
MAX_TARGET_CLASSES = 50


def validate_target(
    df: pd.DataFrame, target_col: str, max_classes: int = MAX_TARGET_CLASSES,
) -> int:
    """Validate the target column and return n_classes.

    Raises if <2 classes or >max_classes unique values (default 50).
    """
    if target_col not in df.columns:
        raise AutoDLInputError(
            f"Target column '{target_col}' not found in the uploaded data. "
            f"Why: the column may have been renamed or dropped before upload. "
            f"What to do: choose from the available columns: {list(df.columns)}"
        )

    y = df[target_col].dropna()
    if y.empty:
        raise AutoDLInputError(
            f"Target column '{target_col}' has no non-null values. "
            "Why: every value in this column is missing. "
            "What to do: label your data with at least 2 distinct classes."
        )
    if y.nunique() < MIN_TARGET_CLASSES:
        raise AutoDLInputError(
            f"Target column '{target_col}' has only {y.nunique()} distinct value(s). "
            "Why: classification requires at least 2 distinct classes to learn from. "
            "What to do: add at least 2 distinct class labels to your data."
        )

    n_classes = int(y.nunique())
    if n_classes > max_classes:
        raise AutoDLInputError(
            f"Target column '{target_col}' has {n_classes} distinct classes, "
            f"exceeding the AutoDL maximum of {max_classes}. "
            f"Why: beyond {max_classes} classes, model quality degrades and training slows significantly. "
            f"What to do: merge rare classes into an 'Other' category, or "
            f"choose a different target column with fewer classes."
        )
    return n_classes


def validate_no_all_nan(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """Drop all-NaN columns with a warning. Raise if result is empty."""
    nan_cols = [c for c in df.columns if df[c].isna().all()]
    if nan_cols:
        logger.warning(
            "Dropping %d all-null column(s): %s. "
            "These columns contain no data.",
            len(nan_cols), nan_cols,
        )
        df = df.drop(columns=nan_cols)

    if df.empty or df.shape[1] == 0:
        raise AutoDLInputError(
            f"After removing all-null columns, the {name} has no columns left. "
            "Why: all columns in the uploaded file were entirely empty. "
            "What to do: add feature columns and re-upload."
        )
    return df


# ---------------------------------------------------------------------------
# Prediction-mode validation
# ---------------------------------------------------------------------------
def validate_prediction_columns(
    df: pd.DataFrame, feature_cols: List[str],
) -> None:
    """Check prediction dataset has the same columns as training data.

    Raises AutoDLInputError for missing columns AND for extra columns.
    """
    missing = sorted(set(feature_cols) - set(df.columns))
    extra = sorted(set(df.columns) - set(feature_cols))

    messages: List[str] = []
    if missing:
        messages.append(
            f"Missing {len(missing)} required column(s): {missing}. "
            "Why: training and prediction data must have identical feature columns. "
            "What to do: add these columns (even if empty) and re-upload."
        )
    if extra:
        messages.append(
            f"Found {len(extra)} unexpected column(s): {extra}. "
            "Why: the prediction dataset must only contain the same columns used in training. "
            "What to do: remove or rename these extra columns to match the training schema."
        )
    if messages:
        raise AutoDLInputError(" ".join(messages))