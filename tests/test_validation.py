"""Tests for input validation layer (F1)."""
import os
import tempfile

import pandas as pd
import numpy as np
import pytest

from src.core.exceptions import AutoDLInputError
from src.core.validation import (
    validate_file_exists,
    validate_non_empty,
    validate_min_rows,
    validate_target,
    validate_no_all_nan,
    validate_prediction_columns,
)


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------
def test_validate_file_exists_ok():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a,b\n1,2\n")
        path = f.name
    try:
        validate_file_exists(path)  # should not raise
    finally:
        os.unlink(path)


def test_validate_file_exists_missing():
    with pytest.raises(AutoDLInputError, match="not found"):
        validate_file_exists("/tmp/__nope__nope__nope__42.xyz")


def test_validate_file_exists_not_a_file():
    with pytest.raises(AutoDLInputError, match="not a regular file"):
        validate_file_exists(os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Non-empty
# ---------------------------------------------------------------------------
def test_non_empty_raises_on_empty():
    with pytest.raises(AutoDLInputError, match="empty"):
        validate_non_empty(pd.DataFrame())


def test_non_empty_ok():
    validate_non_empty(pd.DataFrame({"x": [1]}))


# ---------------------------------------------------------------------------
# Min rows
# ---------------------------------------------------------------------------
def test_min_rows_raises():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(AutoDLInputError, match="only 3 row"):
        validate_min_rows(df, n=10)


def test_min_rows_ok():
    df = pd.DataFrame({"x": range(20), "y": range(20)})
    validate_min_rows(df, n=10)


# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
def test_target_not_found():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(AutoDLInputError, match="not found"):
        validate_target(df, "missing")


def test_target_single_class():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})
    with pytest.raises(AutoDLInputError, match="fewer than 2|at least 2|only 1 distinct"):
        validate_target(df, "a")


def test_target_ok():
    df = pd.DataFrame({"target": ["a", "b", "a", "b", "a", "b"]})
    n = validate_target(df, "target")
    assert n == 2


def test_target_too_many_classes():
    df = pd.DataFrame({"target": list(range(60))})
    with pytest.raises(AutoDLInputError, match="50"):
        validate_target(df, "target", max_classes=50)


# ---------------------------------------------------------------------------
# All-NaN columns
# ---------------------------------------------------------------------------
def test_all_nan_dropped_with_warning():
    df = pd.DataFrame({"good": [1, 2, 3], "all_nan": [np.nan, np.nan, np.nan]})
    result = validate_no_all_nan(df)
    assert "good" in result.columns
    assert "all_nan" not in result.columns


def test_all_nan_results_in_empty_raises():
    df = pd.DataFrame({"all_nan": [np.nan, np.nan]})
    with pytest.raises(AutoDLInputError, match="no columns left"):
        validate_no_all_nan(df)


# ---------------------------------------------------------------------------
# Prediction columns
# ---------------------------------------------------------------------------
def test_predict_missing_columns():
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    with pytest.raises(AutoDLInputError, match="missing|required"):
        validate_prediction_columns(df, ["col_a", "col_b", "col_c"])


def test_predict_columns_ok():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validate_prediction_columns(df, ["a", "b"])  # no raise