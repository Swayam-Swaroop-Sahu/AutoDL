"""Tests for data quality gates (F1)."""
import pandas as pd
import numpy as np

from src.quality.gates import (
    detect_leakage,
    detect_id_columns,
    detect_constant_columns,
    detect_imbalance,
)
from src.quality.summarize import summarize_quality


def test_leakage_detects_perfect_corr_numeric():
    """A feature perfectly correlated with target is flagged."""
    X = pd.DataFrame({
        "good": np.random.RandomState(0).randn(30),
        "leaky": np.random.RandomState(0).randn(30),
    })
    df = X.copy()
    df["target"] = X["leaky"]
    warnings = detect_leakage(df, "target")
    assert any(w["column"] == "leaky" for w in warnings)


def test_leakage_does_not_flag_random():
    """An unrelated feature is not flagged."""
    X = pd.DataFrame({"feat": np.random.RandomState(0).randn(100)})
    df = X.copy()
    df["target"] = np.random.RandomState(1).randn(100)
    warnings = detect_leakage(df, "target")
    assert len(warnings) == 0


def test_id_columns_flagged():
    """A column where n_unique == n_rows is flagged."""
    df = pd.DataFrame({
        "id": range(50),
        "val": np.random.RandomState(0).randn(50),
    })
    warnings = detect_id_columns(df)
    assert any(w["column"] == "id" for w in warnings)


def test_regular_feature_not_flagged_as_id():
    """A column with repeats is not flagged."""
    df = pd.DataFrame({"x": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]})
    warnings = detect_id_columns(df)
    assert len(warnings) == 0


def test_constant_column_flagged():
    """A column with one unique value is flagged."""
    df = pd.DataFrame({"const": [7, 7, 7, 7, 7], "var": [1, 2, 3, 4, 5]})
    warnings = detect_constant_columns(df)
    assert any(w["column"] == "const" for w in warnings)


def test_imbalanced_class_flagged():
    """A class below 5% is flagged."""
    df = pd.DataFrame({"target": ["a"] * 98 + ["b"] * 2})
    warnings = detect_imbalance(df, "target")
    assert any(w["class"] == "b" for w in warnings)


def test_balanced_class_not_flagged():
    """Equal classes are not flagged."""
    df = pd.DataFrame({"target": ["a"] * 50 + ["b"] * 50})
    warnings = detect_imbalance(df, "target")
    assert len(warnings) == 0


def test_summarize_quality_empty_on_clean_data():
    """summarize_quality returns passed=True on truly clean data."""
    df = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,  # 10 values, repeated
        "y": np.random.RandomState(0).randn(20),
        "target": ["a"] * 10 + ["b"] * 10,
    })
    report = summarize_quality(df, "target")
    assert report["passed"] is True, f"Expected passed=True, got {report}"
    assert report["warnings"] == []


def test_summarize_quality_finds_all_issues():
    """summarize_quality finds leakage + ID + constant + imbalance."""
    df = pd.DataFrame({
        "id_col": range(50),
        "const_col": [3] * 50,
        "leaky": [1, 0] * 25,
        "target": [1, 0] * 25,
    })
    report = summarize_quality(df, "target")
    assert report["passed"] is False
    assert len(report["warnings"]) >= 3  # ID, constant, leakage