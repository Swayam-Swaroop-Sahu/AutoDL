"""Unit tests for target detection (FINAL_PROJECT_PLAN.md §2 + §7 edge cases).

Tests both the new API (target_likelihood, resolve_target) and the legacy
API (score_targets, detect_target) for backward compatibility.
"""
import numpy as np
import pandas as pd
import pytest

from src.target_detection import (
    # New API
    target_likelihood,
    resolve_target,
    # Legacy API (backward-compatible)
    detect_target,
    score_targets,
    ESCALATION_MARGIN,
    ColumnScore,
)


def _make_df(n=120, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n).round(1),
        "income": rng.normal(50000, 15000, n).round(),
        "city": rng.choice(["NY", "LA", "SF", "CHI"], n),
        "is_member": rng.choice([0, 1], n),
    })
    return df, rng


# ---------------------------------------------------------------------------
# New API: target_likelihood
# ---------------------------------------------------------------------------
def test_target_likelihood_churn_scores_high():
    """A column named 'churn' built from features should score higher than 'customer_id'."""
    rng = np.random.RandomState(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n).round(1),
        "income": rng.normal(50000, 15000, n).round(),
        "churn": ((rng.normal(0, 1, n) > 0).astype(int)),  # noisy, but named 'churn'
        "customer_id": np.arange(n),
    })
    score_churn = target_likelihood(df, "churn")
    score_id = target_likelihood(df, "customer_id")
    assert score_churn > score_id, f"churn={score_churn:.3f} vs id={score_id:.3f}"
    assert 0.0 <= score_churn <= 1.0
    assert score_id == 0.0  # unique-per-row ID columns get 0.0


def test_target_likelihood_name_match():
    """Columns with target-like names score higher."""
    df = pd.DataFrame({
        "target": ["a"] * 50 + ["b"] * 50,
        "feature": np.random.randn(100),
    })
    assert target_likelihood(df, "target") > target_likelihood(df, "feature")


def test_target_likelihood_bad_column():
    """Non-existent column raises KeyError."""
    df, _ = _make_df()
    with pytest.raises(KeyError, match="not in DataFrame"):
        target_likelihood(df, "nonexistent")


def test_target_likelihood_constant_column():
    """All-same column returns 0.0."""
    df = pd.DataFrame({"x": [1.0] * 50, "label": ["a", "b"] * 25})
    assert target_likelihood(df, "x") == 0.0


# ---------------------------------------------------------------------------
# New API: resolve_target
# ---------------------------------------------------------------------------
def test_clear_target_named_column():
    df, rng = _make_df(120)
    df["churn"] = ((df["age"] > 50) & (df["income"] < 40000)).astype(int).astype("category")
    col, status = resolve_target(df)
    assert col == "churn"
    assert status in ("strong_auto", "weak_auto")


def test_override_short_circuits():
    df, _ = _make_df()
    df["target"] = np.random.RandomState(1).choice(["a", "b"], len(df))
    col, status = resolve_target(df, target_col="age")
    assert col == "age"
    assert status == "override"


def test_override_must_exist():
    df, _ = _make_df()
    with pytest.raises(ValueError, match="not found in data"):
        resolve_target(df, target_col="nope")


def test_two_binary_columns_human_required():
    """Two equally-plausible binary columns → human_required with both names."""
    rng = np.random.RandomState(7)
    n = 200
    base_signal = rng.normal(0, 1, n)
    is_a = (base_signal + rng.normal(0, 0.2, n) > 0).astype(int)
    is_b = (base_signal + rng.normal(0, 0.2, n) > 0).astype(int)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "binary_a": is_a,
        "binary_b": is_b,
    })
    col, status = resolve_target(df)
    assert status == "human_required"
    assert isinstance(col, list)
    assert set(col) == {"binary_a", "binary_b"}


def test_only_id_columns_not_classification():
    """Only ID cols → not_classification."""
    rng = np.random.RandomState(3)
    df = pd.DataFrame({
        "id": np.arange(50.0),
        "nulls": [np.nan] * 50,
    })
    col, status = resolve_target(df)
    assert status == "not_classification"
    assert col is None


def test_named_target_wins_over_features():
    """A column named 'churn' built predictably from features clearly wins."""
    rng = np.random.RandomState(0)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
        "x4": rng.normal(0, 1, n),
        "x5": rng.normal(0, 1, n),
        "city": rng.choice(["NY", "LA", "SF", "CHI", "BOS"], n),
        "region": rng.choice(["N", "S", "E", "W"], n),
    })
    # Churn is deterministically derived from x1 and x2 -> highly predictable
    df["churn"] = ((df["x1"] > 0.0) & (df["x2"] < 0.5)).astype(int).astype("category")
    col, status = resolve_target(df)
    assert col == "churn"
    assert status in ("strong_auto", "weak_auto")


def test_id_column_excluded():
    df, _ = _make_df(50)
    df["row_id"] = np.arange(50)
    df["flag"] = np.where(df["age"] > 40, "y", "n")
    col, _ = resolve_target(df)
    assert col == "flag"


# ---------------------------------------------------------------------------
# Legacy API tests (backward compat)
# ---------------------------------------------------------------------------
def test_legacy_detect_target_ambiguous_auto_select_off():
    """Legacy: ambiguous case with auto_select=False → escalated."""
    rng = np.random.RandomState(7)
    n = 200
    base_signal = rng.normal(0, 1, n)
    is_a = (base_signal + rng.normal(0, 0.2, n) > 0).astype(int)
    is_b = (base_signal + rng.normal(0, 0.2, n) > 0).astype(int)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "binary_a": is_a,
        "binary_b": is_b,
    })
    res = detect_target(df, auto_select=False)
    assert res.ambiguous is True
    assert res.escalated is True
    assert res.chosen_column is None
    assert set(res.top_candidates) == {"binary_a", "binary_b"}
    assert "ambiguous" in res.reason.lower()


def test_legacy_signals_normalize_to_unit_interval():
    df, _ = _make_df(60)
    df["target"] = np.random.RandomState(2).choice(["x", "y", "z"], len(df))
    for s in score_targets(df):
        assert 0.0 <= s.name_signal <= 1.0
        assert 0.0 <= s.cardinality_signal <= 1.0
        assert 0.0 <= s.dtype_signal <= 1.0
        assert 0.0 <= s.predictability_signal <= 1.0
        assert 0.0 <= s.total <= 1.0


def test_legacy_no_candidates_raises():
    df = pd.DataFrame({"id": np.arange(50.0), "nulls": [np.nan] * 50})
    with pytest.raises(ValueError, match="No plausible target"):
        detect_target(df)


def test_legacy_escalation_margin_respected():
    df, rng = _make_df(200)
    base = rng.normal(0, 1, 200)
    df["a"] = pd.Series((base + rng.normal(0, 0.1, 200) > 0).astype(int))
    df["b"] = pd.Series((base + rng.normal(0, 0.1, 200) > 0).astype(int))
    res = detect_target(df, auto_select=False)
    top, second = res.scores[0].total, res.scores[1].total
    assert top - second < ESCALATION_MARGIN + 1e-6