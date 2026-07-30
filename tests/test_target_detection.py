"""Unit tests for target detection (FINAL_PROJECT_PLAN.md §2 + §7 edge cases)."""
import numpy as np
import pandas as pd
import pytest

from src.target_detection import (
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


def test_clear_target_named_column():
    df, rng = _make_df(120)
    # build a clearly-predictable target named 'churn' from the features
    df["churn"] = ((df["age"] > 50) & (df["income"] < 40000)).astype(int).astype("category")
    res = detect_target(df, auto_select=True)
    assert res.chosen_column == "churn"
    assert not res.escalated


def test_override_short_circuits():
    df, _ = _make_df()
    df["target"] = np.random.RandomState(1).choice(["a", "b"], len(df))
    res = detect_target(df, override="age")  # explicitly override to a feature column
    assert res.chosen_column == "age"
    assert res.reason.startswith("Target column set by explicit override")


def test_override_must_exist():
    df, _ = _make_df()
    with pytest.raises(ValueError, match="not found in data"):
        detect_target(df, override="nope")


def test_ambiguous_two_binary_columns_fails_over_auto_select():
    """§7 explicit edge case: two equally-plausible binary columns → escalation."""
    rng = np.random.RandomState(7)
    n = 200
    # Two binary columns that weakly predict each other from the same features.
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
    # Ambiguous → escalated, chosen_column None, both candidates surfaced.
    assert res.ambiguous is True
    assert res.escalated is True
    assert res.chosen_column is None
    assert set(res.top_candidates) == {"binary_a", "binary_b"}
    assert "ambiguous" in res.reason.lower()


def test_ambiguous_two_binary_columns_auto_selects_best():
    """Same ambiguous case under auto_select=True → must still pick the top one."""
    rng = np.random.RandomState(7)
    n = 200
    base_signal = rng.normal(0, 1, n)
    is_a = (base_signal + rng.normal(0, 0.2, n) > 0).astype(int)
    is_b = (base_signal + rng.normal(0, 0.2, n) > 0).astype(int)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n),
                       "binary_a": is_a, "binary_b": is_b})
    res = detect_target(df, auto_select=True)
    assert res.chosen_column in {"binary_a", "binary_b"}
    assert res.ambiguous is True


def test_named_target_wins_over_features():
    df, _ = _make_df(80)
    df["label"] = pd.Series(["red"] * 40 + ["blue"] * 40).astype("category")
    res = detect_target(df, auto_select=True)
    assert res.chosen_column == "label"


def test_id_column_excluded():
    df, _ = _make_df(50)
    df["row_id"] = np.arange(50)            # unique → anti-target
    df["flag"] = np.where(df["age"] > 40, "y", "n")  # clear binary target
    scores = {s.name: s for s in score_targets(df)}
    assert scores["row_id"].n_unique == 50
    assert scores["row_id"].cardinality_signal < 0.2
    res = detect_target(df, auto_select=True)
    assert res.chosen_column == "flag"


def test_signals_normalize_to_unit_interval():
    df, _ = _make_df(60)
    df["target"] = np.random.RandomState(2).choice(["x", "y", "z"], len(df))
    for s in score_targets(df):
        assert 0.0 <= s.name_signal <= 1.0
        assert 0.0 <= s.cardinality_signal <= 1.0
        assert 0.0 <= s.dtype_signal <= 1.0
        assert 0.0 <= s.predictability_signal <= 1.0
        assert 0.0 <= s.total <= 1.0


def test_no_candidates_raises():
    rng = np.random.RandomState(3)
    # every column is a unique ID OR all-null → no candidate
    df = pd.DataFrame({"id": np.arange(50.0), "nulls": [np.nan] * 50})
    with pytest.raises(ValueError, match="No plausible target"):
        detect_target(df, auto_select=True)


def test_escalation_margin_respected():
    df, rng = _make_df(200)
    base = rng.normal(0, 1, 200)
    df["a"] = pd.Series((base + rng.normal(0, 0.1, 200) > 0).astype(int))
    df["b"] = pd.Series((base + rng.normal(0, 0.1, 200) > 0).astype(int))
    res = detect_target(df, auto_select=False)
    # Both binary cols with same underlying signal → gap < ESCALATION_MARGIN
    top, second = res.scores[0].total, res.scores[1].total
    assert top - second < ESCALATION_MARGIN + 1e-6
