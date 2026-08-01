"""Tests for simplified target detection (name + cardinality only).

Tests:
  - is_churned scores higher than customer_id
  - binary unnamed column scores moderate
  - unique-per-row column scores 0
  - ranking puts best first
  - pipeline rejects single-class target
  - pipeline rejects >50-class target
"""

import pytest
import pandas as pd
import numpy as np

from src.target_detection import target_likelihood, rank_target_candidates
from src.core.exceptions import AutoDLInputError
from src.core.validation import validate_target


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _df(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# target_likelihood
# ---------------------------------------------------------------------------
def test_is_churned_scores_higher_than_customer_id():
    """Column named 'is_churned' must out-rank 'customer_id'."""
    df = _df({
        "customer_id": list(range(100)),
        "is_churned": [0, 1] * 50,
        "age": np.random.randint(18, 80, 100),
    })
    score_churn = target_likelihood(df, "is_churned")
    score_id = target_likelihood(df, "customer_id")
    assert score_churn > score_id, (
        f"is_churned ({score_churn}) should beat customer_id ({score_id})"
    )
    # id columns should score 0.0 because name is penalized and cardinality is 0
    assert score_id == 0.0
    # is_churned is binary + has positive name → high score
    assert score_churn > 0.7


def test_binary_unnamed_column_scores_moderate():
    """Binary columns without keyword names get moderate scores from cardinality."""
    df = _df({
        "feature_a": np.random.randn(100),
        "col_x": [0, 1] * 50,  # binary but no special name
    })
    score = target_likelihood(df, "col_x")
    # Name=0, card=1.0 → expected 0.5
    assert 0.4 <= score <= 0.6, f"binary unnamed column should score ~0.5, got {score}"


def test_unique_column_scores_zero():
    """Column with n_unique == n_rows (like an ID) scores 0.0."""
    df = _df({
        "unique_id": list(range(200)),
        "target": [0, 1] * 100,
    })
    assert target_likelihood(df, "unique_id") == 0.0


def test_negative_name_patterns_heavily_penalized():
    """Columns with 'id' in name get strong penalty."""
    df = _df({
        "PassengerId": list(range(100)),
        "Survived": [0, 1] * 50,
        "Timestamp": list(range(100)),
        "created_at": list(range(100)),
    })
    for col in ["PassengerId", "Timestamp", "created_at"]:
        s = target_likelihood(df, col)
        assert s <= 0.1, f"{col} should score ≤0.1 for negative names, got {s}"
    # Survived has positive name + binary → high
    assert target_likelihood(df, "Survived") > 0.7


def test_mid_cardinality_gets_intermediate_score():
    """Moderate cardinality (e.g., ~40 unique in 200 rows) yields ~0.3 card."""
    df = _df({
        "region": [f"R{i}" for i in range(40)] * 5,  # 40 unique, 200 rows
    })
    score = target_likelihood(df, "region")
    # name=0, card between 0.3 → ~0.15 overall (since n_unique > 5% but < 50%)
    # Wait: 40 unique / 200 = 0.2 ratio. 5% = 10. So 2 < 40 <= 50 → card = 0.8
    # Still fits "low cardinal categorical" since 40 <= Min(50,10)… hmm
    # 40 > min(50, 10) = 10, so falls into else → card = 0.3 or ambiguous
    # Actually: not binary, not unique, > min(50,0.05*200=10), so card=0.3 (else branch).
    assert 0.1 <= score <= 0.5, f"moderate cardinality got {score}"


# ---------------------------------------------------------------------------
# rank_target_candidates
# ---------------------------------------------------------------------------
def test_ranking_puts_best_first():
    """The top candidate must have the highest score."""
    df = _df({
        "id": list(range(100)),
        "age": np.random.randint(18, 80, 100),
        "churn_flag": [0, 1] * 50,
    })
    ranked = rank_target_candidates(df)
    assert ranked[0]["col"] == "churn_flag", f"Expected churn_flag first, got {ranked[0]['col']}"
    assert ranked[0]["score"] >= ranked[1]["score"], "Scores should be descending"


def test_ranking_includes_all_columns():
    """Every column appears exactly once in the ranked list."""
    df = _df({
        "a": range(30),
        "b": [1, 2, 3] * 10,
        "c": np.arange(30),
    })
    ranked = rank_target_candidates(df)
    assert len(ranked) == 3
    cols = {r["col"] for r in ranked}
    assert cols == {"a", "b", "c"}


def test_ranking_keys_are_present():
    """Every entry has the required keys."""
    df = _df({"target": [0, 1] * 5})
    ranked = rank_target_candidates(df)
    for r in ranked:
        for k in ("col", "score", "name_score", "card_score", "n_unique"):
            assert k in r, f"Missing key {k} in {r}"


# ---------------------------------------------------------------------------
# validate_target
# ---------------------------------------------------------------------------
def test_validate_target_rejects_single_class():
    df = _df({"label": ["A"] * 20})
    with pytest.raises(AutoDLInputError, match="fewer than 2|at least 2|only 1 distinct"):
        validate_target(df, "label")


def test_validate_target_rejects_too_many_classes():
    df = _df({"label": list(range(51))})  # 51 classes
    with pytest.raises(AutoDLInputError, match="exceeds.*50|maximum of 50|beyond 50"):
        validate_target(df, "label")


def test_validate_target_missing_column():
    df = _df({"x": [1, 2]})
    with pytest.raises(AutoDLInputError, match="not found"):
        validate_target(df, "does_not_exist")


def test_validate_target_all_nan():
    df = _df({"label": [None, None, None]})
    with pytest.raises(AutoDLInputError, match="no non-null"):
        validate_target(df, "label")


def test_validate_target_minimum_two_classes_passes():
    df = _df({"label": [0, 1, 0, 1]})
    n = validate_target(df, "label")
    assert n == 2


def test_validate_target_maximum_50_passes():
    df = _df({"label": list(range(50))})
    n = validate_target(df, "label")
    assert n == 50