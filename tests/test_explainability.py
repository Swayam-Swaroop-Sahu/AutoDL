"""Tests for explainability modules: importance and narrative."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.explainability.importance import compute_permutation_importance
from src.explainability.narrative import generate_narrative


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iris_data():
    """Small synthetic 3-class dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(150, 4)
    y = rng.randint(0, 3, size=150)
    return X, y


@pytest.fixture
def trained_sklearn_model(iris_data):
    """A trained LogisticRegression model."""
    X, y = iris_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# compute_permutation_importance tests
# ---------------------------------------------------------------------------

def test_permutation_importance_returns_non_empty(trained_sklearn_model):
    """Permutation importance should return a non-empty sorted list."""
    model, X, y = trained_sklearn_model
    result = compute_permutation_importance(model, X, y, n_repeats=3)
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == X.shape[1]
    # Should be sorted descending by importance
    for i in range(len(result) - 1):
        assert result[i]["importance"] >= result[i + 1]["importance"]
    # Each entry must have expected keys
    for entry in result:
        assert "feature" in entry
        assert "importance" in entry
        assert "std" in entry
        assert isinstance(entry["feature"], str)
        assert isinstance(entry["importance"], float)
        assert isinstance(entry["std"], float)


def test_permutation_importance_with_feature_names(trained_sklearn_model):
    """Feature names should appear in the result when provided."""
    model, X, y = trained_sklearn_model
    names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    result = compute_permutation_importance(
        model, X, y, n_repeats=2, feature_names=names,
    )
    assert result is not None
    for entry, expected_name in zip(result, sorted(names)):
        assert entry["feature"] in names


def test_permutation_importance_returns_none_for_unsupported_model():
    """Model without .predict() should return None gracefully."""
    class BadModel:
        pass

    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20)
    result = compute_permutation_importance(BadModel(), X, y, n_repeats=2)
    assert result is None


def test_permutation_importance_empty_data_returns_none(trained_sklearn_model):
    """Empty validation data should return None gracefully."""
    model, _, _ = trained_sklearn_model
    result = compute_permutation_importance(
        model, np.array([]), np.array([]), n_repeats=2,
    )
    assert result is None


# ---------------------------------------------------------------------------
# generate_narrative tests
# ---------------------------------------------------------------------------

def test_narrative_produces_string_with_expected_content():
    """Narrative should contain model name, accuracy, and top feature."""
    meta = {
        "model_name": "RandomForest",
        "metrics": {"accuracy": 0.92},
        "feature_importance": [
            {"feature": "age", "importance": 0.25, "std": 0.02},
            {"feature": "income", "importance": 0.15, "std": 0.01},
        ],
    }
    quality = {"passed": True, "warnings": [], "summary": "All clear."}
    narrative = generate_narrative(meta, quality)
    assert isinstance(narrative, str)
    assert len(narrative) > 0
    assert "RandomForest" in narrative
    assert "92%" in narrative
    assert "age" in narrative
    assert "No major data quality issues" in narrative


def test_narrative_with_warnings():
    """When quality has warnings, the narrative should mention them."""
    meta = {
        "model_name": "LogReg",
        "metrics": {"accuracy": 0.75},
        "feature_importance": [
            {"feature": "f0", "importance": 0.1, "std": 0.01},
        ],
    }
    quality = {
        "passed": False,
        "summary": "Found 1 issue: leakage.",
        "warnings": [
            {"column": "leaky_col", "issue": "feature_leakage", "detail": "r=0.99"},
        ],
    }
    narrative = generate_narrative(meta, quality=quality)
    assert isinstance(narrative, str)
    assert "leakage" in narrative.lower()


def test_narrative_without_feature_importance():
    """When feature_importance is missing, a fallback sentence is used."""
    meta = {
        "model_name": "KerasModel",
        "metrics": {"accuracy": 0.88},
    }
    narrative = generate_narrative(meta)
    assert isinstance(narrative, str)
    assert "most influential features were not computed" in narrative


def test_narrative_without_accuracy():
    """When accuracy is missing, the narrative should still produce a result."""
    meta = {
        "model_name": "BestModel",
        "metrics": {},
        "feature_importance": None,
    }
    narrative = generate_narrative(meta)
    assert isinstance(narrative, str)
    assert "BestModel" in narrative
    assert "trained successfully" in narrative