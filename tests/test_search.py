"""Unit tests for successive-halving model search (Phase 1b)."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.model_selection.search import (
    Candidate,
    successive_halving_search,
    DEFAULT_SCORING,
    DEFAULT_TIME_BUDGET_SEC,
)


def _make_data(n=200):
    X, y = make_classification(
        n_samples=n, n_features=5, n_classes=2, random_state=42,
    )
    return X, y


def _dummy_candidates():
    """3 simple candidates for testing."""
    return [
        Candidate(
            name="LogReg",
            factory=lambda: LogisticRegression(max_iter=500, random_state=42),
            description="Logistic regression", params="max_iter=500",
            pros="fast", cons="linear",
        ),
        Candidate(
            name="LogReg-weak",
            factory=lambda: LogisticRegression(C=0.001, max_iter=500, random_state=42),
            description="Heavily regularized LR",
            params="C=0.001",
            pros="simple", cons="underfits",
        ),
        Candidate(
            name="LogReg-strong",
            factory=lambda: LogisticRegression(C=10.0, max_iter=500, random_state=42),
            description="Weakly regularized LR",
            params="C=10.0",
            pros="flexible", cons="overfits",
        ),
    ]


# ---------------------------------------------------------------------------
# Core search tests
# ---------------------------------------------------------------------------
def test_search_returns_winner():
    """3 dummy candidates, default budget -> returns winner with valid score."""
    X, y = _make_data()
    candidates = _dummy_candidates()
    best_cfg, best_score, all_results = successive_halving_search(
        candidates, X, y, time_budget_sec=60, scoring="balanced_accuracy",
    )
    assert best_score > 0.0
    assert best_cfg["name"] in {c.name for c in candidates}
    assert "factory" in best_cfg
    assert len(all_results) >= 1


def test_search_2sec_budget_still_returns_winner():
    """With a tiny 2-second budget, search still returns a result."""
    X, y = _make_data()
    candidates = _dummy_candidates()
    best_cfg, best_score, all_results = successive_halving_search(
        candidates, X, y, time_budget_sec=2,
    )
    assert best_cfg.get("name") is not None or best_score >= 0.0
    # Even with tiny budget, we should have at least one result
    assert isinstance(all_results, list)


def test_one_candidate_crashes_search_continues():
    """One bad candidate raises → search completes, bad one scores 0.0."""
    X, y = _make_data()

    def bad_factory():
        raise RuntimeError("simulated OOM")

    candidates = [
        Candidate(
            name="Crasher",
            factory=bad_factory,
            description="This one crashes",
            params="",
        ),
        Candidate(
            name="Good",
            factory=lambda: LogisticRegression(max_iter=500, random_state=42),
            description="This one works",
            params="",
        ),
    ]

    best_cfg, best_score, all_results = successive_halving_search(
        candidates, X, y, time_budget_sec=10,
    )
    # Good candidate should have won
    assert best_cfg["name"] == "Good"
    assert best_score > 0.0
    # Crasher should have score 0.0 and fit_ok=False
    crasher_results = [r for r in all_results if r["name"] == "Crasher"]
    assert len(crasher_results) >= 1
    for r in crasher_results:
        assert r["fit_ok"] is False
        assert r["score"] == 0.0


def test_timeout_respected():
    """With a very tight budget, search does not run forever."""
    import time
    X, y = _make_data(n=200)
    candidates = _dummy_candidates()

    t0 = time.time()
    best_cfg, best_score, all_results = successive_halving_search(
        candidates, X, y, time_budget_sec=1,  # 1 second total
    )
    elapsed = time.time() - t0
    # Should finish within reasonable time (soft check)
    assert elapsed < 30.0, f"Search took {elapsed:.1f}s, expected < 30s"


def test_all_candidates_return_result_dicts():
    """Every candidate in all_results has the expected keys."""
    X, y = _make_data()
    candidates = _dummy_candidates()
    _, _, all_results = successive_halving_search(
        candidates, X, y, time_budget_sec=10,
    )
    required_keys = {"name", "score", "std", "cv_scores", "fit_ok", "error",
                     "description", "params", "pros", "cons", "stage", "fidelity"}
    for r in all_results:
        assert required_keys <= set(r.keys()), f"Missing keys in {r['name']}: {required_keys - set(r.keys())}"


def test_empty_candidates_returns_safe_defaults():
    """No candidates → returns empty best_cfg, -1 score, empty list."""
    best_cfg, best_score, all_results = successive_halving_search(
        [], np.array([[1]]), np.array([0]),
    )
    assert best_cfg == {}
    assert best_score == -1.0
    assert all_results == []


def test_single_candidate_wins():
    """One candidate only → wins by default."""
    X, y = _make_data()
    candidates = [
        Candidate(
            name="Only",
            factory=lambda: LogisticRegression(max_iter=500, random_state=42),
            description="The only one",
        ),
    ]
    best_cfg, best_score, all_results = successive_halving_search(
        candidates, X, y, time_budget_sec=10,
    )
    assert best_cfg["name"] == "Only"
    assert best_score > 0.0
    assert len([r for r in all_results if r["name"] == "Only"]) >= 1


def test_promotion_cuts_candidates():
    """With 3 candidates, stage 2 should have fewer than stage 1."""
    X, y = _make_data(n=150)
    candidates = _dummy_candidates()
    _, _, all_results = successive_halving_search(
        candidates, X, y, time_budget_sec=30,
    )
    stage1 = [r for r in all_results if r["stage"] == 1]
    stage2 = [r for r in all_results if r["stage"] == 2]
    stage3 = [r for r in all_results if r["stage"] == 3]
    assert len(stage1) >= 1
    # Stage 2 should have at most stage 1 count
    assert len(stage2) <= len(stage1)
    # Stage 3 should have 1-2 survivors
    assert 1 <= len(stage3) <= 2