"""Integration test for tabular end-to-end on messy data (Phase 1f)."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.target_detection import rank_target_candidates
from src.preprocessing.tabular_preprocessor import TabularPreprocessor
from src.model_selection.tabular_candidates import get_tabular_candidates
from src.model_selection.search import successive_halving_search


def _make_messy_csv(path, n=100, seed=0):
    """100-row CSV: two plausible binary targets, numeric target-like, 20% missing,
    5 duplicate rows, non-English text feature."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n).round(1),
        "income": rng.normal(50000, 15000, n).round(),
        "tenure_months": rng.choice([3, 6, 12, 24, 36], n),
        "is_churned": rng.choice([0, 1], n),
        "is_premium": rng.choice([0, 1], n),
        "score": rng.normal(0.5, 0.2, n).round(3),  # numeric target-like
        "city": rng.choice(["NY", "SF", "LA", "CHI"], n),
        "city_2": rng.choice(["NY", "SF", "LA", "CHI"], n),  # duplicate name; will get _1 suffix
    })
    # Chinese text feature
    df["comment"] = rng.choice([
        "很好", "很差", "我喜欢", "讨厌", "一般", "完美", "满意",
    ], n)
    # 20% missing values sprinkled across columns
    for col in ["age", "income", "tenure_months"]:
        mask = rng.choice([True, False], n, p=[0.2, 0.8])
        df.loc[mask, col] = np.nan
    # 5 duplicate rows appended
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def test_rank_target_candidates_puts_binary_targets_first():
    """Binary target columns should rank higher than continuous features."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "messy.csv")
        _make_messy_csv(csv)
        df = pd.read_csv(csv)

        ranked = rank_target_candidates(df)
        top_cols = [r["col"] for r in ranked[:3]]

        # Binary targets should be near the top
        assert "is_churned" in top_cols
        assert "is_premium" in top_cols
        # Continuous features should rank lower
        continuous_cols = ["age", "income", "tenure_months", "score"]
        for c in continuous_cols:
            idx = next((i for i, r in enumerate(ranked) if r["col"] == c), None)
            assert idx is not None
            # They should be ranked after the binary targets
            assert idx >= 2


def test_explicit_target_col_preprocesses_correctly():
    """Preprocessing with explicit target column should work."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "messy.csv")
        _make_messy_csv(csv)
        df = pd.read_csv(csv)

        target_col = "is_churned"
        tp = TabularPreprocessor()
        X, y = tp.fit_transform(df, target_col=target_col)
        assert X.shape[0] == len(df)
        assert set(np.unique(y).tolist()) <= {0, 1}

        # Run search
        candidates = get_tabular_candidates(n_samples=len(df))
        best_cfg, best_score, all_results = successive_halving_search(
            candidates, X.to_numpy() if hasattr(X, "to_numpy") else X,
            y, time_budget_sec=15, scoring="balanced_accuracy",
        )
        assert best_cfg.get("name") is not None or best_score >= 0.0


def test_four_class_categorical_target_trains():
    """A 4-class categorical target should train and produce valid predictions."""
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
        "category": rng.choice(["a", "b", "c", "d"], n),
        "target_cat": rng.choice(["red", "blue", "green", "yellow"], n),
    })
    tp = TabularPreprocessor()
    X, y = tp.fit_transform(df, target_col="target_cat")
    assert len(set(y.tolist())) == 4

    candidates = get_tabular_candidates(n_samples=len(df))
    best_cfg, best_score, all_results = successive_halving_search(
        candidates, X.to_numpy() if hasattr(X, "to_numpy") else X,
        y, time_budget_sec=15, scoring="balanced_accuracy",
    )
    assert best_cfg.get("name") is not None
    assert best_score >= 0.0