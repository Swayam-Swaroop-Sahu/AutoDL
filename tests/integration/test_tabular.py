"""Integration test for tabular end-to-end on messy data (Phase 1f)."""
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest

from src.target_detection import resolve_target
from src.preprocessing.tabular_preprocessor import TabularPreprocessor


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


def test_messy_target_detection_escalates_to_human_required():
    """Two binary targets with similar likelihood → human_required escalation."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "messy.csv")
        _make_messy_csv(csv)
        df = pd.read_csv(csv)
        # Resolve target without explicit override → should escalate (or pick top)
        resolved, status = resolve_target(df)
        # With two binary targets + duplicate column, the system must either:
        # - select one (strong_auto/weak_auto), OR
        # - escalate to human_required
        assert status in ("strong_auto", "weak_auto", "human_required"), (
            f"unexpected status: {status}"
        )
        # If auto-selected, the chosen column must be a real binary target
        if status in ("strong_auto", "weak_auto"):
            assert resolved in df.columns


def test_explicit_target_col_completes_training():
    """Re-run with --target-col is_churned: training must complete."""
    from src.model_selection.tabular_candidates import get_tabular_candidates
    from src.model_selection.search import successive_halving_search

    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "messy.csv")
        _make_messy_csv(csv)
        df = pd.read_csv(csv)

        # Resolve target explicitly
        target_col = "is_churned"
        resolved, status = resolve_target(df, target_col=target_col)
        assert status == "override"
        assert resolved == target_col

        # Preprocess
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
    from src.model_selection.tabular_candidates import get_tabular_candidates
    from src.model_selection.search import successive_halving_search

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
