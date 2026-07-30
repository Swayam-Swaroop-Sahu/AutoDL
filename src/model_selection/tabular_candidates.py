"""Tabular candidate factories for the successive-halving search.

Decision (FINAL_PROJECT_PLAN.md §2 "Tabular model family"): drop the Keras MLP; use
Logistic Regression, Random Forest, Gradient Boosting — all sklearn-family.

Deviation logged in PHASE_1_SUMMARY.md: LightGBM (the plan's first-listed candidate)
is blocked by an Application Control policy on this build machine, so it is
substituted with `GradientBoostingClassifier` (same tree-boosting family, pure
sklearn). The factory is structured so LightGBM can be re-added behind a feature
flag later without touching the search loop.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .search import Candidate


def get_tabular_candidates(n_samples: int = 0) -> list:
    """Return the ranked candidate list for tabular classification."""
    rf_n = max(100, min(400, n_samples))  # keep RF cheap on tiny data
    return [
        Candidate(
            name="GradientBoosting",
            factory=lambda: GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            description="Gradient-boosted decision trees (log-loss).",
            params="~n_estimators=100, depth=3",
            pros="Strong tabular baseline; handles non-linearities.",
            cons="Slower than LR; overfits tiny data if unchecked.",
        ),
        Candidate(
            name="RandomForest",
            factory=lambda: RandomForestClassifier(
                n_estimators=rf_n, max_depth=None, n_jobs=1, random_state=42),
            description="Bagged decision trees.",
            params=f"~n_estimators={rf_n}",
            pros="Robust; needs little tuning.",
            cons="Less accurate than boosting on messy tabular.",
        ),
        Candidate(
            name="LogisticRegression",
            factory=lambda: LogisticRegression(
                max_iter=500, n_jobs=1, random_state=42),
            description="Linear baseline with L2.",
            params="max_iter=500, L2",
            pros="Fast; deterministic; good multiclass floor.",
            cons="Underfits non-linear data.",
        ),
    ]
