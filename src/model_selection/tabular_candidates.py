"""Tabular candidate factories for the successive-halving search.

Decision (FINAL_PROJECT_PLAN.md §2 "Tabular model family"): include both deep
learning (MLPs) and classical ML (Logistic Regression, Random Forest, Gradient
Boosting). All candidates return sklearn-style objects or Keras models that the
search loop can score with cross_val_score.

Deviation logged in PHASE_1_SUMMARY.md: LightGBM is blocked by an Application
Control policy on this build machine, so it is substituted with
`GradientBoostingClassifier` (same tree-boosting family, pure sklearn). The
factory is structured so LightGBM can be re-added behind a feature flag later
without touching the search loop.

DL models (MLPs) are built lazily with the correct num_features/num_classes at
search time via `_build_mlp_candidate`.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .search import Candidate


def _build_mlp_candidate(name, hidden_layers, activation, alpha, max_iter, desc, pros, cons):
    """Factory for an sklearn MLPClassifier wrapped as a Candidate."""
    return Candidate(
        name=name,
        factory=lambda: MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
        ),
        description=desc,
        params=f"hidden={hidden_layers}, alpha={alpha}",
        pros=pros,
        cons=cons,
    )


def get_tabular_candidates(n_samples: int = 0) -> list:
    """Return the ranked candidate list for tabular classification.

    Includes:
      - Deep Learning (MLP): MLP-Small, MLP-Medium, MLP-Deep
      - Classical ML: GradientBoosting, RandomForest, LogisticRegression
    """
    rf_n = max(100, min(400, n_samples))  # keep RF cheap on tiny data

    return [
        # --- Deep Learning (MLPs) ---
        _build_mlp_candidate(
            name="MLP-Small",
            hidden_layers=(64,),
            activation="relu",
            alpha=1e-3,
            max_iter=150,
            desc="1 hidden layer (64 units), ReLU, dropout via early stopping.",
            pros="Fast DL baseline; works on small data.",
            cons="Limited capacity for complex patterns.",
        ),
        _build_mlp_candidate(
            name="MLP-Medium",
            hidden_layers=(128, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=200,
            desc="2 hidden layers (128->64), ReLU, L2 regularization.",
            pros="Better capacity; still fast.",
            cons="Needs more data than linear models.",
        ),
        _build_mlp_candidate(
            name="MLP-Deep",
            hidden_layers=(256, 128, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=250,
            desc="3 hidden layers (256->128->64), ReLU, low L2.",
            pros="High capacity; captures complex non-linearities.",
            cons="Slowest; overfits on tiny data.",
        ),

        # --- Classical ML ---
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
