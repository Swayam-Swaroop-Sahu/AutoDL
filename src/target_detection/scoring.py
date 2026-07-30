"""Target Likelihood Score (TLS) — per-column scoring.

Provides `target_likelihood(df, col) → float` — a pure function returning a
score in [0, 1] for how likely `col` is to be a classification target.

Signals (weighted fusion):
  1. name          (0.25) — column-name heuristic against known target vocab
  2. cardinality   (0.30) — low distinct-values/rows ratio → target-like
  3. dtype         (0.20) — object/bool/int → target-like; float → feature-like
  4. predictability(0.25) — fast 3-fold LR CV AUC: can other columns predict this one?
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.core.config import RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable weights
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    "name": 0.25,
    "cardinality": 0.30,
    "dtype": 0.20,
    "predictability": 0.25,
}

TARGET_NAME_VOCAB = (
    "target", "label", "class", "y", "outcome", "response", "diagnosis",
    "survived", "churn", "fraud", "spam", "is_", "flag",
)

# dtype signal constants
_MAX_CARDINALITY_RATIO = 0.5
_CONTINUOUS_FLOAT_PENALTY = 0.6


# ---------------------------------------------------------------------------
# Signal scorers (pure functions)
# ---------------------------------------------------------------------------
def _name_signal(col_name: str) -> float:
    """Score column name for target-like keywords; return 0.0 or 1.0."""
    name = (col_name or "").lower().strip()
    for v in TARGET_NAME_VOCAB:
        if v in name:
            return 1.0
    if name.endswith("id"):
        return 0.0
    return 0.0


def _cardinality_signal(series: pd.Series, n_rows: int) -> float:
    """Few unique values relative to rows → target-like (0..1)."""
    n_unique = series.nunique(dropna=True)
    if n_rows == 0:
        return 0.0
    ratio = n_unique / n_rows
    if ratio >= _MAX_CARDINALITY_RATIO:
        return max(0.0, 1.0 - (ratio - _MAX_CARDINALITY_RATIO)
                   / (1.0 - _MAX_CARDINALITY_RATIO))
    within = ratio / _MAX_CARDINALITY_RATIO
    if n_unique <= 2:
        return 1.0
    return max(0.0, 1.0 - within)


def _dtype_signal(series: pd.Series) -> float:
    """Score dtype suitability for a classification target (0..1)."""
    dtype = str(series.dtype)
    if dtype in ("bool", "boolean"):
        return 1.0
    if dtype in ("object", "string", "category") or dtype.startswith("string"):
        return 0.8
    if dtype.startswith("int"):
        return 0.6
    if dtype.startswith("float"):
        return max(0.0, 1.0 - _CONTINUOUS_FLOAT_PENALTY)  # 0.4
    return 0.3


def _predictability_signal(df: pd.DataFrame, col: str, n_rows: int) -> float:
    """Fast 3-fold LR CV AUC: how well do *other* columns predict `col`?

    Returns score in [0, 1]. Capped at 100 iterations on LR, subsampled
    to ≤800 rows.  Returns 0.0 on any failure (never raises).
    """
    other_cols = [c for c in df.columns if c != col]
    if not other_cols or n_rows < 8:
        return 0.0
    try:
        y = df[col].astype(str)
        if y.nunique() > 20:
            return 0.0

        X_parts = []
        for c in other_cols:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                X_parts.append(
                    s.fillna(s.median() if not s.dropna().empty else 0)
                    .to_numpy().reshape(-1, 1)
                )
            else:
                uniq = s.nunique(dropna=True)
                if uniq == 0:
                    continue
                enc = OneHotEncoder(
                    max_categories=20, handle_unknown="ignore", sparse_output=False,
                )
                try:
                    oh = enc.fit_transform(
                        s.astype(str).fillna("<NA>").to_numpy().reshape(-1, 1)
                    )
                except Exception:
                    continue
                X_parts.append(oh)

        if not X_parts:
            return 0.0
        X = np.hstack([p for p in X_parts if p.shape[0] == n_rows])
        try:
            X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        except Exception:
            pass

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)

        if X.shape[0] > 800:
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(X.shape[0], 800, replace=False)
            X, y_enc = X[idx], y_enc[idx]

        if len(np.unique(y_enc)) < 2:
            return 0.0

        estimator = LogisticRegression(max_iter=100, n_jobs=1)
        scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
        scores = cross_val_score(
            estimator, X, y_enc, cv=3, scoring=scoring, error_score="raise",
        )
        auc = float(np.mean(scores))
        return max(0.0, min(1.0, (auc - 0.5) / 0.5))
    except Exception as exc:
        logger.warning(
            "predictability signal failed for column '%s' (%s); returning 0.0",
            col, exc,
        )
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def target_likelihood(df: pd.DataFrame, col: str) -> float:
    """Return a score 0–1 for how likely `col` is the classification target.

    Returns 0.0 if the column is an obvious anti-target (unique ID, all-null,
    constant, or float with too many distinct values).
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in DataFrame; columns: {list(df.columns)}")

    series = df[col]
    n_rows = len(df)
    n_unique = series.nunique(dropna=True)

    # Hard exclude obvious non-targets
    if n_unique < 2 or series.isna().all():
        return 0.0
    if n_unique == n_rows and n_rows > 8:
        return 0.0  # unique per row → ID, not a class label

    name_sig = _name_signal(col)
    card_sig = _cardinality_signal(series, n_rows)
    dtype_sig = _dtype_signal(series)
    pred_sig = _predictability_signal(df, col, n_rows)

    score = (
        SIGNAL_WEIGHTS["name"] * name_sig
        + SIGNAL_WEIGHTS["cardinality"] * card_sig
        + SIGNAL_WEIGHTS["dtype"] * dtype_sig
        + SIGNAL_WEIGHTS["predictability"] * pred_sig
    )
    return float(max(0.0, min(1.0, score)))