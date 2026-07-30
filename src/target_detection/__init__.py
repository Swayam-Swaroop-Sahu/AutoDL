"""Target column detection for AutoDL v2.

Replaces the v1 "last column is the target" default with a Target Likelihood Score
(TLS) computed for every column, then ranked, then either auto-selected (clear
winner) or escalated (ambiguous → human confirms).

TLS signals (FINAL_PROJECT_PLAN.md §2 "Target column detection" row):
  1. name       — column-name match against known target/label vocabulary
  2. cardinality — target-like columns have low-ish distinct values relative to rows
                  (classification targets are categorical, few classes)
  3. dtype      — object/bool/integer-with-few-unions read as more target-like than
                  float (continuous floats are features, not class labels)
  4. predictability — a quick AUC-style "+ 1 Rest" one-vs-rest score: how well the
                  remaining columns predict THIS column. A high predictability means
                  this column is "explainable" by features → it behaves like a label.

Scoring is normalized to [0, 1] per signal and combined with documented weights
(below). The scorer is pure (no side effects) so it is unit-testable deterministically.

Escalation rules (FINAL_PROJECT_PLAN.md §7 ambiguous-two-binary-columns test case):
  - If the top-2 scores are within `ESCALATION_MARGIN`, the run is flagged
    "ambiguous" and the caller (UI/CLI) must confirm the target before training.
  - The CLI override (`target_col` arg) short-circuits the scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.config import RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable weights (documented, single source of truth for the scorer)
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    "name": 0.25,
    "cardinality": 0.25,
    "dtype": 0.20,
    "predictability": 0.30,
}

# A score gap below this between rank-1 and rank-2 → escalation ("ambiguous").
ESCALATION_MARGIN = 0.10

# Known target/label name vocabulary. Match is case-insensitive + substring.
TARGET_NAME_VOCAB = (
    "target", "label", "class", "y", "outcome", "response", "diagnosis",
    "survived", "churn", "fraud", "spam", "is_", "flag",
)

# dtype signal ceilings
_MAX_CARDINALITY_RATIO = 0.5   # >50% distinct values → almost certainly an ID/feature
_CONTINUOUS_FLOAT_PENALTY = 0.6  # floats get dtype signal capped below this


@dataclass
class ColumnScore:
    """Per-column TLS breakdown (for ranking + UI display + tests)."""
    name: str
    total: float
    name_signal: float
    cardinality_signal: float
    dtype_signal: float
    predictability_signal: float
    dtype: str
    n_unique: int
    is_target_candidate: bool = True

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "total": round(self.total, 4),
            "name_signal": round(self.name_signal, 4),
            "cardinality_signal": round(self.cardinality_signal, 4),
            "dtype_signal": round(self.dtype_signal, 4),
            "predictability_signal": round(self.predictability_signal, 4),
            "dtype": self.dtype,
            "n_unique": self.n_unique,
        }


@dataclass
class TargetDetectionResult:
    chosen_column: Optional[str]
    scores: List[ColumnScore]
    ambiguous: bool
    top_candidates: List[str]
    reason: str
    escalated: bool = False

    def as_dict(self) -> dict:
        return {
            "chosen_column": self.chosen_column,
            "ambiguous": self.ambiguous,
            "escalated": self.escalated,
            "top_candidates": self.top_candidates,
            "reason": self.reason,
            "scores": [s.as_dict() for s in self.scores],
        }


# ---------------------------------------------------------------------------
# Signal scorers — all pure functions
# ---------------------------------------------------------------------------
def _name_signal(col_name: str) -> float:
    name = (col_name or "").lower().strip()
    for v in TARGET_NAME_VOCAB:
        if v in name:
            return 1.0
    # partial credit: column ends with '_label'/'_target' style suffixes
    if name.endswith(("id",)):
        return 0.0  # explicit ID columns are anti-targets
    return 0.0


def _cardinality_signal(series: pd.Series, n_rows: int) -> float:
    n_unique = series.nunique(dropna=True)
    if n_rows == 0:
        return 0.0
    ratio = n_unique / n_rows
    # Many unique values (relative to rows) → ID/feature, score low.
    if ratio >= _MAX_CARDINALITY_RATIO:
        # at 100% unique (IDs) score 0; decays as ratio grows beyond threshold
        return max(0.0, 1.0 - (ratio - _MAX_CARDINALITY_RATIO) * (1.0 / (1.0 - _MAX_CARDINALITY_RATIO)))
    # Below the ceiling: fewer classes relative to rows → more target-like.
    # Score 1.0 for very few classes; ramps down as cardinality climbs toward the ceiling.
    within = ratio / _MAX_CARDINALITY_RATIO  # 0..1
    # nicely target-shaped at 2-20 classes; reward low cardinality
    if n_unique <= 2:
        return 1.0
    return max(0.0, 1.0 - within)


def _dtype_signal(series: pd.Series) -> float:
    dtype = str(series.dtype)
    if dtype in ("bool", "boolean"):
        return 1.0
    if dtype == "object" or dtype.startswith("string") or dtype == "category":
        # string/categorical columns are plausible class labels
        return 0.8
    if dtype.startswith("int"):
        # integer with few unique values reads categorical-ish; capped higher than float
        return 0.6
    if dtype.startswith("float"):
        # continuous numeric is a feature, not a label
        return max(0.0, 1.0 - _CONTINUOUS_FLOAT_PENALTY)  # 0.4
    return 0.3


def _predictability_signal(df: pd.DataFrame, candidate: str, n_rows: int) -> Tuple[float, str]:
    """How well do the OTHER columns predict THIS column (AUC of one-vs-rest LR).

    Returns (score in [0,1], dtype_kind). Failed/too-slow computations degrade to 0
    with a logged warning — never raise. Capped by sample size for speed.
    """
    target_dtype = str(df[candidate].dtype)
    other = [c for c in df.columns if c != candidate]
    if not other or n_rows < 8:
        return 0.0, target_dtype
    try:
        y = df[candidate].astype(str)
        # If candidate has >20 classes, predictability scoring is unreliable & slow → 0.
        if y.nunique() > 20:
            return 0.0, target_dtype

        # Encode features: numeric passthrough, categorical one-hot (bounded).
        X_parts = []
        for c in other:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                X_parts.append(s.fillna(s.median()).to_numpy().reshape(-1, 1))
            else:
                uniq = s.nunique(dropna=True)
                if uniq == 0:
                    continue
                enc = OneHotEncoder(max_categories=20, handle_unknown="ignore", sparse_output=False)
                try:
                    oh = enc.fit_transform(s.astype(str).fillna("<NA>").to_numpy().reshape(-1, 1))
                except Exception:
                    continue
                X_parts.append(oh)
        if not X_parts:
            return 0.0, target_dtype
        X = np.hstack([p for p in X_parts if p.shape[0] == n_rows])
        # Scale numeric-ish features for LR stability.
        try:
            X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        except Exception:
            pass

        # AUC via 3-fold cross-val; binary uses roc_auc, multiclass uses_ovr roc_auc.
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)
        # Subsample to keep this fast (≤800 rows) regardless of dataset size.
        if X.shape[0] > 800:
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(X.shape[0], 800, replace=False)
            X, y_enc = X[idx], y_enc[idx]
        # Need at least 2 classes present in the subsample.
        if len(np.unique(y_enc)) < 2:
            return 0.0, target_dtype
        estimator = LogisticRegression(max_iter=200, n_jobs=1)
        # roc_auc supports multiclass via 'ovr'/'ovo'; binary expects 1D.
        scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
        scores = cross_val_score(estimator, X, y_enc, cv=3, scoring=scoring, error_score="raise")
        auc = float(np.mean(scores))
        # AUC ~0.5 (random) → no predictability signal; AUC 1.0 → strong.
        return max(0.0, min(1.0, (auc - 0.5) / 0.5)), target_dtype
    except Exception as e:
        logger.warning("predictability signal failed for column '%s' (%s); using 0.0", candidate, e)
        return 0.0, target_dtype


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def score_targets(df: pd.DataFrame, sample: int = 800) -> List[ColumnScore]:
    """Score every column of `df` for target-likelihood. Pure, deterministic given RANDOM_SEED."""
    if df is None or df.empty:
        return []
    n_rows = len(df)
    scores: List[ColumnScore] = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        name_sig = _name_signal(col)
        card_sig = _cardinality_signal(s, n_rows)
        dtype_sig = _dtype_signal(s)
        pred_sig, dtype_str = _predictability_signal(df, col, n_rows)
        total = (
            SIGNAL_WEIGHTS["name"] * name_sig
            + SIGNAL_WEIGHTS["cardinality"] * card_sig
            + SIGNAL_WEIGHTS["dtype"] * dtype_sig
            + SIGNAL_WEIGHTS["predictability"] * pred_sig
        )
        # ID / constant / all-null columns cannot be targets.
        is_candidate = not (n_unique == n_rows and n_rows > 8) and n_unique >= 2 and not s.isna().all()
        scores.append(ColumnScore(
            name=col, total=total,
            name_signal=name_sig, cardinality_signal=card_sig,
            dtype_signal=dtype_sig, predictability_signal=pred_sig,
            dtype=dtype_str, n_unique=n_unique,
            is_target_candidate=is_candidate,
        ))
    scores.sort(key=lambda c: c.total, reverse=True)
    return scores


def detect_target(
    df: pd.DataFrame,
    override: Optional[str] = None,
    auto_select: bool = True,
) -> TargetDetectionResult:
    """Detect the target column.

    Args:
        df: input DataFrame.
        override: explicit target column name (CLI/UI override). Short-circuits scoring.
        auto_select: when False and ambiguous, does NOT auto-pick; returns chosen=None + escalated.

    Returns TargetDetectionResult with `.chosen_column` set or `.escalated=True`.
    """
    if override is not None:
        if override not in df.columns:
            raise ValueError(f"Override target column '{override}' not found in data; "
                             f"available: {list(df.columns)}")
        logger.info("target overridden by caller: %s", override)
        return TargetDetectionResult(
            chosen_column=override,
            scores=[],
            ambiguous=False,
            top_candidates=[override],
            reason=f"Target column set by explicit override: '{override}'.",
            escalated=False,
        )

    scores = score_targets(df)
    candidates = [s for s in scores if s.is_target_candidate]
    if not candidates:
        raise ValueError("No plausible target column found. Every column is either "
                         "all-null, constant, or a unique ID. Pass an explicit target_col override.")

    top = candidates[0]
    if len(candidates) == 1:
        return TargetDetectionResult(chosen_column=top.name, scores=scores,
                                     ambiguous=False, top_candidates=[top.name],
                                     reason=f"Only one plausible target column found: '{top.name}'.")

    second = candidates[1]
    gap = top.total - second.total
    ambiguous = gap < ESCALATION_MARGIN
    top_two = [top.name, second.name]

    if ambiguous and not auto_select:
        return TargetDetectionResult(
            chosen_column=None, scores=scores, ambiguous=True,
            top_candidates=top_two,
            reason=(f"Target column is ambiguous: '{top.name}' (TLS={top.total:.3f}) vs "
                    f"'{second.name}' (TLS={second.total:.3f}); gap={gap:.3f} < {ESCALATION_MARGIN}. "
                    "Confirm the target before training."),
            escalated=True,
        )

    chosen = top.name
    reason = (
        f"Ambiguous target (top-2 gap {gap:.3f} < {ESCALATION_MARGIN}); auto-selected highest TLS "
        f"'{chosen}' ({top.total:.3f}). Confirm to accept."
        if ambiguous else
        f"Auto-selected '{chosen}' (TLS={top.total:.3f}); next best '{second.name}' "
        f"(TLS={second.total:.3f}), gap={gap:.3f}."
    )
    logger.info("target detected: %s (ambiguous=%s, escalated=%s)", chosen, ambiguous,
                ambiguous and not auto_select)
    return TargetDetectionResult(chosen_column=chosen, scores=scores, ambiguous=ambiguous,
                                  top_candidates=top_two, reason=reason,
                                  escalated=ambiguous and not auto_select)
