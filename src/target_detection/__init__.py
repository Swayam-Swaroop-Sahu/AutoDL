"""Target column detection for AutoDL v2.

Replaces the v1 "last column is the target" default with a Target Likelihood Score
(TLS) computed for every column, then ranked, then either auto-selected (clear
winner) or escalated (ambiguous → human confirms).

Two public API surfaces:

  **New API (preferred)** — lightweight, status-driven:
    - target_likelihood(df, col) → float 0–1  (from .scoring)
    - resolve_target(df, target_col=None) → (col_or_list, status)  (from .escalation)

  **Legacy API** (backward-compatible, preserved for tests):
    - score_targets(df) → List[ColumnScore]
    - detect_target(df, override, auto_select) → TargetDetectionResult
    - ColumnScore, TargetDetectionResult, ESCALATION_MARGIN
"""

from __future__ import annotations

# New recommended API
from src.target_detection.scoring import target_likelihood
from src.target_detection.escalation import resolve_target

# Rebuild the legacy API from scoring.py's internals.
# Import the signal functions and weights to construct score_targets / detect_target.
from src.target_detection.scoring import (
    SIGNAL_WEIGHTS,
    TARGET_NAME_VOCAB,
    _name_signal as _scoring_name_signal,
    _cardinality_signal as _scoring_cardinality_signal,
    _dtype_signal as _scoring_dtype_signal,
    _predictability_signal as _scoring_predictability_signal,
)

# We re-export the legacy types as well (they are still used by tests and UI).
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.config import RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Legacy constants (preserved for tests)
# ---------------------------------------------------------------------------
ESCALATION_MARGIN = 0.10


# ---------------------------------------------------------------------------
# Legacy dataclasses
# ---------------------------------------------------------------------------
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
# Legacy public functions (thin wrappers around scoring internals)
# ---------------------------------------------------------------------------
def score_targets(df: pd.DataFrame, sample: int = 800) -> List[ColumnScore]:
    """Score every column of `df` for target-likelihood. (Legacy API)"""
    if df is None or df.empty:
        return []
    n_rows = len(df)
    scores: List[ColumnScore] = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        name_sig = _scoring_name_signal(col)
        card_sig = _scoring_cardinality_signal(s, n_rows)
        dtype_sig = _scoring_dtype_signal(s)
        pred_sig = _scoring_predictability_signal(df, col, n_rows)
        total = (
            SIGNAL_WEIGHTS["name"] * name_sig
            + SIGNAL_WEIGHTS["cardinality"] * card_sig
            + SIGNAL_WEIGHTS["dtype"] * dtype_sig
            + SIGNAL_WEIGHTS["predictability"] * pred_sig
        )
        is_candidate = (not (n_unique == n_rows and n_rows > 8)
                        and n_unique >= 2
                        and not s.isna().all())
        scores.append(ColumnScore(
            name=col, total=total,
            name_signal=name_sig, cardinality_signal=card_sig,
            dtype_signal=dtype_sig, predictability_signal=pred_sig,
            dtype=str(s.dtype), n_unique=n_unique,
            is_target_candidate=is_candidate,
        ))
    scores.sort(key=lambda c: c.total, reverse=True)
    return scores


def detect_target(
    df: pd.DataFrame,
    override: Optional[str] = None,
    auto_select: bool = True,
) -> TargetDetectionResult:
    """Detect the target column. (Legacy API, preserved for tests.)

    For new code, prefer resolve_target() from .escalation.
    """
    if override is not None:
        if override not in df.columns:
            raise ValueError(
                f"Override target column '{override}' not found in data; "
                f"available: {list(df.columns)}"
            )
        logger.info("target overridden by caller: %s", override)
        return TargetDetectionResult(
            chosen_column=override, scores=[], ambiguous=False,
            top_candidates=[override],
            reason=f"Target column set by explicit override: '{override}'.",
            escalated=False,
        )

    scores = score_targets(df)
    candidates = [s for s in scores if s.is_target_candidate]
    if not candidates:
        raise ValueError(
            "No plausible target column found. Every column is either "
            "all-null, constant, or a unique ID. Pass an explicit target_col override."
        )

    top = candidates[0]
    if len(candidates) == 1:
        return TargetDetectionResult(
            chosen_column=top.name, scores=scores,
            ambiguous=False, top_candidates=[top.name],
            reason=f"Only one plausible target column found: '{top.name}'."
        )

    second = candidates[1]
    gap = top.total - second.total
    ambiguous = gap < ESCALATION_MARGIN
    top_two = [top.name, second.name]

    if ambiguous and not auto_select:
        return TargetDetectionResult(
            chosen_column=None, scores=scores, ambiguous=True,
            top_candidates=top_two,
            reason=(
                f"Target column is ambiguous: '{top.name}' (TLS={top.total:.3f}) vs "
                f"'{second.name}' (TLS={second.total:.3f}); gap={gap:.3f} < {ESCALATION_MARGIN}. "
                "Confirm the target before training."
            ),
            escalated=True,
        )

    chosen = top.name
    if ambiguous:
        reason = (
            f"Ambiguous target (top-2 gap {gap:.3f} < {ESCALATION_MARGIN}); "
            f"auto-selected highest TLS '{chosen}' ({top.total:.3f}). Confirm to accept."
        )
    else:
        reason = (
            f"Auto-selected '{chosen}' (TLS={top.total:.3f}); next best '{second.name}' "
            f"(TLS={second.total:.3f}), gap={gap:.3f}."
        )
    logger.info("target detected: %s (ambiguous=%s, escalated=%s)",
                chosen, ambiguous, ambiguous and not auto_select)
    return TargetDetectionResult(
        chosen_column=chosen, scores=scores, ambiguous=ambiguous,
        top_candidates=top_two, reason=reason,
        escalated=ambiguous and not auto_select,
    )


__all__ = [
    # New API (recommended)
    "target_likelihood",
    "resolve_target",
    # Legacy API
    "score_targets",
    "detect_target",
    "ColumnScore",
    "TargetDetectionResult",
    "ESCALATION_MARGIN",
]