"""Target detection escalation logic.

Provides `resolve_target(df, target_col=None) → (col_or_list, status)` that
encapsulates the full decision flow:

  - Explicit override     → ("col_name",  "override")
  - Strong auto-detect    → ("best_col",  "strong_auto")   TLS > 0.80 & gap > 0.20
  - Weak auto-detect      → ("best_col",  "weak_auto")     TLS ≥ 0.40 but not strong
  - Ambiguous candidates  → (candidates, "human_required")  top scores too close
  - Not classification    → (None,        "not_classification")  best TLS < 0.40

Callers should:
  - Switch on status: "override", "strong_auto", "weak_auto" → use col_str directly
  - "human_required" → raise ValueError with ranked candidates
  - "not_classification" → raise ValueError
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import pandas as pd

from src.target_detection.scoring import target_likelihood
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Escalation thresholds
STRONG_AUTO_THRESHOLD = 0.80    # top TLS must be > this for strong_auto
STRONG_AUTO_GAP = 0.20          # gap between 1st and 2nd must be > this for strong_auto
AMBIGUITY_GAP = 0.10            # gap below this between 1st and 2nd → human_required
NOT_CLASSIFICATION_THRESHOLD = 0.40  # top TLS < this → not_classification


def resolve_target(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Tuple[Optional[Union[str, List[str]]], str]:
    """Resolve the target column for `df`.

    Args:
        df: Input DataFrame.
        target_col: Explicit override from CLI/UI. Short-circuits all scoring.

    Returns:
        (column_or_list, status) tuple:
          - ("col", "override")        — explicit override
          - ("col", "strong_auto")     — clear winner, high confidence
          - ("col", "weak_auto")       — acceptable but not top-tier confidence
          - (["col1","col2"], "human_required") — ambiguous, user must pick
          - (None, "not_classification")       — no viable target found

    Raises:
        ValueError: if explicit target_col is not in df columns.
    """
    # --- override path ---
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"Override target column '{target_col}' not found in data; "
                f"available columns: {list(df.columns)}"
            )
        logger.info("resolve_target: override → '%s'", target_col)
        return target_col, "override"

    # --- score every column ---
    scores = {}
    for col in df.columns:
        scores[col] = target_likelihood(df, col)

    # sort by score descending, filter out zero-score columns
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [(c, s) for c, s in ranked if s > 0.0]

    if not candidates:
        logger.warning("resolve_target: no candidates with TLS > 0")
        return None, "not_classification"

    top_name, top_score = candidates[0]

    # --- not_classification ---
    if top_score < NOT_CLASSIFICATION_THRESHOLD:
        logger.warning(
            "resolve_target: top score %.3f < %.2f → not_classification",
            top_score, NOT_CLASSIFICATION_THRESHOLD,
        )
        return None, "not_classification"

    # --- single candidate ---
    if len(candidates) == 1:
        logger.info("resolve_target: single candidate '%s' (%.3f) → weak_auto",
                    top_name, top_score)
        return top_name, "weak_auto"

    second_name, second_score = candidates[1]
    gap = top_score - second_score

    # --- strong_auto ---
    if top_score > STRONG_AUTO_THRESHOLD and gap > STRONG_AUTO_GAP:
        logger.info("resolve_target: strong_auto → '%s' (TLS=%.3f, gap=%.3f)",
                    top_name, top_score, gap)
        return top_name, "strong_auto"

    # --- human_required (ambiguous) ---
    if gap < AMBIGUITY_GAP:
        top_candidates = [c for c, _ in candidates if c in (top_name, second_name)]
        logger.warning(
            "resolve_target: ambiguous (gap=%.3f < %.2f) → human_required; "
            "candidates: %s", gap, AMBIGUITY_GAP, top_candidates,
        )
        return top_candidates, "human_required"

    # --- weak_auto (fall-through) ---
    logger.info("resolve_target: weak_auto → '%s' (TLS=%.3f, gap=%.3f)",
                top_name, top_score, gap)
    return top_name, "weak_auto"