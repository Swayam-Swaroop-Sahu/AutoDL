"""Optional binary threshold optimization (Phase 1d).

Replaces the hardcoded `>0.5` literals for binary problems. For multiclass,
classes are decoded via `argmax` and threshold optimization is a no-op.

Strategies supported (Phase 1d):
  - "youden": maximize (TPR - FPR) via ROC sweep
  - "f1":     maximize F1 across candidate thresholds

Failures degrade to `DEFAULT_BINARY_THRESHOLD` with a logged warning — never raise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_curve

from src.core.config import DEFAULT_BINARY_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def optimize_threshold(
    y_true,
    y_score_positive: np.ndarray,
    strategy: str = "youden",
) -> Tuple[float, float, float]:
    """Pick the binary decision threshold.

    Args:
        y_true: ground-truth labels (0/1).
        y_score_positive: predicted probability (or score) for the positive class.
        strategy: "youden" (default) → max(TPR - FPR), "f1" → max F1.

    Returns:
        (threshold, score_metric, roc_auc). On failure, returns
        (DEFAULT_BINARY_THRESHOLD, 0.0, 0.5).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score_positive).astype(float).ravel()

    if y_true.size == 0 or y_score.size == 0:
        logger.warning(
            "optimize_threshold received empty input; using default %.2f",
            DEFAULT_BINARY_THRESHOLD,
        )
        return DEFAULT_BINARY_THRESHOLD, 0.0, 0.5

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "only one class present in y_true; using default threshold %.2f",
            DEFAULT_BINARY_THRESHOLD,
        )
        return DEFAULT_BINARY_THRESHOLD, 0.0, 0.5

    strategy = strategy.lower().strip()
    if strategy not in ("youden", "f1"):
        logger.warning("unknown strategy '%s'; falling back to 'youden'", strategy)
        strategy = "youden"

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # Compute AUC (numpy 2.0+ uses trapezoid; trapz kept as fallback)
        try:
            auc = float(np.trapezoid(tpr, fpr))
        except AttributeError:
            auc = float(np.trapz(tpr, fpr))

        if strategy == "f1":
            # Sweep candidate thresholds; for binary, midpoints between unique scores work well
            best_threshold = float(DEFAULT_BINARY_THRESHOLD)
            best_f1 = -1.0
            for thr in np.unique(np.concatenate([[0.0, 1.0], thresholds])):
                y_pred_at_thr = (y_score >= thr).astype(int)
                if len(np.unique(y_pred_at_thr)) < 2:
                    continue
                f1 = f1_score(y_true, y_pred_at_thr, zero_division=0)
                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_threshold = float(thr)
            best_threshold = float(np.clip(best_threshold, 0.0, 1.0))
            logger.info(
                "threshold optimized via F1: %.4f (F1=%.4f, AUC=%.4f)",
                best_threshold, best_f1, auc,
            )
            return best_threshold, best_f1, auc

        # Default: Youden's J
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_threshold = float(thresholds[best_idx])
        best_threshold = float(np.clip(best_threshold, 0.0, 1.0))
        logger.info(
            "threshold optimized via Youden's J: %.4f (J=%.4f, AUC=%.4f)",
            best_threshold, j[best_idx], auc,
        )
        return best_threshold, float(j[best_idx]), auc
    except Exception as e:
        logger.warning(
            "threshold optimization failed (%s); using default %.2f",
            e, DEFAULT_BINARY_THRESHOLD,
        )
        return DEFAULT_BINARY_THRESHOLD, 0.0, 0.5


def apply_threshold(proba, threshold: float, num_classes: int):
    """Apply the unified decode: binary→thresholded, multiclass→argmax.

    `proba` shape may be (N,) for binary-ish single score, (N, 2) for binary
    proba, or (N, K). Always returns integer class indices.
    """
    proba = np.asarray(proba)
    if num_classes > 2:
        return np.argmax(proba, axis=1)
    # binary
    if proba.ndim == 1:
        return (proba > threshold).astype(int)
    if proba.ndim == 2 and proba.shape[1] == 1:
        return (proba.ravel() > threshold).astype(int)
    # (N, 2): use positive-class column
    return (proba[:, 1] > threshold).astype(int)
