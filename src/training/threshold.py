"""Optional binary threshold optimization (Phase 1, §2 "Binary vs multiclass").

Replaces the hardcoded `>0.5` literals (audit §4 item #4) for binary problems.
For multiclass, classes are decoded via `argmax` and threshold optimization is a
no-op. For binary, we sweep the ROC and pick the threshold maximizing Youden's J
(sensitivity + specificity - 1). Failures degrade to `DEFAULT_BINARY_THRESHOLD`
with a logged warning — never raise.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve

from src.core.config import DEFAULT_BINARY_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def optimize_threshold(y_true, y_score_positive: np.ndarray) -> Tuple[float, float, float]:
    """Pick the binary decision threshold via Youden's J.

    Args:
        y_true: ground-truth labels (0/1).
        y_score_positive: predicted probability (or score) for the positive class.
    Returns:
        (threshold, youdens_j, roc_auc). On failure, returns (DEFAULT_BINARY_THRESHOLD, 0.0, 0.5).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score_positive).astype(float).ravel()
    if y_true.size == 0 or y_score.size == 0:
        logger.warning("optimize_threshold received empty input; using default %.2f", DEFAULT_BINARY_THRESHOLD)
        return DEFAULT_BINARY_THRESHOLD, 0.0, 0.5
    # need both classes present to compute a meaningful ROC
    if len(np.unique(y_true)) < 2:
        logger.warning("only one class present in y_true; using default threshold %.2f", DEFAULT_BINARY_THRESHOLD)
        return DEFAULT_BINARY_THRESHOLD, 0.0, 0.5
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        threshold = float(thresholds[best_idx])
        auc = float(np.trapz(tpr, fpr))
        threshold = float(np.clip(threshold, 0.0, 1.0))
        logger.info("threshold optimized via Youden's J: %.4f (J=%.4f, AUC=%.4f)",
                    threshold, j[best_idx], auc)
        return threshold, float(j[best_idx]), auc
    except Exception as e:
        logger.warning("threshold optimization failed (%s); using default %.2f", e, DEFAULT_BINARY_THRESHOLD)
        return DEFAULT_BINARY_THRESHOLD, 0.0, 0.5


def apply_threshold(proba, threshold: float, num_classes: int):
    """Apply the unified decode: binary→thresholded, multiclass→argmax.

    `proba` shape is (N,) for binary-ish single score, (N, 2) for binary proba, or (N, K).
    Always returns integer class indices.
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
