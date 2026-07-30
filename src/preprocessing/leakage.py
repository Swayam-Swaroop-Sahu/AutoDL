"""Leakage detection for tabular datasets (BUGFIX Phase 1e item 14).

Detects features that are essentially duplicates of the target (high
correlation / Cramér's V). Such features will inflate CV scores but
generalize poorly. We log WARNING; the pipeline does not auto-drop because
in some real-world cases the user genuinely wants to use them as features.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _cramers_v(confusion_matrix: np.ndarray) -> float:
    """Compute Cramér's V from a contingency table."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    if n == 0:
        return 0.0
    r, k = confusion_matrix.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def detect_leakage(
    X: pd.DataFrame,
    y_raw,
    numeric_threshold: float = 0.95,
    categorical_threshold: float = 0.95,
) -> List[dict]:
    """Return a list of leakage flags for features highly correlated with target.

    Args:
        X: feature DataFrame.
        y_raw: raw target Series / array.
        numeric_threshold: warn if |Pearson| > this.
        categorical_threshold: warn if Cramér's V > this.

    Returns:
        list of dicts: [{"feature": "...", "metric": "pearson" or "cramers_v",
                          "value": float, "threshold": float}]
    """
    y_series = pd.Series(y_raw)
    flags: List[dict] = []

    for col in X.columns:
        s = X[col]
        try:
            if pd.api.types.is_numeric_dtype(s):
                # Pearson correlation — drop NaN pairs
                pairs = pd.concat([s, y_series], axis=1).dropna()
                if pairs.shape[0] < 5 or pairs.iloc[:, 1].nunique() < 2:
                    continue
                # Target must be numeric OR mapped to integers
                target_numeric = pd.to_numeric(pairs.iloc[:, 1], errors="coerce")
                if target_numeric.isna().all():
                    continue
                r = float(np.corrcoef(pairs.iloc[:, 0].astype(float),
                                      target_numeric.astype(float))[0, 1])
                if abs(r) > numeric_threshold:
                    flags.append({
                        "feature": col,
                        "metric": "pearson",
                        "value": round(r, 4),
                        "threshold": numeric_threshold,
                    })
            else:
                # Categorical: contingency table + Cramér's V
                pairs = pd.concat([s.astype(str), y_series.astype(str)], axis=1)
                pairs.columns = ["f", "t"]
                pairs = pairs.dropna()
                if pairs.shape[0] < 5:
                    continue
                ct = pd.crosstab(pairs["f"], pairs["t"]).values
                if ct.size == 0:
                    continue
                v = _cramers_v(ct)
                if v > categorical_threshold:
                    flags.append({
                        "feature": col,
                        "metric": "cramers_v",
                        "value": round(v, 4),
                        "threshold": categorical_threshold,
                    })
        except Exception as e:
            logger.debug("leakage detection skipped for column '%s': %s", col, e)
            continue

    if flags:
        logger.warning(
            "BUGFIX Phase 1e item 14: %d potential target-leakage feature(s) detected: %s",
            len(flags), flags,
        )
    return flags
