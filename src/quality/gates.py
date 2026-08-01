"""Data quality gates for AutoDL (F1 trust layer).

Pure functions that examine a DataFrame and return structured warnings.
None of them modify data or raise exceptions — they return dicts that
the caller (UI / report / CLI) can present however it wants.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds
_PEARSON_R_THRESHOLD = 0.95
_CRAMERS_V_THRESHOLD = 0.95
_IMBALANCE_THRESHOLD = 0.05  # classes with <5% are flagged


def _cramers_v(ct: np.ndarray) -> float:
    """Cramer's V from a contingency table."""
    chi2 = chi2_contingency(ct, lambda_="")[0]  # silence warning
    n = ct.sum()
    if n == 0:
        return 0.0
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


# ---------------------------------------------------------------------------
# Individual gate functions
# ---------------------------------------------------------------------------
def detect_leakage(
    df: pd.DataFrame, target_col: str,
) -> List[Dict]:
    """Return a list of leakage warnings for features highly correlated with target.

    Numeric features: |Pearson r| > 0.95.
    Categorical features: Cramer's V > 0.95.
    """
    warnings: List[Dict] = []
    y = df[target_col]
    X = df.drop(columns=[target_col])

    for col in X.columns:
        try:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                pairs = pd.concat([s, y], axis=1).dropna()
                if pairs.shape[0] < 5 or pairs.iloc[:, 1].nunique() < 2:
                    continue
                r = float(np.corrcoef(
                    pairs.iloc[:, 0].astype(float),
                    pd.to_numeric(pairs.iloc[:, 1], errors="coerce").astype(float),
                )[0, 1])
                if abs(r) > _PEARSON_R_THRESHOLD:
                    warnings.append({
                        "column": col,
                        "issue": "feature_leakage",
                        "detail": f"Pearson r={r:.4f} with target (threshold {_PEARSON_R_THRESHOLD})",
                    })
            else:
                pairs = pd.concat([s.astype(str), y.astype(str)], axis=1).dropna()
                pairs.columns = ["f", "t"]
                if pairs.shape[0] < 5:
                    continue
                ct = pd.crosstab(pairs["f"], pairs["t"]).values
                if ct.size == 0:
                    continue
                v = _cramers_v(ct)
                if v > _CRAMERS_V_THRESHOLD:
                    warnings.append({
                        "column": col,
                        "issue": "feature_leakage",
                        "detail": f"Cramers V={v:.4f} with target (too high)",
                    })
        except Exception:
            continue

    return warnings


def detect_id_columns(df: pd.DataFrame) -> List[Dict]:
    """Flag columns where n_unique == n_rows AND dtype is int-like / string.

    Float columns with high precision are often features, not IDs,
    even if they are unique per row.
    """
    warnings: List[Dict] = []
    n = df.shape[0]
    if n == 0:
        return warnings
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique(dropna=True)
        if n_unique == n and n > 8:
            dtype_str = str(s.dtype)
            # Only flag if int-like or object/string (IDs), not float (features)
            if dtype_str.startswith("int") or dtype_str in ("object", "string", "category"):
                warnings.append({
                    "column": col,
                    "issue": "id_column",
                    "detail": (
                        f"Every row has a unique value & dtype is {dtype_str} — "
                        "this is likely an ID, not a feature."
                    ),
                })
    return warnings


def detect_constant_columns(df: pd.DataFrame) -> List[Dict]:
    """Flag columns with only 1 unique value."""
    warnings: List[Dict] = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            warnings.append({
                "column": col,
                "issue": "constant_column",
                "detail": "Column has only 1 unique value — adds no information",
            })
    return warnings


def detect_imbalance(
    df: pd.DataFrame, target_col: str,
) -> List[Dict]:
    """Return warnings for rare classes (<5% of rows)."""
    warnings: List[Dict] = []
    y = df[target_col].dropna()
    if y.empty:
        return warnings
    total = len(y)
    for label, count in y.value_counts().items():
        frac = count / total
        if frac < _IMBALANCE_THRESHOLD:
            warnings.append({
                "class": str(label),
                "issue": "class_imbalance",
                "detail": (
                    f"'{label}' has {count} samples ({frac:.1%}) — "
                    f"fewer than {_IMBALANCE_THRESHOLD:.0%}. "
                    "This class may be hard to learn."
                ),
            })
    return warnings