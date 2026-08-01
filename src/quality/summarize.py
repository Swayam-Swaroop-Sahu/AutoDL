"""Quality summary — aggregates all gate results into one dict.

Provides `summarize_quality(df, target_col)` that returns a single dict with
keys: "warnings" (list of dicts), "passed" (bool), "summary" (one-line English).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.quality.gates import (
    detect_leakage,
    detect_id_columns,
    detect_constant_columns,
    detect_imbalance,
)


def summarize_quality(
    df: pd.DataFrame, target_col: Optional[str] = None,
) -> Dict:
    """Run all quality gates and return a summary dict."""
    all_warnings: List[Dict] = []
    if target_col and target_col in df.columns:
        all_warnings.extend(detect_leakage(df, target_col))
        all_warnings.extend(detect_imbalance(df, target_col))
    all_warnings.extend(detect_id_columns(df))
    all_warnings.extend(detect_constant_columns(df))

    # Deduplicate (same column + same issue)
    seen = set()
    unique = []
    for w in all_warnings:
        key = (w.get("column", w.get("class", "")), w.get("issue", ""))
        if key not in seen:
            seen.add(key)
            unique.append(w)

    passed = len(unique) == 0
    if passed:
        summary = "All quality checks passed. No data issues detected."
    else:
        issue_types = set(w["issue"] for w in unique)
        nice = {
            "feature_leakage": "Leakage",
            "id_column": "ID columns",
            "constant_column": "Constant columns",
            "class_imbalance": "Imbalanced classes",
        }
        names = [nice.get(t, t) for t in issue_types]
        summary = f"Found {len(unique)} issue(s): {', '.join(names)}."

    return {
        "warnings": unique,
        "passed": passed,
        "summary": summary,
    }