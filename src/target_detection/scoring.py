"""Target detection — simple name + cardinality ranking.

Philosophy:
  - Name catches obvious targets (columns named target, label, is_churned, etc.)
  - Cardinality catches structural signals (binary = very likely target, unique-per-row = ID)
  - The user ALWAYS confirms. No auto-pick, no "weak_auto" fallback.
  - No dtype, no predictability signals — they add complexity without value.

Provides:
  - target_likelihood(df, col) → float 0.0–1.0  (per-column score)
  - rank_target_candidates(df) → List[dict]      (all columns ranked)
"""

from __future__ import annotations

from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Name heuristic constants
# ---------------------------------------------------------------------------
POSITIVE_NAME_PATTERNS = (
    "target", "label", "y", "class", "category", "type",
    "status", "outcome", "flag", "is_", "has_", "will_",
    "survived", "churn", "fraud",
)

NEGATIVE_NAME_PATTERNS = (
    "id", "uuid", "guid", "pk", "sk", "index", "timestamp",
    "date", "time", "created", "updated", "url", "email",
    "phone", "first", "last",
)

# ---------------------------------------------------------------------------
# PassengerId is the ugliest ID-column naming convention.
# It trips the "id" substring matcher but only because NEGATIVE_PATTERNS
# match against the whole lowercase name.  That is fine — anything with "id"
# in its name is not a classification target.
# ---------------------------------------------------------------------------


def target_likelihood(df: pd.DataFrame, col: str) -> float:
    """Score how "target-like" column `col` is.  Returns float 0.0–1.0.

    Used ONLY for ranking suggestions — the user always picks.
    """
    n_rows = len(df)
    series = df[col]
    n_unique = int(series.nunique(dropna=True))
    name_lower = col.lower().strip()

    # ----- Name signal (0.5 weight) ---------------------------------------
    name_score = 1.0 if any(p in name_lower for p in POSITIVE_NAME_PATTERNS) else 0.0
    if any(n in name_lower for n in NEGATIVE_NAME_PATTERNS):
        name_score -= 0.7  # strong penalty for ID / temporal / name columns
    name_score = max(0.0, name_score)

    # ----- Cardinality signal (0.5 weight) --------------------------------
    if n_unique == n_rows and n_rows > 8:
        card_score = 0.0       # unique per row → ID, not target
    elif n_unique == 2:
        card_score = 1.0       # binary → very strong target signal
    elif 2 < n_unique <= min(50, int(0.05 * n_rows)):
        card_score = 0.8       # low-card categorical
    elif n_unique > 0.5 * n_rows:
        card_score = 0.1       # high cardinality → probably continuous feature
    else:
        card_score = 0.3       # medium cardinality → ambiguous

    return round(name_score * 0.5 + card_score * 0.5, 4)


def rank_target_candidates(df: pd.DataFrame) -> List[dict]:
    """Return all columns ranked by target_likelihood, highest first.

    Each entry:
        {
            "col": str,
            "score": float,
            "name_score": float,
            "card_score": float,
            "n_unique": int,
        }
    """
    candidates = []
    for col in df.columns:
        name_lower = col.lower().strip()
        name_score = 1.0 if any(p in name_lower for p in POSITIVE_NAME_PATTERNS) else 0.0
        if any(n in name_lower for n in NEGATIVE_NAME_PATTERNS):
            name_score -= 0.7
        name_score = max(0.0, name_score)

        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        n_rows = len(df)

        if n_unique == n_rows and n_rows > 8:
            card_score = 0.0
        elif n_unique == 2:
            card_score = 1.0
        elif 2 < n_unique <= min(50, int(0.05 * n_rows)):
            card_score = 0.8
        elif n_unique > 0.5 * n_rows:
            card_score = 0.1
        else:
            card_score = 0.3

        score = name_score * 0.5 + card_score * 0.5
        candidates.append({
            "col": col,
            "score": round(score, 4),
            "name_score": round(name_score, 4),
            "card_score": round(card_score, 4),
            "n_unique": n_unique,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates