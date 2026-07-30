"""Successive-halving model search.

Replaces v1's deterministic `selector.py` pick AND keras-tuner (`tuner.py`).

Design (Phase 1b spec):
  - 3 stages: fidelity 0.1 (25% time), 0.5 (35% time), 1.0 (40% time)
  - 3-fold stratified CV per trial
  - Promote top 50% each stage
  - Timeout: if stage budget exceeded, stop starting new trials
  - Crash resilience: one bad candidate scores 0.0, search continues
  - Returns (best_cfg, best_score, all_results) where all_results is list of dicts

Per-modality candidate FACTORIES live in:
  - tabular_candidates.py
  - text_candidates.py
  - image_candidates.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.core.config import RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SCORING = "balanced_accuracy"
DEFAULT_TIME_BUDGET_SEC = 600  # 10 min
STAGE_FIDELITIES = [0.1, 0.5, 1.0]
STAGE_TIME_FRACTIONS = [0.25, 0.35, 0.40]
CV_FOLDS = 3
PROMOTION_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Candidate descriptor
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A search candidate: name + factory returning a fresh unfitted estimator."""
    name: str
    factory: Callable[[], object]
    description: str = ""
    pros: str = ""
    cons: str = ""
    params: str = ""
    extras: dict = field(default_factory=dict)

    def build(self):
        return self.factory()


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------
def successive_halving_search(
    candidates: Sequence[Candidate],
    X: np.ndarray,
    y: np.ndarray,
    time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC,
    scoring: str = DEFAULT_SCORING,
    cv: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict, float, List[Dict]]:
    """Run 3-stage successive-halving model search.

    Args:
        candidates: List of Candidate descriptors with factory_score callables.
        X, y: Training data (numpy arrays).
        time_budget_sec: Total time budget in seconds (default 600 = 10 min).
        scoring: sklearn-compatible scoring string (default "balanced_accuracy").
        cv: Number of CV folds per trial (default 3).
        seed: Random seed for reproducibility.

    Returns:
        (best_cfg, best_score, all_results) where:
          - best_cfg: dict with name, factory, params of the winner
          - best_score: float, winner's final-stage mean CV score
          - all_results: list of dicts with full per-candidate results
    """
    if not candidates:
        return {}, -1.0, []

    start_time = time.time()
    n = len(y)
    n_classes = len(np.unique(y))
    # Adjust CV based on data: floor is 2, ceiling is min of cv and min-class-size.
    actual_cv = min(cv, max(2, min(np.bincount(y.astype(np.int64))) if n_classes > 1 else 2))

    # --- Fidelity sizes ---
    fid_samples = [
        max(2, int(n * f)) for f in STAGE_FIDELITIES
    ]
    # Ensure fidelity 2 is larger than fidelity 1, etc.
    fid_samples[1] = max(fid_samples[1], fid_samples[0] + 1)
    fid_samples[2] = max(fid_samples[2], fid_samples[1] + 1)

    all_results: List[Dict] = []
    alive_indices = list(range(len(candidates)))
    best_cfg: Dict = {}
    best_score: float = -1.0

    for stage_idx in range(3):
        stage_start = time.time()
        stage_budget = STAGE_TIME_FRACTIONS[stage_idx] * time_budget_sec
        n_sub = fid_samples[stage_idx]

        logger.info(
            "Stage %d/%d: fidelity=%.1f, n_sub=%d, n_candidates=%d, budget=%.1fs",
            stage_idx + 1, 3, STAGE_FIDELITIES[stage_idx], n_sub,
            len(alive_indices), stage_budget,
        )

        rng = np.random.RandomState(seed + stage_idx)
        idx = rng.choice(n, n_sub, replace=False)
        X_sub = X[idx] if not hasattr(X, "iloc") else X.iloc[idx]
        y_sub = np.asarray(y)[idx]

        stage_results: List[Tuple[int, float, Dict]] = []

        for cand_idx in alive_indices:
            # Budget check before each new trial
            elapsed_stage = time.time() - stage_start
            if elapsed_stage > stage_budget:
                logger.warning(
                    "Stage %d budget exhausted (%.1fs > %.1fs); stopping new trials.",
                    stage_idx + 1, elapsed_stage, stage_budget,
                )
                break

            cand = candidates[cand_idx]
            try:
                est = cand.build()
                kf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=seed + stage_idx)
                scores = cross_val_score(
                    est, X_sub, y_sub, cv=kf, scoring=scoring, error_score="raise",
                )
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))
                result = {
                    "name": cand.name,
                    "stage": stage_idx + 1,
                    "score": round(mean_score, 4),
                    "std": round(std_score, 4),
                    "cv_scores": [round(float(s), 4) for s in scores],
                    "fit_ok": True,
                    "error": "",
                    "description": cand.description,
                    "params": cand.params,
                    "pros": cand.pros,
                    "cons": cand.cons,
                    "n_sub": n_sub,
                    "fidelity": STAGE_FIDELITIES[stage_idx],
                }
            except Exception as exc:
                logger.warning(
                    "Candidate '%s' crashed in stage %d (%s); scoring 0.0",
                    cand.name, stage_idx + 1, exc,
                )
                result = {
                    "name": cand.name,
                    "stage": stage_idx + 1,
                    "score": 0.0,
                    "std": 0.0,
                    "cv_scores": [],
                    "fit_ok": False,
                    "error": str(exc),
                    "description": cand.description,
                    "params": cand.params,
                    "pros": cand.pros,
                    "cons": cand.cons,
                    "n_sub": n_sub,
                    "fidelity": STAGE_FIDELITIES[stage_idx],
                }
            stage_results.append((cand_idx, result["score"], result))
            all_results.append(result)

        # Sort descending by score
        stage_results.sort(key=lambda t: t[1], reverse=True)
        keep = max(1, int(len(stage_results) * PROMOTION_FRACTION))
        alive_indices = [t[0] for t in stage_results[:keep]]

        logger.info(
            "Stage %d done: %d evaluated, %d promoted (cutoff %.4f), elapsed %.1fs",
            stage_idx + 1, len(stage_results), len(alive_indices),
            stage_results[keep - 1][1] if keep <= len(stage_results) else
            (stage_results[-1][1] if stage_results else -1.0),
            time.time() - stage_start,
        )

    # ---- Winner is top survivor from last stage ----
    final_stage_results = [
        r for r in all_results
        if any(candidates[i].name == r["name"] for i in alive_indices)
    ]
    # Re-score living candidates at full fidelity in final stage results
    candidates_scored = [
        r for r in all_results if r["stage"] == 3 and r["fit_ok"]
    ]
    candidates_scored.sort(key=lambda r: r["score"], reverse=True)

    if candidates_scored:
        winner = candidates_scored[0]
        best_cfg = {
            "name": winner["name"],
            "factory": next(
                (c for c in candidates if c.name == winner["name"]), None
            ),
            "params": winner["params"],
        }
        best_score = winner["score"]
        logger.info(
            "Search winner: '%s' best_score=%.4f (%d candidates tested)",
            winner["name"], best_score, len(all_results),
        )
    else:
        logger.warning("No candidate survived all stages with fit_ok=True")
        best_cfg, best_score = {}, -1.0

    total_elapsed = time.time() - start_time
    logger.info(
        "Search finished in %.1fs (budget=%.1fs), winner=%s score=%.4f",
        total_elapsed, time_budget_sec,
        best_cfg.get("name", "none"), best_score,
    )

    return best_cfg, best_score, all_results