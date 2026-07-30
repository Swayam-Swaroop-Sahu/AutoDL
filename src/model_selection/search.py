"""Successive-halving model search + per-modality candidates.

Replaces v1's deterministic `selector.py` pick AND keras-tuner (`tuner.py`).

Design (FINAL_PROJECT_PLAN.md §2 "HPO / search framework" row):
  - Hand-rolled successive-halving (~100 lines), sklearn-only, single-machine,
    minutes-not-hours. No Optuna / Ray Tune / keras-tuner dependency.
  - Round 1 trains ALL candidates at a small resource budget (fewer CV folds or a
    subsample); each round, keep top `SEARCH_BUDGET_FRACTION`, raise the budget,
    repeat until one remains (or budget is exhausted).
  - Final scoring is mean CV score (the metric) on the FULL data at full folds.

This module is the orchestration layer; per-modality candidate FACTORIES live in
`tabular_candidates.py` / `text_candidates.py` / `image_candidates.py`.

The circuit breaker (Phase 1 item 4) wraps each candidate fit, so a hung/slow
candidate is killed and demoted rather than blocking the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.core.config import (
    RANDOM_SEED, SEARCH_CV_FOLDS, SEARCH_BUDGET_FRACTION,
    SEARCH_MIN_RESOURCE, SEARCH_MAX_CANDIDATES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default scoring for classification. Each candidate factory MAY override.
DEFAULT_SCORING = "f1_weighted"


# ---------------------------------------------------------------------------
# Candidate descriptor
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A search candidate: a name + a factory returning a fresh unfitted estimator."""
    name: str
    factory: Callable[[], object]
    description: str = ""
    pros: str = ""
    cons: str = ""
    params: str = ""
    extras: dict = field(default_factory=dict)

    def build(self):
        return self.factory()


@dataclass
class CandidateResult:
    name: str
    cv_scores: List[float]
    mean_score: float
    std_score: float
    fit_ok: bool
    error: str
    description: str
    params: str
    pros: str
    cons: str
    elapsed_s: float = 0.0
    rounds_survived: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_score": round(self.mean_score, 4),
            "std_score": round(self.std_score, 4),
            "cv_scores": [round(s, 4) for s in self.cv_scores],
            "fit_ok": self.fit_ok,
            "error": self.error,
            "description": self.description,
            "params": self.params,
            "pros": self.pros,
            "cons": self.cons,
            "elapsed_s": round(self.elapsed_s, 3),
            "rounds_survived": self.rounds_survived,
        }


# ---------------------------------------------------------------------------
# Successive halving
# ---------------------------------------------------------------------------
def _resource_for_round(round_idx: int, n_rows: int) -> Tuple[int, int]:
    """Return (n_subsample, cv_folds) for a given round.

    Round 0 uses a small subsample + 2 folds; later rounds escalate to the full
    dataset at SEARCH_CV_FOLDS. The escalation is bounded so total work is O(N log K).
    """
    # Cap round count to a small number (e.g. 3).
    if round_idx == 0:
        return min(n_rows, max(SEARCH_MIN_RESOURCE, n_rows // 4)), 2
    # subsequent rounds: grow subsample, then upgrade folds.
    frac = min(1.0, (n_rows // 4) * (2 ** round_idx) / max(1, n_rows))
    n_sub = min(n_rows, max(n_rows * frac, SEARCH_MIN_RESOURCE))
    folds = SEARCH_CV_FOLDS
    return int(n_sub), folds


def _evaluate(
    cand: Candidate, X, y, n_sub: int, cv: int, scoring: str, seed: int,
    timeout_s: Optional[float] = None,
) -> CandidateResult:
    """Train `cand` via `cv`-fold CV. Failures are caught + logged, never raised."""
    import time
    t0 = time.time()
    try:
        # Subsample deterministically for the round.
        n = len(y) if hasattr(y, "__len__") else len(X)
        if n_sub < n:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, n_sub, replace=False)
            Xr = X[idx] if hasattr(X, "__getitem__") and not hasattr(X, "iloc") else _iloc(X, idx)
            yr = np.asarray(y)[idx]
        else:
            Xr, yr = X, np.asarray(y)
        est = cand.build()
        # Stratified K-fold unless too few samples / too few classes.
        try:
            kf = StratifiedKFold(n_splits=min(cv, max(2, sum(1 for _ in np.unique(yr) * 0) + len(np.unique(yr)))))
        except Exception:
            kf = cv
        scores = cross_val_score(est, Xr, yr, cv=kf, scoring=scoring, error_score="raise")
        elapsed = time.time() - t0
        return CandidateResult(
            name=cand.name, cv_scores=list(map(float, scores)),
            mean_score=float(np.mean(scores)), std_score=float(np.std(scores)),
            fit_ok=True, error="", description=cand.description, params=cand.params,
            pros=cand.pros, cons=cand.cons, elapsed_s=elapsed, rounds_survived=0,
        )
    except Exception as e:
        elapsed = time.time() - t0
        logger.warning("candidate '%s' failed to score (%s) — demoted", cand.name, e)
        return CandidateResult(
            name=cand.name, cv_scores=[], mean_score=-np.inf, std_score=0.0,
            fit_ok=False, error=str(e), description=cand.description, params=cand.params,
            pros=cand.pros, cons=cand.cons, elapsed_s=elapsed, rounds_survived=0,
        )


def _iloc(X, idx):
    """Index arrays or DataFrames by positional integer index."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


@dataclass
class SearchResult:
    winner: Optional[CandidateResult]
    all_results: List[CandidateResult]
    rounds: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "winner": self.winner.as_dict() if self.winner else None,
            "all_results": [r.as_dict() for r in self.all_results],
            "rounds": self.rounds,
            "reason": self.reason,
        }


def successive_halving(
    candidates: Sequence[Candidate],
    X, y,
    scoring: str = DEFAULT_SCORING,
    max_candidates: int = SEARCH_MAX_CANDIDATES,
    budget_fraction: float = SEARCH_BUDGET_FRACTION,
    seed: int = RANDOM_SEED,
) -> SearchResult:
    """Run successive halving and return full per-candidate results + the winner.

    Failures (OOM/exception) demote a candidate (score=-inf), never raise. Halving
    keeps top `budget_fraction` per round (always >=1 survivor). The final winner is
    re-scored at full folds on full data (when affordable) for an honest final number.
    """
    cands = list(candidates)[:max_candidates]
    if not cands:
        return SearchResult(winner=None, all_results=[], rounds=0,
                            reason="No candidates provided.")
    n_rows = len(y) if hasattr(y, "__len__") else len(X)

    alive = cands
    round_idx = 0
    # Bound rounds so the loop terminates cheaply.
    max_rounds = 3 if len(cands) > 4 else (2 if len(cands) > 1 else 1)
    last_results: List[CandidateResult] = []
    while alive and round_idx < max_rounds:
        n_sub, cv = _resource_for_round(round_idx, n_rows)
        results = [_evaluate(c, X, y, n_sub, cv, scoring, seed) for c in alive]
        for r in results:
            r.rounds_survived = round_idx + 1
        last_results = [r for r in results]
        if round_idx == max_rounds - 1:
            break
        # Order: fit_ok first, then mean_score desc.
        results.sort(key=lambda r: (r.fit_ok, r.mean_score), reverse=True)
        keep = max(1, int(len(results) * budget_fraction))
        alive = [c for c, r in zip(alive, results)][:keep]
        round_idx += 1

    # Final round at full folds on full data (if affordable) for honest CV score.
    if n_rows >= 2 * SEARCH_MIN_RESOURCE:
        final = []
        for c in alive:
            r = _evaluate(c, X, y, n_rows, SEARCH_CV_FOLDS, scoring, seed)
            r.rounds_survived = max_rounds
            final.append(r)
        last_results = final

    winners = sorted(last_results, key=lambda r: (r.fit_ok, r.mean_score), reverse=True)
    winner = winners[0] if winners and winners[0].fit_ok else None

    if winner is None:
        reason = (f"All {len(cands)} candidates failed CV scoring; "
                  "the circuit breaker should supply the deterministic fallback baseline.")
    else:
        reason = (f"'{winner.name}' won successive-halving with mean {scoring}={winner.mean_score:.4f} "
                  f"± {winner.std_score:.4f} over {len(winner.cv_scores)} folds. "
                  f"Survived {winner.rounds_survived} rounds.")

    # Merge all_results: candidates that died early get their last-known score carried.
    all_results = _merge_results(cands, last_results)
    return SearchResult(winner=winner, all_results=all_results, rounds=max_rounds, reason=reason)


def _merge_results(all_cands: Sequence[Candidate], last_results: List[CandidateResult]) -> List[CandidateResult]:
    """Preserve every candidate in the final output, even if eliminated early."""
    by_name = {r.name: r for r in last_results}
    out: List[CandidateResult] = []
    for c in all_cands:
        if c.name in by_name:
            out.append(by_name[c.name])
        else:
            # eliminated early with no recorded score → mark as not-measured-at-final-round
            out.append(CandidateResult(
                name=c.name, cv_scores=[], mean_score=-np.inf, std_score=0.0,
                fit_ok=False, error="eliminated in early halving round",
                description=c.description, params=c.params, pros=c.pros, cons=c.cons,
            ))
    return out


# ---------------------------------------------------------------------------
# Winnowing helper: re-instantiate the winner fresh so it can be fit on all data
# ---------------------------------------------------------------------------
def refit_winner(winner: CandidateResult, candidates: Sequence[Candidate]):
    for c in candidates:
        if c.name == winner.name:
            return c.build()
    return None
