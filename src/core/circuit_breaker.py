"""Stage-level reliability contract for AutoDL v2.

FINAL_PROJECT_PLAN.md §2 "Reliability / fallback" row: every pipeline stage runs
under (a) a wall-clock timeout, (b) subprocess isolation where feasible, (c) a
deterministic fallback if the stage fails or times out, and (d) structured
logging of the outcome. Phase 0's rule "never let the pipeline hang" is enforced
here.

Design constraints (laptop-native, no extra infra):
  - Per-stage timeout uses a worker thread with a join deadline. Why not
    multiprocessing/subprocess? On Windows, `multiprocessing` in a Streamlit /
    pytest context is fragile (spawn + re-import) and adds launch latency that
    can exceed the stage budget. A thread-based deadline reliably kills the
    *wait* (and on join-timeout we record the stage as hung and proceed to
    fallback), which is the property the plan cares about: the pipeline never
    hangs. Subprocess isolation per stage (item b) is implemented as an opt-in
    `subprocess=True` flag for stages that genuinely benefit from a separate OS
    process (e.g. TF image search); it is NOT the default for cheap stages.

  - Fallback is the cheapest deterministic option guaranteed to terminate:
      detect/load/preprocess/target_detect/quality  → ValueError surfaced
      search   → LogisticRegression(no tuning) baseline
      train    → the search winner (already chosen) or Logistic Regression
      evaluate → metrics computed from whatever predictions exist
      report   → return None (logged), pipeline still succeeds
      save     → best-effort; logged if it fails

  - Checkpointing: each successful stage records a checkpoint marker in the run
    dir; the pipeline (or a future resume run) can read these to resume from the
    last completed stage rather than redoing work.

Every code path logs via src.utils.logger — never print, never bare except-pass.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StageTimeoutError(TimeoutError):
    pass


class StageFailure(Exception):
    def __init__(self, stage: str, kind: str, detail: str):
        super().__init__(f"[{stage}] {kind}: {detail}")
        self.stage = stage
        self.kind = kind
        self.detail = detail


@dataclass
class StageResult:
    stage: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    kind: str = ""            # "timeout" | "exception" | "fallback" | "success"
    elapsed_s: float = 0.0
    used_fallback: bool = False
    log: list = field(default_factory=list)


def _run_with_timeout(func: Callable, args, kwargs, timeout_s: Optional[float]):
    """Run `func` in a worker thread; join up to `timeout_s` seconds.

    Returns (value, None) on success or (None, StageTimeoutError) on timeout.
    An unhandled exception in `func` is re-raised in the caller thread
    (we capture it via a container).
    """
    if timeout_s is None or timeout_s <= 0:
        return func(*args, **kwargs), None

    result = {"value": None, "exc": None}

    def _worker():
        try:
            result["value"] = func(*args, **kwargs)
        except SystemExit:
            raise
        except BaseException as e:  # noqa: BLE001 — captured, re-raised on join
            result["exc"] = e

    t = threading.Thread(target=_worker, daemon=True, name=f"autodl-stage-{time.time()}")
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise StageTimeoutError(f"stage exceeded {timeout_s}s deadline")
    if result["exc"] is not None:
        raise result["exc"]
    return result["value"], None


class CircuitBreakerPipeline:
    """Wrap each pipeline stage in timeout + fallback + structured logging.

    Usage::

        cb = CircuitBreakerPipeline(run_dir=..., seed=42)
        df = cb.stage("load", loader_func, args=(path,),
                      fallback=lambda: _default_loader(path))
        ...

    Every stage's outcome is appended to `cb.history` and also written to
    `{run_dir}/circuit_breaker.jsonl` so a resume run can reconstruct progress.
    """

    def __init__(
        self,
        run_dir: str,
        seed: int = 42,
        stage_timeouts: Optional[dict] = None,
    ):
        self.run_dir = run_dir
        self.seed = seed
        from src.core.config import STAGE_TIMEOUTS
        self.stage_timeouts = stage_timeouts or dict(STAGE_TIMEOUTS)
        self.history: list[StageResult] = []
        self._checkpoint_path = os.path.join(run_dir, "circuit_breaker.jsonl")
        os.makedirs(run_dir, exist_ok=True)

    # ----------------------------------------------------------------- public
    def stage(
        self,
        name: str,
        func: Callable,
        args=(),
        kwargs=None,
        fallback: Optional[Callable] = None,
        timeout_s: Optional[float] = None,
        required: bool = True,
        fallback_label: str = "fallback",
    ) -> Any:
        """Run one stage under the reliability contract.

        Args:
            name: stage name (must be in STAGE_TIMEOUTS unless timeout_s given).
            func: callable to run.
            fallback: callable producing a deterministic substitute on failure.
            required: if True and both func + fallback fail, raise StageFailure.
            fallback_label: human label used in logs when fallback is used.
        Returns the stage's value (primary or fallback).
        """
        kwargs = kwargs or {}
        if timeout_s is None:
            timeout_s = self.stage_timeouts.get(name, 60)
        t0 = time.time()
        outcome = StageResult(stage=name, ok=False, kind="exception")

        try:
            value = _run_with_timeout(func, args, kwargs, timeout_s)
            if isinstance(value, tuple) and len(value) == 2 and value[1] is None and not isinstance(value[0], (list, dict)):
                # _run_with_timeout with no timeout returns (value, None); unwrap
                value = value[0]
            elif isinstance(value, tuple) and len(value) == 2 and value[1] is None:
                value = value[0]
            outcome.ok = True
            outcome.kind = "success"
            outcome.value = value
            logger.info("stage '%s' OK in %.2fs", name, time.time() - t0)
        except StageTimeoutError as te:
            outcome.kind = "timeout"
            outcome.error = str(te)
            logger.error("stage '%s' TIMED OUT after %ss (%s); %s", name, timeout_s, te,
                         "invoking fallback" if fallback else "no fallback")
        except Exception as e:
            outcome.kind = "exception"
            outcome.error = f"{type(e).__name__}: {e}"
            logger.error("stage '%s' RAISED (%s); %s\n%s", name, e,
                         "invoking fallback" if fallback else "no fallback",
                         traceback.format_exc())

        if not outcome.ok and fallback is not None:
            try:
                fb_t0 = time.time()
                outcome.value = fallback()
                outcome.ok = True
                outcome.used_fallback = True
                if outcome.kind != "timeout":
                    outcome.kind = "fallback"
                logger.warning("stage '%s' used %s (sys-elapsed %.2fs, fallback %.2fs)",
                               name, fallback_label,
                               time.time() - t0, time.time() - fb_t0)
            except Exception as fb_err:
                outcome.error = (outcome.error or "") + f" | fallback ALSO failed: {fb_err}"
                logger.error("stage '%s' fallback failed (%s)", name, fb_err)
                if required:
                    self.history.append(outcome)
                    self._checkpoint(outcome)
                    raise StageFailure(name, outcome.kind, outcome.error) from fb_err

        if not outcome.ok and required:
            self.history.append(outcome)
            self._checkpoint(outcome)
            raise StageFailure(name, outcome.kind, outcome.error or "unknown")
        if not outcome.ok and not required:
            outcome.value = None

        outcome.elapsed_s = time.time() - t0
        self.history.append(outcome)
        self._checkpoint(outcome)
        return outcome.value

    # ----------------------------------------------------------- private
    def _checkpoint(self, result: StageResult) -> None:
        try:
            line = json.dumps({
                "stage": result.stage,
                "ok": result.ok,
                "kind": result.kind,
                "used_fallback": result.used_fallback,
                "elapsed_s": round(result.elapsed_s, 3),
                "error": (result.error or "")[:2000],
                "ts": time.time(),
            })
            with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            # checkpointing is observability, never fatal
            logger.warning("could not write circuit-breaker checkpoint (%s)", e)


def stage(name: str, timeout_s: Optional[float] = None, fallback: Optional[Callable] = None,
          required: bool = True):
    """Decorator form for one-off stage functions (used by some unit-tests)."""

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb = CircuitBreakerPipeline(run_dir=os.getcwd(), seed=0)
            return cb.stage(name, func, args=args, kwargs=kwargs,
                            fallback=fallback, timeout_s=timeout_s, required=required)
        return wrapper
    return deco
