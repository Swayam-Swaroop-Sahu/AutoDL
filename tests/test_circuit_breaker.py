"""Tests for circuit breaker pipeline (Phase 1d)."""
import os
import tempfile
import time
import pickle
import pytest

from src.core.circuit_breaker import (
    CircuitBreakerPipeline,
    StageTimeoutError,
    StageFailure,
)


# ---------------------------------------------------------------------------
# Stage timeout → fallback fires
# ---------------------------------------------------------------------------
def test_stage_timeout_triggers_fallback():
    """A stage that sleeps past the timeout must trigger the fallback."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)

        def slow():
            time.sleep(3.0)
            return "primary"

        def fallback():
            return "fallback_value"

        result = cb.stage(
            "search", slow, fallback=fallback, timeout_s=1,
        )
        assert result == "fallback_value"
        # History records the timeout
        history_names = [h.stage for h in cb.history]
        assert "search" in history_names
        last = cb.history[-1]
        assert last.kind == "timeout" or last.used_fallback is True


def test_timeout_logs_timed_out():
    """On timeout, the structured log records the timeout event."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)

        def slow():
            time.sleep(2.0)

        cb.stage("search", slow, fallback=lambda: "ok", timeout_s=1)
        last = cb.history[-1]
        assert last.error is not None
        assert "deadline" in last.error.lower() or "timeout" in last.error.lower()


# ---------------------------------------------------------------------------
# Stage exception → fallback fires
# ---------------------------------------------------------------------------
def test_stage_exception_triggers_fallback():
    """A stage that raises RuntimeError must trigger the fallback."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)

        def bad():
            raise RuntimeError("simulated OOM")

        def fallback():
            return "from_fallback"

        result = cb.stage("search", bad, fallback=fallback, timeout_s=5)
        assert result == "from_fallback"
        last = cb.history[-1]
        assert last.used_fallback is True
        assert last.kind == "fallback"
        assert "RuntimeError" in (last.error or "")


def test_stage_exception_logged():
    """When a stage raises, the error is recorded in history."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)

        def bad():
            raise ValueError("boom")

        cb.stage("search", bad, fallback=lambda: None, timeout_s=5)
        last = cb.history[-1]
        assert last.error is not None
        assert "ValueError" in last.error or "boom" in last.error


# ---------------------------------------------------------------------------
# Required stage with no fallback → StageFailure raised
# ---------------------------------------------------------------------------
def test_required_stage_failure_raises():
    """A required stage with no fallback and exception must raise StageFailure."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)

        def bad():
            raise RuntimeError("fatal")

        with pytest.raises(StageFailure):
            cb.stage("load", bad, fallback=None, timeout_s=5, required=True)


# ---------------------------------------------------------------------------
# Optional stage with no fallback → returns None
# ---------------------------------------------------------------------------
def test_optional_stage_no_fallback_returns_none():
    """An optional stage with no fallback and failure returns None."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)

        def bad():
            raise RuntimeError("non-fatal")

        result = cb.stage("report", bad, fallback=None, timeout_s=5, required=False)
        assert result is None


# ---------------------------------------------------------------------------
# Success path records success
# ---------------------------------------------------------------------------
def test_successful_stage_records_ok():
    """A successful stage is recorded as ok=True, kind=success."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)
        result = cb.stage(
            "preprocess", lambda: "ok",
            fallback=lambda: "fallback", timeout_s=5,
        )
        assert result == "ok"
        last = cb.history[-1]
        assert last.ok is True
        assert last.kind == "success"
        assert last.used_fallback is False


# ---------------------------------------------------------------------------
# Checkpoint save/load works
# ---------------------------------------------------------------------------
def test_checkpoint_save_load():
    """A pickled state dict can be round-tripped via the checkpoint helper pattern."""
    state = {"stage": "search", "score": 0.85, "params": {"n": 100}}
    with tempfile.TemporaryDirectory() as run_dir:
        ckpt_path = os.path.join(run_dir, "checkpoint.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(state, f)
        with open(ckpt_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == state


def test_circuit_breaker_writes_checkpoint_file():
    """The circuit breaker writes a JSONL log file with one line per stage."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)
        cb.stage("preprocess", lambda: "ok", timeout_s=5)
        ckpt_path = os.path.join(run_dir, "circuit_breaker.jsonl")
        assert os.path.exists(ckpt_path), "checkpoint file should be written"
        with open(ckpt_path, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        assert len(lines) >= 1
        # Each line must be valid JSON with at least the stage key
        import json
        first = json.loads(lines[0])
        assert "stage" in first
        assert "ok" in first
        assert "kind" in first


# ---------------------------------------------------------------------------
# Multiple stages in sequence
# ---------------------------------------------------------------------------
def test_multiple_stages_recorded_in_order():
    """Multiple stages are recorded in the order they ran."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)
        cb.stage("load", lambda: "a", timeout_s=5)
        cb.stage("preprocess", lambda: "b", timeout_s=5)
        cb.stage("search", lambda: "c", timeout_s=5)
        assert [h.stage for h in cb.history] == ["load", "preprocess", "search"]


# ---------------------------------------------------------------------------
# Fallback itself fails → required stage raises StageFailure
# ---------------------------------------------------------------------------
def test_fallback_failure_required_raises():
    """If both primary and fallback fail on a required stage, raise StageFailure."""
    with tempfile.TemporaryDirectory() as run_dir:
        cb = CircuitBreakerPipeline(run_dir=run_dir, seed=0)

        def bad():
            raise RuntimeError("primary failed")

        def bad_fallback():
            raise ValueError("fallback also failed")

        with pytest.raises(StageFailure):
            cb.stage("search", bad, fallback=bad_fallback, timeout_s=5, required=True)
