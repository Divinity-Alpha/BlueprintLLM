"""
pipeline_logger.py
------------------
Unified logging for the Blueprint LLM pipeline.

Provides:
- PipelineLogger: step-based logging with start/complete, progress, ETA prediction
- get_logger(): factory that returns a singleton or _NoOpLogger for standalone use
- Live state file for dashboard consumption
- Timing history for ETA calculation across runs

Output destinations:
- logs/pipeline_live.log  (append, shared by orchestrator + subprocesses)
- logs/pipeline_live_state.json  (atomic replace, current step info)
- logs/step_timing_history.json  (accumulated averages per step)
"""

import json
import os
import time
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Lock

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(os.environ.get("BLUEPRINT_LLM_ROOT", r"C:\BlueprintLLM"))
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LIVE_LOG = _LOGS_DIR / "pipeline_live.log"
_LIVE_STATE = _LOGS_DIR / "pipeline_live_state.json"
_TIMING_HISTORY = _LOGS_DIR / "step_timing_history.json"

_MAX_HISTORY_PER_STEP = 10  # keep last N timings per step_id for ETA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        whole_min = int(minutes)
        secs = int(seconds % 60)
        return f"{whole_min}m {secs}s" if secs else f"{whole_min}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"


def _load_timing_history() -> dict:
    """Load step timing history from disk."""
    if _TIMING_HISTORY.exists():
        try:
            with open(_TIMING_HISTORY, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_timing_history(history: dict):
    """Save step timing history atomically."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _TIMING_HISTORY.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        # Atomic replace (Windows: need to remove first)
        if _TIMING_HISTORY.exists():
            _TIMING_HISTORY.unlink()
        tmp.rename(_TIMING_HISTORY)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _get_eta(step_id: str, history: dict) -> str | None:
    """Estimate time for a step based on historical averages."""
    timings = history.get(step_id, [])
    if not timings:
        return None
    avg = sum(timings) / len(timings)
    return format_duration(avg)


# ---------------------------------------------------------------------------
# PipelineLogger
# ---------------------------------------------------------------------------

class PipelineLogger:
    """Unified pipeline logger with step tracking, ETA, and live state."""

    def __init__(self, step_prefix: str = "", run_id: str = ""):
        self.step_prefix = step_prefix
        self.run_id = run_id or os.environ.get("PIPELINE_RUN_ID", "")
        self._step_starts: dict[str, float] = {}
        self._timing_history = _load_timing_history()
        self._last_progress_time: float = 0.0
        self._progress_interval = 5.0  # seconds between progress emissions
        self._lock = Lock()
        self._current_step_id: str | None = None
        self._current_step_desc: str | None = None
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # -- Core logging -------------------------------------------------------

    def log(self, msg: str, level: str = "INFO"):
        """Write a timestamped line to live log and stdout."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line, flush=True)
        self._append_live_log(line)

    def _append_live_log(self, line: str):
        """Append a line to the shared live log file."""
        try:
            with open(_LIVE_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # -- Step lifecycle -----------------------------------------------------

    def start_step(self, step_id, description: str, details: str = ""):
        """Mark the beginning of a pipeline step."""
        step_id = str(step_id)
        ts = datetime.now().strftime("%H:%M:%S")
        self._step_starts[step_id] = time.time()
        self._current_step_id = step_id
        self._current_step_desc = description

        eta = _get_eta(step_id, self._timing_history)
        eta_str = f" | ETA: {eta}" if eta else ""

        line = f"[{ts}] [STEP {step_id}] STARTING: {description}{eta_str}"
        print(line, flush=True)
        self._append_live_log(line)

        if details:
            detail_line = f"           {details}"
            print(detail_line, flush=True)
            self._append_live_log(detail_line)

        self.write_live_state(step_id, description, status="running", eta=eta)

    def complete_step(self, step_id, description: str = "", details: str = ""):
        """Mark the end of a pipeline step and record timing."""
        step_id = str(step_id)
        ts = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self._step_starts.pop(step_id, time.time())
        dur = format_duration(elapsed)
        desc = description or self._current_step_desc or ""

        line = f"[{ts}] [STEP {step_id}] COMPLETE: {desc} ({dur})"
        print(line, flush=True)
        self._append_live_log(line)

        if details:
            detail_line = f"           {details}"
            print(detail_line, flush=True)
            self._append_live_log(detail_line)

        # Record timing for future ETA
        self._record_timing(step_id, elapsed)

        if step_id == self._current_step_id:
            self._current_step_id = None
            self._current_step_desc = None

    # -- Progress -----------------------------------------------------------

    def progress(self, step_id, current: int, total: int, detail: str = ""):
        """Emit a progress update, rate-limited to every N seconds."""
        now = time.time()
        with self._lock:
            if now - self._last_progress_time < self._progress_interval:
                return
            self._last_progress_time = now

        step_id = str(step_id)
        ts = datetime.now().strftime("%H:%M:%S")
        pct = int(100 * current / max(total, 1))
        detail_str = f" — {detail}" if detail else ""
        line = f"[{ts}]   [{step_id}] {current}/{total} ({pct}%){detail_str}"
        print(line, flush=True)
        self._append_live_log(line)

        self.write_live_state(step_id, self._current_step_desc or "",
                              status="running", progress_current=current,
                              progress_total=total)

    # -- Live state ---------------------------------------------------------

    def write_live_state(self, step_id: str = "", description: str = "",
                         status: str = "running", eta: str | None = None,
                         progress_current: int | None = None,
                         progress_total: int | None = None):
        """Write current pipeline state to JSON for dashboard consumption."""
        state = {
            "status": status,
            "run_id": self.run_id,
            "step_id": step_id,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        }
        if eta:
            state["eta"] = eta
        if progress_current is not None:
            state["progress_current"] = progress_current
            state["progress_total"] = progress_total

        try:
            tmp = _LIVE_STATE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            if _LIVE_STATE.exists():
                _LIVE_STATE.unlink()
            tmp.rename(_LIVE_STATE)
        except OSError:
            pass

    # -- Timing history -----------------------------------------------------

    def _record_timing(self, step_id: str, elapsed: float):
        """Accumulate timing for a step, keeping last N entries."""
        timings = self._timing_history.setdefault(step_id, [])
        timings.append(round(elapsed, 2))
        if len(timings) > _MAX_HISTORY_PER_STEP:
            self._timing_history[step_id] = timings[-_MAX_HISTORY_PER_STEP:]
        _save_timing_history(self._timing_history)


# ---------------------------------------------------------------------------
# NoOp fallback
# ---------------------------------------------------------------------------

class _NoOpLogger:
    """Silent fallback so scripts never need `if plog:` guards."""

    def log(self, msg: str, level: str = "INFO"):
        pass

    def start_step(self, step_id, description: str, details: str = ""):
        pass

    def complete_step(self, step_id, description: str = "", details: str = ""):
        pass

    def progress(self, step_id, current: int, total: int, detail: str = ""):
        pass

    def write_live_state(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_singleton: PipelineLogger | _NoOpLogger | None = None


def get_logger(step_prefix: str = "", run_id: str = "") -> PipelineLogger | _NoOpLogger:
    """Return a PipelineLogger singleton, or _NoOpLogger if not in a pipeline run.

    If PIPELINE_RUN_ID is set in the environment, returns a real logger.
    Otherwise returns a _NoOpLogger so standalone script runs stay silent.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    env_run_id = run_id or os.environ.get("PIPELINE_RUN_ID", "")
    if env_run_id:
        _singleton = PipelineLogger(step_prefix=step_prefix, run_id=env_run_id)
    else:
        _singleton = _NoOpLogger()
    return _singleton
