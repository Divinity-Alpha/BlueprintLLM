"""
pipeline_logger.py
------------------
Unified logging for the Blueprint LLM pipeline.

Provides:
- PipelineLogger: step-based logging with start/complete, progress, ETA prediction
- get_logger(): factory that returns a singleton or _NoOpLogger for standalone use
- Live state file for dashboard consumption
- Timing history for ETA calculation across runs

Step numbering (9-step hierarchy):
  1.x  Data Foundation      (analyze, translate, synthetic, lessons, merge+dedup)
  2.x  Pre-Flight Checks    (validate DSL, validate JSONL, system prompt, data hash)
  3.x  Model Setup          (system prompt, dataset, GPU detect, model load, LoRA, trainer)
  4.x  Training             (backup, init, training loop, save weights/config/prompt, backup)
  5.x  Post-Training        (verify model, summary, health check, report, history, archive)
  6.x  Exam                 (load lesson, load model, run prompts, validate, score, save, backup)
  7.x  Evaluation           (load model, run tests, score, report, save)
  8.x  Lesson Integration   (context=8 for 13_lesson_to_training.py)
  9.x  Dashboard & Finalize (update dashboard)

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

        # Enriched state tracking
        self._completed_steps: list[dict] = []
        self._latest_metrics: dict = {}
        self._loss_history: list[dict] = []
        self._cycle_start_time: float = time.time()
        self._current_step_start_time: float | None = None
        self._step_plan: list[dict] = []
        self._upcoming_steps: list[dict] = []

        # Error/retry tracking
        self._error_history: list[dict] = []  # last 10 errors
        self._latest_error: dict | None = None
        self._latest_retry: dict | None = None

        _LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Inherit state from disk for cross-process continuity
        self._inherit_state_from_disk()

    # -- Cross-process state inheritance ------------------------------------

    def _inherit_state_from_disk(self):
        """Read existing live state and inherit accumulated data if run_id matches.

        This is the key to cross-process continuity: when the orchestrator
        spawns a subprocess (e.g. 04_train), the subprocess's PipelineLogger
        inherits completed_steps, loss_history, etc. from the orchestrator's
        state file.
        """
        try:
            if _LIVE_STATE.exists():
                with open(_LIVE_STATE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("run_id") == self.run_id:
                    self._completed_steps = state.get("completed_steps", [])
                    self._loss_history = state.get("loss_history", [])
                    self._latest_metrics = state.get("latest_metrics", {})
                    if state.get("cycle_start_time"):
                        self._cycle_start_time = state["cycle_start_time"]
                    self._step_plan = state.get("step_plan", [])
                    self._upcoming_steps = state.get("upcoming_steps", [])
                    self._error_history = state.get("error_history", [])
                    self._latest_error = state.get("error_info")
                    self._latest_retry = state.get("retry_info")
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # -- Step plan ----------------------------------------------------------

    def set_step_plan(self, plan: list[dict]):
        """Register the full ordered step sequence for this pipeline run.

        Args:
            plan: list of {step, name} dicts describing all steps in order.
        """
        self._step_plan = plan
        self._cycle_start_time = time.time()
        # Build upcoming steps with ETA estimates
        self._upcoming_steps = []
        for item in plan:
            eta = self._estimate_eta_seconds(item["step"])
            self._upcoming_steps.append({
                "step": item["step"],
                "name": item["name"],
                "eta_seconds": eta,
            })
        self.write_live_state(status="running")

    # -- Metrics ------------------------------------------------------------

    def update_metrics(self, metrics: dict):
        """Merge new metrics into the latest metrics dict."""
        self._latest_metrics.update(metrics)

    # -- Error/retry tracking -------------------------------------------------

    def record_error(self, step_id: str, category: str, message: str):
        """Record an error for dashboard display."""
        error = {
            "step_id": step_id,
            "category": category,
            "message": message[:500],
            "timestamp": datetime.now().isoformat(),
        }
        self._latest_error = error
        self._error_history.append(error)
        # Keep last 10
        if len(self._error_history) > 10:
            self._error_history = self._error_history[-10:]
        self.write_live_state(status="error")

    def record_retry(self, step_id: str, attempt: int, max_retries: int):
        """Record a retry attempt for dashboard display."""
        self._latest_retry = {
            "step_id": step_id,
            "attempt": attempt,
            "max_retries": max_retries,
            "timestamp": datetime.now().isoformat(),
        }
        self.write_live_state(status="retrying")

    def append_loss_history(self, step: int, loss: float):
        """Append a loss data point for the mini loss chart.

        Downsamples if >200 entries: thin first half, keep recent half intact.
        """
        self._loss_history.append({"step": step, "loss": round(loss, 6)})
        if len(self._loss_history) > 200:
            half = len(self._loss_history) // 2
            # Keep every other point in first half, all points in second half
            thinned = self._loss_history[:half:2] + self._loss_history[half:]
            self._loss_history = thinned

    # -- ETA helpers --------------------------------------------------------

    def _estimate_eta_seconds(self, step_id: str) -> float | None:
        """Return average duration for a step from timing history, or None."""
        timings = self._timing_history.get(str(step_id), [])
        if not timings:
            return None
        return round(sum(timings) / len(timings), 1)

    def _calculate_remaining_eta(self, step_id: str,
                                  progress_current: int | None = None,
                                  progress_total: int | None = None) -> float | None:
        """Estimate total remaining seconds (current step + upcoming steps)."""
        remaining = 0.0
        has_estimate = False

        # Current step: extrapolate from elapsed + progress
        if (self._current_step_start_time and progress_current and progress_total
                and progress_current > 0):
            elapsed = time.time() - self._current_step_start_time
            fraction_done = progress_current / max(progress_total, 1)
            if fraction_done > 0.01:
                estimated_total = elapsed / fraction_done
                remaining += max(0, estimated_total - elapsed)
                has_estimate = True
        elif self._current_step_start_time:
            # No progress info — use historical average for remaining
            eta = self._estimate_eta_seconds(str(step_id))
            if eta is not None:
                elapsed = time.time() - self._current_step_start_time
                remaining += max(0, eta - elapsed)
                has_estimate = True

        # Sum upcoming step estimates
        for upcoming in self._upcoming_steps:
            if upcoming.get("eta_seconds") is not None:
                remaining += upcoming["eta_seconds"]
                has_estimate = True

        return round(remaining, 1) if has_estimate else None

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
        self._current_step_start_time = time.time()

        # Remove from upcoming steps
        self._upcoming_steps = [
            s for s in self._upcoming_steps if s["step"] != step_id
        ]

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

        # Add to completed steps
        self._completed_steps.append({
            "step": step_id,
            "name": desc,
            "duration": round(elapsed, 2),
            "status": "complete",
        })

        if step_id == self._current_step_id:
            self._current_step_id = None
            self._current_step_desc = None
            self._current_step_start_time = None

        # Write updated state so next step (or subprocess) sees it
        self.write_live_state(status="running")

    # -- Progress -----------------------------------------------------------

    def progress(self, step_id, current: int, total: int, detail: str = "",
                 metrics: dict | None = None):
        """Emit a progress update, rate-limited to every N seconds."""
        # Always accept metrics even if rate-limited
        if metrics:
            self._latest_metrics.update(metrics)

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
        now = time.time()
        elapsed = now - self._cycle_start_time

        state = {
            "status": status,
            "run_id": self.run_id,
            "step_id": step_id or (self._current_step_id or ""),
            "description": description or (self._current_step_desc or ""),
            "timestamp": datetime.now().isoformat(),
            "cycle_start_time": self._cycle_start_time,
            "elapsed_seconds": round(elapsed, 1),
            "completed_steps": self._completed_steps,
            "upcoming_steps": self._upcoming_steps,
            "step_plan": self._step_plan,
            "latest_metrics": self._latest_metrics,
            "loss_history": self._loss_history,
            "error_info": self._latest_error,
            "retry_info": self._latest_retry,
            "error_history": self._error_history[-10:],
        }

        # Step timing
        if self._current_step_start_time and status == "running":
            state["step_start_time"] = self._current_step_start_time

        # ETA
        if eta:
            state["eta"] = eta
        if progress_current is not None:
            state["progress_current"] = progress_current
            state["progress_total"] = progress_total

        # Calculate remaining ETA
        eta_remaining = self._calculate_remaining_eta(
            state["step_id"], progress_current, progress_total
        )
        if eta_remaining is not None:
            state["eta_remaining_seconds"] = eta_remaining

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

    def progress(self, step_id, current: int, total: int, detail: str = "",
                 metrics: dict | None = None):
        pass

    def write_live_state(self, *args, **kwargs):
        pass

    def set_step_plan(self, plan: list[dict]):
        pass

    def update_metrics(self, metrics: dict):
        pass

    def append_loss_history(self, step: int, loss: float):
        pass

    def record_error(self, step_id: str, category: str, message: str):
        pass

    def record_retry(self, step_id: str, attempt: int, max_retries: int):
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
