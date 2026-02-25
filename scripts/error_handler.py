"""
error_handler.py
----------------
Central error handling, retry logic, stall detection, and resume capability
for the Blueprint LLM pipeline.

Provides:
- ErrorCategory: classification of subprocess failures
- classify_error(): parse stderr/exceptions into categories
- RetryConfig / STEP_RETRY_CONFIGS: per-step retry policies
- SubprocessMonitor: background thread watching for stalled subprocesses
- per_prompt_timeout(): thread-based timeout for model.generate() calls
- cuda_oom_retry(): catch OOM, reduce config, retry
- save/load/clear_resume_state(): pipeline resume checkpoint management
"""

import enum
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Error Classification
# ---------------------------------------------------------------------------

class ErrorCategory(enum.Enum):
    TIMEOUT = "timeout"
    CUDA_OOM = "cuda_oom"
    DISK_FULL = "disk_full"
    NETWORK = "network"
    CORRUPT_CHECKPOINT = "corrupt_checkpoint"
    ENCODING = "encoding"
    UNKNOWN = "unknown"


# Patterns checked against stderr (case-insensitive)
_ERROR_PATTERNS = [
    (ErrorCategory.CUDA_OOM, [
        r"cuda out of memory",
        r"out of memory",
        r"torch\.cuda\.OutOfMemoryError",
        r"CUDA error: out of memory",
        r"RuntimeError:.*CUDA.*memory",
    ]),
    (ErrorCategory.DISK_FULL, [
        r"no space left on device",
        r"not enough disk space",
        r"OSError.*Errno 28",
        r"disk quota exceeded",
    ]),
    (ErrorCategory.NETWORK, [
        r"connectionerror",
        r"connection refused",
        r"connection reset",
        r"timeout.*connect",
        r"urlopen error",
        r"requests\.exceptions",
        r"HTTPSConnectionPool",
        r"SSLError",
        r"NewConnectionError",
    ]),
    (ErrorCategory.CORRUPT_CHECKPOINT, [
        r"corrupted",
        r"invalid checkpoint",
        r"safetensors.*error",
        r"PeftModel.*not found",
        r"can't load weights",
    ]),
    (ErrorCategory.ENCODING, [
        r"unicodeencodeerror",
        r"unicodedecodeerror",
        r"codec can't encode",
        r"codec can't decode",
        r"charmap.*codec",
    ]),
]

# Categories that should NOT be retried
NON_RETRYABLE = {ErrorCategory.DISK_FULL}


def classify_error(returncode: int = 1, stderr: str = "",
                   exception: Exception | None = None) -> ErrorCategory:
    """Classify a failure into an ErrorCategory by inspecting stderr and exception."""
    text = stderr.lower()
    if exception:
        text += " " + str(exception).lower() + " " + type(exception).__name__.lower()

    for category, patterns in _ERROR_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return category

    return ErrorCategory.UNKNOWN


def is_retryable(category: ErrorCategory) -> bool:
    """Return True if this error category is worth retrying."""
    return category not in NON_RETRYABLE


# ---------------------------------------------------------------------------
# Retry Configuration
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    max_retries: int = 1
    backoff: tuple = (10,)          # seconds to wait before each retry
    timeout: int = 600              # subprocess timeout in seconds

    def get_backoff(self, attempt: int) -> int:
        """Return backoff seconds for the given retry attempt (0-indexed)."""
        if attempt < len(self.backoff):
            return self.backoff[attempt]
        return self.backoff[-1]


STEP_RETRY_CONFIGS = {
    "data":     RetryConfig(max_retries=2, backoff=(10, 30),   timeout=600),
    "validate": RetryConfig(max_retries=1, backoff=(5,),       timeout=300),
    "training": RetryConfig(max_retries=2, backoff=(60, 120),  timeout=14400),
    "eval":     RetryConfig(max_retries=1, backoff=(30,),      timeout=14400),
    "exam":     RetryConfig(max_retries=1, backoff=(30,),      timeout=14400),
    "utility":  RetryConfig(max_retries=1, backoff=(10,),      timeout=600),
}


# ---------------------------------------------------------------------------
# Heartbeat File (lightweight alternative to JSON state for liveness)
# ---------------------------------------------------------------------------

_HEARTBEAT_FILE = Path(os.environ.get(
    "BLUEPRINT_LLM_ROOT", r"C:\BlueprintLLM"
)) / "logs" / "pipeline_heartbeat"


def write_heartbeat():
    """Touch the heartbeat file to signal the subprocess is alive.

    This is a lightweight alternative to updating pipeline_live_state.json.
    SubprocessMonitor checks both — whichever was updated more recently wins.
    Cannot fail due to file locking (just overwrites a tiny file).
    """
    try:
        _HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HEARTBEAT_FILE.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def _get_heartbeat_age() -> float | None:
    """Return seconds since the heartbeat file was last written."""
    try:
        if _HEARTBEAT_FILE.exists():
            ts = float(_HEARTBEAT_FILE.read_text(encoding="utf-8").strip())
            return time.time() - ts
    except (OSError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Subprocess Stall Detection
# ---------------------------------------------------------------------------

class SubprocessMonitor(threading.Thread):
    """Background thread that watches pipeline_live_state.json for staleness.

    If the timestamp hasn't been updated within warn_seconds, logs a warning.
    If it exceeds kill_seconds, terminates the subprocess.

    Also checks the heartbeat file as a secondary liveness signal — if either
    the state JSON or heartbeat file was updated recently, the subprocess is
    considered alive.
    """

    def __init__(self, process, state_file: Path,
                 warn_seconds: int = 300, kill_seconds: int = 600,
                 check_interval: int = 30, log_func=None):
        super().__init__(daemon=True)
        self.process = process
        self.state_file = state_file
        self.warn_seconds = warn_seconds
        self.kill_seconds = kill_seconds
        self.check_interval = check_interval
        self.log_func = log_func or print
        self.was_stalled = False
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _get_state_age(self) -> float | None:
        """Return seconds since last state file update, or None if unavailable.

        Checks both the JSON state timestamp and the heartbeat file, returning
        the minimum (most recent signal).
        """
        ages = []

        # Check JSON state timestamp
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                ts_str = data.get("timestamp")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                    ages.append((datetime.now() - ts).total_seconds())
        except (json.JSONDecodeError, OSError, ValueError):
            pass

        # Check heartbeat file
        hb_age = _get_heartbeat_age()
        if hb_age is not None:
            ages.append(hb_age)

        return min(ages) if ages else None

    def run(self):
        warned = False
        while not self._stop_event.is_set():
            self._stop_event.wait(self.check_interval)
            if self._stop_event.is_set():
                break

            # Check if process already exited
            if self.process.poll() is not None:
                break

            age = self._get_state_age()
            if age is None:
                continue

            if age >= self.kill_seconds:
                self.log_func(
                    f"  STALL DETECTED: No state update for {int(age)}s "
                    f"(limit: {self.kill_seconds}s). Terminating subprocess."
                )
                self.was_stalled = True
                try:
                    self.process.terminate()
                except OSError:
                    pass
                break
            elif age >= self.warn_seconds and not warned:
                self.log_func(
                    f"  WARNING: No state update for {int(age)}s "
                    f"(kill at {self.kill_seconds}s)"
                )
                warned = True


# ---------------------------------------------------------------------------
# Per-Prompt Timeout
# ---------------------------------------------------------------------------

def per_prompt_timeout(func, timeout_seconds: int = 300):
    """Run func() in a thread with a timeout.

    Returns (result, False) on success, or (None, True) on timeout.
    On Windows, we can't kill threads stuck in C extensions — the thread
    will linger. The subprocess-level timeout is the ultimate safety net.
    """
    result_container = [None]
    error_container = [None]

    def target():
        try:
            result_container[0] = func()
        except Exception as e:
            error_container[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running — timed out
        return None, True

    if error_container[0] is not None:
        raise error_container[0]

    return result_container[0], False


# ---------------------------------------------------------------------------
# CUDA OOM Recovery
# ---------------------------------------------------------------------------

def cuda_oom_retry(func, config: dict, max_retries: int = 2, log_func=None):
    """Wrap a training function with CUDA OOM recovery.

    On OOM:
    1. Clears CUDA cache
    2. Reduces max_seq_length by half (min 512)
    3. If still failing, reduces lora_r by half (min 16)
    4. Rebuilds and retries

    Args:
        func: callable(config) -> result
        config: mutable training config dict
        max_retries: max OOM retries
        log_func: logging function

    Returns:
        Result from func(config)
    """
    log = log_func or print
    import torch

    for attempt in range(max_retries + 1):
        try:
            return func(config)
        except torch.cuda.OutOfMemoryError:
            if attempt >= max_retries:
                log(f"  CUDA OOM: Exhausted {max_retries} retries. Giving up.")
                raise

            log(f"  CUDA OOM on attempt {attempt + 1}/{max_retries + 1}. Recovering...")

            # Clear GPU memory
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

            # Reduce config
            cur_seq = config.get("max_seq_length", 2048)
            cur_lora = config.get("lora_r", 64)

            if cur_seq > 512:
                new_seq = max(512, cur_seq // 2)
                log(f"  Reducing max_seq_length: {cur_seq} -> {new_seq}")
                config["max_seq_length"] = new_seq
            elif cur_lora > 16:
                new_lora = max(16, cur_lora // 2)
                log(f"  Reducing lora_r: {cur_lora} -> {new_lora}")
                config["lora_r"] = new_lora
                config["lora_alpha"] = new_lora * 2
            else:
                log(f"  Cannot reduce config further. Giving up.")
                raise

            log(f"  Retrying with reduced config (attempt {attempt + 2}/{max_retries + 1})...")
            time.sleep(5)  # Brief pause for GPU to settle


# ---------------------------------------------------------------------------
# Resume State Management
# ---------------------------------------------------------------------------

_RESUME_STATE_FILE = Path(os.environ.get(
    "BLUEPRINT_LLM_ROOT", r"C:\BlueprintLLM"
)) / "logs" / "pipeline_resume_state.json"


def save_resume_state(completed_steps: list[str], failed_step: str,
                      error_info: dict, state_snapshot: dict | None = None):
    """Save pipeline resume state atomically.

    Args:
        completed_steps: list of step function names that completed successfully
        failed_step: name of the step that failed
        error_info: dict with category, message, timestamp, attempt
        state_snapshot: copy of pipeline state at time of failure
    """
    resume = {
        "completed_steps": completed_steps,
        "failed_step": failed_step,
        "error_info": error_info,
        "state_snapshot": state_snapshot or {},
        "timestamp": datetime.now().isoformat(),
    }

    _RESUME_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _RESUME_STATE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(resume, f, indent=2)
        if _RESUME_STATE_FILE.exists():
            _RESUME_STATE_FILE.unlink()
        tmp.rename(_RESUME_STATE_FILE)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_resume_state() -> dict | None:
    """Load pipeline resume state, or None if no resume file exists."""
    try:
        if _RESUME_STATE_FILE.exists():
            with open(_RESUME_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def clear_resume_state():
    """Delete the resume state file after successful pipeline completion."""
    try:
        if _RESUME_STATE_FILE.exists():
            _RESUME_STATE_FILE.unlink()
    except OSError:
        pass
