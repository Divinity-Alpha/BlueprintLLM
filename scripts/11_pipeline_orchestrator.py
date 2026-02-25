"""
11_pipeline_orchestrator.py
---------------------------
Master orchestrator that chains the entire Blueprint LLM pipeline.
Designed to run unattended via Windows Task Scheduler.

All interactive prompts are suppressed. Everything is logged.

9-Step Hierarchy:
    1  Data Foundation       (analyze, translate, synthetic, lessons, merge+dedup)
    2  Pre-Flight Checks     (validate DSL, validate JSONL, system prompt, data hash)
    3  Model Setup           (subprocess: 04_train steps 3.x)
    4  Training              (subprocess: 04_train steps 4.x)
    5  Post-Training         (verify, summary, health check, report, history, archive)
    6  Exam                  (run exams against latest lessons)
    7  Evaluation            (run tier-based test suite)
    8  Lesson Integration    (convert lessons for next cycle)
    9  Dashboard & Finalize  (update dashboard)

Usage:
    python scripts/11_pipeline_orchestrator.py --full
    python scripts/11_pipeline_orchestrator.py --full --dry-run
    python scripts/11_pipeline_orchestrator.py --data-only
    python scripts/11_pipeline_orchestrator.py --train-only --force
    python scripts/11_pipeline_orchestrator.py --eval-only

Exit codes: 0=Success, 1=Failure, 2=Skipped (no new data)
"""

import os, sys, json, re, shutil, subprocess, hashlib, argparse, uuid, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from stop_signal_utils import is_stop_requested, clear_signal
from pipeline_logger import PipelineLogger, format_duration
from error_handler import (
    classify_error, is_retryable, ErrorCategory,
    RetryConfig, STEP_RETRY_CONFIGS,
    SubprocessMonitor, save_resume_state, load_resume_state, clear_resume_state,
)

# ---------------------------------------------------------------------------
# Step plans for each pipeline mode — used by PipelineLogger for timeline/ETA
# ---------------------------------------------------------------------------

STEP_PLANS = {
    "full": [
        {"step": "1", "name": "Data Foundation", "parent": None},
        {"step": "1.1", "name": "Analyze new exports", "parent": "1"},
        {"step": "1.2", "name": "Auto-translate to DSL", "parent": "1"},
        {"step": "1.3", "name": "Generate synthetic data", "parent": "1"},
        {"step": "1.4", "name": "Convert lessons", "parent": "1"},
        {"step": "1.5", "name": "Merge + Deduplicate", "parent": "1"},
        {"step": "2", "name": "Pre-Flight Checks", "parent": None},
        {"step": "2.1", "name": "Validate DSL files", "parent": "2"},
        {"step": "2.2", "name": "Validate merged dataset", "parent": "2"},
        {"step": "2.3", "name": "Generate system prompt", "parent": "2"},
        {"step": "2.4", "name": "Compute data hash", "parent": "2"},
        {"step": "3", "name": "Model Setup", "parent": None},
        {"step": "3.1", "name": "Load system prompt", "parent": "3"},
        {"step": "3.2", "name": "Load training dataset", "parent": "3"},
        {"step": "3.3", "name": "Detect GPU + precision", "parent": "3"},
        {"step": "3.4", "name": "Load base model + quantize", "parent": "3"},
        {"step": "3.5", "name": "Attach LoRA adapter", "parent": "3"},
        {"step": "3.6", "name": "Configure SFT Trainer", "parent": "3"},
        {"step": "4", "name": "Training", "parent": None},
        {"step": "4.1", "name": "Pre-training backup", "parent": "4"},
        {"step": "4.2", "name": "Initialize training", "parent": "4"},
        {"step": "4.3", "name": "Training loop", "parent": "4"},
        {"step": "4.4", "name": "Save model weights", "parent": "4"},
        {"step": "4.5", "name": "Save training config", "parent": "4"},
        {"step": "4.6", "name": "Save system prompt", "parent": "4"},
        {"step": "4.7", "name": "Post-training backup", "parent": "4"},
        {"step": "5", "name": "Post-Training", "parent": None},
        {"step": "5.1", "name": "Verify model output", "parent": "5"},
        {"step": "5.2", "name": "Training summary", "parent": "5"},
        {"step": "5.3", "name": "Run health checks", "parent": "5"},
        {"step": "5.4", "name": "Generate health report", "parent": "5"},
        {"step": "5.5", "name": "Update training history", "parent": "5"},
        {"step": "5.6", "name": "Archive logs", "parent": "5"},
        {"step": "6", "name": "Exam", "parent": None},
        {"step": "6.1", "name": "Load lesson", "parent": "6"},
        {"step": "6.2", "name": "Load model for exam", "parent": "6"},
        {"step": "6.3", "name": "Run prompts", "parent": "6"},
        {"step": "6.4", "name": "Validate responses", "parent": "6"},
        {"step": "6.5", "name": "Compare & score", "parent": "6"},
        {"step": "6.6", "name": "Save exam results", "parent": "6"},
        {"step": "6.7", "name": "Post-exam backup", "parent": "6"},
        {"step": "7", "name": "Evaluation", "parent": None},
        {"step": "7.1", "name": "Load model", "parent": "7"},
        {"step": "7.2", "name": "Run test suite", "parent": "7"},
        {"step": "7.3", "name": "Score results", "parent": "7"},
        {"step": "7.4", "name": "Generate report", "parent": "7"},
        {"step": "7.5", "name": "Save report", "parent": "7"},
        {"step": "8", "name": "Lesson Integration", "parent": None},
        {"step": "8.1", "name": "Load lessons", "parent": "8"},
        {"step": "8.2", "name": "Generate variations", "parent": "8"},
        {"step": "8.3", "name": "Validate entries", "parent": "8"},
        {"step": "8.4", "name": "Merge into dataset", "parent": "8"},
        {"step": "8.5", "name": "Post-merge backup", "parent": "8"},
        {"step": "9", "name": "Dashboard & Finalize", "parent": None},
        {"step": "9.1", "name": "Collect metrics", "parent": "9"},
        {"step": "9.2", "name": "Generate charts", "parent": "9"},
        {"step": "9.3", "name": "Build HTML", "parent": "9"},
        {"step": "9.4", "name": "Write dashboard", "parent": "9"},
        {"step": "9.5", "name": "Finalize", "parent": "9"},
    ],
    "data-only": [
        {"step": "1", "name": "Data Foundation", "parent": None},
        {"step": "1.1", "name": "Analyze new exports", "parent": "1"},
        {"step": "1.2", "name": "Auto-translate to DSL", "parent": "1"},
        {"step": "1.3", "name": "Generate synthetic data", "parent": "1"},
        {"step": "1.4", "name": "Convert lessons", "parent": "1"},
        {"step": "1.5", "name": "Merge + Deduplicate", "parent": "1"},
        {"step": "2", "name": "Pre-Flight Checks", "parent": None},
        {"step": "2.1", "name": "Validate DSL files", "parent": "2"},
        {"step": "2.2", "name": "Validate merged dataset", "parent": "2"},
        {"step": "2.3", "name": "Generate system prompt", "parent": "2"},
        {"step": "2.4", "name": "Compute data hash", "parent": "2"},
    ],
    "train-only": [
        {"step": "3", "name": "Model Setup", "parent": None},
        {"step": "3.1", "name": "Load system prompt", "parent": "3"},
        {"step": "3.2", "name": "Load training dataset", "parent": "3"},
        {"step": "3.3", "name": "Detect GPU + precision", "parent": "3"},
        {"step": "3.4", "name": "Load base model + quantize", "parent": "3"},
        {"step": "3.5", "name": "Attach LoRA adapter", "parent": "3"},
        {"step": "3.6", "name": "Configure SFT Trainer", "parent": "3"},
        {"step": "4", "name": "Training", "parent": None},
        {"step": "4.1", "name": "Pre-training backup", "parent": "4"},
        {"step": "4.2", "name": "Initialize training", "parent": "4"},
        {"step": "4.3", "name": "Training loop", "parent": "4"},
        {"step": "4.4", "name": "Save model weights", "parent": "4"},
        {"step": "4.5", "name": "Save training config", "parent": "4"},
        {"step": "4.6", "name": "Save system prompt", "parent": "4"},
        {"step": "4.7", "name": "Post-training backup", "parent": "4"},
        {"step": "5", "name": "Post-Training", "parent": None},
        {"step": "5.1", "name": "Verify model output", "parent": "5"},
        {"step": "5.2", "name": "Training summary", "parent": "5"},
        {"step": "5.3", "name": "Run health checks", "parent": "5"},
        {"step": "5.4", "name": "Generate health report", "parent": "5"},
        {"step": "5.5", "name": "Update training history", "parent": "5"},
        {"step": "5.6", "name": "Archive logs", "parent": "5"},
    ],
    "eval-only": [
        {"step": "7", "name": "Evaluation", "parent": None},
        {"step": "7.1", "name": "Load model", "parent": "7"},
        {"step": "7.2", "name": "Run test suite", "parent": "7"},
        {"step": "7.3", "name": "Score results", "parent": "7"},
        {"step": "7.4", "name": "Generate report", "parent": "7"},
        {"step": "7.5", "name": "Save report", "parent": "7"},
    ],
}


class PipelineConfig:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("BLUEPRINT_LLM_ROOT", r"C:\BlueprintLLM"))
        self.venv_python = self.root / "venv" / "Scripts" / "python.exe"
        self.scripts = self.root / "scripts"
        self.clipboard_inbox = self.root / "raw-data" / "clipboard-exports"
        self.clipboard_processed = self.root / "raw-data" / "clipboard-exports" / "processed"
        self.dsl_dir = self.root / "cleaned-data" / "parsed-blueprints"
        self.manual_data = self.root / "datasets" / "manual_examples.jsonl"
        self.lesson_data = self.root / "datasets" / "lesson_data.jsonl"
        self.synthetic_data = self.root / "datasets" / "synthetic_train.jsonl"
        self.train_data = self.root / "datasets" / "train.jsonl"
        self.struct_dedup_cap = 5  # max entries per structural fingerprint
        self.val_data = self.root / "datasets" / "validation.jsonl"
        self.models_dir = self.root / "models"
        self.results_dir = self.root / "results"
        self.logs_dir = self.root / "logs"
        self.state_file = self.root / ".pipeline_state.json"
        self.base_model = self._detect_best_model()
        self.epochs = 3
        self.batch_size = 1
        self.lora_r = 64
        self.learning_rate = 2e-4
        self.synthetic_count = 500
        # Error handling
        self.retry_config = STEP_RETRY_CONFIGS
        self.stall_warn_seconds = 300
        self.stall_kill_seconds = 600
        # Training has legitimate long pauses (eval checkpoints, saves,
        # gradient accumulation, CUDA sync) so use a much higher threshold.
        self.stall_kill_seconds_training = 1800

    def _detect_best_model(self):
        """Pick the best model based on available GPU VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu = torch.cuda.get_device_name(0)
                if vram >= 12:
                    print(f"GPU: {gpu} ({vram:.0f} GB) -> Using 8B model")
                    return "meta-llama/Meta-Llama-3.1-8B"
                else:
                    print(f"GPU: {gpu} ({vram:.0f} GB) -> Using 3B model")
                    return "meta-llama/Llama-3.2-3B"
        except Exception:
            pass
        return "meta-llama/Llama-3.2-3B"  # Safe fallback

    def ensure_dirs(self):
        for d in [self.clipboard_inbox, self.clipboard_processed, self.dsl_dir,
                  self.train_data.parent, self.models_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)


class Logger:
    """Hybrid logger: per-run log file + PipelineLogger for step tracking."""

    def __init__(self, logs_dir, run_id=""):
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or str(uuid.uuid4())[:8]
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"pipeline_{self.ts}.log"
        self.fh = open(self.path, "w", encoding="utf-8")
        self.errors = []
        self.retry_counts = {}          # {desc: attempt_number}
        self.completed_step_names = []  # for resume state
        self.error_details = []         # [{step, category, message, timestamp, attempt}]
        # Truncate the shared live log — but only if no other run is active
        live_log = logs_dir / "pipeline_live.log"
        live_state = logs_dir / "pipeline_live_state.json"
        another_running = False
        try:
            if live_state.exists():
                import json as _json
                state = _json.loads(live_state.read_text(encoding="utf-8"))
                if state.get("status") == "running" and state.get("run_id") != self.run_id:
                    another_running = True
        except (OSError, ValueError):
            pass
        if not another_running:
            try:
                live_log.write_text("", encoding="utf-8")
            except OSError:
                pass
        # Set env so subprocesses pick up the run_id
        os.environ["PIPELINE_RUN_ID"] = self.run_id
        self.plog = PipelineLogger(step_prefix="", run_id=self.run_id)

    def log(self, msg, level="INFO"):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}"
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()
        if level == "ERROR":
            self.errors.append(msg)

    def section(self, title):
        self.log("=" * 60)
        self.log(f"  {title}")
        self.log("=" * 60)

    def start_step(self, step_id, description, details=""):
        self.plog.start_step(step_id, description, details)
        self.fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] [STEP {step_id}] STARTING: {description}\n")
        self.fh.flush()

    def complete_step(self, step_id, description="", details=""):
        self.plog.complete_step(step_id, description, details)
        self.fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] [STEP {step_id}] COMPLETE: {description}\n")
        self.fh.flush()

    def close(self):
        # Only write idle if this run owns the live state
        try:
            import json as _json
            live_state = self.path.parent / "pipeline_live_state.json"
            if live_state.exists():
                state = _json.loads(live_state.read_text(encoding="utf-8"))
                if state.get("run_id", "") == self.run_id:
                    self.plog.write_live_state(status="idle")
            else:
                self.plog.write_live_state(status="idle")
        except (OSError, ValueError):
            self.plog.write_live_state(status="idle")
        self.fh.close()
        return self.path


def load_state(cfg):
    if cfg.state_file.exists():
        with open(cfg.state_file) as f:
            return json.load(f)
    return {"last_run": None, "last_training_data_hash": None, "last_model_version": 0, "runs": []}


def save_state(cfg, state):
    with open(cfg.state_file, "w") as f:
        json.dump(state, f, indent=2)


def hash_file(path):
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_script(cfg, log, script, args, desc, dry_run=False, allow_fail=False,
               timeout=7200, extra_env=None, step_category="utility"):
    """Run a subprocess with retry logic, error classification, and stall detection.

    Args:
        step_category: one of "data", "validate", "training", "eval", "exam", "utility"
    """
    cmd = [str(cfg.venv_python), str(cfg.scripts / script)] + args
    log.log(f"Running: {desc}")
    log.log(f"  Cmd: {' '.join(cmd)}")
    if dry_run:
        log.log("  [DRY RUN] Skipped")
        return True

    retry_cfg = cfg.retry_config.get(step_category, RetryConfig())
    effective_timeout = timeout or retry_cfg.timeout
    use_monitor = step_category in ("training", "eval", "exam")

    for attempt in range(retry_cfg.max_retries + 1):
        if attempt > 0:
            backoff = retry_cfg.get_backoff(attempt - 1)
            log.log(f"  [RETRY {attempt}/{retry_cfg.max_retries}] Waiting {backoff}s before retry...")
            log.retry_counts[desc] = attempt
            log.plog.record_retry(
                log.plog._current_step_id or "", attempt, retry_cfg.max_retries)
            time.sleep(backoff)

        try:
            sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if log.run_id:
                sub_env["PIPELINE_RUN_ID"] = log.run_id
            if extra_env:
                sub_env.update(extra_env)

            if use_monitor:
                # Use Popen + SubprocessMonitor for long-running steps
                proc = subprocess.Popen(
                    cmd, cwd=str(cfg.root), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, env=sub_env)
                state_file = cfg.logs_dir / "pipeline_live_state.json"
                # Training gets a much higher stall threshold because of
                # legitimate long pauses (eval checkpoints, saves, CUDA sync).
                kill_secs = (cfg.stall_kill_seconds_training
                             if step_category == "training"
                             else cfg.stall_kill_seconds)
                monitor = SubprocessMonitor(
                    proc, state_file,
                    warn_seconds=cfg.stall_warn_seconds,
                    kill_seconds=kill_secs,
                    log_func=lambda msg: log.log(msg, "WARN"),
                )
                monitor.start()
                try:
                    stdout, stderr = proc.communicate(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    stdout, stderr = proc.communicate(timeout=30)
                    monitor.stop()
                    monitor.join(timeout=5)
                    category = ErrorCategory.TIMEOUT
                    msg = f"TIMED OUT ({effective_timeout / 3600:.1f}h limit)"
                    log.log(f"  {msg}", "ERROR")
                    log.plog.record_error(
                        log.plog._current_step_id or "", category.value, msg)
                    _record_error_detail(log, desc, category, msg, attempt)
                    if not is_retryable(category) or attempt >= retry_cfg.max_retries:
                        return allow_fail
                    continue
                finally:
                    monitor.stop()
                    monitor.join(timeout=5)

                if monitor.was_stalled:
                    category = ErrorCategory.TIMEOUT
                    msg = "Subprocess stalled and was terminated"
                    log.log(f"  {msg}", "ERROR")
                    log.plog.record_error(
                        log.plog._current_step_id or "", category.value, msg)
                    _record_error_detail(log, desc, category, msg, attempt)
                    if attempt >= retry_cfg.max_retries:
                        return allow_fail
                    continue

                r_returncode = proc.returncode
                r_stdout = stdout or ""
                r_stderr = stderr or ""
            else:
                r = subprocess.run(cmd, cwd=str(cfg.root), capture_output=True,
                                   text=True, timeout=effective_timeout, env=sub_env)
                r_returncode = r.returncode
                r_stdout = r.stdout or ""
                r_stderr = r.stderr or ""

            if r_stdout:
                for line in r_stdout.strip().splitlines():
                    log.log(f"  | {line}")

            if r_returncode != 0:
                # Classify the error
                category = classify_error(r_returncode, r_stderr)
                log.log(f"  FAILED (exit {r_returncode}, category={category.value})", "ERROR")

                # Show stderr
                if r_stderr:
                    stderr_lines = r_stderr.strip().splitlines()
                    real_errors = [l for l in stderr_lines
                                   if not any(x in l for x in ["|", "it/s]", "Fetching",
                                              "Loading weights:", "Downloading"])]
                    for line in (real_errors or stderr_lines[-30:])[:50]:
                        log.log(f"  ERR| {line}", "ERROR")

                msg = f"exit {r_returncode}: {category.value}"
                log.plog.record_error(
                    log.plog._current_step_id or "", category.value, msg)
                _record_error_detail(log, desc, category, msg, attempt)

                # Check if retryable
                if is_retryable(category) and attempt < retry_cfg.max_retries:
                    log.log(f"  Error is retryable ({category.value}). Will retry.")
                    continue
                elif not is_retryable(category):
                    log.log(f"  Error is NOT retryable ({category.value}). Stopping.")
                    return False

                return allow_fail

            # Success
            if attempt > 0:
                log.log(f"  Succeeded on retry {attempt}")
            return True

        except subprocess.TimeoutExpired:
            category = ErrorCategory.TIMEOUT
            msg = f"TIMED OUT ({effective_timeout / 3600:.1f}h limit)"
            log.log(f"  {msg}", "ERROR")
            log.plog.record_error(
                log.plog._current_step_id or "", category.value, msg)
            _record_error_detail(log, desc, category, msg, attempt)
            if attempt >= retry_cfg.max_retries:
                return False
            continue
        except Exception as e:
            category = classify_error(1, "", e)
            msg = f"EXCEPTION: {e}"
            log.log(f"  {msg}", "ERROR")
            log.plog.record_error(
                log.plog._current_step_id or "", category.value, msg)
            _record_error_detail(log, desc, category, msg, attempt)
            if is_retryable(category) and attempt < retry_cfg.max_retries:
                continue
            return False

    return False


def _record_error_detail(log, desc, category, message, attempt):
    """Append to log.error_details for resume state tracking."""
    log.error_details.append({
        "step": desc,
        "category": category.value,
        "message": message[:500],
        "timestamp": datetime.now().isoformat(),
        "attempt": attempt,
    })


def get_latest_model(cfg):
    dirs = sorted(cfg.models_dir.glob("blueprint-lora-v*/final"), reverse=True)
    return dirs[0] if dirs else None


def _dedup_entries(entries, struct_cap, log):
    """Deduplicate training entries: exact output dedup + structural near-dedup cap."""
    from collections import Counter

    def structural_fingerprint(output):
        s = re.sub(r'"[^"]*"', '"X"', output)
        s = re.sub(r'=\d+\.?\d*', '=N', s)
        return s

    before = len(entries)

    # Phase 1: Exact output dedup (keep first occurrence)
    seen_hashes = set()
    phase1 = []
    for entry_line in entries:
        try:
            obj = json.loads(entry_line) if isinstance(entry_line, str) else entry_line
            output = obj.get("output", "")
        except (json.JSONDecodeError, AttributeError):
            phase1.append(entry_line)
            continue
        h = hashlib.md5(output.encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        phase1.append(entry_line)

    exact_removed = before - len(phase1)

    # Phase 2: Structural near-dedup
    if struct_cap > 0:
        fp_counts = Counter()
        phase2 = []
        for entry_line in phase1:
            try:
                obj = json.loads(entry_line) if isinstance(entry_line, str) else entry_line
                output = obj.get("output", "")
            except (json.JSONDecodeError, AttributeError):
                phase2.append(entry_line)
                continue
            fp = structural_fingerprint(output)
            fp_counts[fp] += 1
            if fp_counts[fp] > struct_cap:
                continue
            phase2.append(entry_line)
        struct_removed = len(phase1) - len(phase2)
        final = phase2
    else:
        struct_removed = 0
        final = phase1

    total_removed = before - len(final)
    log.log(f"  Dedup: {before} -> {len(final)} (removed {exact_removed} exact, {struct_removed} structural)")
    return final


# ===========================================================================
# STEP 1: Data Foundation
# ===========================================================================

def step_data_foundation(cfg, log, dry_run):
    """Steps 1.x: Analyze, translate, synthetic, lessons, merge+dedup."""
    log.section("STEP 1: Data Foundation")
    log.start_step("1", "Data Foundation")

    # 1.1 + 1.2: Analyze & Auto-Translate
    new_files = []
    for f in cfg.clipboard_inbox.glob("*.txt"):
        if f.name.startswith("processed_"):
            continue
        if ".dsl" in f.name or ".analysis" in f.name:
            continue
        new_files.append(f)
    if not new_files:
        log.log("No new exports found.")
    else:
        log.log(f"Found {len(new_files)} new export(s)")
        done = 0
        for f in new_files:
            ok = run_script(cfg, log, "01_analyze_blueprint_clipboard.py", [str(f)],
                            f"Analyzing {f.name}", dry_run, allow_fail=True,
                            step_category="data")
            if ok:
                run_script(cfg, log, "05_auto_translate_export.py",
                           [str(f), "--training",
                            "--dsl-dir", str(cfg.dsl_dir),
                            "--jsonl", str(cfg.root / "datasets" / "auto_translated.jsonl")],
                           f"Auto-translating {f.name}", dry_run, allow_fail=True,
                           step_category="data")
                done += 1
                if not dry_run:
                    cfg.clipboard_processed.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(cfg.clipboard_processed / f.name))
        log.log(f"Analyzed and translated {done}/{len(new_files)}")

    # 1.3: Generate synthetic data
    run_script(cfg, log, "03_generate_synthetic_data.py",
               ["--count", str(cfg.synthetic_count), "--output", str(cfg.synthetic_data), "--seed", "42"],
               f"Generating {cfg.synthetic_count} synthetic examples", dry_run,
               step_category="data")

    # 1.4: Convert lessons
    lesson_dir = cfg.root / "lessons"
    if lesson_dir.exists() and list(lesson_dir.glob("lesson_*.json")):
        run_script(cfg, log, "13_lesson_to_training.py",
                   ["--lesson-dir", str(lesson_dir), "--output", str(cfg.lesson_data), "--no-append"],
                   "Converting lessons to training data", dry_run, allow_fail=True,
                   extra_env={"PIPELINE_STEP_CONTEXT": "1"}, step_category="data")

    # 1.5: Merge + Deduplicate
    if not dry_run:
        entries = []
        if cfg.synthetic_data.exists():
            with open(cfg.synthetic_data) as f:
                entries = [l.strip() for l in f if l.strip()]
            log.log(f"  {len(entries)} synthetic examples")
        # Auto-translated entries
        auto_path = cfg.root / "datasets" / "auto_translated.jsonl"
        ac = 0
        if auto_path.exists():
            with open(auto_path) as f:
                for l in f:
                    if l.strip():
                        entries.append(l.strip())
                        ac += 1
            log.log(f"  {ac} auto-translated examples")
        mc = 0
        if cfg.manual_data.exists():
            with open(cfg.manual_data) as f:
                for l in f:
                    if l.strip():
                        entries.append(l.strip())
                        mc += 1
            log.log(f"  {mc} manual examples (highest priority)")
        # Lesson data
        lc = 0
        if cfg.lesson_data.exists():
            with open(cfg.lesson_data) as f:
                for l in f:
                    if l.strip():
                        entries.append(l.strip())
                        lc += 1
            log.log(f"  {lc} lesson examples")
        # Deduplicate
        entries = _dedup_entries(entries, cfg.struct_dedup_cap, log)
        with open(cfg.train_data, "w") as f:
            f.write("\n".join(entries) + "\n")
        log.log(f"  Total: {len(entries)} training examples")
        sv = cfg.synthetic_data.with_name("validation.jsonl")
        if sv.exists() and sv.resolve() != cfg.val_data.resolve():
            shutil.copy2(str(sv), str(cfg.val_data))

    log.complete_step("1", "Data Foundation")
    return True


# ===========================================================================
# STEP 2: Pre-Flight Checks
# ===========================================================================

def step_preflight(cfg, log, state, dry_run):
    """Steps 2.x: Validate DSL, validate dataset, generate prompt, compute hash."""
    log.section("STEP 2: Pre-Flight Checks")
    log.start_step("2", "Pre-Flight Checks")

    # 2.1: Validate DSL files
    dsl_files = list(cfg.dsl_dir.glob("*.dsl"))
    if dsl_files:
        for f in dsl_files:
            run_script(cfg, log, "06_validate_dsl.py", [str(f)],
                       f"Validating {f.name}", dry_run, allow_fail=True,
                       step_category="validate")

    # 2.2: Validate merged dataset
    if cfg.train_data.exists():
        run_script(cfg, log, "06_validate_dsl.py", [str(cfg.train_data)],
                   "Validating merged dataset", dry_run, allow_fail=True,
                   extra_env={"PIPELINE_STEP_ID": "2.2"}, step_category="validate")

    # 2.3: Generate system prompt
    run_script(cfg, log, "08_generate_system_prompt.py",
               ["--output", str(cfg.scripts / "system_prompt.txt")],
               "Generating system prompt", dry_run, step_category="utility")

    # 2.4: Compute data hash
    log.start_step("2.4", "Compute data hash")
    cur_hash = hash_file(cfg.train_data)
    prev_hash = state.get("last_training_data_hash", "")
    if cur_hash == prev_hash:
        log.log(f"Data hash unchanged: {cur_hash}")
    else:
        log.log(f"Data hash changed: {prev_hash} -> {cur_hash}")
    log.complete_step("2.4", "Compute data hash", f"hash={cur_hash}")

    log.complete_step("2", "Pre-Flight Checks")
    return cur_hash


# ===========================================================================
# STEPS 3+4: Training (subprocess handles 3.x and 4.x)
# ===========================================================================

def step_train(cfg, log, state, dry_run, force, data_hash=None):
    """Steps 3.x + 4.x: Training subprocess (04_train_blueprint_lora.py)."""
    log.section("STEPS 3-4: Model Setup & Training")
    cur_hash = data_hash or hash_file(cfg.train_data)
    if cur_hash == state.get("last_training_data_hash", "") and not force:
        log.log("Training data unchanged. Skipping. (Use --force to override)")
        log.start_step("3", "Model Setup")
        log.complete_step("3", "Model Setup", "Skipped (data unchanged)")
        log.start_step("4", "Training")
        log.complete_step("4", "Training", "Skipped (data unchanged)")
        m = get_latest_model(cfg)
        return str(m) if m else None
    ver = state.get("last_model_version", 0) + 1
    model_dir = cfg.models_dir / f"blueprint-lora-v{ver}"
    log.log(f"Training v{ver} (data hash: {cur_hash})")
    ok = run_script(cfg, log, "04_train_blueprint_lora.py", [
        "--base_model", cfg.base_model, "--dataset", str(cfg.train_data),
        "--output", str(model_dir), "--epochs", str(cfg.epochs),
        "--batch_size", str(cfg.batch_size), "--lr", str(cfg.learning_rate),
        "--lora_r", str(cfg.lora_r)
    ], f"Fine-tuning v{ver}", dry_run, timeout=14400, step_category="training")
    if ok or dry_run:
        state["last_training_data_hash"] = cur_hash
        state["last_model_version"] = ver
        return str(model_dir / "final")
    return None


# ===========================================================================
# STEP 5: Post-Training
# ===========================================================================

def step_post_training(cfg, log, model_path, state, dry_run):
    """Steps 5.x: Verify, summary, health check, history, archive."""
    log.section("STEP 5: Post-Training")
    log.start_step("5", "Post-Training")

    # 5.1: Verify model output
    log.start_step("5.1", "Verify model output")
    if model_path:
        model_final = Path(model_path)
        if model_final.exists():
            log.log(f"Model output verified at {model_final}")
        else:
            log.log(f"Model output not found at {model_final}", "WARN")
    else:
        log.log("No model path. Skipping verification.", "WARN")
    log.complete_step("5.1", "Verify model output")

    # 5.2: Training summary
    if model_path:
        parent = str(Path(model_path).parent) if model_path.endswith("final") else model_path
        run_script(cfg, log, "10_training_dashboard.py", ["--summary", parent],
                   "Training summary", dry_run, allow_fail=True,
                   step_category="utility")

    # 5.3 + 5.4: Health check
    ver = state.get("last_model_version", 0)
    if ver >= 1:
        run_script(cfg, log, "19_training_health_monitor.py",
                   ["--version", f"v{ver}", "--project-root", str(cfg.root)],
                   f"Health check v{ver}", dry_run, allow_fail=True,
                   step_category="utility")
    else:
        log.log("No model version to check. Skipping health monitor.")

    log.complete_step("5", "Post-Training")


# ===========================================================================
# STEP 6: Exam
# ===========================================================================

def step_exam(cfg, log, model_path, dry_run):
    """Steps 6.x: Auto-detect latest lessons and run exams."""
    log.section("STEP 6: Exam")
    log.start_step("6", "Exam")

    if not model_path:
        log.log("No model available. Skipping exams.", "WARN")
        log.complete_step("6", "Exam", "Skipped (no model)")
        return

    lesson_dir = cfg.root / "lessons"
    lesson_files = sorted(lesson_dir.glob("lesson_*.json")) if lesson_dir.exists() else []
    if not lesson_files:
        log.log("No lesson files found. Skipping exams.")
        log.complete_step("6", "Exam", "No lessons")
        return

    log.log(f"Found {len(lesson_files)} lesson(s) to examine")
    for lf in lesson_files:
        run_script(cfg, log, "12_run_exam.py",
                   ["--lesson", str(lf), "--model", model_path],
                   f"Exam: {lf.name}", dry_run, allow_fail=True, timeout=14400,
                   step_category="exam")

    log.complete_step("6", "Exam", f"{len(lesson_files)} lessons examined")


# ===========================================================================
# STEP 7: Evaluation (grading)
# ===========================================================================

def step_grading(cfg, log, model_path, state, dry_run):
    """Steps 7.x: Run tier-based evaluation test suite."""
    log.section("STEP 7: Evaluation")
    log.start_step("7", "Evaluation")
    if not model_path:
        log.log("No model. Skipping.", "WARN")
        log.complete_step("7", "Evaluation", "Skipped (no model)")
        return None
    ver = state.get("last_model_version", 0)
    rpt = cfg.results_dir / f"eval_v{ver}_{datetime.now().strftime('%Y%m%d')}.json"
    run_script(cfg, log, "09_evaluate_model.py",
               ["--model", model_path, "--report", str(rpt)],
               f"Evaluating {model_path}", dry_run, timeout=14400,
               step_category="eval")
    if rpt.exists():
        with open(rpt) as f:
            report = json.load(f)
        s = report.get("summary", {})
        log.complete_step("7", "Evaluation", f"{s.get('passed','?')}/{s.get('total','?')} passed")
        return report
    log.complete_step("7", "Evaluation")
    return None


# ===========================================================================
# STEP 8: Lesson Integration
# ===========================================================================

def step_lesson_integration(cfg, log, dry_run):
    """Steps 8.x: Convert lessons for next training cycle (context=8)."""
    log.section("STEP 8: Lesson Integration")
    log.start_step("8", "Lesson Integration")

    lesson_dir = cfg.root / "lessons"
    if not lesson_dir.exists() or not list(lesson_dir.glob("lesson_*.json")):
        log.log("No lessons found. Skipping.")
        log.complete_step("8", "Lesson Integration", "No lessons")
        return

    run_script(cfg, log, "13_lesson_to_training.py",
               ["--lesson-dir", str(lesson_dir), "--output", str(cfg.lesson_data), "--no-append"],
               "Integrating lessons for next cycle", dry_run, allow_fail=True,
               extra_env={"PIPELINE_STEP_CONTEXT": "8"}, step_category="data")

    log.complete_step("8", "Lesson Integration")


# ===========================================================================
# STEP 9: Dashboard & Finalize
# ===========================================================================

def step_dashboard_finalize(cfg, log, dry_run):
    """Steps 9.x: Update dashboard."""
    log.section("STEP 9: Dashboard & Finalize")
    log.start_step("9", "Dashboard & Finalize")
    run_script(cfg, log, "14_update_dashboard.py", [],
               "Updating dashboard", dry_run, allow_fail=True, step_category="utility")
    log.complete_step("9", "Dashboard & Finalize")


# ===========================================================================
# PIPELINE RUNNER
# ===========================================================================

def run_pipeline(mode, dry_run=False, force=False, root=None, resume=False):
    cfg = PipelineConfig(root)
    cfg.ensure_dirs()
    state = load_state(cfg)
    log = Logger(cfg.logs_dir)
    start = datetime.now()

    # Resume support: check for saved resume state
    resume_state = None
    skip_steps = set()
    if resume:
        resume_state = load_resume_state()
        if resume_state:
            skip_steps = set(resume_state.get("completed_steps", []))
            log.log(f"RESUMING from previous run. Skipping {len(skip_steps)} completed steps.")
            log.log(f"  Failed step was: {resume_state.get('failed_step', '?')}")
            # Restore state snapshot if available
            snapshot = resume_state.get("state_snapshot")
            if snapshot:
                for key in ("last_training_data_hash", "last_model_version"):
                    if key in snapshot:
                        state[key] = snapshot[key]
        else:
            log.log("--resume specified but no resume state found. Starting fresh.")

    # Register step plan for timeline and ETA tracking
    log.plog.set_step_plan(STEP_PLANS.get(mode, []))

    log.section(f"BLUEPRINT LLM PIPELINE — {mode.upper()}")
    log.log(f"Root: {cfg.root}")
    log.log(f"Time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"Last run: {state.get('last_run', 'Never')}")
    log.log(f"Model: v{state.get('last_model_version', 0)}")
    if dry_run:
        log.log("*** DRY RUN ***")
    if resume and skip_steps:
        log.log(f"Resume: skipping {sorted(skip_steps)}")

    model_path = None
    eval_report = None
    data_hash = None
    stopped = False
    failed_step_name = None

    def check_stop():
        """Return True if a graceful stop was requested between stages."""
        if is_stop_requested():
            log.log("GRACEFUL STOP requested between pipeline stages.")
            clear_signal()
            return True
        return False

    def should_skip(step_name):
        """Return True if this step was already completed in a prior run."""
        return step_name in skip_steps

    def mark_completed(step_name):
        """Record step as completed for resume state."""
        log.completed_step_names.append(step_name)

    try:
        # Steps 1-2: Data Foundation + Pre-Flight
        if mode in ("full", "data-only"):
            if not should_skip("step_data_foundation"):
                step_data_foundation(cfg, log, dry_run)
                mark_completed("step_data_foundation")
            else:
                log.log("Skipping step_data_foundation (already completed)")
            if check_stop(): stopped = True
            if not stopped:
                if not should_skip("step_preflight"):
                    data_hash = step_preflight(cfg, log, state, dry_run)
                    mark_completed("step_preflight")
                else:
                    log.log("Skipping step_preflight (already completed)")
                    data_hash = hash_file(cfg.train_data)
            if check_stop(): stopped = True

        # Steps 3-5: Training + Post-Training
        if not stopped and mode in ("full", "train-only"):
            if check_stop(): stopped = True
            if not stopped:
                if not should_skip("step_train"):
                    model_path = step_train(cfg, log, state, dry_run, force, data_hash)
                    if model_path:
                        mark_completed("step_train")
                    else:
                        failed_step_name = "step_train"
                else:
                    log.log("Skipping step_train (already completed)")
                    model_path = str(get_latest_model(cfg)) if get_latest_model(cfg) else None
            if not stopped and model_path and not check_stop():
                if not should_skip("step_post_training"):
                    step_post_training(cfg, log, model_path, state, dry_run)
                    mark_completed("step_post_training")
                else:
                    log.log("Skipping step_post_training (already completed)")
            else:
                stopped = stopped or not model_path

        # Steps 6-9: Exam, Evaluation, Lesson Integration, Dashboard (full only)
        if not stopped and mode == "full":
            if not check_stop():
                if not should_skip("step_exam"):
                    step_exam(cfg, log, model_path, dry_run)
                    mark_completed("step_exam")
                else:
                    log.log("Skipping step_exam (already completed)")
            if not check_stop():
                if not should_skip("step_grading"):
                    eval_report = step_grading(cfg, log, model_path, state, dry_run)
                    mark_completed("step_grading")
                else:
                    log.log("Skipping step_grading (already completed)")
            if not check_stop():
                if not should_skip("step_lesson_integration"):
                    step_lesson_integration(cfg, log, dry_run)
                    mark_completed("step_lesson_integration")
                else:
                    log.log("Skipping step_lesson_integration (already completed)")
            if not check_stop():
                if not should_skip("step_dashboard_finalize"):
                    step_dashboard_finalize(cfg, log, dry_run)
                    mark_completed("step_dashboard_finalize")
                else:
                    log.log("Skipping step_dashboard_finalize (already completed)")

        # Eval-only mode
        if not stopped and mode == "eval-only":
            m = get_latest_model(cfg)
            if m:
                eval_report = step_grading(cfg, log, str(m), state, dry_run)
            else:
                log.log("No model found.", "ERROR")
    except Exception as e:
        log.log(f"Pipeline exception: {e}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
        failed_step_name = failed_step_name or "unknown"

    # Save resume state on failure so --resume can pick up
    if log.errors and not dry_run:
        save_resume_state(
            completed_steps=log.completed_step_names,
            failed_step=failed_step_name or "unknown",
            error_info={
                "errors": log.error_details[-5:] if log.error_details else [],
                "message": log.errors[-1] if log.errors else "",
            },
            state_snapshot={
                "last_training_data_hash": state.get("last_training_data_hash"),
                "last_model_version": state.get("last_model_version", 0),
            },
        )
        log.log(f"Resume state saved. Re-run with --resume to continue.")
    elif not log.errors:
        # Pipeline succeeded — clear any old resume state
        clear_resume_state()

    elapsed = (datetime.now() - start).total_seconds()
    log.section("PIPELINE STOPPED EARLY" if stopped else "PIPELINE COMPLETE")
    log.log(f"Time: {int(elapsed//60)}m {int(elapsed%60)}s")
    log.log(f"Version: v{state.get('last_model_version', '?')}")
    log.log(f"Errors: {len(log.errors)}")
    if log.error_details:
        log.log(f"Error categories: {', '.join(set(e['category'] for e in log.error_details))}")
    if eval_report:
        s = eval_report.get("summary", {})
        log.log(f"Eval: {s.get('passed','?')}/{s.get('total','?')} ({s.get('avg_score',0)*100:.0f}%)")

    state.setdefault("runs", []).append({
        "timestamp": start.isoformat(), "duration": round(elapsed),
        "version": state.get("last_model_version", 0),
        "errors": len(log.errors),
        "eval": eval_report.get("summary") if eval_report else None,
    })
    state["runs"] = state["runs"][-50:]
    state["last_run"] = start.isoformat()

    lp = log.close()
    if not dry_run:
        save_state(cfg, state)
    print(f"\nLog: {lp}")
    sys.exit(1 if log.errors else (2 if mode in ("full", "train-only") and not model_path else 0))


def main():
    p = argparse.ArgumentParser(description="Blueprint LLM Pipeline")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--full", action="store_true")
    m.add_argument("--data-only", action="store_true")
    m.add_argument("--train-only", action="store_true")
    m.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last failure (skip already-completed steps)")
    p.add_argument("--project-root", type=str)
    a = p.parse_args()
    mode = "full" if a.full else "data-only" if a.data_only else "train-only" if a.train_only else "eval-only"
    run_pipeline(mode, a.dry_run, a.force, a.project_root, a.resume)


if __name__ == "__main__":
    main()
