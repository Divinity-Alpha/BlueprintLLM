"""
11_pipeline_orchestrator.py
---------------------------
Master orchestrator that chains the entire Blueprint LLM pipeline.
Designed to run unattended via Windows Task Scheduler.

All interactive prompts are suppressed. Everything is logged.

Usage:
    python scripts/11_pipeline_orchestrator.py --full
    python scripts/11_pipeline_orchestrator.py --full --dry-run
    python scripts/11_pipeline_orchestrator.py --data-only
    python scripts/11_pipeline_orchestrator.py --train-only --force
    python scripts/11_pipeline_orchestrator.py --eval-only

Exit codes: 0=Success, 1=Failure, 2=Skipped (no new data)
"""

import os, sys, json, re, shutil, subprocess, hashlib, argparse, uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from stop_signal_utils import is_stop_requested, clear_signal
from pipeline_logger import PipelineLogger, format_duration


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
        # Truncate the shared live log at pipeline start
        live_log = logs_dir / "pipeline_live.log"
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


def run_script(cfg, log, script, args, desc, dry_run=False, allow_fail=False, timeout=7200):
    cmd = [str(cfg.venv_python), str(cfg.scripts / script)] + args
    log.log(f"Running: {desc}")
    log.log(f"  Cmd: {' '.join(cmd)}")
    if dry_run:
        log.log("  [DRY RUN] Skipped")
        return True
    try:
        sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if log.run_id:
            sub_env["PIPELINE_RUN_ID"] = log.run_id
        r = subprocess.run(cmd, cwd=str(cfg.root), capture_output=True, text=True,
                           timeout=timeout, env=sub_env)
        if r.stdout:
            for line in r.stdout.strip().splitlines():
                log.log(f"  | {line}")
        if r.returncode != 0:
            log.log(f"  FAILED (exit {r.returncode})", "ERROR")
            if r.stderr:
                # Filter out progress bar lines (contain \r or |###) to find real errors
                stderr_lines = r.stderr.strip().splitlines()
                real_errors = [l for l in stderr_lines
                               if not any(x in l for x in ["|", "it/s]", "Fetching", "Loading weights:", "Downloading"])]
                # Show real errors first (up to 50 lines)
                for line in real_errors[:50]:
                    log.log(f"  ERR| {line}", "ERROR")
                # If no real errors found, show last 30 raw lines
                if not real_errors:
                    log.log("  (Only progress bars in stderr, showing last 30 lines:)", "ERROR")
                    for line in stderr_lines[-30:]:
                        log.log(f"  ERR| {line}", "ERROR")
            return allow_fail
        return True
    except subprocess.TimeoutExpired:
        hours = timeout / 3600
        log.log(f"  TIMED OUT ({hours:.0f}h limit)", "ERROR")
        return False
    except Exception as e:
        log.log(f"  EXCEPTION: {e}", "ERROR")
        return False


def get_latest_model(cfg):
    dirs = sorted(cfg.models_dir.glob("blueprint-lora-v*/final"), reverse=True)
    return dirs[0] if dirs else None


def step_analyze(cfg, log, dry_run):
    log.section("STEP 1: Analyze & Auto-Translate New Blueprint Exports")
    log.start_step(1, "Analyze & Auto-Translate")
    # Only pick up actual clipboard exports — skip .dsl.txt, .analysis.json, and other artifacts
    new_files = []
    for f in cfg.clipboard_inbox.glob("*.txt"):
        if f.name.startswith("processed_"):
            continue
        if ".dsl" in f.name or ".analysis" in f.name:
            continue  # Skip artifacts from previous analyzer runs
        new_files.append(f)
    if not new_files:
        log.log("No new exports found. Skipping.")
        log.complete_step(1, "Analyze & Auto-Translate", "No new exports")
        return 0
    log.log(f"Found {len(new_files)} new export(s)")
    done = 0
    for f in new_files:
        # Step 1a: Analyze (raw structure dump)
        ok = run_script(cfg, log, "01_analyze_blueprint_clipboard.py", [str(f)],
                        f"Analyzing {f.name}", dry_run, allow_fail=True)
        # Step 1b: Auto-translate to clean DSL + training entries
        if ok:
            run_script(cfg, log, "05_auto_translate_export.py",
                       [str(f), "--training",
                        "--dsl-dir", str(cfg.dsl_dir),
                        "--jsonl", str(cfg.root / "datasets" / "auto_translated.jsonl")],
                       f"Auto-translating {f.name}", dry_run, allow_fail=True)
            done += 1
            if not dry_run:
                cfg.clipboard_processed.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(cfg.clipboard_processed / f.name))
    log.log(f"Analyzed and translated {done}/{len(new_files)}")
    log.complete_step(1, "Analyze & Auto-Translate", f"{done}/{len(new_files)} files")
    return done


def step_validate(cfg, log, dry_run):
    log.section("STEP 2: Validate DSL Files")
    log.start_step(2, "Validate DSL Files")
    dsl_files = list(cfg.dsl_dir.glob("*.dsl"))
    if not dsl_files:
        log.log("No DSL files. Skipping.")
        log.complete_step(2, "Validate DSL Files", "No DSL files")
        return True
    ok = True
    for f in dsl_files:
        if not run_script(cfg, log, "06_validate_dsl.py", [str(f)],
                          f"Validating {f.name}", dry_run, allow_fail=True):
            ok = False
    log.complete_step(2, "Validate DSL Files", f"{len(dsl_files)} files checked")
    return ok


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


def step_merge(cfg, log, dry_run):
    log.section("STEP 3: Build Training Dataset")
    log.start_step(3, "Build Training Dataset")
    run_script(cfg, log, "03_generate_synthetic_data.py",
               ["--count", str(cfg.synthetic_count), "--output", str(cfg.synthetic_data), "--seed", "42"],
               f"Generating {cfg.synthetic_count} synthetic examples", dry_run)
    # Generate lesson data if lessons exist
    lesson_dir = cfg.root / "lessons"
    if lesson_dir.exists() and list(lesson_dir.glob("lesson_*.json")):
        run_script(cfg, log, "13_lesson_to_training.py",
                   ["--lesson-dir", str(lesson_dir), "--output", str(cfg.lesson_data), "--no-append"],
                   "Converting lessons to training data", dry_run, allow_fail=True)
    if not dry_run:
        entries = []
        if cfg.synthetic_data.exists():
            with open(cfg.synthetic_data) as f:
                entries = [l.strip() for l in f if l.strip()]
            log.log(f"  {len(entries)} synthetic examples")
        # Auto-translated entries (from 05_auto_translate_export.py)
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
    run_script(cfg, log, "06_validate_dsl.py", [str(cfg.train_data)],
               "Validating merged dataset", dry_run, allow_fail=True)
    log.complete_step(3, "Build Training Dataset")
    return True


def step_prompt(cfg, log, dry_run):
    log.section("STEP 4: Generate System Prompt")
    log.start_step(4, "Generate System Prompt")
    result = run_script(cfg, log, "08_generate_system_prompt.py",
                        ["--output", str(cfg.scripts / "system_prompt.txt")],
                        "Generating system prompt", dry_run)
    log.complete_step(4, "Generate System Prompt")
    return result


def step_train(cfg, log, state, dry_run, force):
    log.section("STEP 5: Train Model")
    log.start_step(5, "Train Model")
    cur_hash = hash_file(cfg.train_data)
    if cur_hash == state.get("last_training_data_hash", "") and not force:
        log.log("Training data unchanged. Skipping. (Use --force to override)")
        log.complete_step(5, "Train Model", "Skipped (data unchanged)")
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
    ], f"Fine-tuning v{ver}", dry_run, timeout=14400)
    if ok or dry_run:
        state["last_training_data_hash"] = cur_hash
        state["last_model_version"] = ver
        log.complete_step(5, "Train Model", f"v{ver}")
        return str(model_dir / "final")
    log.complete_step(5, "Train Model", "FAILED")
    return None


def step_evaluate(cfg, log, model_path, state, dry_run):
    log.section("STEP 6: Evaluate Model")
    log.start_step(6, "Evaluate Model")
    if not model_path:
        log.log("No model. Skipping.", "WARN")
        log.complete_step(6, "Evaluate Model", "Skipped (no model)")
        return None
    ver = state.get("last_model_version", 0)
    rpt = cfg.results_dir / f"eval_v{ver}_{datetime.now().strftime('%Y%m%d')}.json"
    run_script(cfg, log, "09_evaluate_model.py",
               ["--model", model_path, "--report", str(rpt)],
               f"Evaluating {model_path}", dry_run, timeout=14400)
    if rpt.exists():
        with open(rpt) as f:
            report = json.load(f)
        s = report.get("summary", {})
        log.complete_step(6, "Evaluate Model", f"{s.get('passed','?')}/{s.get('total','?')} passed")
        return report
    log.complete_step(6, "Evaluate Model")
    return None


def step_summary(cfg, log, model_path, dry_run):
    log.section("STEP 7: Training Summary")
    log.start_step(7, "Training Summary")
    if not model_path:
        log.complete_step(7, "Training Summary", "Skipped (no model)")
        return
    parent = str(Path(model_path).parent) if model_path.endswith("final") else model_path
    run_script(cfg, log, "10_training_dashboard.py", ["--summary", parent],
               "Training summary", dry_run, allow_fail=True)
    log.complete_step(7, "Training Summary")


def step_health_monitor(cfg, log, state, dry_run):
    log.section("STEP 8: Training Health Monitor")
    log.start_step(8, "Training Health Monitor")
    ver = state.get("last_model_version", 0)
    if ver < 1:
        log.log("No model version to check. Skipping.")
        log.complete_step(8, "Training Health Monitor", "Skipped")
        return
    run_script(cfg, log, "19_training_health_monitor.py",
               ["--version", f"v{ver}", "--project-root", str(cfg.root)],
               f"Health check v{ver}", dry_run, allow_fail=True)
    log.complete_step(8, "Training Health Monitor")


def step_dashboard(cfg, log, dry_run):
    log.section("STEP 9: Update Dashboard")
    log.start_step(9, "Update Dashboard")
    run_script(cfg, log, "14_update_dashboard.py", [],
               "Updating dashboard", dry_run, allow_fail=True)
    log.complete_step(9, "Update Dashboard")


def run_pipeline(mode, dry_run=False, force=False, root=None):
    cfg = PipelineConfig(root)
    cfg.ensure_dirs()
    state = load_state(cfg)
    log = Logger(cfg.logs_dir)
    start = datetime.now()

    log.section(f"BLUEPRINT LLM PIPELINE — {mode.upper()}")
    log.log(f"Root: {cfg.root}")
    log.log(f"Time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"Last run: {state.get('last_run', 'Never')}")
    log.log(f"Model: v{state.get('last_model_version', 0)}")
    if dry_run:
        log.log("*** DRY RUN ***")

    model_path = None
    eval_report = None
    stopped = False

    def check_stop():
        """Return True if a graceful stop was requested between stages."""
        if is_stop_requested():
            log.log("GRACEFUL STOP requested between pipeline stages.")
            clear_signal()
            return True
        return False

    try:
        if mode in ("full", "data-only"):
            step_analyze(cfg, log, dry_run)
            if check_stop(): stopped = True
            if not stopped:
                step_validate(cfg, log, dry_run)
            if not stopped and not check_stop():
                step_merge(cfg, log, dry_run)
            else:
                stopped = True
            if not stopped and not check_stop():
                step_prompt(cfg, log, dry_run)
            else:
                stopped = True
        if not stopped and mode in ("full", "train-only"):
            if check_stop(): stopped = True
            if not stopped:
                model_path = step_train(cfg, log, state, dry_run, force)
            if not stopped and model_path and not check_stop():
                eval_report = step_evaluate(cfg, log, model_path, state, dry_run)
            if not stopped and model_path and not check_stop():
                step_summary(cfg, log, model_path, dry_run)
            if not stopped and not check_stop():
                step_health_monitor(cfg, log, state, dry_run)
            if not stopped and not check_stop():
                step_dashboard(cfg, log, dry_run)
        if not stopped and mode == "eval-only":
            m = get_latest_model(cfg)
            if m:
                eval_report = step_evaluate(cfg, log, str(m), state, dry_run)
            else:
                log.log("No model found.", "ERROR")
    except Exception as e:
        log.log(f"Pipeline exception: {e}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")

    elapsed = (datetime.now() - start).total_seconds()
    log.section("PIPELINE STOPPED EARLY" if stopped else "PIPELINE COMPLETE")
    log.log(f"Time: {int(elapsed//60)}m {int(elapsed%60)}s")
    log.log(f"Version: v{state.get('last_model_version', '?')}")
    log.log(f"Errors: {len(log.errors)}")
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
    p.add_argument("--project-root", type=str)
    a = p.parse_args()
    mode = "full" if a.full else "data-only" if a.data_only else "train-only" if a.train_only else "eval-only"
    run_pipeline(mode, a.dry_run, a.force, a.project_root)


if __name__ == "__main__":
    main()
