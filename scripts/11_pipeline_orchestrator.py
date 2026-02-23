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

import os, sys, json, shutil, subprocess, hashlib, argparse
from pathlib import Path
from datetime import datetime


class PipelineConfig:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("BLUEPRINT_LLM_ROOT", r"C:\BlueprintLLM"))
        self.venv_python = self.root / "venv" / "Scripts" / "python.exe"
        self.scripts = self.root / "scripts"
        self.clipboard_inbox = self.root / "raw-data" / "clipboard-exports"
        self.clipboard_processed = self.root / "raw-data" / "clipboard-exports" / "processed"
        self.dsl_dir = self.root / "cleaned-data" / "parsed-blueprints"
        self.manual_data = self.root / "datasets" / "manual_examples.jsonl"
        self.synthetic_data = self.root / "datasets" / "synthetic_train.jsonl"
        self.train_data = self.root / "datasets" / "train.jsonl"
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
    def __init__(self, logs_dir):
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"pipeline_{self.ts}.log"
        self.fh = open(self.path, "w", encoding="utf-8")
        self.errors = []

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

    def close(self):
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


def run_script(cfg, log, script, args, desc, dry_run=False, allow_fail=False):
    cmd = [str(cfg.venv_python), str(cfg.scripts / script)] + args
    log.log(f"Running: {desc}")
    log.log(f"  Cmd: {' '.join(cmd)}")
    if dry_run:
        log.log("  [DRY RUN] Skipped")
        return True
    try:
        r = subprocess.run(cmd, cwd=str(cfg.root), capture_output=True, text=True,
                           timeout=7200, env={**os.environ, "PYTHONUNBUFFERED": "1"})
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
        log.log("  TIMED OUT (2h limit)", "ERROR")
        return False
    except Exception as e:
        log.log(f"  EXCEPTION: {e}", "ERROR")
        return False


def get_latest_model(cfg):
    dirs = sorted(cfg.models_dir.glob("blueprint-lora-v*/final"), reverse=True)
    return dirs[0] if dirs else None


def step_analyze(cfg, log, dry_run):
    log.section("STEP 1: Analyze & Auto-Translate New Blueprint Exports")
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
    return done


def step_validate(cfg, log, dry_run):
    log.section("STEP 2: Validate DSL Files")
    dsl_files = list(cfg.dsl_dir.glob("*.dsl"))
    if not dsl_files:
        log.log("No DSL files. Skipping.")
        return True
    ok = True
    for f in dsl_files:
        if not run_script(cfg, log, "06_validate_dsl.py", [str(f)],
                          f"Validating {f.name}", dry_run, allow_fail=True):
            ok = False
    return ok


def step_merge(cfg, log, dry_run):
    log.section("STEP 3: Build Training Dataset")
    run_script(cfg, log, "03_generate_synthetic_data.py",
               ["--count", str(cfg.synthetic_count), "--output", str(cfg.synthetic_data), "--seed", "42"],
               f"Generating {cfg.synthetic_count} synthetic examples", dry_run)
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
        with open(cfg.train_data, "w") as f:
            f.write("\n".join(entries) + "\n")
        log.log(f"  Total: {len(entries)} training examples")
        sv = cfg.synthetic_data.with_name("validation.jsonl")
        if sv.exists() and sv.resolve() != cfg.val_data.resolve():
            shutil.copy2(str(sv), str(cfg.val_data))
    run_script(cfg, log, "06_validate_dsl.py", [str(cfg.train_data)],
               "Validating merged dataset", dry_run, allow_fail=True)
    return True


def step_prompt(cfg, log, dry_run):
    log.section("STEP 4: Generate System Prompt")
    return run_script(cfg, log, "08_generate_system_prompt.py",
                      ["--output", str(cfg.scripts / "system_prompt.txt")],
                      "Generating system prompt", dry_run)


def step_train(cfg, log, state, dry_run, force):
    log.section("STEP 5: Train Model")
    cur_hash = hash_file(cfg.train_data)
    if cur_hash == state.get("last_training_data_hash", "") and not force:
        log.log("Training data unchanged. Skipping. (Use --force to override)")
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
    ], f"Fine-tuning v{ver}", dry_run)
    if ok or dry_run:
        state["last_training_data_hash"] = cur_hash
        state["last_model_version"] = ver
        return str(model_dir / "final")
    return None


def step_evaluate(cfg, log, model_path, state, dry_run):
    log.section("STEP 6: Evaluate Model")
    if not model_path:
        log.log("No model. Skipping.", "WARN")
        return None
    ver = state.get("last_model_version", 0)
    rpt = cfg.results_dir / f"eval_v{ver}_{datetime.now().strftime('%Y%m%d')}.json"
    run_script(cfg, log, "09_evaluate_model.py",
               ["--model", model_path, "--report", str(rpt)],
               f"Evaluating {model_path}", dry_run)
    if rpt.exists():
        with open(rpt) as f:
            return json.load(f)
    return None


def step_summary(cfg, log, model_path, dry_run):
    log.section("STEP 7: Training Summary")
    if not model_path:
        return
    parent = str(Path(model_path).parent) if model_path.endswith("final") else model_path
    run_script(cfg, log, "10_training_dashboard.py", ["--summary", parent],
               "Training summary", dry_run, allow_fail=True)


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

    try:
        if mode in ("full", "data-only"):
            step_analyze(cfg, log, dry_run)
            step_validate(cfg, log, dry_run)
            step_merge(cfg, log, dry_run)
            step_prompt(cfg, log, dry_run)
        if mode in ("full", "train-only"):
            model_path = step_train(cfg, log, state, dry_run, force)
            if model_path:
                eval_report = step_evaluate(cfg, log, model_path, state, dry_run)
                step_summary(cfg, log, model_path, dry_run)
        if mode == "eval-only":
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
    log.section("PIPELINE COMPLETE")
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
