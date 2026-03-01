"""
Run holdout exam lessons against the latest model.
Loads the model ONCE, then runs all holdout lessons in-process.
Results are saved after EACH lesson completes.

Usage:
    python scripts/run_holdout_exams.py                  # wait for pipeline idle, then run
    python scripts/run_holdout_exams.py --no-wait         # run immediately
    python scripts/run_holdout_exams.py --no-push         # skip git push
"""
import json
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "logs" / "pipeline_live_state.json"
HOLDOUT_DIR = ROOT / "lessons" / "holdout"
OUTPUT_DIR = ROOT / "results" / "exams"

# Ensure scripts/ is on path for imports
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Per-prompt timeout (seconds)
PROMPT_TIMEOUT = 300


def log(msg):
    print(msg, flush=True)


def wait_for_pipeline():
    """Block until pipeline_live_state.json shows idle."""
    log("Waiting for pipeline to finish...")
    while True:
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if state.get("status") == "idle":
                log("Pipeline is idle.")
                return
            log(f"  Pipeline still running: step {state.get('step_id', '?')} - {state.get('description', '?')}")
        except Exception:
            pass
        time.sleep(30)


def detect_model():
    """Find the latest model version directory."""
    models_dir = ROOT / "models"
    versions = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("blueprint-lora-v")],
        key=lambda d: int(d.name.split("-v")[-1]) if d.name.split("-v")[-1].isdigit() else 0,
    )
    if versions:
        final = versions[-1] / "final"
        if final.exists():
            return str(final)
    return None


def load_model_once(model_path):
    """Load model and tokenizer once, return them for reuse."""
    from run_exam_lib import load_model
    log(f"Loading model: {model_path}")
    t0 = time.time()
    model, tokenizer, base_model = load_model(model_path)
    log(f"Model loaded in {time.time() - t0:.0f}s (base: {base_model})")
    return model, tokenizer, base_model


def run_single_exam(lesson_path, model, tokenizer, model_path):
    """Run all prompts from a lesson, save results immediately. Returns summary dict."""
    from run_exam_lib import generate, validate_dsl, compare_outputs, GENERATE_TIMEOUT
    from error_handler import per_prompt_timeout

    with open(lesson_path, encoding="utf-8") as f:
        lesson = json.load(f)

    lesson_id = lesson["lesson_id"]
    lesson_name = lesson["lesson_name"]
    prompts = lesson["prompts"]

    log(f"\n{'='*60}")
    log(f"  HOLDOUT EXAM: {lesson_name}")
    log(f"  {len(prompts)} prompts | Lesson: {lesson_id}")
    log(f"{'='*60}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"exam_{lesson_id}_{ts}.jsonl"
    summary_file = OUTPUT_DIR / f"exam_{lesson_id}_{ts}_summary.json"

    results = []
    valid_count = 0
    total_score = 0
    timeout_count = 0

    for i, prompt in enumerate(prompts):
        log(f"[{i+1}/{len(prompts)}] {prompt['id']}: {prompt['instruction'][:60]}...")
        start = time.time()

        gen_result, timed_out = per_prompt_timeout(
            lambda p=prompt: generate(model, tokenizer, p["instruction"]),
            timeout_seconds=PROMPT_TIMEOUT,
        )

        if timed_out:
            elapsed = time.time() - start
            cleaned_dsl = "[TIMEOUT]"
            raw_output = f"[Generation timed out after {PROMPT_TIMEOUT}s]"
            timeout_count += 1
            log(f"  [TIMEOUT] after {PROMPT_TIMEOUT}s — skipping")
        else:
            cleaned_dsl, raw_output = gen_result
            elapsed = time.time() - start

        validation = validate_dsl(cleaned_dsl)
        comparison = compare_outputs(prompt["expected_dsl"], cleaned_dsl)

        if timed_out:
            log(f"  [TIMEOUT] Score: 0% | {elapsed:.1f}s")
        else:
            status = "[OK]" if validation["valid"] else "[X]"
            log(f"  {status} Score: {comparison['score']:.0%} | "
                f"Nodes: {validation['nodes']} | "
                f"Connections: {validation['connections']} | "
                f"{elapsed:.1f}s")
            if not validation["valid"] and validation["error"]:
                first_line = validation["error"].split('\n')[0][:80]
                log(f"  Error: {first_line}")

        result = {
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "instruction": prompt["instruction"],
            "expected_dsl": prompt["expected_dsl"],
            "actual_dsl": cleaned_dsl,
            "raw_output": raw_output,
            "validation": validation,
            "comparison": comparison,
            "time_seconds": round(elapsed, 1),
            "status": "timeout_skipped" if timed_out else "completed",
        }
        results.append(result)

        if validation["valid"] and not timed_out:
            valid_count += 1
        if not timed_out:
            total_score += comparison["score"]

        # Write incrementally
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    avg_score = total_score / max(len(results), 1)
    summary = {
        "lesson_id": lesson_id,
        "lesson_name": lesson_name,
        "model": str(model_path),
        "timestamp": ts,
        "total_prompts": len(results),
        "valid_syntax": valid_count,
        "valid_syntax_pct": round(valid_count / max(len(results), 1) * 100, 1),
        "avg_similarity_score": round(avg_score * 100, 1),
        "timeout_count": timeout_count,
        "holdout": True,
        "per_category": {},
    }

    for r in results:
        cat = r["category"]
        if cat not in summary["per_category"]:
            summary["per_category"][cat] = {"count": 0, "valid": 0, "avg_score": 0}
        summary["per_category"][cat]["count"] += 1
        if r["validation"]["valid"]:
            summary["per_category"][cat]["valid"] += 1
        summary["per_category"][cat]["avg_score"] += r["comparison"]["score"]

    for cat in summary["per_category"]:
        s = summary["per_category"][cat]
        s["avg_score"] = round(s["avg_score"] / max(s["count"], 1) * 100, 1)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    timeout_msg = f", {timeout_count} timed out" if timeout_count else ""
    log(f"\n{'='*60}")
    log(f"  RESULTS: {lesson_name}")
    log(f"  Valid syntax:     {valid_count}/{len(results)} ({summary['valid_syntax_pct']}%)")
    log(f"  Avg similarity:   {summary['avg_similarity_score']}%")
    log(f"  Results:          {results_file}{timeout_msg}")
    log(f"  Summary:          {summary_file}")
    log(f"{'='*60}\n")

    return summary


def update_grade_me(model_path):
    """Append holdout exam URLs to GRADE_ME.txt."""
    grade_file = ROOT / "GRADE_ME.txt"
    repo_base = "https://raw.githubusercontent.com/Divinity-Alpha/BlueprintLLM/main"

    holdout_exams = []
    for ef in sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if ef.suffix == ".jsonl" and ("lesson_11" in ef.name or "lesson_12" in ef.name):
            holdout_exams.append(ef)
            summary = ef.with_name(ef.stem + "_summary.json")
            if summary.exists():
                holdout_exams.append(summary)
        if len(holdout_exams) >= 4:
            break

    if holdout_exams:
        existing = grade_file.read_text(encoding="utf-8") if grade_file.exists() else ""
        lines = ["\nHoldout (unseen) exam results:"]
        for ef in holdout_exams:
            lines.append(f"{repo_base}/results/exams/{ef.name}")
        grade_file.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
        log(f"Updated GRADE_ME.txt with {len(holdout_exams)} holdout URLs")


def git_push():
    """Stage holdout results and push."""
    try:
        subprocess.run(["git", "add", "results/exams/", "GRADE_ME.txt",
                         "lessons/holdout/", "scripts/run_holdout_exams.py"],
                        cwd=str(ROOT), capture_output=True, timeout=30)

        status = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                cwd=str(ROOT), capture_output=True, timeout=30)
        if status.returncode == 0:
            log("Nothing new to commit.")
            return

        subprocess.run(["git", "commit", "-m",
                         "v5 holdout exam results (lessons 11-12)"],
                        cwd=str(ROOT), capture_output=True, text=True, timeout=60)

        push = subprocess.run(["git", "push"], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=120)
        if push.returncode == 0:
            log("Pushed to remote.")
        else:
            log(f"Push failed: {push.stderr.strip()}")
    except Exception as e:
        log(f"Git error: {e}")


def notify():
    """Desktop notification."""
    try:
        subprocess.Popen(
            ["powershell", "-Command",
             "Add-Type -AssemblyName System.Windows.Forms; "
             "[System.Windows.Forms.MessageBox]::Show("
             "'Holdout exams complete! Results pushed. Open GRADE_ME.txt.', "
             "'BlueprintLLM', 'OK', 'Information')"],
            creationflags=0x00000008,
        )
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run holdout exams")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for pipeline idle")
    parser.add_argument("--no-push", action="store_true", help="Skip git push")
    args = parser.parse_args()

    if not args.no_wait:
        wait_for_pipeline()

    model_path = detect_model()
    if not model_path:
        log("ERROR: No model found")
        sys.exit(1)
    log(f"Model: {model_path}")

    lessons = sorted(HOLDOUT_DIR.glob("lesson_*.json"))
    log(f"Found {len(lessons)} holdout lessons: {[l.name for l in lessons]}")

    if not lessons:
        log("No holdout lessons found.")
        sys.exit(0)

    # Load model ONCE — this is the expensive operation (~2-30 min for 70B)
    model, tokenizer, base_model = load_model_once(model_path)

    # Run each lesson in-process, saving results after each
    summaries = []
    for lesson in lessons:
        try:
            summary = run_single_exam(lesson, model, tokenizer, model_path)
            summaries.append(summary)
        except Exception as e:
            log(f"ERROR running {lesson.name}: {e}")
            import traceback
            traceback.print_exc()

    # Print overall summary
    if summaries:
        log(f"\n{'='*60}")
        log(f"  HOLDOUT EXAM SUMMARY ({len(summaries)} lessons)")
        log(f"{'='*60}")
        for s in summaries:
            log(f"  {s['lesson_id']:>12}: {s['valid_syntax_pct']:5.1f}% syntax | {s['avg_similarity_score']:5.1f}% similarity")
        log(f"{'='*60}\n")

    if not args.no_push:
        update_grade_me(model_path)
        git_push()
    notify()
    log("\nDone!")
