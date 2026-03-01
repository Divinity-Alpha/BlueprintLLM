"""
Re-run all exams (standard + holdout) with fixed DSLStoppingCriteria.
Loads the model ONCE and runs all lessons in-process.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_PATH = str(ROOT / "models" / "blueprint-lora-v5" / "final")


def log(msg):
    print(msg, flush=True)


def run_all_exams():
    from error_handler import per_prompt_timeout
    import importlib.util

    log("Loading model (once)...")
    start_load = time.time()
    spec = importlib.util.spec_from_file_location("run_exam_mod", ROOT / "scripts" / "12_run_exam.py")
    exam_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exam_mod)

    model, tokenizer, base_model = exam_mod.load_model(MODEL_PATH)
    log(f"Model loaded in {time.time() - start_load:.0f}s")

    # Gather all lesson files
    lesson_dir = ROOT / "lessons"
    holdout_dir = ROOT / "lessons" / "holdout"
    lessons = sorted(lesson_dir.glob("lesson_*.json"))
    if holdout_dir.exists():
        lessons += sorted(holdout_dir.glob("lesson_*.json"))
    # Also include correction files for exam
    lessons += sorted(lesson_dir.glob("correction_*.json"))

    log(f"\nRunning {len(lessons)} exams: {[l.name for l in lessons]}")

    output_dir = ROOT / "results" / "exams"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for li, lesson_path in enumerate(lessons):
        with open(lesson_path, encoding="utf-8") as f:
            lesson = json.load(f)

        lesson_id = lesson.get("lesson_id", lesson_path.stem)
        lesson_name = lesson.get("lesson_name", lesson_id)
        prompts = lesson.get("prompts", [])

        log(f"\n{'='*60}")
        log(f"  [{li+1}/{len(lessons)}] EXAM: {lesson_name}")
        log(f"  {len(prompts)} prompts")
        log(f"{'='*60}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"exam_{lesson_id}_{ts}.jsonl"
        summary_file = output_dir / f"exam_{lesson_id}_{ts}_summary.json"

        results = []
        total_score = 0
        valid_count = 0
        timeout_count = 0

        for i, prompt in enumerate(prompts):
            log(f"  [{i+1}/{len(prompts)}] {prompt['id']}: {prompt['instruction'][:60]}...")

            start = time.time()

            gen_result, timed_out = per_prompt_timeout(
                lambda p=prompt: exam_mod.generate(model, tokenizer, p["instruction"]),
                timeout_seconds=300,
            )

            if timed_out:
                elapsed = time.time() - start
                cleaned_dsl = "[TIMEOUT]"
                raw_output = f"[Generation timed out after 300s]"
                timeout_count += 1
                log(f"    [TIMEOUT] {elapsed:.0f}s")
            else:
                cleaned_dsl, raw_output = gen_result
                elapsed = time.time() - start

            validation = exam_mod.validate_dsl(cleaned_dsl)
            comparison = exam_mod.compare_outputs(prompt["expected_dsl"], cleaned_dsl)

            if not timed_out:
                status = "OK" if validation["valid"] else "FAIL"
                log(f"    [{status}] Score: {comparison['score']:.0%} | "
                    f"Nodes: {validation['nodes']} | {elapsed:.1f}s")

            if validation["valid"] and not timed_out:
                valid_count += 1
            if not timed_out:
                total_score += comparison["score"]

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

            with open(results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Summary
        avg_score = total_score / max(len(results), 1)
        summary = {
            "lesson_id": lesson_id,
            "lesson_name": lesson_name,
            "model": MODEL_PATH,
            "base_model": base_model,
            "timestamp": ts,
            "total_prompts": len(results),
            "valid_syntax": valid_count,
            "valid_syntax_pct": round(valid_count / max(len(results), 1) * 100, 1),
            "avg_similarity_score": round(avg_score * 100, 1),
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
        log(f"  Result: {valid_count}/{len(results)} valid ({summary['valid_syntax_pct']}%){timeout_msg}")
        log(f"  Avg similarity: {summary['avg_similarity_score']}%")
        all_summaries.append(summary)

    # Final report
    log(f"\n{'='*60}")
    log(f"  ALL EXAMS COMPLETE")
    log(f"{'='*60}")
    for s in all_summaries:
        log(f"  {s['lesson_id']:20s} {s['valid_syntax']:2d}/{s['total_prompts']:2d} valid ({s['valid_syntax_pct']:5.1f}%)  sim={s['avg_similarity_score']:5.1f}%")

    return all_summaries


def update_grade_me():
    repo_base = "https://raw.githubusercontent.com/Divinity-Alpha/BlueprintLLM/main"
    exams_dir = ROOT / "results" / "exams"
    today = time.strftime("%Y%m%d")

    exam_files = sorted(
        [f for f in exams_dir.iterdir() if today in f.name and f.name.startswith("exam_")],
        key=lambda p: p.name,
    )

    lines = ["Grade these v5 results (re-run with fixed early stopping):"]
    for ef in exam_files:
        lines.append(f"{repo_base}/results/exams/{ef.name}")

    results_dir = ROOT / "results"
    eval_files = sorted(
        [f for f in results_dir.iterdir() if f.name.startswith("eval_v5") and f.suffix == ".json"],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if eval_files:
        lines.append(f"{repo_base}/results/{eval_files[0].name}")

    grade_path = ROOT / "GRADE_ME.txt"
    grade_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nGRADE_ME.txt updated with {len(lines) - 1} URLs")


def git_push():
    try:
        subprocess.run(["git", "add", "results/", "GRADE_ME.txt",
                         "scripts/07_inference.py", "scripts/12_run_exam.py",
                         "scripts/rerun_all_exams.py", "scripts/run_holdout_exams.py",
                         "scripts/13_lesson_to_training.py", "lessons/holdout/"],
                        cwd=str(ROOT), capture_output=True, timeout=30)

        status = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                cwd=str(ROOT), capture_output=True, timeout=30)
        if status.returncode == 0:
            log("Nothing to commit.")
            return

        subprocess.run(["git", "commit", "-m",
                         "v5 exam re-run — fixed DSLStoppingCriteria truncation bug"],
                        cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        push = subprocess.run(["git", "push"], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=120)
        log("Pushed." if push.returncode == 0 else f"Push failed: {push.stderr.strip()}")
    except Exception as e:
        log(f"Git error: {e}")


def notify():
    try:
        subprocess.Popen(
            ["powershell", "-Command",
             "Add-Type -AssemblyName System.Windows.Forms; "
             "[System.Windows.Forms.MessageBox]::Show("
             "'v5 exams re-run complete with fixed early stopping! "
             "Results pushed. Open GRADE_ME.txt.', "
             "'BlueprintLLM', 'OK', 'Information')"],
            creationflags=0x00000008,
        )
    except Exception:
        pass


if __name__ == "__main__":
    run_all_exams()
    update_grade_me()
    git_push()
    notify()
    log("\nAll done!")
