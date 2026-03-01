"""
Dashboard Writer — atomic JSON writes for Mission Control dashboard.

Three files:
  dashboard/state.json          — live pipeline state (every step transition)
  dashboard/exam_progress.json  — per-prompt exam results (during exams)
  dashboard/version_history.json — accumulates across training versions

All writes are atomic (write tmp then os.replace) so the dashboard
never reads a partial file.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

DASHBOARD_DIR = Path(r"C:\BlueprintLLM\dashboard")

# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _write_atomic(path, data):
    """Write JSON atomically via tmp + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(path))


def _read_json(path):
    """Read JSON file, return None if missing or corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# state.json — live pipeline state
# ---------------------------------------------------------------------------

# Module-level state kept in memory so callers can do incremental updates
_state = None


def reset_run(version, steps=None):
    """Initialize state.json for a new pipeline run.

    Args:
        version: e.g. "v6"
        steps: list of step dicts [{number, name, status, ...}]
               If None, a default 9-step skeleton is created.
    """
    global _state

    if steps is None:
        steps = [
            {"number": 1, "name": "Data Foundation", "status": "pending"},
            {"number": 2, "name": "Pre-Flight Checks", "status": "pending"},
            {"number": 3, "name": "Validate data", "status": "pending"},
            {"number": 4, "name": "Training", "status": "pending"},
            {"number": 5, "name": "Merge LoRA weights", "status": "pending"},
            {"number": 6, "name": "Run exams", "status": "pending"},
            {"number": 7, "name": "Run eval suite", "status": "pending"},
            {"number": 8, "name": "Generate reports", "status": "pending"},
            {"number": 9, "name": "Finalize", "status": "pending"},
        ]

    _state = {
        "version": version,
        "status": "running",
        "pipeline_start_time": time.time(),
        "current_step": {"number": 1, "total_steps": len(steps),
                         "name": steps[0]["name"], "started_at": time.time()},
        "steps": steps,
        "training": None,
        "exam_active": None,
    }
    _write_atomic(DASHBOARD_DIR / "state.json", _state)

    # Clear exam progress for the new run
    _write_atomic(DASHBOARD_DIR / "exam_progress.json", {
        "version": version,
        "lessons": [],
        "aggregate": {
            "total_prompts_completed": 0,
            "total_prompts_remaining": 0,
            "overall_valid_syntax_pct": 0,
            "overall_avg_similarity": 0,
            "estimated_remaining_seconds": 0,
        },
    })
    return _state


def _ensure_state():
    """Load state from disk if not in memory."""
    global _state
    if _state is None:
        _state = _read_json(DASHBOARD_DIR / "state.json")
    if _state is None:
        _state = {"version": "?", "status": "idle", "steps": [],
                  "current_step": {}, "training": None, "exam_active": None}
    return _state


def write_state(state_dict=None):
    """Write the current state to dashboard/state.json.

    If state_dict is provided, it replaces the module-level state.
    Otherwise the existing module-level state is flushed to disk.
    """
    global _state
    if state_dict is not None:
        _state = state_dict
    else:
        _ensure_state()
    _write_atomic(DASHBOARD_DIR / "state.json", _state)


def step_start(step_number, name=None, detail=None):
    """Mark step *step_number* as active, update current_step."""
    s = _ensure_state()
    for st in s["steps"]:
        if st["number"] == step_number:
            st["status"] = "active"
            st["started_at"] = time.time()
            if detail:
                st["detail"] = detail
            if name:
                st["name"] = name
            s["current_step"] = {
                "number": step_number,
                "total_steps": len(s["steps"]),
                "name": name or st.get("name", ""),
                "started_at": st["started_at"],
            }
            break
    write_state()


def step_done(step_number, duration_seconds=None, detail=None):
    """Mark step *step_number* as done."""
    s = _ensure_state()
    for st in s["steps"]:
        if st["number"] == step_number:
            st["status"] = "done"
            if duration_seconds is not None:
                st["duration_seconds"] = round(duration_seconds, 1)
            elif "started_at" in st:
                st["duration_seconds"] = round(time.time() - st["started_at"], 1)
            if detail:
                st["detail"] = detail
            break
    write_state()


def update_training(current_step, total_steps, loss, examples=None,
                    epochs=None, learning_rate=None, elapsed_seconds=None):
    """Update the training progress block in state.json."""
    s = _ensure_state()
    s["training"] = {
        "total_steps": total_steps,
        "current_step": current_step,
        "loss": round(loss, 4) if loss is not None else None,
        "examples": examples,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "elapsed_seconds": round(elapsed_seconds, 1) if elapsed_seconds else None,
    }
    # Also update step detail
    for st in s["steps"]:
        if st.get("status") == "active" and st["number"] == 4:
            st["detail"] = f"{current_step}/{total_steps}, loss {round(loss, 4) if loss else '?'}"
            break
    write_state()


def update_exam_active(lesson_id, lesson_name, current_prompt,
                       total_prompts, last_prompt_time=None, avg_prompt_time=None):
    """Update the exam_active block in state.json (live exam indicator)."""
    s = _ensure_state()
    s["exam_active"] = {
        "lesson_id": lesson_id,
        "lesson_name": lesson_name,
        "current_prompt": current_prompt,
        "total_prompts": total_prompts,
        "last_prompt_time_seconds": round(last_prompt_time, 1) if last_prompt_time else None,
        "avg_prompt_time_seconds": round(avg_prompt_time, 1) if avg_prompt_time else None,
    }
    # Update step detail
    for st in s["steps"]:
        if st.get("status") == "active":
            st["detail"] = f"{lesson_id} {current_prompt}/{total_prompts}"
            break
    write_state()


def set_idle():
    """Set pipeline status to idle."""
    s = _ensure_state()
    s["status"] = "idle"
    s["exam_active"] = None
    write_state()


# ---------------------------------------------------------------------------
# exam_progress.json — per-prompt exam results
# ---------------------------------------------------------------------------

def update_exam_progress(lesson_id, lesson_name, prompt_result,
                         total_exam_prompts=None):
    """Append a single prompt result to exam_progress.json.

    Args:
        lesson_id: e.g. "lesson_01"
        lesson_name: e.g. "Core Patterns"
        prompt_result: dict with keys:
            prompt_id, category, valid (bool), similarity (float 0-100),
            time_seconds (float), total_in_lesson (int)
        total_exam_prompts: total prompts across ALL lessons (for ETA)
    """
    path = DASHBOARD_DIR / "exam_progress.json"
    data = _read_json(path)
    if data is None:
        data = {"version": "?", "lessons": [], "aggregate": {}}

    # Normalise lesson_id to short form for display
    short_id = lesson_id.replace("lesson_", "L").replace("lesson", "L")
    if not short_id.startswith("L"):
        short_id = lesson_id  # fallback

    # Find or create lesson entry
    lesson = next((l for l in data["lessons"] if l["lesson_id"] == short_id), None)
    if lesson is None:
        lesson = {
            "lesson_id": short_id,
            "lesson_name": lesson_name,
            "status": "running",
            "valid_count": 0,
            "total_count": prompt_result.get("total_in_lesson", 20),
            "completed_prompts": 0,
            "prompts": [],
            "weakest_categories": [],
            "strongest_categories": [],
        }
        data["lessons"].append(lesson)

    # Append prompt
    lesson["prompts"].append({
        "prompt_id": prompt_result.get("prompt_id", ""),
        "category": prompt_result.get("category", ""),
        "valid": prompt_result.get("valid", False),
        "similarity": round(prompt_result.get("similarity", 0), 1),
        "time_seconds": round(prompt_result.get("time_seconds", 0), 1),
    })
    lesson["completed_prompts"] = len(lesson["prompts"])
    lesson["valid_count"] = sum(1 for p in lesson["prompts"] if p["valid"])

    # Update aggregate
    all_prompts = [p for l in data["lessons"] for p in l["prompts"]]
    total_done = len(all_prompts)
    total_remaining = (total_exam_prompts or 0) - total_done
    if total_remaining < 0:
        total_remaining = 0
    avg_time = sum(p["time_seconds"] for p in all_prompts) / max(total_done, 1)

    data["aggregate"] = {
        "total_prompts_completed": total_done,
        "total_prompts_remaining": total_remaining,
        "overall_valid_syntax_pct": round(
            sum(1 for p in all_prompts if p["valid"]) / max(total_done, 1) * 100, 1),
        "overall_avg_similarity": round(
            sum(p["similarity"] for p in all_prompts) / max(total_done, 1), 1),
        "estimated_remaining_seconds": int(avg_time * total_remaining),
    }

    _write_atomic(path, data)


def finalize_lesson(lesson_id):
    """Mark a lesson as done and compute weakest/strongest categories."""
    path = DASHBOARD_DIR / "exam_progress.json"
    data = _read_json(path)
    if data is None:
        return

    short_id = lesson_id.replace("lesson_", "L").replace("lesson", "L")
    if not short_id.startswith("L"):
        short_id = lesson_id

    lesson = next((l for l in data["lessons"] if l["lesson_id"] == short_id), None)
    if lesson is None:
        return

    lesson["status"] = "done"
    prompts = lesson["prompts"]
    if prompts:
        lesson["valid_syntax_pct"] = round(
            lesson["valid_count"] / max(len(prompts), 1) * 100, 1)
        lesson["avg_similarity"] = round(
            sum(p["similarity"] for p in prompts) / len(prompts), 1)
        lesson["duration_seconds"] = round(
            sum(p["time_seconds"] for p in prompts), 1)
        lesson["avg_prompt_time"] = round(
            lesson["duration_seconds"] / len(prompts), 1)

        # Category aggregation
        cat_map = {}
        for p in prompts:
            cat = p.get("category", "unknown")
            cat_map.setdefault(cat, []).append(p["similarity"])
        cat_avgs = [(c, round(sum(v) / len(v), 1))
                    for c, v in cat_map.items()]
        cat_avgs.sort(key=lambda x: x[1])
        lesson["weakest_categories"] = [
            {"category": c, "similarity": s} for c, s in cat_avgs[:3]
        ]
        lesson["strongest_categories"] = [
            {"category": c, "similarity": s} for c, s in cat_avgs[-3:]
        ]

    _write_atomic(path, data)


# ---------------------------------------------------------------------------
# version_history.json — persistent across training versions
# ---------------------------------------------------------------------------

def append_version_history(version_tag, eval_results=None,
                           exam_summaries=None, training_info=None):
    """Append (or update) a version entry in version_history.json.

    Args:
        version_tag: e.g. "v5"
        eval_results: dict with "summary" key containing
            {total, passed, avg_score, avg_time} and "results" list
        exam_summaries: list of per-lesson summary dicts from exam runs
        training_info: dict with total_examples, training_seconds, etc.
    """
    path = DASHBOARD_DIR / "version_history.json"
    data = _read_json(path)
    if data is None:
        data = {"versions": []}

    # Build lesson scores
    lesson_scores = {}
    total_valid = 0
    total_prompts = 0
    total_sim = 0.0

    if exam_summaries:
        for summary in exam_summaries:
            lid = summary.get("lesson_id", "").replace("lesson_", "L")
            if not lid.startswith("L"):
                lid = summary.get("lesson_id", lid)
            syntax_pct = summary.get("valid_syntax_pct",
                                     summary.get("syntax", 0))
            sim_score = summary.get("avg_similarity",
                                    summary.get("similarity", 0))
            lesson_scores[lid] = {
                "syntax": round(syntax_pct, 1),
                "similarity": round(sim_score, 1),
            }
            sv = summary.get("valid_count",
                             summary.get("valid_syntax", 0))
            tp = summary.get("total_count",
                             summary.get("total_prompts", 0))
            total_valid += sv
            total_prompts += tp
            total_sim += sim_score * tp

    # Build eval section
    eval_section = {}
    if eval_results and "summary" in eval_results:
        es = eval_results["summary"]
        total = es.get("total", 1)
        eval_section = {
            "pass_rate": round(es.get("passed", 0) / max(total, 1), 3),
            "avg_score": round(es.get("avg_score", 0), 3),
            "perfect_scores": sum(
                1 for r in eval_results.get("results", [])
                if r.get("score", 0) == 1.0
            ),
            "avg_gen_time": round(es.get("avg_time", 0), 1),
        }

    entry = {
        "version": version_tag,
        "date": datetime.now().isoformat(),
        "training_examples": (training_info or {}).get("total_examples", 0),
        "training_time_seconds": (training_info or {}).get(
            "training_seconds", 0),
        "system_prompt_chars": (training_info or {}).get(
            "system_prompt_chars", 0),
        "eval": eval_section,
        "exam_summary": {
            "total_prompts": total_prompts,
            "valid_syntax_pct": round(
                total_valid / max(total_prompts, 1) * 100, 1),
            "avg_similarity": round(
                total_sim / max(total_prompts, 1), 1),
        },
        "lessons_tested": list(lesson_scores.keys()),
        "lesson_scores": lesson_scores,
    }

    # Replace if version exists, else append
    data["versions"] = [v for v in data["versions"]
                        if v.get("version") != version_tag]
    data["versions"].append(entry)
    data["versions"].sort(key=lambda v: v.get("version", ""))
    _write_atomic(path, data)
