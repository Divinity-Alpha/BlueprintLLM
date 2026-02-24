"""
19_training_health_monitor.py
-----------------------------
Analyzes training health across 6 dimensions after each training cycle.
Surfaces actionable alerts for overfitting, accuracy plateaus, catastrophic
forgetting, dataset imbalance, and resource usage trends.

Usage:
    python scripts/19_training_health_monitor.py
    python scripts/19_training_health_monitor.py --version v2
    python scripts/19_training_health_monitor.py --project-root C:\\BlueprintLLM

Outputs:
    health_report.json          Machine-readable report at project root
    logs/health_alerts.log      Append-only human-readable alert log
    logs/training_history.json  Accumulating cycle-over-cycle history
"""

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="5")


# ═══════════════════════════════════════════════════
# Alert data model
# ═══════════════════════════════════════════════════

ALERT_LEVELS = ["INFO", "SUGGESTION", "WARNING", "CRITICAL"]


@dataclass
class Alert:
    level: str        # INFO, SUGGESTION, WARNING, CRITICAL
    dimension: str    # epoch_efficiency, overfitting, learning_rate, dataset_quality, node_mastery, resource_usage
    title: str
    detail: str
    metric: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


# ═══════════════════════════════════════════════════
# Data loading functions
# ═══════════════════════════════════════════════════

def find_latest_version(models_dir: Path) -> Optional[str]:
    """Find the latest model version directory (e.g. 'v2')."""
    dirs = sorted(models_dir.glob("blueprint-lora-v*"), key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs:
        m = re.search(r'v(\d+)', d.name)
        if m:
            return f"v{m.group(1)}"
    return None


def find_version_dir(models_dir: Path, version: str) -> Optional[Path]:
    """Get model directory for a specific version string like 'v2'."""
    num = version.lstrip("v")
    d = models_dir / f"blueprint-lora-v{num}"
    return d if d.exists() else None


def load_training_metrics(model_dir: Path) -> Optional[dict]:
    """Parse trainer_state.json from the latest checkpoint for loss/accuracy curves."""
    # Find latest checkpoint
    checkpoints = sorted(model_dir.glob("checkpoint-*"), key=lambda d: d.stat().st_mtime, reverse=True)
    if not checkpoints:
        return None

    state_path = checkpoints[0] / "trainer_state.json"
    if not state_path.exists():
        return None

    with open(state_path, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    train_steps = []
    eval_steps = []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append({
                "step": entry.get("step", 0),
                "epoch": entry.get("epoch", 0),
                "loss": entry.get("loss", 0),
                "accuracy": entry.get("mean_token_accuracy", 0),
                "learning_rate": entry.get("learning_rate", 0),
                "grad_norm": entry.get("grad_norm", 0),
            })
        elif "eval_loss" in entry:
            eval_steps.append({
                "step": entry.get("step", 0),
                "epoch": entry.get("epoch", 0),
                "loss": entry.get("eval_loss", 0),
                "accuracy": entry.get("eval_mean_token_accuracy", 0),
            })

    if not train_steps:
        return None

    return {
        "global_step": state.get("global_step", 0),
        "epoch": state.get("epoch", 0),
        "best_metric": state.get("best_metric"),
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "initial_loss": train_steps[0]["loss"],
        "final_loss": train_steps[-1]["loss"],
        "initial_accuracy": train_steps[0]["accuracy"],
        "final_accuracy": train_steps[-1]["accuracy"],
        "eval_loss": eval_steps[-1]["loss"] if eval_steps else None,
        "eval_accuracy": eval_steps[-1]["accuracy"] if eval_steps else None,
        "total_steps": len(train_steps),
    }


def load_training_config(model_dir: Path) -> Optional[dict]:
    """Read training_config.json for hyperparameters."""
    cfg_path = model_dir / "training_config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r") as f:
        return json.load(f)


def load_pipeline_state(root: Path) -> Optional[dict]:
    """Read .pipeline_state.json for run history and eval summaries."""
    state_path = root / ".pipeline_state.json"
    if not state_path.exists():
        return None
    with open(state_path, "r") as f:
        return json.load(f)


def load_exam_node_scores(exams_dir: Path) -> dict:
    """Parse exam JSONL files for per-node-type scores (latest exam only)."""
    node_scores = {}
    if not exams_dir.exists():
        return node_scores

    # Find the most recent exam JSONL
    jsonl_files = sorted(exams_dir.glob("exam_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not jsonl_files:
        return node_scores

    latest = jsonl_files[0]
    with open(latest, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            score = entry.get("score", 0)
            if score == 0:
                comp = entry.get("comparison", {})
                score = comp.get("score", 0)

            expected = entry.get("expected_dsl", "")
            for text_line in expected.split("\n"):
                m = re.match(r"NODE\s+\w+:\s*(\w+)", text_line)
                if m:
                    node_name = m.group(1)
                    if node_name not in node_scores or score > node_scores[node_name]:
                        node_scores[node_name] = score

    return node_scores


def count_dataset_composition(train_path: Path) -> dict:
    """Count entries by source, detect duplicates via output-field hashing."""
    result = {
        "total": 0,
        "synthetic": 0,
        "lesson": 0,
        "auto_translated": 0,
        "manual": 0,
        "unknown": 0,
        "duplicates": 0,
    }
    if not train_path.exists():
        return result

    seen_hashes = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            result["total"] += 1

            # Detect source from fields
            output_text = entry.get("output", "")
            instruction = entry.get("instruction", "")
            pattern = entry.get("pattern", "")

            # Duplicate detection
            h = hashlib.md5(output_text.encode()).hexdigest()
            if h in seen_hashes:
                result["duplicates"] += 1
            seen_hashes.add(h)

            # Source classification heuristics
            if entry.get("source") == "lesson" or "L0" in entry.get("prompt_id", ""):
                result["lesson"] += 1
            elif pattern:
                result["synthetic"] += 1
            elif entry.get("source") == "auto_translated" or "_auto" in str(entry.get("source_file", "")):
                result["auto_translated"] += 1
            elif entry.get("source") == "manual":
                result["manual"] += 1
            else:
                # Heuristic: synthetic data usually has a pattern field
                # Auto-translated tends to have more complex blueprints
                if "pattern_" in pattern or instruction.startswith("Create a Blueprint"):
                    result["synthetic"] += 1
                else:
                    result["unknown"] += 1

    return result


def compute_node_type_coverage(train_path: Path) -> dict:
    """Extract node type frequency from all output fields in training data."""
    node_counts = {}
    if not train_path.exists():
        return node_counts

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = entry.get("output", "")
            for m in re.finditer(r"NODE\s+\w+:\s*(\w+)", output):
                name = m.group(1)
                node_counts[name] = node_counts.get(name, 0) + 1

    return node_counts


def load_training_history(path: Path) -> list:
    """Load accumulated training history from logs/training_history.json."""
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_training_history(path: Path, data: list):
    """Save training history to logs/training_history.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_exam_summary(exams_dir: Path) -> Optional[dict]:
    """Load the latest exam summary."""
    if not exams_dir.exists():
        return None
    summaries = sorted(exams_dir.glob("exam_*_summary.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not summaries:
        return None
    with open(summaries[0], "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
# Build current cycle snapshot
# ═══════════════════════════════════════════════════

def build_current_cycle(version: str, root: Path, model_dir: Path) -> dict:
    """Assemble all data for the current training cycle."""
    metrics = load_training_metrics(model_dir)
    config = load_training_config(model_dir)
    pipeline = load_pipeline_state(root)
    exams_dir = root / "results" / "exams"
    node_scores = load_exam_node_scores(exams_dir)
    exam_summary = load_exam_summary(exams_dir)
    train_path = root / "datasets" / "train.jsonl"
    dataset_comp = count_dataset_composition(train_path)
    node_coverage = compute_node_type_coverage(train_path)

    # Find training duration from pipeline state
    training_time = None
    eval_summary = None
    if pipeline:
        for run in reversed(pipeline.get("runs", [])):
            if run.get("version") == int(version.lstrip("v")) and run.get("duration", 0) > 100:
                training_time = run["duration"]
                if run.get("eval"):
                    eval_summary = run["eval"]
                break

    current = {
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "dataset_size": dataset_comp["total"],
        "lesson_data_size": dataset_comp["lesson"],
        "dataset_composition": dataset_comp,
        "node_coverage": node_coverage,
        "epochs": config.get("epochs", 0) if config else 0,
        "learning_rate": config.get("learning_rate", 0) if config else 0,
        "training_time_seconds": training_time,
        "total_steps": metrics["total_steps"] if metrics else 0,
        "initial_loss": metrics["initial_loss"] if metrics else None,
        "final_loss": metrics["final_loss"] if metrics else None,
        "eval_loss": metrics["eval_loss"] if metrics else None,
        "train_accuracy": metrics["final_accuracy"] if metrics else None,
        "eval_accuracy": metrics["eval_accuracy"] if metrics else None,
        "train_steps": metrics["train_steps"] if metrics else [],
        "eval_steps": metrics["eval_steps"] if metrics else [],
        "node_scores": node_scores,
        "eval_summary": eval_summary or (
            {"total": exam_summary.get("total_prompts", 0),
             "passed": exam_summary.get("valid_syntax", 0),
             "avg_score": exam_summary.get("avg_similarity_score", 0) / 100.0}
            if exam_summary else None
        ),
    }

    return current


# ═══════════════════════════════════════════════════
# Analysis functions (one per dimension)
# ═══════════════════════════════════════════════════

def analyze_epoch_efficiency(current: dict, history: list) -> List[Alert]:
    """Analyze training time and epoch count efficiency."""
    alerts = []
    ds = current.get("dataset_size", 0)
    acc = current.get("eval_accuracy") or current.get("train_accuracy") or 0
    epochs = current.get("epochs", 0)

    # Suggest reducing epochs for large datasets with high accuracy
    if ds > 2500 and acc > 0.90:
        alerts.append(Alert(
            level="SUGGESTION",
            dimension="epoch_efficiency",
            title="Consider reducing epochs",
            detail=f"Dataset has {ds} entries and accuracy is {acc*100:.1f}%. "
                   f"With {epochs} epochs, the model may be overtraining. Try {max(1, epochs - 1)} epochs.",
            metric="dataset_size_and_accuracy",
            value=ds,
        ))

    # Suggest increasing epochs for low accuracy with lesson data
    lesson_size = current.get("lesson_data_size", 0)
    if acc < 0.85 and lesson_size > 0:
        alerts.append(Alert(
            level="SUGGESTION",
            dimension="epoch_efficiency",
            title="Consider increasing epochs",
            detail=f"Accuracy is {acc*100:.1f}% with {lesson_size} lesson entries. "
                   f"Try {epochs + 1} epochs to improve learning.",
            metric="accuracy_with_lessons",
            value=acc,
            threshold=0.85,
        ))

    # Time trending — compare to previous cycles
    cur_time = current.get("training_time_seconds")
    if cur_time and len(history) >= 1:
        prev_times = [h["training_time_seconds"] for h in history if h.get("training_time_seconds")]
        if prev_times:
            baseline = sum(prev_times) / len(prev_times)
            if baseline > 0 and cur_time > baseline * 1.15:
                pct = ((cur_time - baseline) / baseline) * 100
                alerts.append(Alert(
                    level="WARNING",
                    dimension="epoch_efficiency",
                    title="Training time increased",
                    detail=f"Current cycle took {cur_time}s vs baseline {baseline:.0f}s (+{pct:.0f}%). "
                           f"Check dataset size growth or system load.",
                    metric="training_time_increase_pct",
                    value=pct,
                    threshold=15.0,
                ))

    # Report basic stats
    if cur_time:
        minutes = cur_time / 60
        alerts.append(Alert(
            level="INFO",
            dimension="epoch_efficiency",
            title="Training completed",
            detail=f"Version {current['version']}: {current.get('total_steps', 0)} steps, "
                   f"{epochs} epochs, {minutes:.1f} minutes, {ds} examples.",
            metric="training_time_seconds",
            value=cur_time,
        ))

    return alerts


def analyze_overfitting(current: dict, history: list) -> List[Alert]:
    """Detect train/eval loss gap, accuracy plateau, memorization."""
    alerts = []
    final_loss = current.get("final_loss")
    eval_loss = current.get("eval_loss")
    train_acc = current.get("train_accuracy")
    eval_acc = current.get("eval_accuracy")

    # Memorization detection: final loss extremely low
    if final_loss is not None and final_loss < 0.05:
        alerts.append(Alert(
            level="CRITICAL",
            dimension="overfitting",
            title="Possible memorization",
            detail=f"Final training loss is {final_loss:.4f} (< 0.05 threshold). "
                   f"The model may be memorizing training data rather than learning generalizable patterns.",
            metric="final_loss",
            value=final_loss,
            threshold=0.05,
        ))

    # Train/eval loss gap (overfitting indicator)
    if final_loss is not None and eval_loss is not None and eval_loss > 0:
        if final_loss < eval_loss * 0.5:
            gap = eval_loss - final_loss
            alerts.append(Alert(
                level="WARNING",
                dimension="overfitting",
                title="Train/eval loss gap (overfitting)",
                detail=f"Train loss ({final_loss:.4f}) is less than half of eval loss ({eval_loss:.4f}). "
                       f"Gap: {gap:.4f}. Consider regularization, data augmentation, or fewer epochs.",
                metric="loss_gap_ratio",
                value=final_loss / eval_loss if eval_loss > 0 else 0,
                threshold=0.5,
            ))

    # Accuracy plateau detection across cycles
    if len(history) >= 2:
        recent_accs = [h.get("eval_accuracy") or h.get("train_accuracy", 0) for h in history[-2:]]
        cur_acc = eval_acc or train_acc or 0
        all_accs = recent_accs + [cur_acc]
        all_accs = [a for a in all_accs if a > 0]
        if len(all_accs) >= 3:
            max_diff = max(all_accs) - min(all_accs)
            if max_diff < 0.005:
                alerts.append(Alert(
                    level="WARNING",
                    dimension="overfitting",
                    title="Accuracy plateau",
                    detail=f"Accuracy has changed less than 0.5% over the last 3 cycles "
                           f"({', '.join(f'{a*100:.1f}%' for a in all_accs)}). "
                           f"Consider curriculum changes, learning rate adjustments, or new training data.",
                    metric="accuracy_plateau_range",
                    value=max_diff,
                    threshold=0.005,
                ))

    # Report overfitting summary
    if train_acc is not None and eval_acc is not None:
        gap = train_acc - eval_acc
        status = "healthy" if gap < 0.05 else "moderate gap" if gap < 0.10 else "large gap"
        alerts.append(Alert(
            level="INFO",
            dimension="overfitting",
            title=f"Generalization: {status}",
            detail=f"Train accuracy {train_acc*100:.1f}% vs eval accuracy {eval_acc*100:.1f}% "
                   f"(gap: {gap*100:.1f}%).",
            metric="accuracy_gap",
            value=gap,
        ))

    return alerts


def analyze_learning_rate(current: dict, history: list) -> List[Alert]:
    """Check early convergence speed and loss oscillation."""
    alerts = []
    steps = current.get("train_steps", [])
    if len(steps) < 5:
        return alerts

    total = len(steps)
    lr = current.get("learning_rate", 0)

    # Check if loss is not decreasing in first 20% of steps
    early_cutoff = max(2, int(total * 0.2))
    early_steps = steps[:early_cutoff]
    if len(early_steps) >= 2:
        first_loss = early_steps[0]["loss"]
        early_end_loss = early_steps[-1]["loss"]
        if first_loss > 0 and early_end_loss >= first_loss * 0.95:
            alerts.append(Alert(
                level="WARNING",
                dimension="learning_rate",
                title="Slow early convergence",
                detail=f"Loss barely decreased in the first {early_cutoff} steps "
                       f"({first_loss:.4f} -> {early_end_loss:.4f}). "
                       f"Learning rate ({lr}) may be too low. Try increasing to {lr * 2}.",
                metric="early_loss_reduction",
                value=(first_loss - early_end_loss) / first_loss if first_loss > 0 else 0,
                threshold=0.05,
            ))

    # Loss oscillation: count direction changes
    if total >= 10:
        losses = [s["loss"] for s in steps]
        direction_changes = 0
        for i in range(2, len(losses)):
            prev_dir = losses[i-1] - losses[i-2]
            cur_dir = losses[i] - losses[i-1]
            if prev_dir * cur_dir < 0:  # sign change
                direction_changes += 1

        oscillation_ratio = direction_changes / (len(losses) - 2)
        if oscillation_ratio > 0.7:
            alerts.append(Alert(
                level="WARNING",
                dimension="learning_rate",
                title="High loss oscillation",
                detail=f"Loss changed direction {direction_changes}/{len(losses)-2} times "
                       f"({oscillation_ratio*100:.0f}% oscillation). "
                       f"Learning rate ({lr}) may be too high. Try reducing to {lr * 0.5}.",
                metric="oscillation_ratio",
                value=oscillation_ratio,
                threshold=0.7,
            ))

    # LR info
    if lr > 0:
        alerts.append(Alert(
            level="INFO",
            dimension="learning_rate",
            title="Learning rate configuration",
            detail=f"LR: {lr}, total steps: {total}, "
                   f"loss range: {steps[0]['loss']:.4f} -> {steps[-1]['loss']:.4f}.",
            metric="learning_rate",
            value=lr,
        ))

    return alerts


def analyze_dataset_quality(current: dict, root: Path) -> List[Alert]:
    """Check lesson ratio, duplicates, node coverage gaps."""
    alerts = []
    comp = current.get("dataset_composition", {})
    total = comp.get("total", 0)
    if total == 0:
        return alerts

    # Lesson data ratio
    lesson_count = comp.get("lesson", 0)
    lesson_ratio = lesson_count / total
    if lesson_ratio > 0.4:
        alerts.append(Alert(
            level="WARNING",
            dimension="dataset_quality",
            title="High lesson data ratio",
            detail=f"Lesson data is {lesson_count}/{total} ({lesson_ratio*100:.0f}%) of training set. "
                   f"More than 40% lesson data may bias the model toward correction patterns. "
                   f"Consider generating more synthetic data.",
            metric="lesson_ratio",
            value=lesson_ratio,
            threshold=0.4,
        ))

    # Duplicate detection
    dupes = comp.get("duplicates", 0)
    if dupes > 0:
        dupe_pct = (dupes / total) * 100
        level = "WARNING" if dupe_pct > 5 else "SUGGESTION"
        alerts.append(Alert(
            level=level,
            dimension="dataset_quality",
            title="Duplicate training examples",
            detail=f"Found {dupes} duplicate outputs ({dupe_pct:.1f}% of dataset). "
                   f"Duplicates cause the model to over-weight those patterns.",
            metric="duplicate_count",
            value=dupes,
        ))

    # Node coverage gaps — key nodes with zero training examples
    coverage = current.get("node_coverage", {})
    key_nodes = [
        "Event_BeginPlay", "PrintString", "Branch", "Delay", "Sequence",
        "SetActorLocation", "GetActorLocation", "SpawnActor", "DestroyActor",
        "CastTo", "SetTimer", "PlaySound", "Timeline", "ForEachLoop",
        "CustomEvent", "EventTick", "OnOverlap", "OnHit",
    ]
    missing = [n for n in key_nodes if coverage.get(n, 0) == 0]
    if missing:
        alerts.append(Alert(
            level="SUGGESTION",
            dimension="dataset_quality",
            title="Missing node types in training data",
            detail=f"{len(missing)} key node types have zero training examples: "
                   f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}. "
                   f"Consider adding synthetic examples for these.",
            metric="missing_key_nodes",
            value=len(missing),
        ))

    # Dataset composition summary
    alerts.append(Alert(
        level="INFO",
        dimension="dataset_quality",
        title="Dataset composition",
        detail=f"Total: {total} | Synthetic: {comp.get('synthetic', 0)} | "
               f"Lesson: {lesson_count} | Auto-translated: {comp.get('auto_translated', 0)} | "
               f"Manual: {comp.get('manual', 0)} | Duplicates: {dupes}",
        metric="dataset_total",
        value=total,
    ))

    return alerts


def analyze_node_mastery_velocity(current: dict, history: list) -> List[Alert]:
    """Detect stagnation and regression in per-node-type scores."""
    alerts = []
    cur_scores = current.get("node_scores", {})
    if not cur_scores:
        return alerts

    # Compare with previous cycles for regression/stagnation
    if history:
        prev = history[-1]
        prev_scores = prev.get("node_scores", {})

        regressed = []
        stagnant_long = []

        for node, cur_score in cur_scores.items():
            prev_score = prev_scores.get(node)
            if prev_score is not None:
                # Regression: score dropped > 10%
                if cur_score < prev_score - 0.10:
                    regressed.append((node, prev_score, cur_score))

        # Check for nodes stagnant across 3+ cycles
        if len(history) >= 2:
            for node, cur_score in cur_scores.items():
                scores_over_time = []
                for h in history[-2:]:
                    s = h.get("node_scores", {}).get(node)
                    if s is not None:
                        scores_over_time.append(s)
                scores_over_time.append(cur_score)
                if len(scores_over_time) >= 3:
                    spread = max(scores_over_time) - min(scores_over_time)
                    if spread < 0.05 and cur_score < 0.85:
                        stagnant_long.append((node, cur_score))

        # Regression alerts
        if regressed:
            for node, old, new in regressed:
                alerts.append(Alert(
                    level="CRITICAL",
                    dimension="node_mastery",
                    title=f"Node regression: {node}",
                    detail=f"{node} score dropped from {old*100:.0f}% to {new*100:.0f}% "
                           f"(-{(old-new)*100:.0f}%). Possible catastrophic forgetting. "
                           f"Check if new lesson data displaced training for this node type.",
                    metric=f"node_regression_{node}",
                    value=new - old,
                    threshold=-0.10,
                ))

        # Stagnation alerts
        if stagnant_long:
            node_list = ", ".join(f"{n} ({s*100:.0f}%)" for n, s in stagnant_long[:5])
            alerts.append(Alert(
                level="WARNING",
                dimension="node_mastery",
                title="Node scores stagnant for 3+ cycles",
                detail=f"{len(stagnant_long)} node(s) unchanged: {node_list}"
                       f"{'...' if len(stagnant_long) > 5 else ''}. "
                       f"Consider targeted lessons or more examples for these types.",
                metric="stagnant_node_count",
                value=len(stagnant_long),
                threshold=1,
            ))

    # Mastery summary
    mastered = sum(1 for s in cur_scores.values() if s >= 0.85)
    learning = sum(1 for s in cur_scores.values() if 0.6 <= s < 0.85)
    struggling = sum(1 for s in cur_scores.values() if s < 0.6)
    total = len(cur_scores)
    alerts.append(Alert(
        level="INFO",
        dimension="node_mastery",
        title="Node mastery snapshot",
        detail=f"{total} nodes scored: {mastered} mastered (>=85%), "
               f"{learning} learning (60-84%), {struggling} struggling (<60%).",
        metric="nodes_mastered",
        value=mastered,
    ))

    return alerts


def analyze_resource_usage(current: dict, history: list) -> List[Alert]:
    """Analyze training time trends and estimate future run times."""
    alerts = []
    cur_time = current.get("training_time_seconds")
    ds = current.get("dataset_size", 0)

    if cur_time and ds > 0:
        time_per_example = cur_time / ds
        alerts.append(Alert(
            level="INFO",
            dimension="resource_usage",
            title="Training throughput",
            detail=f"{time_per_example:.2f}s per training example, "
                   f"{cur_time/60:.1f} minutes total for {ds} examples.",
            metric="time_per_example",
            value=time_per_example,
        ))

    # Trend over history
    if len(history) >= 2 and cur_time:
        times = [h["training_time_seconds"] for h in history if h.get("training_time_seconds")]
        times.append(cur_time)
        if len(times) >= 3:
            # Simple linear trend
            avg_recent = sum(times[-2:]) / 2
            avg_older = sum(times[:-2]) / max(len(times) - 2, 1)
            if avg_older > 0:
                trend_pct = ((avg_recent - avg_older) / avg_older) * 100
                if abs(trend_pct) > 10:
                    direction = "increasing" if trend_pct > 0 else "decreasing"
                    alerts.append(Alert(
                        level="INFO",
                        dimension="resource_usage",
                        title=f"Training time {direction}",
                        detail=f"Recent avg: {avg_recent:.0f}s vs earlier avg: {avg_older:.0f}s "
                               f"({trend_pct:+.0f}%).",
                        metric="time_trend_pct",
                        value=trend_pct,
                    ))

    return alerts


# ═══════════════════════════════════════════════════
# Output functions
# ═══════════════════════════════════════════════════

def determine_overall_health(alerts: List[Alert]) -> str:
    """Determine overall health status from alert levels."""
    levels = {a.level for a in alerts}
    if "CRITICAL" in levels:
        return "critical"
    if "WARNING" in levels:
        return "warning"
    if "SUGGESTION" in levels:
        return "suggestion"
    return "healthy"


def generate_health_report(alerts: List[Alert], current: dict, history: list) -> dict:
    """Build the machine-readable health_report.json structure."""
    by_level = {}
    for level in ALERT_LEVELS:
        by_level[level] = sum(1 for a in alerts if a.level == level)

    # Build trends from history + current
    loss_trend = [h.get("final_loss") for h in history if h.get("final_loss") is not None]
    if current.get("final_loss") is not None:
        loss_trend.append(current["final_loss"])

    acc_trend = [h.get("eval_accuracy") or h.get("train_accuracy") for h in history
                 if (h.get("eval_accuracy") or h.get("train_accuracy")) is not None]
    cur_acc = current.get("eval_accuracy") or current.get("train_accuracy")
    if cur_acc is not None:
        acc_trend.append(cur_acc)

    ds_trend = [h.get("dataset_size") for h in history if h.get("dataset_size")]
    if current.get("dataset_size"):
        ds_trend.append(current["dataset_size"])

    time_trend = [h.get("training_time_seconds") for h in history if h.get("training_time_seconds")]
    if current.get("training_time_seconds"):
        time_trend.append(current["training_time_seconds"])

    # Determine cycle number
    cycle = len(history) + 1

    # Build current_cycle (without bulky train_steps)
    current_clean = {k: v for k, v in current.items() if k not in ("train_steps", "eval_steps", "node_coverage")}

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "version": current.get("version", "unknown"),
        "cycle": cycle,
        "summary": {
            "total_alerts": len(alerts),
            "by_level": by_level,
            "overall_health": determine_overall_health(alerts),
        },
        "alerts": [asdict(a) for a in alerts],
        "current_cycle": current_clean,
        "trends": {
            "loss": loss_trend,
            "accuracy": acc_trend,
            "dataset_size": ds_trend,
            "training_time": time_trend,
        },
    }
    return report


def write_health_alerts_log(alerts: List[Alert], path: Path):
    """Append alerts to a human-readable log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Health Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        for a in alerts:
            f.write(f"[{a.level:10s}] [{a.dimension}] {a.title}\n")
            f.write(f"             {a.detail}\n")
            if a.metric and a.value is not None:
                threshold_str = f" (threshold: {a.threshold})" if a.threshold is not None else ""
                f.write(f"             metric={a.metric} value={a.value}{threshold_str}\n")
        f.write(f"\nTotal: {len(alerts)} alerts\n")


def print_health_summary(alerts: List[Alert], current: dict):
    """Print a concise console summary."""
    overall = determine_overall_health(alerts)
    version = current.get("version", "?")

    level_counts = {}
    for a in alerts:
        level_counts[a.level] = level_counts.get(a.level, 0) + 1

    status_icon = {
        "healthy": "[OK]",
        "suggestion": "[OK]",
        "warning": "[!!]",
        "critical": "[XX]",
    }

    print(f"\n{'='*50}")
    print(f"  Training Health Monitor — {version}")
    print(f"{'='*50}")
    print(f"  Status: {status_icon.get(overall, '?')} {overall.upper()}")
    print(f"  Alerts: {len(alerts)} total", end="")
    parts = []
    for level in ALERT_LEVELS:
        c = level_counts.get(level, 0)
        if c > 0:
            parts.append(f"{c} {level}")
    if parts:
        print(f" ({', '.join(parts)})")
    else:
        print()
    print()

    # Print non-INFO alerts
    important = [a for a in alerts if a.level != "INFO"]
    if important:
        for a in important:
            prefix = {"CRITICAL": "!!!", "WARNING": " ! ", "SUGGESTION": " > "}
            print(f"  {prefix.get(a.level, '   ')} [{a.level}] {a.title}")
            print(f"      {a.detail}")
    else:
        print("  All clear — no issues detected.")

    print(f"\n{'='*50}\n")


# ═══════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════

def run_health_check(version: str = None, project_root: str = None) -> dict:
    """
    Run a full health check. Returns the health report dict.

    Args:
        version: Model version string (e.g. "v2"). Auto-detected if None.
        project_root: Project root path. Defaults to C:\\BlueprintLLM.
    """
    root = Path(project_root or os.environ.get("BLUEPRINT_LLM_ROOT", r"C:\BlueprintLLM"))
    models_dir = root / "models"
    history_path = root / "logs" / "training_history.json"

    # Resolve version
    if not version:
        version = find_latest_version(models_dir)
        if not version:
            print("No model versions found. Nothing to check.")
            return {}

    model_dir = find_version_dir(models_dir, version)
    if not model_dir:
        print(f"Model directory not found for {version}")
        return {}

    print(f"Training Health Monitor — checking {version}")
    print(f"  Model dir: {model_dir}")

    # Load history
    history = load_training_history(history_path)

    # Build current cycle data
    current = build_current_cycle(version, root, model_dir)

    # Run all 6 analysis dimensions
    alerts: List[Alert] = []
    alerts.extend(analyze_epoch_efficiency(current, history))
    alerts.extend(analyze_overfitting(current, history))
    alerts.extend(analyze_learning_rate(current, history))
    alerts.extend(analyze_dataset_quality(current, root))
    alerts.extend(analyze_node_mastery_velocity(current, history))
    alerts.extend(analyze_resource_usage(current, history))

    # Sort: CRITICAL first, then WARNING, SUGGESTION, INFO
    level_order = {l: i for i, l in enumerate(ALERT_LEVELS)}
    alerts.sort(key=lambda a: level_order.get(a.level, 99))

    # Console output
    print_health_summary(alerts, current)

    # Write health report
    report = generate_health_report(alerts, current, history)
    report_path = root / "health_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    # Append to alerts log
    alerts_log = root / "logs" / "health_alerts.log"
    write_health_alerts_log(alerts, alerts_log)
    print(f"  Alerts log: {alerts_log}")

    # Update training history — append current cycle (without bulky step data)
    history_entry = {
        "cycle": len(history) + 1,
        "version": current["version"],
        "date": current["date"],
        "dataset_size": current["dataset_size"],
        "lesson_data_size": current["lesson_data_size"],
        "epochs": current["epochs"],
        "learning_rate": current["learning_rate"],
        "training_time_seconds": current.get("training_time_seconds"),
        "total_steps": current["total_steps"],
        "initial_loss": current.get("initial_loss"),
        "final_loss": current.get("final_loss"),
        "eval_loss": current.get("eval_loss"),
        "train_accuracy": current.get("train_accuracy"),
        "eval_accuracy": current.get("eval_accuracy"),
        "node_scores": current.get("node_scores", {}),
        "eval_summary": current.get("eval_summary"),
    }

    # Don't append duplicate if last entry is same version
    if not history or history[-1].get("version") != version:
        history.append(history_entry)
    else:
        history[-1] = history_entry  # Update in place

    save_training_history(history_path, history)
    print(f"  History: {history_path} ({len(history)} cycles)")

    return report


def main():
    parser = argparse.ArgumentParser(description="Training Health Monitor for BlueprintLLM")
    parser.add_argument("--version", type=str, default=None,
                        help="Model version to check (e.g. v2). Auto-detected if omitted.")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Project root directory")
    args = parser.parse_args()

    plog.start_step("5.3", "Run health checks",
                      args.version or "auto-detect")
    report = run_health_check(version=args.version, project_root=args.project_root)
    if not report:
        plog.complete_step("5.3", "Run health checks", "FAILED")
        sys.exit(1)

    overall = report.get("summary", {}).get("overall_health", "?")
    plog.complete_step("5.3", "Run health checks", f"Health: {overall}")

    # Exit with non-zero if critical alerts
    critical = report.get("summary", {}).get("by_level", {}).get("CRITICAL", 0)
    if critical > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
