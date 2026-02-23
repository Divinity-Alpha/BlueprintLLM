"""
10_training_dashboard.py
------------------------
Reads training logs and displays a real-time (or post-hoc) dashboard showing
how your training is progressing. Works with the log files that the Hugging Face
Trainer automatically creates.

Usage:
    # Watch training in real-time (run in a SECOND PowerShell window while training)
    python scripts/10_training_dashboard.py --watch models/blueprint-lora

    # View completed training summary
    python scripts/10_training_dashboard.py --summary models/blueprint-lora

    # Export training history as CSV for spreadsheet analysis
    python scripts/10_training_dashboard.py --export models/blueprint-lora --output training_history.csv

    # Compare training runs
    python scripts/10_training_dashboard.py --compare models/blueprint-lora-v1 models/blueprint-lora-v2
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================
# LOG PARSING
# ============================================================

def find_trainer_state(model_dir: str) -> Path | None:
    """Find the most recent trainer_state.json in checkpoint directories."""
    model_path = Path(model_dir)

    # Check checkpoints in order (most recent first)
    checkpoints = sorted(model_path.glob("checkpoint-*/trainer_state.json"), reverse=True)
    if checkpoints:
        return checkpoints[0]

    # Also check the output dir itself
    direct = model_path / "trainer_state.json"
    if direct.exists():
        return direct

    return None


def parse_training_logs(model_dir: str) -> list[dict]:
    """Extract training metrics from Hugging Face trainer logs."""
    state_path = find_trainer_state(model_dir)

    if state_path is None:
        # Fall back: look for any log files
        print(f"  No trainer_state.json found in {model_dir}")
        print(f"  (Training may not have started yet, or logs are in a different location)")
        return []

    with open(state_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    # Separate train and eval logs
    entries = []
    for entry in log_history:
        parsed = {
            "step": entry.get("step", 0),
            "epoch": entry.get("epoch", 0),
        }

        if "loss" in entry:
            parsed["type"] = "train"
            parsed["loss"] = entry["loss"]
            parsed["learning_rate"] = entry.get("learning_rate", 0)
            parsed["grad_norm"] = entry.get("grad_norm", 0)
        elif "eval_loss" in entry:
            parsed["type"] = "eval"
            parsed["eval_loss"] = entry["eval_loss"]
        else:
            continue

        entries.append(parsed)

    return entries


def parse_training_config(model_dir: str) -> dict:
    """Load the training configuration."""
    config_path = Path(model_dir) / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_summary(model_dir: str):
    """Show a complete post-training summary."""
    entries = parse_training_logs(model_dir)
    config = parse_training_config(model_dir)

    if not entries:
        print("No training data found. Make sure training has started and produced checkpoints.")
        return

    train_entries = [e for e in entries if e.get("type") == "train"]
    eval_entries = [e for e in entries if e.get("type") == "eval"]

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    # Config info
    if config:
        print(f"\nModel: {config.get('base_model', '?')}")
        print(f"Dataset: {config.get('dataset', '?')}")
        print(f"Epochs: {config.get('epochs', '?')}")
        print(f"LoRA rank: {config.get('lora_r', '?')}")
        print(f"Learning rate: {config.get('learning_rate', '?')}")
        print(f"Batch size: {config.get('batch_size', '?')} x {config.get('gradient_accumulation_steps', '?')} grad accum")
        print(f"System prompt: {config.get('system_prompt_type', '?')}")

    # Loss trajectory
    if train_entries:
        first_loss = train_entries[0]["loss"]
        last_loss = train_entries[-1]["loss"]
        min_loss = min(e["loss"] for e in train_entries)
        max_loss = max(e["loss"] for e in train_entries)
        total_steps = train_entries[-1]["step"]

        print(f"\n--- Training Loss ---")
        print(f"  Starting loss:  {first_loss:.4f}")
        print(f"  Final loss:     {last_loss:.4f}")
        print(f"  Minimum loss:   {min_loss:.4f}")
        print(f"  Total reduction: {first_loss - last_loss:.4f} ({(first_loss - last_loss)/first_loss*100:.1f}%)")
        print(f"  Total steps:    {total_steps}")

        # Is the model learning?
        print(f"\n--- Diagnosis ---")
        reduction_pct = (first_loss - last_loss) / first_loss * 100

        if reduction_pct > 60:
            print(f"  STRONG LEARNING — Loss dropped {reduction_pct:.0f}%.")
            print(f"  The model is absorbing the patterns well.")
        elif reduction_pct > 30:
            print(f"  GOOD LEARNING — Loss dropped {reduction_pct:.0f}%.")
            print(f"  The model is learning. More epochs or data may help further.")
        elif reduction_pct > 10:
            print(f"  SLOW LEARNING — Loss only dropped {reduction_pct:.0f}%.")
            print(f"  Consider: higher learning rate, more epochs, or check data quality.")
        elif reduction_pct > 0:
            print(f"  MINIMAL LEARNING — Loss barely moved ({reduction_pct:.0f}%).")
            print(f"  Possible issues: learning rate too low, data too small, or data format issues.")
        else:
            print(f"  NO LEARNING — Loss didn't decrease or went up.")
            print(f"  Check: data format, system prompt consistency, learning rate.")

        # Overfitting check
        if eval_entries:
            last_train = train_entries[-1]["loss"]
            last_eval = eval_entries[-1]["eval_loss"]
            gap = last_eval - last_train

            print(f"\n  Train/Eval gap: {gap:.4f}")
            if gap > 0.5:
                print(f"  WARNING: Large gap suggests overfitting.")
                print(f"  -> Add more training data or reduce epochs.")
            elif gap > 0.2:
                print(f"  MILD OVERFITTING — Monitoring recommended.")
            else:
                print(f"  HEALTHY — Train and eval losses are close.")

        # Still improving at the end?
        if len(train_entries) >= 10:
            last_10_avg = sum(e["loss"] for e in train_entries[-10:]) / 10
            prev_10_avg = sum(e["loss"] for e in train_entries[-20:-10]) / min(10, len(train_entries[-20:-10]))

            if last_10_avg < prev_10_avg - 0.01:
                print(f"\n  STILL IMPROVING at the end. More epochs could help.")
                print(f"  (Last 10 avg: {last_10_avg:.4f} vs previous 10 avg: {prev_10_avg:.4f})")
            else:
                print(f"\n  CONVERGED — Loss plateaued. More epochs may not help.")
                print(f"  To improve further, add more training data or increase model capacity.")

    # Loss curve visualization (ASCII)
    if train_entries and len(train_entries) > 5:
        print(f"\n--- Loss Curve (ASCII) ---")
        display_ascii_chart(train_entries, eval_entries)

    # Eval results
    if eval_entries:
        print(f"\n--- Validation Loss ---")
        for e in eval_entries:
            print(f"  Step {e['step']:>6}: eval_loss = {e['eval_loss']:.4f}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print(f"  1. Run evaluation: python scripts/09_evaluate_model.py --model {model_dir}/final")
    print(f"  2. Test interactively: python scripts/07_inference.py --model {model_dir}/final --interactive")
    print("=" * 70)


def display_ascii_chart(train_entries: list, eval_entries: list, width: int = 60, height: int = 15):
    """Display a simple ASCII loss curve."""
    losses = [e["loss"] for e in train_entries]
    steps = [e["step"] for e in train_entries]

    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max_loss - min_loss
    if loss_range == 0:
        loss_range = 1

    # Resample to fit width
    if len(losses) > width:
        step_size = len(losses) / width
        resampled = []
        for i in range(width):
            idx = int(i * step_size)
            resampled.append(losses[idx])
        losses = resampled
    
    # Build chart
    chart = []
    for row in range(height):
        threshold = max_loss - (row / (height - 1)) * loss_range
        line = ""
        for col_val in losses:
            if abs(col_val - threshold) < loss_range / height:
                line += "#"
            elif col_val < threshold:
                line += " "
            else:
                line += " "
        
        # Y-axis label
        label = f"{threshold:.3f}"
        chart.append(f"  {label:>7} │{line}")
    
    # X-axis
    chart.append(f"          └{'─' * len(losses)}")
    x_label = f"          Step 0{' ' * (len(losses) - 10)}Step {steps[-1]}"
    chart.append(x_label)
    
    for line in chart:
        print(line)

    # Legend
    if eval_entries:
        eval_points = {e["step"]: e["eval_loss"] for e in eval_entries}
        print(f"\n  Eval checkpoints: ", end="")
        for step, loss in eval_points.items():
            print(f"Step {step}={loss:.4f}  ", end="")
        print()


def display_watch(model_dir: str, interval: int = 15):
    """Watch training progress in real-time by polling log files."""
    print(f"Watching training progress in: {model_dir}")
    print(f"Refreshing every {interval} seconds. Press Ctrl+C to stop.\n")

    last_step = 0
    last_loss = None

    try:
        while True:
            entries = parse_training_logs(model_dir)
            train_entries = [e for e in entries if e.get("type") == "train"]

            if not train_entries:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training data...", end="\r")
                time.sleep(interval)
                continue

            latest = train_entries[-1]
            current_step = latest["step"]
            current_loss = latest["loss"]
            current_epoch = latest.get("epoch", 0)
            current_lr = latest.get("learning_rate", 0)

            if current_step != last_step:
                # Calculate trend
                if last_loss is not None:
                    delta = current_loss - last_loss
                    arrow = "v" if delta < 0 else "^" if delta > 0 else "->"
                    trend = f"{arrow} {abs(delta):.4f}"
                else:
                    trend = "—"

                # Progress bar (rough)
                first_loss = train_entries[0]["loss"]
                progress = max(0, min(1, (first_loss - current_loss) / max(first_loss, 0.001)))
                bar_width = 20
                filled = int(bar_width * progress)
                bar = "#" * filled + "-" * (bar_width - filled)

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Step {current_step:>5} | "
                    f"Epoch {current_epoch:.1f} | "
                    f"Loss: {current_loss:.4f} {trend} | "
                    f"LR: {current_lr:.2e} | "
                    f"Progress: [{bar}] {progress:.0%}"
                )

                last_step = current_step
                last_loss = current_loss

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopped watching. Run --summary for a full report.")


# ============================================================
# COMPARISON
# ============================================================

def compare_runs(dirs: list[str]):
    """Compare multiple training runs."""
    print("\n" + "=" * 70)
    print("TRAINING RUN COMPARISON")
    print("=" * 70)

    all_data = []
    for d in dirs:
        entries = parse_training_logs(d)
        config = parse_training_config(d)
        train_entries = [e for e in entries if e.get("type") == "train"]

        if not train_entries:
            print(f"\n  {d}: No training data found")
            continue

        all_data.append({
            "dir": d,
            "config": config,
            "first_loss": train_entries[0]["loss"],
            "final_loss": train_entries[-1]["loss"],
            "min_loss": min(e["loss"] for e in train_entries),
            "total_steps": train_entries[-1]["step"],
            "epochs": config.get("epochs", "?"),
            "lr": config.get("learning_rate", "?"),
            "lora_r": config.get("lora_r", "?"),
            "examples": config.get("dataset", "?"),
        })

    if not all_data:
        print("No valid training runs found.")
        return

    # Table
    print(f"\n{'Run':<35} {'Steps':>6} {'Start':>8} {'Final':>8} {'Min':>8} {'Drop':>8}")
    print("-" * 80)
    for d in all_data:
        drop = f"{(d['first_loss'] - d['final_loss'])/d['first_loss']*100:.0f}%"
        name = Path(d["dir"]).name
        print(f"{name:<35} {d['total_steps']:>6} {d['first_loss']:>8.4f} {d['final_loss']:>8.4f} {d['min_loss']:>8.4f} {drop:>8}")

    # Config comparison
    print(f"\n{'Run':<35} {'Epochs':>7} {'LR':>10} {'LoRA r':>8}")
    print("-" * 65)
    for d in all_data:
        name = Path(d["dir"]).name
        print(f"{name:<35} {d['epochs']:>7} {d['lr']:>10} {d['lora_r']:>8}")

    print("=" * 70)


# ============================================================
# CSV EXPORT
# ============================================================

def export_csv(model_dir: str, output_path: str):
    """Export training history as CSV."""
    entries = parse_training_logs(model_dir)
    if not entries:
        print("No training data to export.")
        return

    path = Path(output_path)
    with open(path, "w") as f:
        f.write("step,epoch,type,loss,eval_loss,learning_rate,grad_norm\n")
        for e in entries:
            f.write(f"{e.get('step',0)},{e.get('epoch',0)},{e.get('type','')},")
            f.write(f"{e.get('loss','')},{e.get('eval_loss','')},")
            f.write(f"{e.get('learning_rate','')},{e.get('grad_norm','')}\n")

    print(f"Exported {len(entries)} entries to {path}")
    print(f"Open in Excel or Google Sheets to create custom charts.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Training progress dashboard")
    parser.add_argument("--watch", type=str, help="Watch training in real-time")
    parser.add_argument("--summary", type=str, help="Show completed training summary")
    parser.add_argument("--compare", nargs="+", help="Compare multiple training runs")
    parser.add_argument("--export", type=str, help="Export training history")
    parser.add_argument("--output", type=str, default="training_history.csv", help="CSV output path")
    parser.add_argument("--interval", type=int, default=15, help="Watch refresh interval (seconds)")
    args = parser.parse_args()

    if args.watch:
        display_watch(args.watch, args.interval)
    elif args.summary:
        display_summary(args.summary)
    elif args.compare:
        compare_runs(args.compare)
    elif args.export:
        export_csv(args.export, args.output)
    else:
        print("Usage:")
        print("  --watch MODEL_DIR     Watch training in real-time")
        print("  --summary MODEL_DIR   View completed training summary")
        print("  --compare DIR1 DIR2   Compare training runs")
        print("  --export MODEL_DIR    Export as CSV")


if __name__ == "__main__":
    main()
