# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blueprint LLM fine-tunes LLaMA models (3.2-3B or 3.1-8B) using QLoRA to generate Unreal Engine 5 Blueprint code in a custom DSL. The system implements a curriculum-based training loop: raw UE5 clipboard exports are parsed into DSL, used to train LoRA adapters, evaluated via tiered exams, and failures feed back into new lessons for the next training iteration.

## Key Commands

```bash
# Activate the virtual environment first
venv\Scripts\activate          # Windows cmd
source venv/Scripts/activate   # bash/git-bash

# Full pipeline (data → training → evaluation)
python scripts/11_pipeline_orchestrator.py --full

# Individual pipeline modes
python scripts/11_pipeline_orchestrator.py --data-only
python scripts/11_pipeline_orchestrator.py --train-only --force
python scripts/11_pipeline_orchestrator.py --eval-only

# Evaluate a specific model
python scripts/09_evaluate_model.py --model models/blueprint-lora-v1/final
python scripts/09_evaluate_model.py --model models/blueprint-lora-v1/final --quick   # smoke test
python scripts/09_evaluate_model.py --model models/blueprint-lora-v1/final --compare models/blueprint-lora-v2/final

# Run an exam against a lesson
python scripts/12_run_exam.py --lesson lessons/lesson_01.json --model models/blueprint-lora-v2/final

# Single-file operations
python scripts/01_analyze_blueprint_clipboard.py <clipboard_export.txt>
python scripts/05_auto_translate_export.py <file.txt> --training
python scripts/03_generate_synthetic_data.py --count 500 --output datasets/train.jsonl
python scripts/06_validate_dsl.py datasets/train.jsonl
python scripts/07_inference.py --model models/blueprint-lora/final --prompt "Create a..."

# Windows Task Scheduler wrapper
.\run_pipeline.ps1 -Mode full

# Graceful shutdown (one-click or CLI)
stop.bat                                        # request stop
resume.bat                                      # clear stop signal
python scripts/15_stop_signal.py stop           # CLI alternative
python scripts/15_stop_signal.py resume
python scripts/15_stop_signal.py status         # check if stop is pending
```

## Graceful Shutdown

A file-based `STOP_SIGNAL` mechanism lets you stop long-running operations without losing work. Double-click `stop.bat` (or run `python scripts/15_stop_signal.py stop`) to request a stop. The running process finishes its current unit of work, saves state, and exits cleanly.

Where it's checked:
- **`04_train_blueprint_lora.py`** — Custom `TrainerCallback` checks at each logging step (~every 10 batches). Saves a resumable HF checkpoint before exiting.
- **`12_run_exam.py`** — Checks between each prompt. Saves partial results already collected to the JSONL output file.
- **`11_pipeline_orchestrator.py`** — Checks between each pipeline stage (analyze, validate, merge, prompt, train, evaluate, summary). Finishes the current stage before stopping.

Key principle: never interrupt mid-operation. The signal is only checked at safe boundaries. The signal file is automatically deleted after being handled. Use `resume.bat` to clear a signal that was set but not yet consumed.

## Architecture

### Data Flow

```
raw-data/clipboard-exports/   UE5 clipboard text
        ↓ (01_analyze → 05_auto_translate)
cleaned-data/parsed-blueprints/   DSL files
        ↓ (02_dsl_to_training_entry + 03_generate_synthetic + 13_lesson_to_training)
datasets/train.jsonl              merged training data (instruction → DSL output)
        ↓ (04_train_blueprint_lora)
models/blueprint-lora-vN/final    QLoRA adapter weights
        ↓ (09_evaluate_model / 12_run_exam)
results/                          eval reports + exam summaries
        ↓ (Claude creates correction lessons)
lessons/lesson_XX.json            feed back into next training cycle
```

### Script Numbering Convention

Scripts are numbered `01`–`15` reflecting pipeline order. The orchestrator (`11_pipeline_orchestrator.py`) chains them together. Each script is self-contained with its own `argparse` CLI and can run independently.

### Key Modules

- **`scripts/utils/dsl_parser.py`** — Parses and validates the Blueprint DSL. Converts between DSL text and Python dataclasses. All generated output must pass this parser.
- **`scripts/utils/blueprint_patterns.py`** — Node type catalog (50+ types) with pin definitions. Used for synthetic data generation and validation.
- **`scripts/11_pipeline_orchestrator.py`** — Master orchestrator. Manages state via `.pipeline_state.json`, detects data changes via hash, auto-increments model versions.
- **`scripts/09_evaluate_model.py`** — Tier-based test suite (Tiers 1–4). Scores on structure/keywords/node-types/connections/parseability. Pass threshold: 70%.

### Training Configuration

Stored per model version at `models/blueprint-lora-vN/training_config.json`. Key settings: QLoRA with 4-bit NF4 quantization, LoRA targeting all attention + MLP projections, gradient checkpointing enabled. The system prompt embeds a full node vocabulary reference (~5660 chars) so the model can use it as a "cheat sheet" rather than memorizing all node types.

## DSL Format

```
BLUEPRINT: BP_Name
PARENT: Actor
VAR Health: Float = 100.0
GRAPH: EventGraph
NODE n1: Event_BeginPlay
NODE n2: PrintString [InString="Hello"]
EXEC n1.Then -> n2.Execute
DATA n3.ReturnValue -> n4.Target [ObjectRef]
```

All generated Blueprint DSL must include `BLUEPRINT:`, `GRAPH:`, properly numbered `NODE` declarations, and `EXEC`/`DATA` connections. The parser in `dsl_parser.py` is the source of truth for valid syntax.

## Training Data Format

JSONL with `instruction` and `output` fields:
```json
{"instruction": "Create a Blueprint that...", "input": "", "output": "BLUEPRINT: BP_...\n..."}
```

## State Tracking

`.pipeline_state.json` at project root tracks run history, training data hashes, and the current model version number. The orchestrator uses this to skip redundant training when data hasn't changed.

## Backup System

An automatic backup system protects datasets, results, lessons, pipeline state, and the system prompt against accidental loss. Models are NOT backed up by default (too large); only pre-training safety backups include the current best model's `final/` directory.

### Commands

```bash
# Manual milestone backup
python scripts/16_backup.py

# List all backups
python scripts/16_backup.py --list

# Run retention cleanup
python scripts/16_backup.py --cleanup

# Start background watchdog (checks every 6 hours)
start_backup_watchdog.bat
python scripts/17_scheduled_backup.py --interval 6
python scripts/17_scheduled_backup.py --once        # single check

# Restore a backup
python scripts/18_restore_backup.py --list
python scripts/18_restore_backup.py --restore <label>
```

### Automatic Triggers

Backups are created automatically at these pipeline milestones:
- **Pre-training** (`04_train_blueprint_lora.py`) — includes current best model weights
- **Post-training** (`04_train_blueprint_lora.py`) — milestone after training completes
- **Post-exam** (`12_run_exam.py`) — milestone after exam results are saved
- **Post-merge** (`13_lesson_to_training.py`) — milestone after lesson data is merged

### Retention Policy

- **Milestone** backups (train_complete, exam_complete, lesson_merged, manual): kept forever
- **Scheduled** backups: last 5 kept
- **Pre-train** backups: last 3 kept

### Restore Safety

The restore utility (`18_restore_backup.py`) always creates a safety backup of the current state before overwriting files, and asks for confirmation before proceeding.
