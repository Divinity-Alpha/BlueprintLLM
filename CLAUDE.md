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

# Training health monitor
python scripts/19_training_health_monitor.py                    # auto-detect latest version
python scripts/19_training_health_monitor.py --version v2       # check specific version

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
        ↓ (19_training_health_monitor)
health_report.json                health alerts + trends
logs/training_history.json        cycle-over-cycle metrics
        ↓ (Claude creates correction lessons)
lessons/lesson_XX.json            feed back into next training cycle
```

### Script Numbering Convention

Scripts are numbered `01`–`19` reflecting pipeline order. The orchestrator (`11_pipeline_orchestrator.py`) chains them together. Each script is self-contained with its own `argparse` CLI and can run independently.

### Key Modules

- **`scripts/utils/dsl_parser.py`** — Parses and validates the Blueprint DSL. Converts between DSL text and Python dataclasses. All generated output must pass this parser.
- **`scripts/utils/blueprint_patterns.py`** — Node type catalog (50+ types) with pin definitions. Used for synthetic data generation and validation.
- **`scripts/pipeline_logger.py`** — Unified pipeline logging with step tracking, ETA prediction, and live state for dashboard consumption. See [Pipeline Logger](#pipeline-logger) section below.
- **`scripts/11_pipeline_orchestrator.py`** — Master orchestrator. Manages state via `.pipeline_state.json`, detects data changes via hash, auto-increments model versions.
- **`scripts/09_evaluate_model.py`** — Tier-based test suite (Tiers 1–4). Scores on structure/keywords/node-types/connections/parseability. Pass threshold: 70%.
- **`scripts/19_training_health_monitor.py`** — Post-training health analysis across 6 dimensions (epoch efficiency, overfitting, learning rate, dataset quality, node mastery velocity, resource usage). Outputs `health_report.json` and `logs/training_history.json`.

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

`logs/training_history.json` accumulates per-cycle metrics (loss, accuracy, node scores, dataset size) used by the health monitor to detect trends, plateaus, and regressions across training cycles.

`health_report.json` at project root is the latest health check output with alerts, trends, and cycle data. Read by the dashboard generator to display the health panel.

`logs/pipeline_live_state.json` is the current pipeline step/status, atomically updated by `PipelineLogger`. Shows `{"status": "idle"}` when no pipeline is running.

`logs/step_timing_history.json` accumulates per-step durations (last 10 per step) used by `PipelineLogger` to predict ETAs for future runs.

## Pipeline Logger

A unified `PipelineLogger` (`scripts/pipeline_logger.py`) provides step-numbered logging, ETA prediction from historical timings, and a live state file for dashboard consumption. All pipeline scripts (01–20) use it.

### How It Works

- The **orchestrator** generates a `PIPELINE_RUN_ID` and passes it to subprocesses via environment variable.
- Each script calls `get_logger(step_prefix="N")` which returns a real `PipelineLogger` when `PIPELINE_RUN_ID` is set, or a silent `_NoOpLogger` for standalone runs (zero extra output).
- The orchestrator's `Logger` class wraps `PipelineLogger` while keeping its own per-run log file.

### Step Numbering

| Orch Step | Script(s)       | Sub-step IDs                                              |
|-----------|-----------------|-----------------------------------------------------------|
| 1         | 01, 05          | 1.1 (analyze), 1.2 (translate)                            |
| 2         | 06              | 2.1 (validate)                                            |
| 3         | 03, 13, merge, 20 | 3.1 (synthetic), 3.2 (lessons), 3.3 (merge+dedup)      |
| 4         | 08              | 4.1 (generate prompt)                                     |
| 5         | 04              | 5.1–5.8 (prompt, dataset, model, lora, backup, train, save, backup) |
| 6         | 09              | 6.1–6.4 (load model, run tests, report, save)             |
| 7         | 10              | 7.1 (summary)                                             |
| 8         | 19              | 8.1 (health check)                                        |
| 9         | 14              | 9.1 (dashboard)                                           |
| E         | 12 (standalone) | E.1–E.5 (lesson, model, exam, summary, backup)            |

### Output Files

- **`logs/pipeline_live.log`** — Append-only log shared by orchestrator and subprocesses. Truncated at each pipeline start.
- **`logs/pipeline_live_state.json`** — Atomically replaced JSON showing current step, status, and progress. Consumed by the dashboard.
- **`logs/step_timing_history.json`** — Accumulated per-step timings (last 10 runs each) used for ETA prediction.

### Output Format

```
[14:30:06] [STEP 5.1] STARTING: Load system prompt
[14:30:06] [STEP 5.1] COMPLETE: Load system prompt (0.8s)
           5,660 chars loaded
[14:30:06] [STEP 5.3] STARTING: Load base model | ETA: 4.2m
[14:34:22] [STEP 5.3] COMPLETE: Load base model (4m 15s)
[14:34:22] [STEP 5.6] STARTING: Training | ETA: 52.3m
           3 epochs, 458 examples
[14:35:10]   [5.6] 20/171 (12%) — loss=2.3412
[14:36:05]   [5.6] 40/171 (23%) — loss=1.8921
```

### Key Design Decisions

- **Progress is rate-limited** to every 5 seconds to prevent log spam from training loops.
- **`_NoOpLogger`** means scripts never need `if plog:` guards — standalone runs are completely silent.
- **Timing history** keeps last 10 runs per step_id for ETA calculation.

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
