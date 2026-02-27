# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blueprint LLM fine-tunes LLaMA models using QLoRA to generate Unreal Engine 5 Blueprint code in a custom DSL. The current production model is **Meta-Llama-3.1-70B** with **8-bit quantization** (bitsandbytes) running on an NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM). Smaller models (3.2-3B, 3.1-8B) are also supported. The system implements a curriculum-based training loop: raw UE5 clipboard exports are parsed into DSL, used to train LoRA adapters, evaluated via tiered exams, and failures feed back into new lessons for the next training iteration.

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

# Resume after failure (skips already-completed steps)
python scripts/11_pipeline_orchestrator.py --full --resume

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
- **`scripts/pipeline_logger.py`** — Unified pipeline logging with step tracking, ETA prediction, error/retry tracking, and live state for dashboard consumption. See [Pipeline Logger](#pipeline-logger) section below.
- **`scripts/error_handler.py`** — Error classification, retry logic, subprocess stall detection, per-prompt timeouts, CUDA OOM recovery, and resume state management. See [Error Handling](#error-handling) section below.
- **`scripts/11_pipeline_orchestrator.py`** — Master orchestrator. Manages state via `.pipeline_state.json`, detects data changes via hash, auto-increments model versions. Retries failed steps based on error category.
- **`scripts/09_evaluate_model.py`** — Tier-based test suite (Tiers 1–4). Scores on structure/keywords/node-types/connections/parseability. Pass threshold: 70%.
- **`scripts/19_training_health_monitor.py`** — Post-training health analysis across 6 dimensions (epoch efficiency, overfitting, learning rate, dataset quality, node mastery velocity, resource usage). Outputs `health_report.json` and `logs/training_history.json`.

### Hardware Configuration (`pipeline_config.json`)

The `pipeline_config.json` file at project root configures hardware-specific settings. Both the orchestrator (`PipelineConfig`) and training script (`04_train_blueprint_lora.py`) load it automatically. CLI arguments still override these values.

```json
{
    "hardware": {
        "cuda_visible_devices": "0",
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell 96GB",
        "gpu_vram_gb": 96
    },
    "model": {
        "base_model": "meta-llama/Meta-Llama-3.1-70B",
        "max_seq_length": 4096
    },
    "quantization": {
        "use_8bit": true,
        "use_4bit": false
    },
    "lora": { "lora_r": 64, "lora_alpha": 128 },
    "training": {
        "learning_rate": 5e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 4
    },
    "stall_detection": {
        "stall_kill_seconds_training": 10800
    }
}
```

**GPU pinning**: `CUDA_VISIBLE_DEVICES` is set in `pipeline_config.json`, the orchestrator's subprocess env, `run_pipeline.ps1`, and `startup_blueprint_llm.ps1`. This ensures GPU 0 (training GPU) is used consistently, keeping GPU 1 (display) free.

**Dual GPU setup**: The system has two GPUs:
- GPU 0 (CUDA device 0): NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM, compute)
- GPU 1 (nvidia-smi index 0): NVIDIA RTX 5070 Ti (16 GB, display)
Note: `CUDA_VISIBLE_DEVICES=0` maps to the PRO 6000 despite nvidia-smi showing it as index 1.

**Quantization**: The 70B model uses 8-bit quantization (bitsandbytes `load_in_8bit=True`), consuming ~68 GB VRAM plus ~30 GB for gradients/optimizer state. GPTQ (INT4) and AWQ are NOT working due to package version incompatibilities with the Blackwell GPU (sm_120) and current toolchain (torch 2.10.0+cu130, transformers 4.57.6).

**70B training performance**: The first training step takes ~96 minutes (measured) due to CUDA kernel JIT compilation, cuBLAS autotuning, bitsandbytes 8-bit warmup, and gradient checkpointing overhead on Blackwell architecture. This happens every training run, not just once. CUDA JIT kernels are cached to `.cuda_cache/` (via `CUDA_CACHE_PATH`) and `torch.backends.cudnn.benchmark = True` caches cuDNN autotuning within each run — both configured in `04_train_blueprint_lora.py`. The stall detection threshold is set to 10800s (3 hours) to accommodate this. Heartbeats are written at step boundaries.

**Zombie process cleanup**: Failed pipeline runs can leave Python processes consuming all GPU VRAM. Before relaunching, always check `nvidia-smi` and kill zombie processes with `taskkill /F /PID <pid>`.

**Model auto-detection**: If `base_model` is not set in `pipeline_config.json`, the orchestrator detects VRAM and picks: 70B (>=48 GB), 8B (>=12 GB), or 3B (fallback).

### Training Configuration

Stored per model version at `models/blueprint-lora-vN/training_config.json`. Key settings: QLoRA with 8-bit quantization (for 70B) or 4-bit NF4 (for smaller models), LoRA targeting all attention + MLP projections, gradient checkpointing enabled. The system prompt embeds a full node vocabulary reference (~5660 chars) so the model can use it as a "cheat sheet" rather than memorizing all node types.

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

`logs/pipeline_resume_state.json` is written when the pipeline fails. Contains which steps completed successfully and what error occurred. Used by `--resume` to skip completed steps on the next run. Automatically deleted on successful completion.

## Pipeline Logger

A unified `PipelineLogger` (`scripts/pipeline_logger.py`) provides step-numbered logging, ETA prediction from historical timings, and a live state file for dashboard consumption. All pipeline scripts (01–20) use it.

### How It Works

- The **orchestrator** generates a `PIPELINE_RUN_ID` and passes it to subprocesses via environment variable.
- Each script calls `get_logger(step_prefix="N")` which returns a real `PipelineLogger` when `PIPELINE_RUN_ID` is set, or a silent `_NoOpLogger` for standalone runs (zero extra output).
- The orchestrator's `Logger` class wraps `PipelineLogger` while keeping its own per-run log file.

### Step Numbering (9-Step Hierarchy)

| Phase | Script(s)            | Sub-steps                                                         |
|-------|----------------------|-------------------------------------------------------------------|
| 1     | 01, 05, 03, 13, 20  | 1.1 analyze, 1.2 translate, 1.3 synthetic, 1.4 lessons, 1.5 merge+dedup |
| 2     | 06, 08, orchestrator | 2.1 validate DSL, 2.2 validate JSONL, 2.3 system prompt, 2.4 data hash |
| 3     | 04 (setup)           | 3.1 prompt, 3.2 dataset, 3.3 GPU, 3.4 model, 3.5 LoRA, 3.6 trainer |
| 4     | 04 (training)        | 4.1 backup, 4.2 init, 4.3 training loop, 4.4–4.6 save, 4.7 backup |
| 5     | 10, 19, orchestrator | 5.1 verify, 5.2 summary, 5.3 health, 5.4 report, 5.5 history, 5.6 archive |
| 6     | 12                   | 6.1 lesson, 6.2 model, 6.3 prompts, 6.4 validate, 6.5 score, 6.6 save, 6.7 backup |
| 7     | 09                   | 7.1 load model, 7.2 run tests, 7.3 score, 7.4 report, 7.5 save  |
| 8     | 13 (context=8)       | 8.1–8.5 lesson integration for next cycle                        |
| 9     | 14                   | 9.1 collect, 9.2 charts, 9.3 build HTML, 9.4 write, 9.5 finalize |

### Output Files

- **`logs/pipeline_live.log`** — Append-only log shared by orchestrator and subprocesses. Truncated at each pipeline start.
- **`logs/pipeline_live_state.json`** — Atomically replaced JSON showing current step, status, and progress. Consumed by the dashboard.
- **`logs/step_timing_history.json`** — Accumulated per-step timings (last 10 runs each) used for ETA prediction.

### Output Format

```
[14:30:06] [STEP 3.1] STARTING: Load system prompt
[14:30:06] [STEP 3.1] COMPLETE: Load system prompt (0.8s)
           5,660 chars loaded
[14:30:06] [STEP 3.4] STARTING: Load base model + quantize | ETA: 4.2m
[14:34:22] [STEP 3.4] COMPLETE: Load base model + quantize (4m 15s)
[14:34:22] [STEP 4.3] STARTING: Training loop | ETA: 52.3m
           3 epochs, 458 examples
[14:35:10]   [4.3] 20/171 (12%) — loss=2.3412
[14:36:05]   [4.3] 40/171 (23%) — loss=1.8921
```

### Key Design Decisions

- **Progress is rate-limited** to every 5 seconds to prevent log spam from training loops.
- **`_NoOpLogger`** means scripts never need `if plog:` guards — standalone runs are completely silent.
- **Timing history** keeps last 10 runs per step_id for ETA calculation.

## Error Handling

The pipeline has robust error handling via `scripts/error_handler.py`, integrated into the orchestrator and subprocess scripts.

### Error Classification

Every subprocess failure is classified into one of 7 categories by inspecting stderr and exception types:

| Category | Pattern Examples | Retryable |
|----------|-----------------|-----------|
| `TIMEOUT` | subprocess timeout, stall detection | Yes |
| `CUDA_OOM` | "CUDA out of memory", `OutOfMemoryError` | Yes |
| `DISK_FULL` | "no space left on device" | **No** (stops immediately) |
| `NETWORK` | `ConnectionError`, `SSLError`, `URLError` | Yes |
| `CORRUPT_CHECKPOINT` | "corrupted", safetensors errors | Yes |
| `ENCODING` | `UnicodeEncodeError`, charmap codec | Yes |
| `UNKNOWN` | anything else | Yes |

### Retry Logic

Each pipeline step has a category that determines its retry policy:

| Step Category | Steps | Max Retries | Backoff | Timeout |
|---------------|-------|-------------|---------|---------|
| `data` | 1.x (analyze, translate, synthetic, lessons) | 2 | 10s, 30s | 10m |
| `validate` | 2.x (DSL validation) | 1 | 5s | 5m |
| `training` | 3.x–4.x (model setup, training) | 2 | 60s, 120s | 4h |
| `eval` | 7.x (evaluation) | 1 | 30s | 4h |
| `exam` | 6.x (exam) | 1 | 30s | 4h |
| `utility` | 5.x, 8.x, 9.x (post-training, dashboard) | 1 | 10s | 10m |

The `run_script()` function in the orchestrator handles the retry loop: classify error → check if retryable → wait backoff → retry or fail.

### Subprocess Stall Detection

For long-running steps (training, eval, exam), a `SubprocessMonitor` daemon thread watches both `pipeline_live_state.json` and `logs/pipeline_heartbeat` for liveness:
- **Warning** at 5 minutes without any update
- **Kill** at 10 minutes for eval/exam, **30 minutes for training** — terminates the subprocess and marks it as stalled

Training uses a higher threshold (1800s vs 600s) because it has legitimate long pauses during eval checkpoints, checkpoint saves, gradient accumulation, and CUDA synchronization.

The training script (`04_train_blueprint_lora.py`) writes heartbeats via `write_heartbeat()` on every training step, logging step, evaluation, and checkpoint save. The heartbeat file is a lightweight fallback that avoids JSON file-locking issues on Windows.

This catches `model.generate()` hangs that would otherwise block the pipeline indefinitely.

### Per-Prompt Timeouts

Both `12_run_exam.py` and `09_evaluate_model.py` wrap `model.generate()` in `per_prompt_timeout(timeout_seconds=300)`. If a single generation takes longer than 5 minutes:
- The prompt is skipped (not the entire script)
- Exam: result marked as `{"status": "timeout_skipped"}`, `actual_dsl = "[TIMEOUT]"`
- Eval: `TestResult` with `score=0`, `errors=["Generation timed out"]`

**Windows limitation**: Python can't kill threads stuck in C extensions. The thread lingers but the script continues. The subprocess-level timeout is the ultimate safety net.

### CUDA OOM Recovery

`cuda_oom_retry()` wraps `trainer.train()` in `04_train_blueprint_lora.py` with automatic config reduction:
1. Catch `torch.cuda.OutOfMemoryError`
2. Clear CUDA cache, delete old model
3. Reduce `max_seq_length` by half (min 512), or reduce `lora_r` by half (min 16)
4. Rebuild model + trainer with reduced config, retry up to 2 times

### Pipeline Resume

When the pipeline fails, it saves `logs/pipeline_resume_state.json` containing:
- Which step functions completed successfully
- What step failed and why
- A snapshot of pipeline state (data hash, model version)

To resume from a failure:
```bash
python scripts/11_pipeline_orchestrator.py --full --resume
```

This skips completed steps and picks up where it left off. On successful completion, the resume state file is automatically deleted.

### Dashboard Error Display

The live dashboard (`dashboard/live.html`) shows error state:
- **Status pill**: `retrying` (amber pulse) or `error` (red)
- **Error panel**: collapsible panel below the progress bar showing last 10 errors with color-coded category tags
- **Retry badge**: shows current retry attempt (e.g., "Retry 2/3")
- State fields in `pipeline_live_state.json`: `error_info`, `retry_info`, `error_history`

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

### Mirror to D:\BlueprintLLMBackup

All backups are automatically mirrored to `D:\BlueprintLLMBackup` via `_mirror_to_external()` in `backup_utils.py`. The mirror is non-fatal — if D: is unavailable, the backup proceeds normally with a warning. Retention cleanup also removes mirror copies to keep them in sync. The `--list` command shows mirror status with `[LM]` indicators (L=local, M=mirror).

### Restore Safety

The restore utility (`18_restore_backup.py`) always creates a safety backup of the current state before overwriting files, and asks for confirmation before proceeding.
