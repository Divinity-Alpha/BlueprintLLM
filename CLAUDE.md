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
```

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

Scripts are numbered `01`–`13` reflecting pipeline order. The orchestrator (`11_pipeline_orchestrator.py`) chains them together. Each script is self-contained with its own `argparse` CLI and can run independently.

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
