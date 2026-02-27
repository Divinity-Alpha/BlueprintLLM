# BlueprintLLM — Claude Code Project Context

> **Last Updated:** 2026-02-26
> **Owner:** Divinity Alpha
> **Repo:** github.com/Divinity-Alpha/BlueprintLLM

---

## What This Project Is

BlueprintLLM is a self-improving AI system that trains LLMs to generate **validated, structurally correct** Unreal Engine 5 Blueprint DSL from natural language descriptions. The core innovation is the **teaching loop** — a closed-loop cycle of train → examine → grade → create lesson → retrain that targets specific weaknesses each iteration.

The long-term vision is a platform that trains validated AI models for ANY structured language (not just Blueprints). Blueprints are the proof-of-concept. The teaching loop infrastructure is language-agnostic.

---

## Hardware Configuration (Current — February 2026)

| Component | Details |
|---|---|
| **GPU 0 (PyTorch cuda:0)** | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96GB VRAM |
| **GPU 1 (PyTorch cuda:1)** | NVIDIA GeForce RTX 5070 Ti, 16GB VRAM |
| **System RAM** | 64GB DDR5 @ 4000 MT/s |
| **nvidia-smi ordering** | GPU 0 = 5070 Ti, GPU 1 = PRO 6000 (reversed from PyTorch) |
| **CUDA Version** | 13.1 |
| **Driver** | 591.74 |
| **Compute Capability** | PRO 6000 = sm_120 (Blackwell), 5070 Ti = sm_100 |

### CRITICAL Hardware Notes

- **Monitors are connected to the 5070 Ti, NOT the PRO 6000.** Display rendering must stay off the training card.
- **PyTorch and nvidia-smi order GPUs differently.** PyTorch cuda:0 = PRO 6000. nvidia-smi GPU 0 = 5070 Ti. Always use PyTorch ordering in scripts.
- **CUDA_VISIBLE_DEVICES=0 targets the PRO 6000** in our scripts (PyTorch ordering).
- **Blackwell (sm_120) has compatibility issues** with bitsandbytes 4-bit NF4 quantization. 4-bit does NOT work. Use 8-bit quantization.

### GPU Role Assignment

| GPU | Role | What Runs Here |
|---|---|---|
| **PRO 6000 (cuda:0)** | ML Training & Inference | 70B model training, exams, inference, all heavy ML work |
| **5070 Ti (cuda:1)** | Display + Light Tasks | Monitor rendering, UE5 editor, dashboard, 8B quick testing, general computing |

### Quantization — What Works and What Doesn't

| Method | Status | Notes |
|---|---|---|
| bitsandbytes 4-bit NF4 | ❌ FAILS | Segfaults or extreme slowdown on Blackwell sm_120 |
| bitsandbytes 8-bit | ✅ WORKS | 67.7GB VRAM, loads in ~125s, 3.4 tok/s inference |
| GPTQ (gptqmodel) | ❌ FAILS | Version incompatibility with transformers 5.x |
| AWQ (autoawq) | ❌ FAILS | Pins old torch version, won't build |
| bfloat16 (no quant) | ✅ WORKS | For small models (3B, 8B) that fit in VRAM |

**ALWAYS use `load_in_8bit=True` for the 70B model. Never attempt 4-bit on this hardware.**

---

## Software Environment

| Package | Version | Notes |
|---|---|---|
| Python | 3.11.9 | Installed to C:\Program Files\Python311 |
| PyTorch | 2.10.0+cu130 | CUDA-enabled build |
| transformers | 4.57.6 | Downgraded from 5.x for gptqmodel compat |
| bitsandbytes | 0.49.2 | 8-bit works, 4-bit does not on Blackwell |
| peft | Installed | QLoRA adapter management |
| trl | Installed | Training with SFTTrainer |
| accelerate | Installed | Multi-device management |
| CUDA Toolkit | 13.1 | Matches driver |
| Git | 2.53.0 | |
| Node.js | 24.14.0 | For Claude Code |
| Visual Studio 2022 | Community | C++ build tools for CUDA compilation |

### Virtual Environment
- Location: `C:\BlueprintLLM\venv\`
- Always activate before running: `.\venv\Scripts\activate`
- Python on PATH requires disabling Windows app execution aliases

---

## Project Structure

```
C:\BlueprintLLM\
├── scripts/                    # All pipeline scripts (numbered 01-20+)
│   ├── utils/                  # Shared utilities
│   ├── 01_analyze_blueprint_clipboard.py
│   ├── 02_dsl_to_training_entry.py
│   ├── 03_generate_synthetic_data.py
│   ├── 04_train_blueprint_lora.py      # Main training script
│   ├── 05_auto_translate_export.py
│   ├── 06_validate_dsl.py              # DSL parser/validator
│   ├── 07_inference.py                 # Interactive inference
│   ├── 08_generate_system_prompt.py
│   ├── 09_evaluate_model.py
│   ├── 10_training_dashboard.py
│   ├── 11_pipeline_orchestrator.py     # Autonomous pipeline
│   ├── 12_run_exam.py                  # Exam runner
│   ├── 13_lesson_to_training.py        # Lesson integration
│   ├── 14_update_dashboard.py          # Dashboard generator
│   ├── 15_stop_signal.py               # Graceful shutdown
│   ├── 16_backup.py                    # Backup utility
│   ├── 17_scheduled_backup.py          # Watchdog backup
│   ├── 18_restore_backup.py            # Restore from backup
│   ├── 19_training_health_monitor.py   # Automated health checks
│   ├── 20_deduplicate_dataset.py
│   ├── error_handler.py                # Retry logic, timeout handling
│   ├── pipeline_logger.py              # Step-numbered logging
│   ├── backup_utils.py
│   └── system_prompt.txt               # Training system prompt
├── datasets/
│   ├── train.jsonl                     # Main training data (~728KB, 1400+ examples)
│   ├── validation.jsonl                # Eval set (~21KB)
│   ├── auto_translated.jsonl           # Auto-translated examples
│   ├── synthetic_train.jsonl           # Synthetic training data
│   └── lesson_02_data.jsonl            # Lesson 2 additions
├── lessons/
│   ├── lesson_01.json                  # First teaching lesson (20 prompts)
│   └── lesson_02.json                  # Second teaching lesson
├── models/                             # Created by training (adapter weights)
├── exams/                              # Exam results
├── logs/                               # Training logs, pipeline logs
│   ├── pipeline_live_state.json        # Live dashboard state file
│   ├── pipeline_heartbeat              # Heartbeat for stall detection
│   └── step_timing_history.json        # Historical step timings for ETA
├── dashboard/
│   ├── index.html                      # Main training observatory
│   └── live.html                       # Live pipeline monitor
├── backups/                            # Local backups (primary)
├── offload/                            # Disk offload folder for model loading
├── pipeline_config.json                # Pipeline configuration
└── .claude/
    └── CLAUDE.md                       # THIS FILE
```

---

## Pipeline Step Numbering System

ALL pipeline output, logs, dashboard, and communication uses this canonical numbering:

| Step | Name | Duration | Auto? |
|---|---|---|---|
| **1** | **Data Foundation** | | |
| 1.1 | Export UE5 Blueprints | Manual | ❌ |
| 1.2 | Convert to DSL format | ~2 min | ✅ |
| 1.3 | Validate with parser | ~1 min | ✅ |
| 1.4 | Format as training JSONL | ~1 min | ✅ |
| 1.5 | Split train/validation sets | ~30 sec | ✅ |
| **2** | **Pre-Flight Checks** | | |
| 2.1 | Check STOP_SIGNAL | ~1 sec | ✅ |
| 2.2 | Verify GPU availability | ~5 sec | ✅ |
| 2.3 | Verify dataset integrity | ~10 sec | ✅ |
| 2.4 | Pre-training backup | ~2 min | ✅ |
| **3** | **Model Loading** | | |
| 3.1 | Load base model from cache | ~2 min (125s for 70B 8-bit) | ✅ |
| 3.2 | Apply quantization (8-bit) | Included in 3.1 | ✅ |
| 3.3 | Apply QLoRA adapters | ~30 sec | ✅ |
| 3.4 | Load tokenizer | ~10 sec | ✅ |
| 3.5 | Load training dataset | ~15 sec | ✅ |
| 3.6 | Configure trainer | ~5 sec | ✅ |
| **4** | **Training** | | |
| 4.1 | Training epoch 1 begin | ~1 sec | ✅ |
| 4.2 | Training epoch 1 steps | ~1-2 hrs | ✅ |
| 4.3 | Epoch 1 eval checkpoint | ~2-5 min | ✅ |
| 4.4 | Training epoch 2 begin | ~1 sec | ✅ |
| 4.5 | Training epoch 2 steps | ~1-2 hrs | ✅ |
| 4.6 | Epoch 2 eval checkpoint | ~2-5 min | ✅ |
| 4.7 | Final eval | ~2-5 min | ✅ |
| **5** | **Post-Training** | | |
| 5.1 | Verify loss health | ~10 sec | ✅ |
| 5.2 | Save LoRA adapter | ~1-2 min | ✅ |
| 5.3 | Save training config | ~2 sec | ✅ |
| 5.4 | Save training log | ~2 sec | ✅ |
| 5.5 | Post-training backup | ~2 min | ✅ |
| 5.6 | Run health monitor | ~10 sec | ✅ |
| **6** | **Examination** | | |
| 6.1 | Load exam prompts | ~5 sec | ✅ |
| 6.2 | Load trained model for inference | ~2-3 min | ✅ |
| 6.3 | Generate DSL prompt N/N | ~15-30 sec each | ✅ |
| 6.N+3 | Compare all outputs vs expected | ~5 sec | ✅ |
| 6.N+4 | Score node mastery | ~5 sec | ✅ |
| 6.N+5 | Categorize errors | ~5 sec | ✅ |
| 6.N+6 | Save exam results | ~2 sec | ✅ |
| **7** | **Claude Grading** | | |
| 7.1 | Upload exam results | Manual | ❌ |
| 7.2 | Analyze error patterns | ~5 min | ❌ Claude |
| 7.3 | Identify weak nodes | ~2 min | ❌ Claude |
| 7.4 | Write corrections | ~10 min | ❌ Claude |
| 7.5 | Create next lesson file | ~10 min | ❌ Claude |
| **8** | **Lesson Integration** | | |
| 8.1 | Load new lesson | ~2 sec | ✅ |
| 8.2 | Generate prompt variations | ~1 min | ✅ |
| 8.3 | Validate all DSL | ~30 sec | ✅ |
| 8.4 | Merge into training set | ~5 sec | ✅ |
| 8.5 | Update dataset stats | ~2 sec | ✅ |
| **9** | **Dashboard & Finalize** | | |
| 9.1 | Update main dashboard | ~5 sec | ✅ |
| 9.2 | Milestone backup | ~2 min | ✅ |
| 9.3 | Log to pipeline history | ~2 sec | ✅ |
| 9.4 | Update step timing history | ~2 sec | ✅ |
| 9.5 | Cycle complete | ~1 sec | ✅ |

**ALWAYS use this numbering in console output, log files, and dashboard displays.**
Format: `[STEP X.Y] STARTING/COMPLETE/PROGRESS: Description`

---

## Training Configuration (70B on PRO 6000)

```json
{
  "gpu": 0,
  "base_model": "meta-llama/Meta-Llama-3.1-70B",
  "epochs": 2,
  "learning_rate": 0.00005,
  "lora_rank": 64,
  "lora_alpha": 128,
  "batch_size": 1,
  "gradient_accumulation_steps": 4,
  "max_seq_length": 2048,
  "quantization": "8bit",
  "use_4bit": false,
  "use_8bit": true,
  "auto_backup": true,
  "max_scheduled_backups": 5,
  "stop_on_error": true,
  "stall_kill_seconds": 1800,
  "prompt_timeout_seconds": 120,
  "heartbeat_interval_seconds": 60
}
```

### Key Training Decisions
- **Learning rate 0.00005** (halved from 8B's 0.0001) — larger models need smaller LR
- **2 epochs** — sweet spot found during 8B training. More = overfitting (v1 lesson learned)
- **LoRA rank 64** — good balance of quality vs training speed
- **Gradient accumulation 4** — effective batch size of 4 without extra VRAM
- **8-bit quantization** — only method that works on Blackwell hardware
- **Stall kill 1800s** (30 min) — training has legitimate long pauses during eval checkpoints

### Overfitting Warning Signs (from v1 experience)
- Train loss < 0.05 = almost certainly memorizing
- Train loss << eval loss (gap > 50%) = memorizing
- Loss 0.013 with regurgitated format rules = classic overfit (v1 failure mode)
- v2 at 2 epochs: loss 0.260, eval accuracy 95.6% > train accuracy 93.5% = healthy generalization

---

## Backup System

### Three-Tier Architecture

**Tier 1: Local Backups (C:\BlueprintLLM\backups\)**
- Milestone backups after training/exam completion (NEVER auto-deleted)
- Pre-training safety backups (keep last 3)
- Scheduled watchdog backups every 6 hours (keep last 5)

**Tier 2: Secondary SSD Backup (D:\BlueprintLLMBackup\)**
- **Mirror of ALL local backups** — automatic duplicate
- Protects against primary drive failure
- MUST be updated every time a local backup runs
- Captures everything NOT in Git:
  - `models/` — trained LoRA adapter weights (the most critical files)
  - `datasets/` — curated training data (train.jsonl, validation.jsonl, etc.)
  - `lessons/` — teaching loop lesson files
  - `exams/` — exam results and history
  - `logs/` — training logs, pipeline history, timing data
  - `backups/backup_manifest.json` — integrity checksums
  - `pipeline_config.json` — current configuration
  - `dashboard/` — generated dashboard HTML

**Tier 3: Git (github.com/Divinity-Alpha/BlueprintLLM)**
- All scripts (scripts/*.py)
- Documentation
- System prompts
- NOT model weights, datasets, or logs (too large / contains training data)

### Backup Implementation

Every backup script MUST:
1. Save to `C:\BlueprintLLM\backups\` (primary)
2. Mirror to `D:\BlueprintLLMBackup\` (secondary SSD)
3. Include SHA256 checksums in backup_manifest.json
4. Log the backup to `logs/backup_log.txt`

```python
# Standard backup paths
PRIMARY_BACKUP = r"C:\BlueprintLLM\backups"
SECONDARY_BACKUP = r"D:\BlueprintLLMBackup"

# After any backup to PRIMARY_BACKUP:
import shutil
shutil.copytree(backup_path, os.path.join(SECONDARY_BACKUP, backup_name), dirs_exist_ok=True)
```

### What Gets Backed Up (per backup)
- `adapter_model.safetensors` (~500-800MB for 70B 8-bit LoRA)
- `adapter_config.json`
- `training_config.json`
- `datasets/train.jsonl` + `validation.jsonl`
- `lessons/lesson_*.json`
- `exams/exam_*.jsonl`
- `logs/*.log`
- `pipeline_config.json`
- `requirements.txt` snapshot

### What Does NOT Need Backup
- Base LLaMA model (re-downloads from HuggingFace, ~130GB cache)
- Python packages (reinstalls from requirements.txt)
- Scripts (in Git)
- venv/ folder (recreate with `python -m venv venv`)

---

## GPU Offloading Strategy

### Tasks for PRO 6000 (cuda:0) ONLY
- All 70B model operations (training, inference, exams)
- Any operation involving the trained LoRA adapter
- Heavy ML compute

### Tasks for 5070 Ti (cuda:1) — Offload These
- Dashboard generation and serving (no GPU needed, CPU task)
- DSL validation/parsing (CPU task)
- Dataset processing and deduplication (CPU task)
- File I/O operations (backups, log writing)
- 8B model quick testing (if needed for comparison)
- UE5 Editor (when user is working in Unreal)
- Display rendering (monitors connected here)

### Implementation
```python
# For training/inference scripts:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # PRO 6000 only

# For quick 8B testing on secondary GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 5070 Ti only

# For non-GPU tasks (dashboard, parsing, backups):
# Don't set CUDA_VISIBLE_DEVICES — these don't use GPU
```

### IMPORTANT: Never run on both GPUs simultaneously for training
Multi-GPU training requires matched VRAM. 96GB + 16GB = broken. Always single-GPU training on the PRO 6000.

---

## Error Handling Philosophy

### Retry Logic
- Default 3 retries with exponential backoff (30s, 60s, 120s)
- Timeout: model load = 600s, training steps = no hard timeout, inference = 120s per prompt
- CUDA OOM: clear cache, reduce batch size, retry
- Single exam prompt timeout: skip and continue (don't fail entire exam)
- Network timeout: retry with backoff

### Stall Detection
- Heartbeat file updated every 60 seconds during training
- Stall threshold: 1800s (30 min) for training steps
- Training has legitimate pauses during eval checkpoints (5-10 min)
- HeartbeatCallback in training fires on: on_log, on_evaluate, on_save

### Graceful Shutdown
- STOP_SIGNAL file approach: `python scripts/15_stop_signal.py stop`
- Pipeline checks between major steps (not mid-training batch)
- Current operation completes, checkpoint saves, then exits
- Resume with: `python scripts/11_pipeline_orchestrator.py --resume`

### Resume Capability
- On fatal error: saves state to `logs/pipeline_resume_state.json`
- `--resume` flag picks up from failed step
- Always creates safety backup before resuming

---

## Teaching Loop — How It Works

The teaching loop is the core methodology. Each cycle targets specific weaknesses found in the previous exam.

### The Loop
1. Train model on dataset (Steps 2-5)
2. Exam: test model against lesson prompts (Step 6)
3. Grade: analyze errors, identify weak nodes (Step 7 — requires Claude)
4. Create lesson: write targeted examples for weak areas (Step 7)
5. Integrate: add lesson data to training set (Step 8)
6. Repeat from step 1

### Error Taxonomy
- **Missing nodes** — model didn't generate a required node type
- **Missing EXEC** — execution flow connections missing
- **Missing DATA** — data pin connections missing
- **Format errors** — DSL syntax violations
- **Extra lines** — model generated unnecessary content

### Node Mastery Tracking
- 42+ target node types tracked individually
- Per-node accuracy from exam scores
- ≥85% = mastered (minimum threshold, not ship quality)
- ≥95% per-node with validation retry = good product quality
- ≥97% per-node = production quality

### Health Monitor Alerts
- Dataset > 2500 examples: suggest reducing to 1 epoch
- Train loss < eval loss × 0.5: overfitting detected
- Node stuck 3+ cycles: change teaching approach
- Node regresses after mastery: catastrophic forgetting
- Lesson data > 40% of total: risk of teaching to the test

---

## Training History

| Version | Date | Model | Hardware | Loss | Accuracy | Notes |
|---|---|---|---|---|---|---|
| v1 | 2026-02-22 | 8B | RTX 4070 12GB | 0.013 | N/A | OVERFIT — too many epochs, regurgitated format rules |
| v2 | 2026-02-23 | 8B | RTX 4070 12GB | 0.260 | 93.5% train / 95.6% eval | Healthy — 2 epochs, good generalization |
| v3 | 2026-02-25 | 3B | RTX 4070 12GB | 1.07 | N/A | FAILED — only 10 steps before graceful stop, model too small |
| v4 | 2026-02-26 | 70B | PRO 6000 96GB | TBD | TBD | FIRST 70B RUN — currently in progress |

---

## Console Output Standards

### Verbose Step Logging
```
[14:30:05] [STEP 2.1] STARTING: Pre-training backup
[14:30:07] [STEP 2.1] COMPLETE: Pre-training backup (2.1s)
           Saved to backups/pre_train_v4_20260226/
[14:30:07] [STEP 3.1] STARTING: Load base model | ETA: 2m 5s
           Model: meta-llama/Meta-Llama-3.1-70B (8-bit)
[14:32:12] [STEP 3.1] COMPLETE: Load base model (2m 5s)
           VRAM: 67.7 GB allocated on cuda:0
```

### Progress Updates During Training
```
[14:45:08] [STEP 4.2] PROGRESS: 7.2% (100/1393) | Elapsed: 10m | Remaining: 2h 5m | loss: 0.8432
```

### Error Format
```
[14:50:00] [STEP 6.3] ERROR: Prompt 7/20 timed out after 120s — skipping
[14:50:00] [STEP 6.3] RETRY 1/3: Attempting with reduced max_tokens
```

---

## Product Roadmap (Summary)

| Phase | Timeline | Goal |
|---|---|---|
| **1: Blueprint Mastery** | Now → Month 3 | 85%+ mastery on all 42+ node types |
| **2: UE5 Plugin** | Months 3-6 | First product, $29/month subscription |
| **3: Multi-System** | Months 6-12 | Add Behavior Trees, Data Tables, Materials, Animation, Niagara, Sequences |
| **4: Platform** | Months 12-18 | Teaching loop as a service for other industries |
| **5: Scale** | Months 18-24 | Multi-industry, $3-10M ARR |

---

## Things That Have Broken Before (Lessons Learned)

1. **Overfitting at low epoch counts looks like success** — v1 had loss 0.013 which seemed great but was memorization. Watch for train loss << eval loss.
2. **Windows PowerShell needs execution policy change** — `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
3. **New PATH entries require new terminal session** — close and reopen PowerShell after installing anything.
4. **Windows app execution aliases intercept `python` command** — disable in Settings → Apps → Advanced app settings → App execution aliases.
5. **bitsandbytes 4-bit does NOT work on Blackwell** — always use 8-bit.
6. **PyTorch and nvidia-smi order GPUs differently** — always verify with `torch.cuda.get_device_name(0)`.
7. **Stall detection too aggressive** — training has legit long pauses. Use 1800s threshold, not 600s.
8. **3B model cannot generate valid Blueprint DSL** — too small for structured output. Minimum 8B, ideally 70B.
9. **transformers 5.x breaks gptqmodel** — if needed, stay on transformers 4.57.x.
10. **HuggingFace login doesn't persist across sessions sometimes** — use `login(add_to_git_credential=False)` and re-login if downloads fail.

---

## Communication Protocol

- **Step numbering is the universal language.** Use it in all logs, dashboard, and communication.
- **Claude.ai (architect)** designs solutions and creates lessons. Share exam results there for grading.
- **Claude Code (builder)** implements solutions and runs the pipeline.
- **This file (CLAUDE.md)** is the shared context between both. Update it when architecture decisions change.
- **The user is the project director**, not a relay between AIs. Minimize copy-pasting between Claude.ai and Claude Code.
