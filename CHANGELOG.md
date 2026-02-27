# Changelog

## 2026-02-26 18:50 — Add dashboard HTTP server

**Changed:**
- `serve_dashboard.py` — New file: HTTP server for dashboard pages. Supports foreground mode, detached background mode (`--background`), and stop (`--stop`). Uses `CREATE_NO_WINDOW` for persistence on Windows. PID tracked in `logs/dashboard_server.pid`.
- `serve_dashboard.bat` — New file: one-click launcher that calls `serve_dashboard.py --background`

**Why:** The dashboard HTML files (`live.html`, `index.html`, `flowchart.html`) use `fetch()` to poll `pipeline_live_state.json`, which requires an HTTP server (won't work with `file://` due to CORS). No server was previously running on port 8080.

**Notes:**
- Server runs on `http://localhost:8080` serving from project root
- Live Monitor: `http://localhost:8080/dashboard/live.html`
- Training Hub: `http://localhost:8080/dashboard/index.html`
- Flowchart: `http://localhost:8080/dashboard/flowchart.html`
- Stop with: `python serve_dashboard.py --stop`

## 2026-02-26 18:40 — Add backup mirroring to D:\BlueprintLLMBackup

**Changed:**
- `scripts/backup_utils.py` — Added `MIRROR_DIR` constant (`D:/BlueprintLLMBackup`); added `_mirror_to_external()` function that copies backups to the mirror drive; integrated mirror call in `auto_backup()` after manifest generation; updated `cleanup_backups()` to remove mirror copies during retention cleanup; added `mirror_path` field to backup history entries
- `scripts/16_backup.py` — Updated `show_list()` to display mirror status with `[LM]` indicators (L=local present, M=mirror present)
- `scripts/17_scheduled_backup.py` — Added `MIRROR_DIR` import from `backup_utils`

**Why:** Protect against local disk failure by mirroring all backups (milestone, scheduled, pre-train) to the D: drive. The mirror is non-fatal — if D: is unavailable, backup proceeds normally with a warning.

**Notes:**
- All backup triggers (manual, scheduled, pre-train, post-train, post-exam, post-merge) automatically mirror via `auto_backup()` in `backup_utils.py`
- Retention cleanup also cleans mirror copies to keep them in sync
- The orchestrator uses `auto_backup()` from `backup_utils.py`, so no changes needed to `11_pipeline_orchestrator.py`

## 2026-02-26 17:35 — Enable 70B model training on Blackwell GPU with 8-bit quantization

**Changed:**
- `pipeline_config.json` — Added `quantization` section (`use_8bit: true, use_4bit: false`); increased `stall_kill_seconds_training` from 1800 to 10800; reduced `logging_steps` from 10 to 5
- `scripts/04_train_blueprint_lora.py` — Added `use_8bit` config option; updated `setup_model_and_tokenizer()` to support 8-bit via BitsAndBytesConfig; added heartbeat writes in `on_train_begin`, `on_step_begin` callbacks, and before `trainer.train()` call; updated `quick_test()` for 70B 8-bit; updated `_apply_pipeline_config()` to read quantization config
- `scripts/07_inference.py` — Updated `load_hf_model()` to use 8-bit for 70B models
- `scripts/09_evaluate_model.py` — Updated model loading to use 8-bit for 70B models
- `scripts/12_run_exam.py` — Updated `load_model()` to use 8-bit for 70B models
- `launch_pipeline.py` — New file: launches pipeline as detached Windows process (CREATE_NO_WINDOW flag) to survive shell session termination

**Why:** Migrating from 3B/8B models to Meta-Llama-3.1-70B on NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM). GPTQ/AWQ quantization methods failed due to version incompatibilities (gptqmodel vs transformers 5.x/4.x, autoawq pinning old torch, llama-cpp-python build failures with VS 2026). bitsandbytes 8-bit quantization works reliably on Blackwell sm_120.

**Config changes:**
- `pipeline_config.json`:
  - `model.base_model`: `meta-llama/Meta-Llama-3.1-70B`
  - `quantization.use_8bit`: `true`
  - `stall_detection.stall_kill_seconds_training`: `10800` (3 hours)
  - `training.logging_steps`: `5`
- `CUDA_VISIBLE_DEVICES=0` maps to the RTX PRO 6000 Blackwell (physical GPU 1 in nvidia-smi)

**Notes:**
- 70B model in 8-bit uses ~68 GB VRAM (97.4 GB with overhead) on the PRO 6000
- Model load time: ~2 minutes
- Inference: ~3.4 tok/s (8.9s for 30 tokens)
- First training step on 70B can take 30+ minutes due to CUDA kernel JIT compilation and gradient checkpointing. Heartbeats added to prevent stall detector false kills.
- System has dual GPUs: RTX 5070 Ti (16 GB, display) + RTX PRO 6000 Blackwell (96 GB, compute)
- Zombie Python processes from failed runs can consume all VRAM. Kill them with `tasklist | grep python` + `taskkill` before relaunching.
- Package versions: torch 2.10.0+cu130, transformers 4.57.6, bitsandbytes (8-bit works), gptqmodel 4.2.5 (installed but GPTQ loading broken)
- GPTQ (INT4) does NOT work: gptqmodel has import errors with both transformers 4.x and 5.x
- AWQ does NOT work: autoawq pins torch 2.3.1, destroying CUDA 13.0/Blackwell support
- llama-cpp-python does NOT work: CUDA toolset build failures with VS 2026 (Build Tools 18)
