"""
backup_utils.py
---------------
Shared backup module for the BlueprintLLM project.

Provides automatic and manual backup of datasets, results, lessons,
pipeline state, and system prompt. Models are NOT backed up by default
(too large) — only pre-training safety backups include the current best
model's final/ directory.

Usage:
    from backup_utils import auto_backup, list_backups, cleanup_backups
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUPS_DIR = PROJECT_ROOT / "backups"
HISTORY_FILE = BACKUPS_DIR / "backup_history.json"

# What gets backed up (relative to PROJECT_ROOT)
BACKUP_DIRS = ["datasets", "results", "lessons"]
BACKUP_FILES = [".pipeline_state.json", "scripts/system_prompt.txt"]

# Retention policy
KEEP_SCHEDULED = 5
KEEP_PRE_TRAIN = 3
# Milestones: keep ALL


# ── Helpers ──────────────────────────────────────────────────

def _sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path):
    """Copy a directory tree, creating parents as needed."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_file(src: Path, dst: Path):
    """Copy a single file, creating parents as needed."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_history() -> list:
    """Load backup history from JSON file."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_history(history: list):
    """Save backup history to JSON file."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _find_latest_model_dir() -> Path | None:
    """Find the latest model version's final/ directory."""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return None
    versions = []
    for d in models_dir.iterdir():
        if d.is_dir() and d.name.startswith("blueprint-lora-v"):
            try:
                v = int(d.name.split("-v")[-1])
                final = d / "final"
                if final.exists():
                    versions.append((v, final))
            except ValueError:
                continue
    if not versions:
        return None
    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0][1]


# ── Public API ───────────────────────────────────────────────

def generate_manifest(backup_dir: Path) -> dict:
    """Walk all files in a backup directory and compute SHA-256 hashes.

    Returns: {relative_path: sha256_hash}
    """
    manifest = {}
    for filepath in sorted(backup_dir.rglob("*")):
        if filepath.is_file() and filepath.name != "manifest.json":
            rel = filepath.relative_to(backup_dir).as_posix()
            manifest[rel] = _sha256(filepath)
    return manifest


def has_changes_since_last(manifest: dict) -> bool:
    """Compare current project file hashes against a stored manifest.

    Returns True if anything has changed (or if no previous manifest exists).
    """
    if not manifest:
        return True

    current = {}
    for d in BACKUP_DIRS:
        src = PROJECT_ROOT / d
        if src.exists():
            for filepath in src.rglob("*"):
                if filepath.is_file():
                    rel = (Path(d) / filepath.relative_to(src)).as_posix()
                    current[rel] = _sha256(filepath)
    for f in BACKUP_FILES:
        src = PROJECT_ROOT / f
        if src.exists():
            current[f] = _sha256(src)

    return current != manifest


def auto_backup(trigger: str, version: str = None, lesson: str = None) -> Path | None:
    """Create an automatic backup.

    Args:
        trigger: What caused this backup (e.g. "pre_train", "train_complete",
                 "exam_complete", "lesson_merged", "scheduled", "manual")
        version: Model version string (e.g. "v3")
        lesson: Lesson ID (e.g. "lesson_01")

    Returns:
        Path to the backup directory, or None if skipped.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build label
    if trigger in ("train_complete", "exam_complete", "lesson_merged", "manual"):
        parts = ["milestone"]
        if version:
            parts.append(version)
        if lesson:
            parts.append(lesson)
        parts.extend([trigger, ts])
        label = "_".join(parts)
        category = "milestone"
    elif trigger == "pre_train":
        parts = ["pre_train"]
        if version:
            parts.append(version)
        parts.append(ts)
        label = "_".join(parts)
        category = "pre_train"
    elif trigger == "scheduled":
        label = f"scheduled_{ts}"
        category = "scheduled"
    else:
        label = f"{trigger}_{ts}"
        category = "milestone"

    backup_dir = BACKUPS_DIR / label
    backup_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {category} backup: {label}")
    print(f"[Backup] Creating {category} backup: {label}")

    # Copy standard directories and files
    for d in BACKUP_DIRS:
        src = PROJECT_ROOT / d
        if src.exists():
            _copy_tree(src, backup_dir / d)

    for f in BACKUP_FILES:
        src = PROJECT_ROOT / f
        if src.exists():
            _copy_file(src, backup_dir / f)

    # Pre-train: also back up the current best model
    if trigger == "pre_train":
        model_final = _find_latest_model_dir()
        if model_final:
            rel = model_final.relative_to(PROJECT_ROOT)
            _copy_tree(model_final, backup_dir / rel)
            logger.info(f"  Included model: {rel}")

    # Generate manifest
    manifest = generate_manifest(backup_dir)
    with open(backup_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Update history
    history = _load_history()
    entry = {
        "label": label,
        "category": category,
        "trigger": trigger,
        "timestamp": ts,
        "version": version,
        "lesson": lesson,
        "file_count": len(manifest),
        "path": str(backup_dir),
    }
    history.append(entry)
    _save_history(history)

    # Cleanup old backups
    cleanup_backups()

    print(f"[Backup] Done: {label} ({len(manifest)} files)")
    return backup_dir


def cleanup_backups():
    """Enforce retention policy.

    Keep ALL milestone backups, last N scheduled, last M pre-train.
    Delete older ones.
    """
    history = _load_history()
    if not history:
        return

    to_remove = []

    # Group by category
    scheduled = [h for h in history if h["category"] == "scheduled"]
    pre_train = [h for h in history if h["category"] == "pre_train"]

    # Scheduled: keep last KEEP_SCHEDULED
    if len(scheduled) > KEEP_SCHEDULED:
        excess = scheduled[:-KEEP_SCHEDULED]
        to_remove.extend(excess)

    # Pre-train: keep last KEEP_PRE_TRAIN
    if len(pre_train) > KEEP_PRE_TRAIN:
        excess = pre_train[:-KEEP_PRE_TRAIN]
        to_remove.extend(excess)

    if not to_remove:
        return

    removed_labels = set()
    for entry in to_remove:
        backup_path = Path(entry["path"])
        if backup_path.exists():
            shutil.rmtree(backup_path)
            logger.info(f"  Cleaned up old backup: {entry['label']}")
        removed_labels.add(entry["label"])

    # Update history
    history = [h for h in history if h["label"] not in removed_labels]
    _save_history(history)

    if removed_labels:
        print(f"[Backup] Cleaned up {len(removed_labels)} old backup(s)")


def list_backups() -> list[dict]:
    """Return all backup entries from history."""
    return _load_history()
