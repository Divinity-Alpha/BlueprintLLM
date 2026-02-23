"""
18_restore_backup.py
--------------------
Restore utility for BlueprintLLM backups.

Lists available backups and restores a selected one, creating a safety
backup of current state before overwriting.

Usage:
    python scripts/18_restore_backup.py --list
    python scripts/18_restore_backup.py --restore milestone_v2_train_complete_20260223_143022
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backup_utils import (
    BACKUP_DIRS,
    BACKUP_FILES,
    BACKUPS_DIR,
    PROJECT_ROOT,
    auto_backup,
    list_backups,
)


def show_list():
    """Print a table of available backups."""
    backups = list_backups()
    if not backups:
        print("No backups found.")
        return

    print(f"\n{'#':<4} {'Label':<55} {'Category':<12} {'Files':>6}  {'Timestamp'}")
    print("-" * 99)
    for i, b in enumerate(backups, 1):
        exists = "[OK]" if Path(b["path"]).exists() else "[MISSING]"
        print(f"{i:<4} {b['label']:<55} {b['category']:<12} {b['file_count']:>6}  {b['timestamp']}  {exists}")
    print(f"\nTotal: {len(backups)} backup(s)")
    print("Use --restore <label> to restore a backup.")


def restore(label: str):
    """Restore a backup by label."""
    backups = list_backups()
    match = [b for b in backups if b["label"] == label]
    if not match:
        print(f"Error: backup '{label}' not found.")
        print("Use --list to see available backups.")
        sys.exit(1)

    entry = match[0]
    backup_path = Path(entry["path"])
    if not backup_path.exists():
        print(f"Error: backup directory missing: {backup_path}")
        sys.exit(1)

    # Show what will be overwritten
    print(f"\nRestoring backup: {label}")
    print(f"  Category:  {entry['category']}")
    print(f"  Timestamp: {entry['timestamp']}")
    print(f"  Files:     {entry['file_count']}")
    print(f"\nThe following will be OVERWRITTEN with backup contents:")

    for d in BACKUP_DIRS:
        src = backup_path / d
        dst = PROJECT_ROOT / d
        if src.exists():
            print(f"  {d}/  {'(exists)' if dst.exists() else '(will create)'}")

    for f in BACKUP_FILES:
        src = backup_path / f
        dst = PROJECT_ROOT / f
        if src.exists():
            print(f"  {f}  {'(exists)' if dst.exists() else '(will create)'}")

    # Check if model files are in the backup
    models_in_backup = backup_path / "models"
    if models_in_backup.exists():
        print(f"  models/  (pre-train backup includes model weights)")

    # Confirm
    print()
    response = input("Proceed? This will create a safety backup first. [y/N] ")
    if response.lower() not in ("y", "yes"):
        print("Restore cancelled.")
        return

    # Safety backup of current state
    print("\nCreating safety backup of current state...")
    safety = auto_backup(trigger="pre_restore")
    if safety:
        print(f"  Safety backup: {safety.name}")

    # Restore
    print(f"\nRestoring from: {label}")

    for d in BACKUP_DIRS:
        src = backup_path / d
        dst = PROJECT_ROOT / d
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Restored {d}/")

    for f in BACKUP_FILES:
        src = backup_path / f
        dst = PROJECT_ROOT / f
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Restored {f}")

    # Restore model if present
    if models_in_backup.exists():
        for model_dir in models_in_backup.iterdir():
            if model_dir.is_dir():
                dst = PROJECT_ROOT / "models" / model_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                final_src = model_dir / "final"
                if final_src.exists():
                    final_dst = dst / "final"
                    if final_dst.exists():
                        shutil.rmtree(final_dst)
                    shutil.copytree(final_src, final_dst)
                    print(f"  Restored models/{model_dir.name}/final/")

    print(f"\nRestore complete. Safety backup available at: {safety}")


def main():
    parser = argparse.ArgumentParser(description="Restore a BlueprintLLM backup")
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--restore", type=str, metavar="LABEL", help="Restore a backup by label")
    args = parser.parse_args()

    if args.list:
        show_list()
    elif args.restore:
        restore(args.restore)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
