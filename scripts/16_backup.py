"""
16_backup.py
------------
Manual backup CLI for the BlueprintLLM project.

Creates milestone backups of datasets, results, lessons, and pipeline state.
Also lists existing backups and runs cleanup per retention policy.

Usage:
    python scripts/16_backup.py                    # create manual milestone backup
    python scripts/16_backup.py --list             # show all backups
    python scripts/16_backup.py --cleanup          # review and prune per policy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backup_utils import auto_backup, list_backups, cleanup_backups


def show_list():
    """Print a table of all backups."""
    backups = list_backups()
    if not backups:
        print("No backups found.")
        return

    print(f"\n{'Label':<55} {'Category':<12} {'Files':>6}  {'Timestamp'}")
    print("-" * 95)
    for b in backups:
        exists = "[OK]" if Path(b["path"]).exists() else "[MISSING]"
        print(f"{b['label']:<55} {b['category']:<12} {b['file_count']:>6}  {b['timestamp']}  {exists}")
    print(f"\nTotal: {len(backups)} backup(s)")


def run_cleanup():
    """Run cleanup and report."""
    print("Running backup cleanup per retention policy...")
    print(f"  Milestones: keep ALL")
    print(f"  Scheduled:  keep last 5")
    print(f"  Pre-train:  keep last 3")
    cleanup_backups()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Manual backup for BlueprintLLM")
    parser.add_argument("--list", action="store_true", help="List all backups")
    parser.add_argument("--cleanup", action="store_true", help="Run retention cleanup")
    args = parser.parse_args()

    if args.list:
        show_list()
    elif args.cleanup:
        run_cleanup()
    else:
        print("Creating manual milestone backup...")
        result = auto_backup(trigger="manual")
        if result:
            print(f"Backup created: {result}")
        else:
            print("Backup skipped (no changes detected).")


if __name__ == "__main__":
    main()
