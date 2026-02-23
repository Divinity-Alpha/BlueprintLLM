"""
17_scheduled_backup.py
----------------------
Background watchdog that periodically checks for file changes and creates
scheduled backups when data has changed.

Usage:
    python scripts/17_scheduled_backup.py                  # default: every 6 hours
    python scripts/17_scheduled_backup.py --interval 12    # every 12 hours
    python scripts/17_scheduled_backup.py --once            # single check then exit
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backup_utils import (
    BACKUPS_DIR,
    auto_backup,
    generate_manifest,
    has_changes_since_last,
    list_backups,
    PROJECT_ROOT,
)

# ── Logging ──────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backup_watchdog.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_last_manifest() -> dict:
    """Load the manifest from the most recent backup."""
    backups = list_backups()
    if not backups:
        return {}

    # Find the last backup that still exists on disk
    for entry in reversed(backups):
        manifest_path = Path(entry["path"]) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                return json.load(f)
    return {}


def run_check():
    """Check for changes and back up if needed. Returns True if backup was made."""
    logger.info("Checking for changes since last backup...")
    last_manifest = get_last_manifest()

    if has_changes_since_last(last_manifest):
        logger.info("Changes detected, creating scheduled backup...")
        result = auto_backup(trigger="scheduled")
        if result:
            logger.info(f"Backup created: {result.name}")
            return True
    else:
        logger.info("No changes detected, skipping backup.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Scheduled backup watchdog for BlueprintLLM")
    parser.add_argument("--interval", type=float, default=6, help="Hours between checks (default: 6)")
    parser.add_argument("--once", action="store_true", help="Run a single check then exit")
    args = parser.parse_args()

    if args.once:
        logger.info("Running single backup check...")
        run_check()
        return

    interval_seconds = args.interval * 3600
    logger.info(f"Backup watchdog started. Checking every {args.interval} hours.")

    while True:
        try:
            run_check()
        except Exception as e:
            logger.error(f"Backup check failed: {e}", exc_info=True)

        logger.info(f"Next check in {args.interval} hours.")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
