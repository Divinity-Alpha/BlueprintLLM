"""
15_stop_signal.py
-----------------
Manages a graceful stop signal for long-running pipeline operations.

Usage:
    python scripts/15_stop_signal.py stop      # Request graceful stop
    python scripts/15_stop_signal.py resume     # Clear the stop signal
    python scripts/15_stop_signal.py status     # Check if a stop is pending
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stop_signal_utils import SIGNAL_FILE, is_stop_requested, clear_signal


def stop():
    SIGNAL_FILE.write_text("stop requested\n", encoding="utf-8")
    print(f"STOP signal created: {SIGNAL_FILE}")
    print("Running training/exam will finish its current step and exit gracefully.")


def resume():
    if is_stop_requested():
        clear_signal()
        print("STOP signal removed. Pipeline can proceed normally.")
    else:
        print("No STOP signal found — nothing to clear.")


def status():
    if is_stop_requested():
        print("STOP PENDING — next checkpoint will trigger graceful shutdown.")
    else:
        print("No stop pending — pipeline running normally.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("stop", "resume", "status"):
        print("Usage: python scripts/15_stop_signal.py {stop|resume|status}")
        sys.exit(1)

    {"stop": stop, "resume": resume, "status": status}[sys.argv[1]]()
