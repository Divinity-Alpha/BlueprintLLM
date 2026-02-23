"""Shared helpers for graceful stop signal. Imported by training and exam scripts."""

from pathlib import Path

SIGNAL_FILE = Path(__file__).resolve().parent.parent / "STOP_SIGNAL"


def is_stop_requested() -> bool:
    """Return True if the STOP_SIGNAL file exists."""
    return SIGNAL_FILE.exists()


def clear_signal():
    """Remove the signal file after handling it."""
    if SIGNAL_FILE.exists():
        SIGNAL_FILE.unlink()
