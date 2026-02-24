#!/usr/bin/env python3
"""Deduplicate training dataset.

Phase 1: Remove exact output duplicates (keep first occurrence).
Phase 2: Cap structural near-duplicates (same structure, different values) at N per fingerprint.

Usage:
    python scripts/20_deduplicate_dataset.py                          # dry run
    python scripts/20_deduplicate_dataset.py --apply                  # write deduped file
    python scripts/20_deduplicate_dataset.py --apply --struct-cap 5   # also cap near-dupes
"""

import argparse
import hashlib
import json
import re
import sys
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="3")


def structural_fingerprint(output: str) -> str:
    """Normalize numbers and string literals to create a structural fingerprint."""
    s = re.sub(r'"[^"]*"', '"X"', output)
    s = re.sub(r'=\d+\.?\d*', '=N', s)
    return s


def deduplicate(train_path: Path, struct_cap: int = 0, apply: bool = False):
    """Deduplicate training JSONL.

    Args:
        train_path: Path to train.jsonl
        struct_cap: Max entries per structural fingerprint (0 = no cap, phase 1 only)
        apply: If True, write the deduplicated file (with backup)
    """
    entries = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    original_count = len(entries)
    print(f"Original dataset: {original_count} entries")

    # Phase 1: Exact output dedup
    seen_hashes = set()
    phase1 = []
    exact_dupes = 0
    for entry in entries:
        h = hashlib.md5(entry["output"].encode()).hexdigest()
        if h in seen_hashes:
            exact_dupes += 1
            continue
        seen_hashes.add(h)
        phase1.append(entry)

    print(f"Phase 1 — Exact output dedup: removed {exact_dupes}, remaining {len(phase1)}")

    # Phase 2: Structural near-dedup (optional)
    if struct_cap > 0:
        fp_counts = Counter()
        phase2 = []
        struct_dupes = 0
        for entry in phase1:
            fp = structural_fingerprint(entry["output"])
            fp_counts[fp] += 1
            if fp_counts[fp] > struct_cap:
                struct_dupes += 1
                continue
            phase2.append(entry)

        print(f"Phase 2 — Structural cap ({struct_cap}/fingerprint): removed {struct_dupes}, remaining {len(phase2)}")

        # Show what got trimmed
        trimmed_fps = defaultdict(int)
        fp_all = Counter()
        for entry in phase1:
            fp = structural_fingerprint(entry["output"])
            fp_all[fp] += 1
        for fp, count in fp_all.most_common():
            if count > struct_cap:
                trimmed_fps[fp] = count - struct_cap

        if trimmed_fps:
            print(f"\nTrimmed {len(trimmed_fps)} over-represented clusters:")
            for fp in sorted(trimmed_fps, key=trimmed_fps.get, reverse=True)[:10]:
                # Find an example instruction
                for entry in phase1:
                    if structural_fingerprint(entry["output"]) == fp:
                        print(f"  -{trimmed_fps[fp]:3d}  {entry['instruction'][:70]}")
                        break

        final = phase2
    else:
        final = phase1

    total_removed = original_count - len(final)
    print(f"\nSummary: {original_count} -> {len(final)} ({total_removed} removed, {total_removed/original_count*100:.1f}%)")

    # Pattern distribution after dedup
    patterns = Counter(e.get("pattern", "unknown") for e in final)
    print(f"\nPattern distribution after dedup:")
    for pat, count in patterns.most_common():
        print(f"  {count:4d}  {pat}")

    if apply:
        # Backup original
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = train_path.parent / f"train_pre_dedup_{timestamp}.jsonl"
        shutil.copy2(train_path, backup_path)
        print(f"\nBacked up original to: {backup_path}")

        # Write deduplicated
        with open(train_path, "w") as f:
            for entry in final:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Wrote {len(final)} entries to {train_path}")
    else:
        print(f"\nDry run — no files modified. Use --apply to write changes.")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate training dataset")
    parser.add_argument("--dataset", default="datasets/train.jsonl", help="Path to training JSONL")
    parser.add_argument("--struct-cap", type=int, default=0,
                        help="Max entries per structural fingerprint (0 = exact dedup only)")
    parser.add_argument("--apply", action="store_true", help="Write deduplicated file (with backup)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_path = root / args.dataset
    if not train_path.exists():
        print(f"Dataset not found: {train_path}")
        return

    plog.start_step("3.3", "Deduplicate dataset", str(train_path))
    deduplicate(train_path, struct_cap=args.struct_cap, apply=args.apply)
    plog.complete_step("3.3", "Deduplicate dataset")


if __name__ == "__main__":
    main()
