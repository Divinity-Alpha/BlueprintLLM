#!/usr/bin/env python3
"""
BlueprintLLM DSL Parser CLI
Usage:
    python run_parser.py --exam exam_lesson_01.jsonl
    python run_parser.py --exam exam_lesson_01.jsonl --save-ir output/
    python run_parser.py --dsl "BLUEPRINT: BP_Hello..."
    python run_parser.py --exam-dir results/exams/ --report
"""

import json
import sys
import os
import argparse
from parser import parse, save_ir


def process_exam(path: str, save_dir: str = None, verbose: bool = False):
    """Process a JSONL exam file. Returns summary stats."""
    results = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            raw = data.get("raw_output", "") or data.get("actual_dsl", "") or data.get("output", "")
            cat = data.get("category", f"prompt_{i}")

            if not raw:
                results.append({"category": cat, "valid": False, "errors": ["No output"], "warnings": []})
                continue

            result = parse(raw)
            result["category"] = cat
            results.append(result)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                ir_path = os.path.join(save_dir, f"{cat}.blueprint.json")
                save_ir(result, ir_path)

            if verbose:
                status = "OK" if not result["errors"] else "ERR"
                unmapped = result["stats"]["unmapped"]
                umap_str = f" [{unmapped} unmapped]" if unmapped else ""
                print(f"  {cat}: {status} | {result['stats']['nodes']}n {result['stats']['connections']}c{umap_str}")
                for e in result["errors"]:
                    print(f"    ERROR: {e}")
                if unmapped:
                    for w in result["warnings"]:
                        if w.startswith("Unmapped"):
                            print(f"    WARN: {w}")

    return results


def print_summary(results: list, label: str = ""):
    """Print aggregate stats for a batch of results."""
    total = len(results)
    valid = sum(1 for r in results if not r.get("errors"))
    total_nodes = sum(r.get("stats", {}).get("nodes", 0) for r in results)
    total_mapped = sum(r.get("stats", {}).get("mapped", 0) for r in results)
    total_unmapped = sum(r.get("stats", {}).get("unmapped", 0) for r in results)
    total_conns = sum(r.get("stats", {}).get("connections", 0) for r in results)

    # Collect all unique unmapped types
    unmapped_types = set()
    for r in results:
        for w in r.get("warnings", []):
            if w.startswith("Unmapped:"):
                ntype = w.split("Unmapped:")[1].strip().split("(")[0].strip()
                unmapped_types.add(ntype)

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  Prompts:     {total}")
    print(f"  Parse valid: {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"  Total nodes: {total_nodes} ({total_mapped} mapped, {total_unmapped} unmapped)")
    print(f"  Total conns: {total_conns}")
    map_pct = 100 * total_mapped / total_nodes if total_nodes else 0
    print(f"  Map rate:    {map_pct:.1f}%")

    if unmapped_types:
        print(f"\n  Unmapped node types ({len(unmapped_types)}):")
        for t in sorted(unmapped_types):
            print(f"    - {t}")

    return {"total": total, "valid": valid, "nodes": total_nodes,
            "mapped": total_mapped, "unmapped": total_unmapped,
            "unmapped_types": sorted(unmapped_types)}


def main():
    ap = argparse.ArgumentParser(description="BlueprintLLM DSL Parser")
    ap.add_argument("--exam", help="Path to exam .jsonl file")
    ap.add_argument("--exam-dir", help="Directory of exam .jsonl files")
    ap.add_argument("--dsl", help="Raw DSL string to parse")
    ap.add_argument("--save-ir", help="Directory to save .blueprint.json IR files")
    ap.add_argument("--report", action="store_true", help="Print summary report only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.dsl:
        result = parse(args.dsl)
        print(json.dumps(result, indent=2))
        return

    if args.exam:
        print(f"Processing: {args.exam}")
        results = process_exam(args.exam, args.save_ir, args.verbose)
        print_summary(results, os.path.basename(args.exam))
        return

    if args.exam_dir:
        all_results = []
        files = sorted(f for f in os.listdir(args.exam_dir) if f.endswith(".jsonl"))
        for fname in files:
            path = os.path.join(args.exam_dir, fname)
            print(f"Processing: {fname}")
            ir_dir = os.path.join(args.save_ir, fname.replace(".jsonl", "")) if args.save_ir else None
            results = process_exam(path, ir_dir, args.verbose)
            stats = print_summary(results, fname)
            all_results.extend(results)

        if len(files) > 1:
            print_summary(all_results, "TOTAL (all exams)")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
