"""
02_dsl_to_training_entry.py
---------------------------
Converts a .dsl file + natural language descriptions into JSONL training entries.
Saves you from manually escaping newlines and quotes.

Usage:
    # Interactive: prompts you for descriptions
    python scripts/02_dsl_to_training_entry.py cleaned-data/parsed-blueprints/hello_world.dsl

    # With a descriptions file (one description per line)
    python scripts/02_dsl_to_training_entry.py cleaned-data/parsed-blueprints/hello_world.dsl --descriptions descriptions.txt

    # Append to existing training file
    python scripts/02_dsl_to_training_entry.py cleaned-data/parsed-blueprints/hello_world.dsl --output datasets/train.jsonl --append

The DSL file is the "answer" and each description becomes a separate training example
with the same answer. More descriptions per Blueprint = better generalization.
"""

import json
import sys
import argparse
from pathlib import Path

# Add parent to path for validation
sys.path.insert(0, str(Path(__file__).parent))


def validate_dsl(dsl_text: str) -> bool:
    """Quick validation check."""
    try:
        from utils.dsl_parser import parse_dsl
        parse_dsl(dsl_text)
        return True
    except Exception as e:
        print(f"  WARNING: DSL validation failed: {e}")
        return False


def create_entries(dsl_text: str, descriptions: list[str]) -> list[dict]:
    """Create JSONL entries from DSL text and descriptions."""
    entries = []
    for desc in descriptions:
        desc = desc.strip()
        if not desc or desc.startswith("#"):
            continue
        entries.append({
            "instruction": desc,
            "input": "",
            "output": dsl_text.strip(),
        })
    return entries


def interactive_descriptions() -> list[str]:
    """Prompt user to enter descriptions interactively."""
    print("\nEnter natural language descriptions for this Blueprint.")
    print("Write what a human would say to request this Blueprint.")
    print("Enter a blank line when done.\n")

    descriptions = []
    count = 1
    while True:
        try:
            desc = input(f"  Description {count}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not desc:
            break
        descriptions.append(desc)
        count += 1

    return descriptions


def main():
    parser = argparse.ArgumentParser(description="Convert DSL files to JSONL training entries")
    parser.add_argument("dsl_file", type=str, help="Path to .dsl file")
    parser.add_argument("--descriptions", type=str, help="File with descriptions (one per line)")
    parser.add_argument("--output", type=str, default="datasets/train.jsonl", help="Output JSONL file")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    args = parser.parse_args()

    # Read DSL
    dsl_path = Path(args.dsl_file)
    if not dsl_path.exists():
        print(f"Error: File not found: {dsl_path}")
        sys.exit(1)

    dsl_text = dsl_path.read_text(encoding="utf-8").strip()
    print(f"Loaded DSL: {dsl_path} ({len(dsl_text)} chars)")

    # Validate
    is_valid = validate_dsl(dsl_text)
    if is_valid:
        print("  [OK] DSL syntax is valid")
    else:
        response = input("  DSL has issues. Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)

    # Get descriptions
    if args.descriptions:
        desc_path = Path(args.descriptions)
        if not desc_path.exists():
            print(f"Error: Descriptions file not found: {desc_path}")
            sys.exit(1)
        descriptions = desc_path.read_text(encoding="utf-8").strip().splitlines()
        print(f"Loaded {len(descriptions)} descriptions from {desc_path}")
    else:
        descriptions = interactive_descriptions()

    if not descriptions:
        print("No descriptions provided. Exiting.")
        sys.exit(1)

    # Create entries
    entries = create_entries(dsl_text, descriptions)
    print(f"\nCreated {len(entries)} training entries")

    # Preview
    print("\n--- Preview of first entry ---")
    preview = json.dumps(entries[0], ensure_ascii=False)
    if len(preview) > 200:
        print(preview[:200] + "...")
    else:
        print(preview)
    print("--- End preview ---\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    action = "Appended" if args.append else "Wrote"

    with open(output_path, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{action} {len(entries)} entries to {output_path}")

    # Count total if appending
    if args.append:
        total = sum(1 for line in open(output_path) if line.strip())
        print(f"Total entries in file: {total}")


if __name__ == "__main__":
    main()
