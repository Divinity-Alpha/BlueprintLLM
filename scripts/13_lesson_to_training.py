"""
13_lesson_to_training.py
------------------------
Converts lesson files into training data entries.
Each lesson prompt + expected DSL becomes a training example.

Also generates variations of each prompt for data diversity:
  - Original instruction
  - Shorter/casual rephrasing
  - More technical rephrasing

Usage:
    python scripts/13_lesson_to_training.py --lesson lessons/lesson_01.json --output datasets/lesson_data.jsonl
    python scripts/13_lesson_to_training.py --lesson-dir lessons/ --output datasets/lesson_data.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def generate_variations(instruction: str, category: str) -> list[str]:
    """Generate 2-3 natural variations of an instruction for training diversity."""
    variations = [instruction]  # Always include original

    # Create a shorter casual version
    casual = instruction.lower()
    casual = casual.replace("create a blueprint that", "make a bp that")
    casual = casual.replace("create a blueprint with", "bp with")
    casual = casual.replace("create a blueprint for", "bp for")
    casual = casual.replace("when the game starts", "on begin play")
    casual = casual.replace("when the player presses", "on pressing")
    casual = casual.replace("input action", "input")
    casual = casual.replace("another actor overlaps", "something overlaps")
    casual = casual.replace("overlaps with the actor", "overlaps this actor")
    casual = casual.replace("boolean variable", "bool var")
    casual = casual.replace("integer variable", "int var")
    casual = casual.replace("using a ", "with ")
    casual = casual.replace(" node", "")
    casual = casual.strip()
    if casual != instruction.lower():
        variations.append(casual)

    # Create a more technical version
    technical = instruction
    technical = technical.replace("prints ", "calls PrintString with text ")
    technical = technical.replace("when the game starts", "on Event BeginPlay")
    technical = technical.replace("waits ", "uses a Delay node for ")
    technical = technical.replace("destroys itself", "calls DestroyActor on self")
    technical = technical.replace("each frame", "every tick via Event Tick")
    if technical != instruction:
        variations.append(technical)

    return variations


def lesson_to_training(lesson_path: str, output_path: str, append: bool = True):
    """Convert a lesson file into JSONL training entries."""

    with open(lesson_path, encoding="utf-8") as f:
        lesson = json.load(f)

    entries = []
    for prompt in lesson["prompts"]:
        variations = generate_variations(prompt["instruction"], prompt["category"])

        for var_instruction in variations:
            entry = {
                "instruction": var_instruction,
                "output": prompt["expected_dsl"],
                "source": f"lesson:{lesson['lesson_id']}:{prompt['id']}",
                "category": prompt["category"],
            }
            entries.append(entry)

    # Write
    mode = "a" if append else "w"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} training entries from {lesson['lesson_id']} to {output}")
    print(f"  ({len(lesson['prompts'])} prompts x ~{len(entries)//len(lesson['prompts'])} variations each)")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Convert lessons to training data")
    parser.add_argument("--lesson", type=str, help="Single lesson file")
    parser.add_argument("--lesson-dir", type=str, help="Directory of lesson files")
    parser.add_argument("--output", default="datasets/lesson_data.jsonl", help="Output JSONL path")
    parser.add_argument("--no-append", action="store_true", help="Overwrite instead of append")
    args = parser.parse_args()

    if args.lesson:
        lesson_to_training(args.lesson, args.output, append=not args.no_append)
    elif args.lesson_dir:
        lesson_dir = Path(args.lesson_dir)
        # Clear output if not appending
        if args.no_append:
            Path(args.output).write_text("")
        for lf in sorted(lesson_dir.glob("lesson_*.json")):
            lesson_to_training(str(lf), args.output, append=True)
    else:
        print("Specify --lesson or --lesson-dir")


if __name__ == "__main__":
    main()
