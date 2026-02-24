"""
08_generate_system_prompt.py
-----------------------------
Generates the enhanced system prompt that includes the full node vocabulary
reference. This prompt is used during BOTH training and inference so the
model always has a "cheat sheet" of valid nodes, pins, and types.

Usage:
    # Generate the prompt and save to a file
    python scripts/08_generate_system_prompt.py --output scripts/system_prompt.txt

    # Preview without saving
    python scripts/08_generate_system_prompt.py --preview

    # Count tokens (approximate) to check it fits
    python scripts/08_generate_system_prompt.py --count-tokens

WHY THIS MATTERS:
    Without this, the model has to memorize every valid node type and pin name
    purely from training examples. If it sees a request that needs a node it
    didn't encounter enough times during training, it hallucinates.

    With this, the model has a reference table on every single request —
    during training AND inference. It learns to "look up" the right node
    from the reference rather than guessing from memory.

    Think of it like the difference between a closed-book exam and an
    open-book exam. Same student, dramatically better results.
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.blueprint_patterns import NODE_CATALOG, get_all_categories, get_nodes_by_category
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="4")


# ============================================================
# PROMPT COMPONENTS
# ============================================================

PROMPT_HEADER = """You are a Blueprint programming assistant for Unreal Engine 5. \
Given a natural language description of desired game behavior, you generate \
valid Blueprint DSL code that implements that behavior.

## DSL FORMAT RULES

Your output MUST follow this exact format:

```
BLUEPRINT: <Name>
PARENT: <ParentClass>

VAR <Name>: <Type> = <DefaultValue>

GRAPH: EventGraph

NODE <id>: <NodeType> [Property=Value]

EXEC <from_id>.<pin> -> <to_id>.<pin>
DATA <from_id>.<pin> -> <to_id>.<pin> [<DataType>]
```

Rules:
- Node IDs use n1, n2, n3... numbering
- EXEC lines = white execution wires
- DATA lines = colored data wires, always include [Type]
- Every exec chain must start from an Event node
- Only use node types and pin names from the reference below"""


PROMPT_FOOTER = """## DATA TYPES

Valid types for VAR declarations and DATA connections:
Bool, Int, Float, String, Vector, Rotator, Transform, Actor, Object, Class, Array

## PARENT CLASSES

Common parent classes for BLUEPRINT declarations:
Actor, Pawn, Character, PlayerController, GameModeBase, ActorComponent, UserWidget

## OUTPUT RULES

- Generate ONLY the DSL code
- No explanations, no commentary
- Use only nodes and pins listed in the reference above
- Every EXEC chain must originate from an Event node
- Every DATA connection must specify its type in [brackets]"""


def generate_node_reference() -> str:
    """Generate the compact node reference section from the catalog."""
    lines = ["\n## NODE REFERENCE\n"]
    lines.append("Valid node types with their pins. Use ONLY these exact names.\n")

    for category in get_all_categories():
        nodes = get_nodes_by_category(category)
        lines.append(f"### {category}\n")

        for node in nodes:
            # Build compact pin notation
            # Input pins shown as ←pin:Type, output pins as pin:Type->
            input_pins = []
            output_pins = []

            for pin in node.pins:
                if pin.direction == "input":
                    input_pins.append(f"{pin.name}:{pin.pin_type}")
                else:
                    output_pins.append(f"{pin.name}:{pin.pin_type}")

            # Format: NodeType — description
            #   IN: pin:Type, pin:Type
            #   OUT: pin:Type, pin:Type
            lines.append(f"**{node.type_name}** — {node.description}")
            if input_pins:
                lines.append(f"  IN: {', '.join(input_pins)}")
            if output_pins:
                lines.append(f"  OUT: {', '.join(output_pins)}")
            lines.append("")

    return "\n".join(lines)


def generate_connection_rules() -> str:
    """Generate rules about valid connections."""
    return """
## CONNECTION RULES

- exec pins connect ONLY to exec pins (EXEC lines)
- Data pins connect ONLY to matching data types (DATA lines)
- Bool->Bool, Float->Float, Int->Float (implicit cast), Actor->Object (upcast OK)
- Each input pin accepts only ONE connection
- Output pins can connect to MULTIPLE inputs
- An EXEC output can only connect to ONE EXEC input (no exec fan-out)"""


def build_full_prompt() -> str:
    """Assemble the complete enhanced system prompt."""
    sections = [
        PROMPT_HEADER,
        generate_node_reference(),
        generate_connection_rules(),
        PROMPT_FOOTER,
    ]
    return "\n".join(sections)


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for English text)."""
    return len(text) // 4


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced system prompt with node reference")
    parser.add_argument("--output", type=str, default="scripts/system_prompt.txt", help="Output file")
    parser.add_argument("--preview", action="store_true", help="Print prompt without saving")
    parser.add_argument("--count-tokens", action="store_true", help="Estimate token count")
    parser.add_argument("--compact", action="store_true", help="Generate compact version (fewer tokens)")
    args = parser.parse_args()

    plog.start_step("4.1", "Generate system prompt")
    prompt = build_full_prompt()

    if args.preview:
        print(prompt)
        print(f"\n{'='*60}")
        print(f"Character count: {len(prompt):,}")
        print(f"Estimated tokens: ~{estimate_tokens(prompt):,}")
        print(f"Node types covered: {len(NODE_CATALOG)}")
        return

    if args.count_tokens:
        tokens = estimate_tokens(prompt)
        print(f"Characters: {len(prompt):,}")
        print(f"Estimated tokens: ~{tokens:,}")
        print(f"Node types: {len(NODE_CATALOG)}")
        print()

        # Context budget analysis
        print("Context budget analysis (2048 max_seq_length):")
        print(f"  System prompt:    ~{tokens:,} tokens")
        print(f"  User instruction: ~50-100 tokens (typical)")
        print(f"  Model output:     ~{2048 - tokens - 100:,} tokens remaining for DSL")
        print()

        if tokens > 800:
            print("WARNING: System prompt is large. Consider:")
            print("  - Increasing max_seq_length to 4096 in training config")
            print("  - Using --compact mode for a shorter prompt")
            print("  - Reducing the node catalog to only the nodes in your training data")
        else:
            print("OK: Prompt fits comfortably within typical context windows.")

        return

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")

    tokens = estimate_tokens(prompt)
    print(f"Enhanced system prompt saved to: {output_path}")
    print(f"  Characters: {len(prompt):,}")
    print(f"  Estimated tokens: ~{tokens:,}")
    print(f"  Node types: {len(NODE_CATALOG)}")
    print()
    print("NEXT STEPS:")
    print("  This prompt is automatically used by the updated training and inference scripts.")
    print("  If you add new nodes to blueprint_patterns.py, re-run this script to regenerate.")
    plog.complete_step("4.1", "Generate system prompt",
                        f"{len(prompt):,} chars, {len(NODE_CATALOG)} nodes")


if __name__ == "__main__":
    main()
