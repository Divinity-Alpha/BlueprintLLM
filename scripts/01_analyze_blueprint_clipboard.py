"""
01_analyze_blueprint_clipboard.py
---------------------------------
Paste Blueprint node data (Ctrl+C from UE5 Editor) into a text file,
then run this script to extract the structure: node types, pins, connections, and properties.

Usage:
    python scripts/01_analyze_blueprint_clipboard.py raw-data/clipboard-exports/my_blueprint.txt

This is your primary tool for Phase 1 — understanding the Blueprint file format.
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="1")


def parse_clipboard_export(text: str) -> list[dict]:
    """Parse UE5 Blueprint clipboard text into structured node data."""
    nodes = []
    current_node = None
    indent_stack = []

    for line in text.splitlines():
        stripped = line.strip()

        # Start of a new object
        if stripped.startswith("Begin Object"):
            match = re.match(
                r'Begin Object Class=([^\s]+)\s+Name="([^"]+)"', stripped
            )
            if match:
                current_node = {
                    "class": match.group(1),
                    "name": match.group(2),
                    "properties": {},
                    "pins": [],
                    "custom_properties": [],
                }
                indent_stack.append(current_node)

        elif stripped == "End Object" and indent_stack:
            finished = indent_stack.pop()
            if not indent_stack:
                nodes.append(finished)
            else:
                # Nested object — attach to parent
                parent = indent_stack[-1]
                if "children" not in parent:
                    parent["children"] = []
                parent["children"].append(finished)
            current_node = indent_stack[-1] if indent_stack else None

        elif current_node and stripped:
            # Parse CustomProperties (pin data)
            if stripped.startswith("CustomProperties Pin"):
                pin_data = parse_pin_line(stripped)
                if pin_data:
                    current_node["pins"].append(pin_data)

            # Parse regular properties
            elif "=" in stripped and not stripped.startswith("#"):
                key, _, value = stripped.partition("=")
                current_node["properties"][key.strip()] = value.strip()

    return nodes


def parse_pin_line(line: str) -> dict | None:
    """Parse a CustomProperties Pin line into structured pin data."""
    pin = {}

    # Extract key-value pairs from the pin definition
    patterns = {
        "PinId": r'PinId=([A-F0-9]+)',
        "PinName": r'PinName="([^"]*)"',
        "PinType": r'PinType\.PinCategory="([^"]*)"',
        "PinSubCategory": r'PinType\.PinSubCategoryObject=([^\s,]+)',
        "Direction": r'Direction="([^"]*)"',
        "DefaultValue": r'DefaultValue="([^"]*)"',
        "LinkedTo": r'LinkedTo=\(([^)]*)\)',
        "PinFriendlyName": r'PinFriendlyName="([^"]*)"',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            pin[key] = match.group(1)

    # Parse linked pins into a list
    if "LinkedTo" in pin and pin["LinkedTo"]:
        pin["LinkedTo"] = [
            link.strip() for link in pin["LinkedTo"].split(",") if link.strip()
        ]
    else:
        pin["LinkedTo"] = []

    return pin if pin else None


def extract_node_summary(nodes: list[dict]) -> dict:
    """Create a high-level summary of the Blueprint structure."""
    summary = {
        "total_nodes": len(nodes),
        "node_types": defaultdict(int),
        "unique_pin_types": set(),
        "connections": [],
        "events": [],
        "function_calls": [],
        "variables_referenced": set(),
    }

    for node in nodes:
        # Classify node type
        class_name = node["class"].split(".")[-1]
        summary["node_types"][class_name] += 1

        # Identify events
        if "Event" in class_name or "K2Node_Event" in node["class"]:
            event_name = node["properties"].get("EventReference", class_name)
            summary["events"].append(event_name)

        # Identify function calls
        if "CallFunction" in class_name:
            func_ref = node["properties"].get("FunctionReference", "Unknown")
            summary["function_calls"].append(func_ref)

        # Catalog pin types and connections
        for pin in node.get("pins", []):
            if "PinType" in pin:
                summary["unique_pin_types"].add(pin["PinType"])
            if pin.get("LinkedTo"):
                for link in pin["LinkedTo"]:
                    summary["connections"].append({
                        "from_node": node["name"],
                        "from_pin": pin.get("PinName", "?"),
                        "to": link,
                    })

    # Convert sets to lists for JSON serialization
    summary["unique_pin_types"] = sorted(summary["unique_pin_types"])
    summary["node_types"] = dict(summary["node_types"])
    summary["variables_referenced"] = sorted(summary["variables_referenced"])

    return summary


def generate_dsl_sketch(nodes: list[dict]) -> str:
    """Attempt to generate a rough DSL representation from parsed nodes."""
    lines = ["# Auto-generated DSL sketch (review and refine manually)", ""]

    # Node definitions
    lines.append("# --- NODES ---")
    for i, node in enumerate(nodes):
        class_name = node["class"].split(".")[-1]
        props = []
        for key, val in node["properties"].items():
            if key not in ("NodePosX", "NodePosY", "NodeGuid", "ErrorType"):
                props.append(f"{key}={val}")
        prop_str = f" [{', '.join(props)}]" if props else ""
        lines.append(f"NODE n{i}: {class_name}{prop_str}")

    lines.append("")
    lines.append("# --- CONNECTIONS ---")

    # Connection mapping
    pin_to_node = {}
    for i, node in enumerate(nodes):
        for pin in node.get("pins", []):
            if "PinId" in pin:
                pin_to_node[pin["PinId"]] = (f"n{i}", pin.get("PinName", "?"))

    for i, node in enumerate(nodes):
        for pin in node.get("pins", []):
            direction = pin.get("Direction", "")
            if direction == "EGPD_Output" and pin.get("LinkedTo"):
                pin_type = pin.get("PinType", "exec")
                prefix = "EXEC" if pin_type == "exec" else "DATA"
                from_pin_name = pin.get("PinName", "?")
                for link in pin["LinkedTo"]:
                    # Try to resolve the link target
                    link_id = link.split(" ")[0] if " " in link else link
                    lines.append(
                        f"{prefix} n{i}.{from_pin_name} -> {link} [{pin_type}]"
                    )

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_analyze_blueprint_clipboard.py <clipboard_file.txt>")
        print()
        print("Steps:")
        print("  1. Open a Blueprint in UE5 Editor")
        print("  2. Select all nodes (Ctrl+A)")
        print("  3. Copy (Ctrl+C)")
        print("  4. Paste into a .txt file")
        print("  5. Run this script on that file")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    plog.start_step("1.1", "Analyze blueprint", input_path.name)
    text = input_path.read_text(encoding="utf-8")
    print(f"Analyzing: {input_path}")
    print(f"File size: {len(text):,} characters")
    print("=" * 60)

    # Parse
    nodes = parse_clipboard_export(text)
    print(f"\nFound {len(nodes)} top-level nodes\n")

    # Summary
    summary = extract_node_summary(nodes)

    print("NODE TYPES:")
    for ntype, count in sorted(summary["node_types"].items()):
        print(f"  {ntype}: {count}")

    print(f"\nPIN TYPES: {', '.join(summary['unique_pin_types'])}")

    print(f"\nEVENTS: {', '.join(summary['events']) or 'None found'}")

    print(f"\nFUNCTION CALLS:")
    for func in summary["function_calls"]:
        print(f"  {func}")

    print(f"\nCONNECTIONS: {len(summary['connections'])}")
    for conn in summary["connections"][:20]:  # Show first 20
        print(f"  {conn['from_node']}.{conn['from_pin']} -> {conn['to']}")
    if len(summary["connections"]) > 20:
        print(f"  ... and {len(summary['connections']) - 20} more")

    # Save detailed JSON
    output_json = input_path.with_suffix(".analysis.json")
    with open(output_json, "w") as f:
        json.dump({
            "source": str(input_path),
            "nodes": nodes,
            "summary": summary,
        }, f, indent=2, default=str)
    print(f"\nDetailed analysis saved to: {output_json}")

    # Generate DSL sketch
    dsl = generate_dsl_sketch(nodes)
    output_dsl = input_path.with_suffix(".dsl.txt")
    output_dsl.write_text(dsl)
    print(f"DSL sketch saved to: {output_dsl}")
    print("\n--- DSL SKETCH ---")
    print(dsl)
    plog.complete_step("1.1", "Analyze blueprint", f"{len(nodes)} nodes")


if __name__ == "__main__":
    main()
