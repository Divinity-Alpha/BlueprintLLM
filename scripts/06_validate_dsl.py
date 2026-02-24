"""
06_validate_dsl.py
------------------
Validates Blueprint DSL files or JSONL datasets for correctness.
Checks syntax, node validity, connection integrity, and execution flow.

Usage:
    python scripts/06_validate_dsl.py datasets/train.jsonl
    python scripts/06_validate_dsl.py my_blueprint.dsl
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Add parent to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.dsl_parser import parse_dsl, DSLParseError, ConnectionType
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="2")


# ============================================================
# KNOWN NODE TYPES (expandable)
# ============================================================
# This is your starting vocabulary. Expand as you catalog more nodes.

KNOWN_NODE_TYPES = {
    # Events
    "Event_BeginPlay", "Event_Tick", "Event_ActorBeginOverlap",
    "Event_ActorEndOverlap", "Event_InputAction", "Event_CustomEvent",
    "Event_AnyDamage", "Event_Hit",

    # Flow Control
    "Branch", "Sequence", "FlipFlop", "DoOnce", "Gate",
    "ForEachLoop", "ForLoop", "WhileLoop", "Delay", "RetriggerableDelay",
    "Select", "Switch", "MultiGate",

    # Functions
    "CallFunction", "PrintString", "SetTimerByFunctionName",
    "ClearTimerByFunctionName",

    # Math
    "AddFloat", "SubtractFloat", "MultiplyFloat", "DivideFloat",
    "AddInt", "SubtractInt", "MultiplyInt",
    "GreaterThan", "LessThan", "GreaterEqualFloat", "LessEqualFloat",
    "EqualEqual", "NotEqual",
    "ClampFloat", "ClampInt", "Abs", "Min", "Max",
    "MakeRotator", "BreakRotator", "MakeVector", "BreakVector",
    "BooleanAND", "BooleanOR", "BooleanNOT",

    # Variables
    "GetVar", "SetVar",

    # Actor
    "DestroyActor", "SetActorHiddenInGame", "SetActorLocation",
    "GetActorLocation", "SetActorRotation", "GetActorRotation",
    "AddActorLocalRotation", "AddActorLocalOffset",
    "GetDistanceTo", "GetActorForwardVector",

    # Components
    "SetVisibility", "SetRelativeRotation", "SetRelativeLocation",
    "SetWorldLocation", "SetWorldRotation",

    # Rendering
    "SetMaterial", "SetStaticMesh",

    # Audio
    "PlaySound", "PlaySoundAtLocation", "StopSound",

    # Physics
    "AddForce", "AddImpulse", "SetSimulatePhysics",

    # Casting
    "CastToCharacter", "CastToPlayerController", "CastToPawn",

    # Input
    "GetInputAxisValue", "IsInputKeyDown",

    # Utility
    "GetWorldDeltaSeconds", "GetGameTimeInSeconds",
    "RandomFloat", "RandomInteger",
    "IsValid", "IsNotValid",
    "MakeString", "Concatenate",

    # Timeline
    "Timeline",

    # UI
    "CreateWidget", "AddToViewport", "RemoveFromParent",
}


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list
    warnings: list
    stats: dict


def validate_dsl(dsl_text: str) -> ValidationResult:
    """Validate a DSL string and return detailed results."""
    errors = []
    warnings = []
    stats = {"nodes": 0, "exec_connections": 0, "data_connections": 0, "variables": 0}

    # 1. Parse
    try:
        bp = parse_dsl(dsl_text)
    except DSLParseError as e:
        return ValidationResult(False, [f"Parse error: {e}"], [], stats)

    # 2. Check required fields
    if not bp.name:
        errors.append("Missing BLUEPRINT name")
    if not bp.parent_class:
        warnings.append("Missing PARENT class (defaulting to Actor)")
    if not bp.graphs:
        errors.append("No GRAPH defined")

    stats["variables"] = len(bp.variables)

    # 3. Validate each graph
    for graph_name, graph in bp.graphs.items():
        node_ids = set()
        node_types = {}

        # Check nodes
        for node in graph["nodes"]:
            stats["nodes"] += 1

            # Duplicate ID check
            if node.id in node_ids:
                errors.append(f"[{graph_name}] Duplicate node ID: {node.id}")
            node_ids.add(node.id)
            node_types[node.id] = node.node_type

            # Known type check
            if node.node_type not in KNOWN_NODE_TYPES:
                warnings.append(
                    f"[{graph_name}] Unknown node type: {node.node_type} "
                    f"(node {node.id}) — may be valid, not in vocabulary yet"
                )

        # Check connections
        for conn in graph["connections"]:
            if conn.conn_type == ConnectionType.EXEC:
                stats["exec_connections"] += 1
            else:
                stats["data_connections"] += 1

            # Source node exists
            if conn.from_node not in node_ids:
                errors.append(
                    f"[{graph_name}] Connection references non-existent source node: "
                    f"{conn.from_node}"
                )

            # Target node exists
            if conn.to_node not in node_ids:
                errors.append(
                    f"[{graph_name}] Connection references non-existent target node: "
                    f"{conn.to_node}"
                )

            # Data connections should have types
            if conn.conn_type == ConnectionType.DATA and not conn.data_type:
                warnings.append(
                    f"[{graph_name}] DATA connection {conn.from_node}.{conn.from_pin} -> "
                    f"{conn.to_node}.{conn.to_pin} missing type annotation"
                )

        # Check for disconnected nodes (no connections at all)
        connected_nodes = set()
        for conn in graph["connections"]:
            connected_nodes.add(conn.from_node)
            connected_nodes.add(conn.to_node)

        for node_id in node_ids:
            if node_id not in connected_nodes:
                node_type = node_types.get(node_id, "?")
                # Events without connections are a bigger deal
                if "Event" in node_type:
                    warnings.append(
                        f"[{graph_name}] Event node {node_id} ({node_type}) has no connections"
                    )
                else:
                    warnings.append(
                        f"[{graph_name}] Node {node_id} ({node_type}) is disconnected"
                    )

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, stats)


def validate_jsonl(path: Path) -> dict:
    """Validate all examples in a JSONL dataset."""
    results = {"total": 0, "valid": 0, "invalid": 0, "errors": [], "warnings_count": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            results["total"] += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                results["invalid"] += 1
                results["errors"].append(f"Line {line_num}: Invalid JSON: {e}")
                continue

            dsl_text = example.get("output", "")
            if not dsl_text:
                results["invalid"] += 1
                results["errors"].append(f"Line {line_num}: Missing 'output' field")
                continue

            result = validate_dsl(dsl_text)
            results["warnings_count"] += len(result.warnings)

            if result.is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                for error in result.errors:
                    instruction = example.get("instruction", "?")[:60]
                    results["errors"].append(f"Line {line_num} ({instruction}...): {error}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python 06_validate_dsl.py <file.jsonl>    # Validate training dataset")
        print("  python 06_validate_dsl.py <file.dsl>      # Validate single DSL file")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    step_id = os.environ.get("PIPELINE_STEP_ID", "2.1")
    plog.start_step(step_id, "Validate DSL", path.name)
    if path.suffix == ".jsonl":
        print(f"Validating JSONL dataset: {path}")
        print("=" * 60)
        results = validate_jsonl(path)

        print(f"\nTotal examples:  {results['total']}")
        print(f"Valid:           {results['valid']} ({results['valid']/max(results['total'],1)*100:.1f}%)")
        print(f"Invalid:         {results['invalid']}")
        print(f"Total warnings:  {results['warnings_count']}")

        if results["errors"]:
            print(f"\nErrors (first 20):")
            for error in results["errors"][:20]:
                print(f"  [X] {error}")
            if len(results["errors"]) > 20:
                print(f"  ... and {len(results['errors']) - 20} more")

    else:
        # Single DSL file
        print(f"Validating DSL file: {path}")
        print("=" * 60)
        dsl_text = path.read_text(encoding="utf-8")
        result = validate_dsl(dsl_text)

        print(f"\nValid: {'YES' if result.is_valid else 'NO'}")
        print(f"\nStats:")
        for key, val in result.stats.items():
            print(f"  {key}: {val}")

        if result.errors:
            print(f"\nErrors:")
            for e in result.errors:
                print(f"  [X] {e}")

        if result.warnings:
            print(f"\nWarnings:")
            for w in result.warnings:
                print(f"  [!] {w}")

        if result.is_valid and not result.warnings:
            print("\n[OK] Blueprint DSL is valid with no warnings!")

    plog.complete_step(step_id, "Validate DSL")


if __name__ == "__main__":
    main()
