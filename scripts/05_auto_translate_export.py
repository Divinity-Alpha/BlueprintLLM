"""
05_auto_translate_export.py
---------------------------
The missing link between "drop .txt in inbox" and "clean training data."

Takes the raw clipboard export (or its .analysis.json) and auto-translates
UE5 internal names into clean DSL, resolves pin connections, and optionally
generates training entries with AI-generated descriptions.

This REDUCES (but does not eliminate) the manual DSL-writing step.

Usage:
    # Translate a clipboard export into clean DSL
    python scripts/05_auto_translate_export.py raw-data/clipboard-exports/rotating_cube.txt

    # Translate and auto-generate training entries (with generic descriptions)
    python scripts/05_auto_translate_export.py raw-data/clipboard-exports/rotating_cube.txt --training

    # Translate all un-translated exports in the inbox
    python scripts/05_auto_translate_export.py --batch

    # Show the translation map (what it knows how to translate)
    python scripts/05_auto_translate_export.py --show-map

QUALITY LEVELS:
    The auto-translated DSL is GOOD ENOUGH for training, but hand-reviewed
    DSL is always better. Think of this as a "first draft" that gets you 80%
    of the way there. You can:
      A) Use it as-is for bulk data (quantity over quality)
      B) Review and fix it before adding to training data (quality)
      C) Skip it entirely and write DSL by hand (gold standard)

    The pipeline supports all three approaches simultaneously via
    manual_examples.jsonl (your gold) + auto_translated.jsonl (machine draft).
"""

import re
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from utils.dsl_parser import parse_dsl, DSLParseError


# ============================================================
# UE5 -> DSL TRANSLATION MAP
# ============================================================
# This is the Rosetta Stone. Every time you analyze a new Blueprint
# and discover a new UE5 internal name, add it here.
#
# Format: "UE5_internal_identifier": "CleanDSLName"
#
# The script matches against class names AND function references.

# Node class translations
NODE_CLASS_MAP = {
    # Events
    "K2Node_Event": "_EVENT_",  # Special: needs further resolution from EventReference
    "K2Node_CustomEvent": "Event_CustomEvent",
    "K2Node_InputAction": "Event_InputAction",

    # Function calls (generic — resolved by FunctionReference)
    "K2Node_CallFunction": "_FUNCTION_",  # Special: resolved by MemberName

    # Flow control
    "K2Node_IfThenElse": "Branch",
    "K2Node_ExecutionSequence": "Sequence",
    "K2Node_FlipFlop": "FlipFlop",
    "K2Node_DoOnce": "DoOnce",
    "K2Node_Delay": "Delay",
    "K2Node_MacroInstance": "_MACRO_",  # Needs further resolution

    # Math operators
    "K2Node_CommutativeAssociativeBinaryOperator": "_MATH_OP_",  # Resolved by function
    "K2Node_PromotableOperator": "_MATH_OP_",

    # Variables
    "K2Node_VariableGet": "GetVar",
    "K2Node_VariableSet": "SetVar",

    # Casting
    "K2Node_DynamicCast": "_CAST_",  # Resolved by target class
}

# Function reference translations (MemberName -> DSL name)
FUNCTION_MAP = {
    # Print / Debug
    "PrintString": "PrintString",
    "PrintText": "PrintString",

    # Actor functions
    "K2_DestroyActor": "DestroyActor",
    "K2_SetActorHiddenInGame": "SetActorHiddenInGame",
    "K2_SetActorLocation": "SetActorLocation",
    "K2_GetActorLocation": "GetActorLocation",
    "K2_SetActorRotation": "SetActorRotation",
    "K2_GetActorRotation": "GetActorRotation",
    "K2_AddActorLocalRotation": "AddActorLocalRotation",
    "K2_AddActorLocalOffset": "AddActorLocalOffset",
    "GetDistanceTo": "GetDistanceTo",
    "GetActorForwardVector": "GetActorForwardVector",

    # Component functions
    "SetVisibility": "SetVisibility",
    "K2_SetRelativeRotation": "SetRelativeRotation",
    "K2_SetRelativeLocation": "SetRelativeLocation",
    "K2_SetWorldLocation": "SetWorldLocation",
    "K2_SetWorldRotation": "SetWorldRotation",

    # Math — construction
    "MakeRotator": "MakeRotator",
    "BreakRotator": "BreakRotator",
    "MakeVector": "MakeVector",
    "BreakVector": "BreakVector",
    "Conv_FloatToString": "MakeString",

    # Math — operations (from CommutativeAssociativeBinaryOperator)
    "Multiply_FloatFloat": "MultiplyFloat",
    "Add_FloatFloat": "AddFloat",
    "Subtract_FloatFloat": "SubtractFloat",
    "Divide_FloatFloat": "DivideFloat",
    "Multiply_IntInt": "MultiplyInt",
    "Add_IntInt": "AddInt",
    "Subtract_IntInt": "SubtractInt",

    # Math — comparison
    "Greater_FloatFloat": "GreaterThan",
    "Less_FloatFloat": "LessThan",
    "GreaterEqual_FloatFloat": "GreaterEqualFloat",
    "LessEqual_FloatFloat": "LessEqualFloat",
    "EqualEqual_FloatFloat": "EqualEqual",
    "NotEqual_FloatFloat": "NotEqual",
    "EqualEqual_BoolBool": "EqualEqual",

    # Math — utility
    "Clamp": "ClampFloat",
    "Abs": "Abs",
    "FMin": "Min",
    "FMax": "Max",
    "RandomFloatInRange": "RandomFloat",
    "RandomIntegerInRange": "RandomInteger",

    # Boolean
    "BooleanAND": "BooleanAND",
    "BooleanOR": "BooleanOR",
    "Not_PreBool": "BooleanNOT",

    # Utility
    "GetWorldDeltaSeconds": "GetWorldDeltaSeconds",
    "GetGameTimeInSeconds": "GetGameTimeInSeconds",
    "IsValid": "IsValid",

    # Timer
    "K2_SetTimerByFunctionName": "SetTimerByFunctionName",
    "K2_ClearTimerByFunctionName": "ClearTimerByFunctionName",

    # Physics
    "AddForce": "AddForce",
    "AddImpulse": "AddImpulse",
    "SetSimulatePhysics": "SetSimulatePhysics",

    # Audio
    "PlaySound2D": "PlaySound",
    "PlaySoundAtLocation": "PlaySoundAtLocation",
    "SpawnSoundAtLocation": "PlaySoundAtLocation",

    # UI
    "Create Widget": "CreateWidget",
    "AddToViewport": "AddToViewport",
    "RemoveFromParent": "RemoveFromParent",
}

# Event reference translations
EVENT_MAP = {
    "ReceiveBeginPlay": "Event_BeginPlay",
    "ReceiveTick": "Event_Tick",
    "ReceiveActorBeginOverlap": "Event_ActorBeginOverlap",
    "ReceiveActorEndOverlap": "Event_ActorEndOverlap",
    "ReceiveHit": "Event_Hit",
    "ReceiveAnyDamage": "Event_AnyDamage",
}

# Pin name normalization
PIN_NAME_MAP = {
    "execute": "Execute",
    "then": "Then",
    "exec": "Execute",
    "self": None,  # Skip self pins
    "WorldContextObject": None,  # Skip context pins
    "bPrintToScreen": None,  # Skip default params
    "bPrintToLog": None,
    "TextColor": None,
    "Duration": None,  # Only skip on PrintString, keep on Delay
    "Key": None,
    "ReturnValue": "ReturnValue",
    "DeltaSeconds": "DeltaSeconds",
    "DeltaRotation": "DeltaRotation",
    "Condition": "Condition",
    "InString": "InString",
    "OtherActor": "OtherActor",
}

# Pin type normalization
PIN_TYPE_MAP = {
    "exec": "exec",
    "bool": "Bool",
    "int": "Int",
    "int64": "Int",
    "real": "Float",
    "float": "Float",
    "double": "Float",
    "string": "String",
    "text": "String",
    "name": "String",
    "struct": "_STRUCT_",  # Resolved by SubCategory
    "object": "Object",
    "class": "Class",
    "delegate": None,  # Skip delegates
    "byte": "Int",
}


# ============================================================
# TRANSLATION ENGINE
# ============================================================

def translate_node_type(node: dict) -> str | None:
    """Translate a UE5 node into a clean DSL node type."""
    class_name = node["class"].split(".")[-1]

    # Direct class match
    if class_name in NODE_CLASS_MAP:
        dsl_type = NODE_CLASS_MAP[class_name]

        # Special: Events — resolve from EventReference
        if dsl_type == "_EVENT_":
            event_ref = node["properties"].get("EventReference", "")
            for ue_name, dsl_name in EVENT_MAP.items():
                if ue_name in event_ref:
                    return dsl_name
            return f"Event_Unknown"

        # Special: Function calls — resolve from FunctionReference
        if dsl_type == "_FUNCTION_":
            func_ref = node["properties"].get("FunctionReference", "")
            for ue_name, dsl_name in FUNCTION_MAP.items():
                if ue_name in func_ref:
                    return dsl_name
            # Extract MemberName as fallback
            match = re.search(r'MemberName="([^"]+)"', func_ref)
            if match:
                return match.group(1)
            return "UnknownFunction"

        # Special: Math operators — resolve from function
        if dsl_type == "_MATH_OP_":
            func_ref = node["properties"].get("FunctionReference", "")
            for ue_name, dsl_name in FUNCTION_MAP.items():
                if ue_name in func_ref:
                    return dsl_name
            return "UnknownMath"

        # Special: Casts — resolve target class
        if dsl_type == "_CAST_":
            target = node["properties"].get("TargetType", "")
            if "Character" in target:
                return "CastToCharacter"
            elif "PlayerController" in target:
                return "CastToPlayerController"
            elif "Pawn" in target:
                return "CastToPawn"
            return "CastToObject"

        return dsl_type

    return None  # Unknown node type


def translate_pin_type(pin: dict) -> str | None:
    """Translate a UE5 pin type to DSL type."""
    raw_type = pin.get("PinType", "")
    if raw_type in PIN_TYPE_MAP:
        dsl_type = PIN_TYPE_MAP[raw_type]
        if dsl_type == "_STRUCT_":
            # Resolve struct type from subcategory
            sub = pin.get("PinSubCategory", "")
            if "Rotator" in sub:
                return "Rotator"
            elif "Vector" in sub:
                return "Vector"
            elif "Transform" in sub:
                return "Transform"
            elif "LinearColor" in sub or "Color" in sub:
                return "Color"
            return "Struct"
        return dsl_type
    return raw_type or None


def extract_node_properties(node: dict, dsl_type: str) -> dict:
    """Extract meaningful properties (default values) from pins."""
    props = {}

    for pin in node.get("pins", []):
        direction = pin.get("Direction", "")
        name = pin.get("PinName", "")
        default = pin.get("DefaultValue", "")

        # Only input pins with non-empty, non-default values
        if direction == "EGPD_Input" and default and name not in ("execute", "self", "WorldContextObject"):
            # Skip pins that have autogenerated defaults matching the actual default
            auto_default = pin.get("AutogeneratedDefaultValue", None)

            # Skip hidden pins
            if "bHidden=True" in str(pin):
                continue

            # Normalize pin name
            clean_name = PIN_NAME_MAP.get(name, name)
            if clean_name is None:
                continue

            # Check if the value differs from the auto default
            if auto_default is not None and default == auto_default:
                continue  # This is just the default, not user-set

            # Clean up the value
            if default.lower() in ("true", "false"):
                props[clean_name] = default.lower()
            else:
                props[clean_name] = default

    return props


def build_clean_dsl(nodes: list[dict], blueprint_name: str = "BP_Translated") -> str:
    """Build clean DSL from parsed UE5 nodes."""
    lines = []
    lines.append(f"BLUEPRINT: {blueprint_name}")
    lines.append("PARENT: Actor")
    lines.append("")
    lines.append("GRAPH: EventGraph")
    lines.append("")

    # Phase 1: Translate nodes
    node_map = {}  # UE5 node name -> (dsl_id, dsl_type)
    pin_index = {}  # PinId -> (dsl_node_id, pin_name, pin_type)
    dsl_id_counter = 1

    for node in nodes:
        dsl_type = translate_node_type(node)
        if dsl_type is None:
            continue  # Skip untranslatable nodes

        dsl_id = f"n{dsl_id_counter}"
        dsl_id_counter += 1

        # Extract properties
        props = extract_node_properties(node, dsl_type)
        prop_str = ""
        if props:
            parts = []
            for k, v in props.items():
                if " " in str(v) or "," in str(v):
                    parts.append(f'{k}="{v}"')
                else:
                    parts.append(f"{k}={v}")
            prop_str = f" [{', '.join(parts)}]"

        lines.append(f"NODE {dsl_id}: {dsl_type}{prop_str}")
        node_map[node["name"]] = (dsl_id, dsl_type)

        # Index all pins for this node
        for pin in node.get("pins", []):
            if "PinId" in pin:
                pin_name = PIN_NAME_MAP.get(pin.get("PinName", ""), pin.get("PinName", ""))
                if pin_name is not None:
                    pin_type = translate_pin_type(pin)
                    pin_index[pin["PinId"]] = (dsl_id, pin_name, pin_type)

    lines.append("")

    # Phase 2: Translate connections
    exec_lines = []
    data_lines = []

    for node in nodes:
        ue_name = node["name"]
        if ue_name not in node_map:
            continue

        from_dsl_id, _ = node_map[ue_name]

        for pin in node.get("pins", []):
            direction = pin.get("Direction", "")
            if direction != "EGPD_Output":
                continue
            if not pin.get("LinkedTo"):
                continue

            from_pin_name = PIN_NAME_MAP.get(pin.get("PinName", ""), pin.get("PinName", ""))
            if from_pin_name is None:
                continue

            from_pin_type = translate_pin_type(pin)
            is_exec = (from_pin_type == "exec")

            for link in pin["LinkedTo"]:
                # Parse the LinkedTo reference to find the target
                # Format is typically: "PinId NodeName PinName"
                parts = link.strip().split(" ")
                target_pin_id = parts[0] if parts else ""
                target_node_name = parts[1] if len(parts) > 1 else ""
                target_pin_name = parts[2] if len(parts) > 2 else ""

                # Resolve target
                to_dsl_id = None
                to_pin_name = None

                if target_node_name in node_map:
                    to_dsl_id, _ = node_map[target_node_name]
                    to_pin_name = PIN_NAME_MAP.get(target_pin_name, target_pin_name)

                # Fallback: try resolving by pin ID
                if to_dsl_id is None and target_pin_id in pin_index:
                    to_dsl_id, to_pin_name, _ = pin_index[target_pin_id]

                if to_dsl_id and to_pin_name:
                    if is_exec:
                        exec_lines.append(f"EXEC {from_dsl_id}.{from_pin_name} -> {to_dsl_id}.{to_pin_name}")
                    else:
                        type_tag = f" [{from_pin_type}]" if from_pin_type else ""
                        data_lines.append(f"DATA {from_dsl_id}.{from_pin_name} -> {to_dsl_id}.{to_pin_name}{type_tag}")

    if exec_lines:
        for line in exec_lines:
            lines.append(line)
    if data_lines:
        if exec_lines:
            lines.append("")
        for line in data_lines:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# AUTO DESCRIPTION GENERATOR
# ============================================================

def generate_descriptions(dsl_text: str, blueprint_name: str) -> list[str]:
    """Generate basic natural language descriptions from DSL structure."""
    descriptions = []

    # Parse to understand structure
    try:
        bp = parse_dsl(dsl_text)
    except DSLParseError:
        return [f"Create a Blueprint called {blueprint_name}."]

    # Detect patterns
    all_nodes = bp.get_all_nodes()
    node_types = [n.node_type for n in all_nodes]
    all_conns = bp.get_all_connections()

    # Identify the trigger
    trigger = "an event"
    if "Event_BeginPlay" in node_types:
        trigger = "when the game starts"
    elif "Event_Tick" in node_types:
        trigger = "every frame"
    elif "Event_ActorBeginOverlap" in node_types:
        trigger = "when a player overlaps with it"
    elif "Event_InputAction" in node_types:
        for n in all_nodes:
            if n.node_type == "Event_InputAction":
                action = n.properties.get("ActionName", "a key")
                trigger = f"when the player presses {action}"

    # Identify the main actions
    actions = []
    for n in all_nodes:
        if n.node_type == "PrintString":
            msg = n.properties.get("InString", "a message")
            actions.append(f"prints '{msg}'")
        elif n.node_type == "DestroyActor":
            actions.append("destroys itself")
        elif n.node_type == "AddActorLocalRotation":
            actions.append("rotates")
        elif n.node_type == "SetVisibility":
            actions.append("toggles visibility")
        elif n.node_type == "PlaySound":
            actions.append("plays a sound")

    if not actions:
        actions = ["performs an action"]

    action_str = " and ".join(actions)

    # Build descriptions
    descriptions.append(f"Create a Blueprint called {blueprint_name} that {action_str} {trigger}.")
    descriptions.append(f"Make an Actor Blueprint that {action_str} {trigger}.")

    # Technical description
    node_list = ", ".join(set(node_types))
    descriptions.append(f"Create a {blueprint_name} Blueprint using the following nodes: {node_list}.")

    return descriptions


# ============================================================
# MAIN
# ============================================================

def parse_clipboard_file(filepath: Path) -> list[dict]:
    """Import the parser from 01_analyze and parse a clipboard file."""
    # Import from the analyzer
    from importlib import import_module
    spec = import_module("importlib").util.spec_from_file_location(
        "analyzer", Path(__file__).parent / "01_analyze_blueprint_clipboard.py"
    )
    analyzer = import_module("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    text = filepath.read_text(encoding="utf-8")
    return analyzer.parse_clipboard_export(text)


def process_export(filepath: Path, generate_training: bool = False,
                   output_dsl_dir: Path = None, output_jsonl: Path = None) -> str:
    """Process a single clipboard export file."""
    print(f"Translating: {filepath.name}")

    # Parse
    text = filepath.read_text(encoding="utf-8")

    # We need the parser from 01_analyze — inline a simplified version
    nodes = []
    current_node = None
    indent_stack = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Begin Object"):
            match = re.match(r'Begin Object Class=([^\s]+)\s+Name="([^"]+)"', stripped)
            if match:
                current_node = {"class": match.group(1), "name": match.group(2),
                                "properties": {}, "pins": []}
                indent_stack.append(current_node)
        elif stripped == "End Object" and indent_stack:
            finished = indent_stack.pop()
            if not indent_stack:
                nodes.append(finished)
            current_node = indent_stack[-1] if indent_stack else None
        elif current_node and stripped:
            if stripped.startswith("CustomProperties Pin"):
                pin = {}
                for key, pattern in {
                    "PinId": r'PinId=([A-F0-9]+)', "PinName": r'PinName="([^"]*)"',
                    "PinType": r'PinType\.PinCategory="([^"]*)"',
                    "PinSubCategory": r'PinType\.PinSubCategoryObject=([^\s,]+)',
                    "Direction": r'Direction="([^"]*)"', "DefaultValue": r'DefaultValue="([^"]*)"',
                    "LinkedTo": r'LinkedTo=\(([^)]*)\)', "AutogeneratedDefaultValue": r'AutogeneratedDefaultValue="([^"]*)"',
                }.items():
                    m = re.search(pattern, stripped)
                    if m:
                        pin[key] = m.group(1)
                if "LinkedTo" in pin and pin["LinkedTo"]:
                    pin["LinkedTo"] = [l.strip() for l in pin["LinkedTo"].split(",") if l.strip()]
                else:
                    pin["LinkedTo"] = []
                if pin:
                    current_node["pins"].append(pin)
            elif "=" in stripped and not stripped.startswith("#"):
                key, _, value = stripped.partition("=")
                current_node["properties"][key.strip()] = value.strip()

    # Derive blueprint name from filename
    bp_name = "BP_" + filepath.stem.replace(" ", "").replace("-", "_").title()

    # Translate
    dsl_text = build_clean_dsl(nodes, bp_name)
    print(f"  Translated {len(nodes)} UE5 nodes into clean DSL")

    # Validate
    try:
        parse_dsl(dsl_text)
        print(f"  DSL validates successfully")
    except DSLParseError as e:
        print(f"  WARNING: DSL has validation issues: {e}")
        print(f"  (Manual review recommended)")

    # Save DSL
    if output_dsl_dir:
        dsl_path = output_dsl_dir / f"{filepath.stem}_auto.dsl"
        dsl_path.write_text(dsl_text, encoding="utf-8")
        print(f"  Saved: {dsl_path}")

    # Generate training entries
    if generate_training and output_jsonl:
        descriptions = generate_descriptions(dsl_text, bp_name)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "a", encoding="utf-8") as f:
            for desc in descriptions:
                entry = {"instruction": desc, "input": "", "output": dsl_text.strip()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  Added {len(descriptions)} training entries to {output_jsonl}")

    return dsl_text


def main():
    parser = argparse.ArgumentParser(description="Auto-translate UE5 clipboard exports to clean DSL")
    parser.add_argument("file", nargs="?", help="Clipboard export .txt file to translate")
    parser.add_argument("--training", action="store_true", help="Also generate training entries")
    parser.add_argument("--batch", action="store_true", help="Translate all un-translated .txt files in inbox")
    parser.add_argument("--show-map", action="store_true", help="Show the translation map")
    parser.add_argument("--dsl-dir", type=str, default="cleaned-data/parsed-blueprints",
                        help="Directory to save translated DSL files")
    parser.add_argument("--jsonl", type=str, default="datasets/auto_translated.jsonl",
                        help="JSONL file for auto-generated training entries")
    args = parser.parse_args()

    if args.show_map:
        print("NODE CLASS TRANSLATIONS:")
        for ue, dsl in sorted(NODE_CLASS_MAP.items()):
            print(f"  {ue:<50} -> {dsl}")
        print(f"\nFUNCTION TRANSLATIONS ({len(FUNCTION_MAP)}):")
        for ue, dsl in sorted(FUNCTION_MAP.items()):
            print(f"  {ue:<40} -> {dsl}")
        print(f"\nEVENT TRANSLATIONS:")
        for ue, dsl in sorted(EVENT_MAP.items()):
            print(f"  {ue:<40} -> {dsl}")
        print(f"\nTotal: {len(NODE_CLASS_MAP)} class rules, {len(FUNCTION_MAP)} function rules, {len(EVENT_MAP)} event rules")
        return

    dsl_dir = Path(args.dsl_dir)
    jsonl_path = Path(args.jsonl)

    if args.batch:
        inbox = Path("raw-data/clipboard-exports")
        if not inbox.exists():
            print(f"Inbox not found: {inbox}")
            sys.exit(1)
        files = [f for f in inbox.glob("*.txt") if not f.name.startswith("processed_")]
        if not files:
            print("No .txt files in inbox.")
            return
        print(f"Batch translating {len(files)} files...\n")
        for f in files:
            process_export(f, args.training, dsl_dir, jsonl_path)
            print()
        print(f"Done. Translated {len(files)} exports.")

    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        dsl = process_export(filepath, args.training, dsl_dir, jsonl_path)
        print(f"\n--- Generated DSL ---")
        print(dsl)
        print(f"--- End ---")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
