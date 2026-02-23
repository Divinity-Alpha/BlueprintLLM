"""
utils/blueprint_patterns.py
----------------------------
Common Blueprint subgraph patterns used for synthetic data generation
and validation. Each pattern defines a reusable graph fragment with
typed pins and expected connections.

This is your expanding library of "Blueprint knowledge" — the more
patterns you add here, the richer your training data becomes.
"""

from dataclasses import dataclass, field


@dataclass
class PinDef:
    name: str
    direction: str  # "input" or "output"
    pin_type: str   # "exec", "Bool", "Float", "Int", "String", "Vector", "Rotator", "Actor", "Object"


@dataclass
class NodePattern:
    """Defines a node type and its expected pins."""
    type_name: str
    category: str
    description: str
    pins: list = field(default_factory=list)


# ============================================================
# NODE TYPE CATALOG
# ============================================================
# This is your "vocabulary" — every node type the LLM should know about.
# Expand this as you discover more nodes from UE5.

NODE_CATALOG = {
    # === EVENTS ===
    "Event_BeginPlay": NodePattern(
        type_name="Event_BeginPlay",
        category="Events",
        description="Fires once when the game starts or the actor is spawned",
        pins=[
            PinDef("Then", "output", "exec"),
        ],
    ),
    "Event_Tick": NodePattern(
        type_name="Event_Tick",
        category="Events",
        description="Fires every frame",
        pins=[
            PinDef("Then", "output", "exec"),
            PinDef("DeltaSeconds", "output", "Float"),
        ],
    ),
    "Event_ActorBeginOverlap": NodePattern(
        type_name="Event_ActorBeginOverlap",
        category="Events",
        description="Fires when another actor overlaps this actor",
        pins=[
            PinDef("Then", "output", "exec"),
            PinDef("OtherActor", "output", "Actor"),
        ],
    ),
    "Event_ActorEndOverlap": NodePattern(
        type_name="Event_ActorEndOverlap",
        category="Events",
        description="Fires when an overlapping actor stops overlapping",
        pins=[
            PinDef("Then", "output", "exec"),
            PinDef("OtherActor", "output", "Actor"),
        ],
    ),
    "Event_InputAction": NodePattern(
        type_name="Event_InputAction",
        category="Events",
        description="Fires when a mapped input action occurs",
        pins=[
            PinDef("Pressed", "output", "exec"),
            PinDef("Released", "output", "exec"),
        ],
    ),
    "Event_CustomEvent": NodePattern(
        type_name="Event_CustomEvent",
        category="Events",
        description="A custom event that can be called by name",
        pins=[
            PinDef("Then", "output", "exec"),
        ],
    ),

    # === FLOW CONTROL ===
    "Branch": NodePattern(
        type_name="Branch",
        category="FlowControl",
        description="If/else conditional branch",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Condition", "input", "Bool"),
            PinDef("True", "output", "exec"),
            PinDef("False", "output", "exec"),
        ],
    ),
    "Sequence": NodePattern(
        type_name="Sequence",
        category="FlowControl",
        description="Executes multiple output pins in order",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Then_0", "output", "exec"),
            PinDef("Then_1", "output", "exec"),
            PinDef("Then_2", "output", "exec"),
        ],
    ),
    "FlipFlop": NodePattern(
        type_name="FlipFlop",
        category="FlowControl",
        description="Alternates between two execution paths",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("A", "output", "exec"),
            PinDef("B", "output", "exec"),
            PinDef("IsA", "output", "Bool"),
        ],
    ),
    "DoOnce": NodePattern(
        type_name="DoOnce",
        category="FlowControl",
        description="Executes only the first time, blocks subsequent calls",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Reset", "input", "exec"),
            PinDef("Completed", "output", "exec"),
        ],
    ),
    "Delay": NodePattern(
        type_name="Delay",
        category="FlowControl",
        description="Pauses execution for a specified duration",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Duration", "input", "Float"),
            PinDef("Completed", "output", "exec"),
        ],
    ),
    "ForEachLoop": NodePattern(
        type_name="ForEachLoop",
        category="FlowControl",
        description="Iterates over each element in an array",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Array", "input", "Array"),
            PinDef("LoopBody", "output", "exec"),
            PinDef("ArrayElement", "output", "Object"),
            PinDef("ArrayIndex", "output", "Int"),
            PinDef("Completed", "output", "exec"),
        ],
    ),
    "Gate": NodePattern(
        type_name="Gate",
        category="FlowControl",
        description="Passes or blocks execution flow based on open/close state",
        pins=[
            PinDef("Enter", "input", "exec"),
            PinDef("Open", "input", "exec"),
            PinDef("Close", "input", "exec"),
            PinDef("Toggle", "input", "exec"),
            PinDef("Exit", "output", "exec"),
        ],
    ),

    # === MATH ===
    "AddFloat": NodePattern(
        type_name="AddFloat",
        category="Math",
        description="Adds two float values",
        pins=[
            PinDef("A", "input", "Float"),
            PinDef("B", "input", "Float"),
            PinDef("ReturnValue", "output", "Float"),
        ],
    ),
    "SubtractFloat": NodePattern(
        type_name="SubtractFloat",
        category="Math",
        description="Subtracts B from A",
        pins=[
            PinDef("A", "input", "Float"),
            PinDef("B", "input", "Float"),
            PinDef("ReturnValue", "output", "Float"),
        ],
    ),
    "MultiplyFloat": NodePattern(
        type_name="MultiplyFloat",
        category="Math",
        description="Multiplies two float values",
        pins=[
            PinDef("A", "input", "Float"),
            PinDef("B", "input", "Float"),
            PinDef("ReturnValue", "output", "Float"),
        ],
    ),
    "GreaterThan": NodePattern(
        type_name="GreaterThan",
        category="Math",
        description="Returns true if A > B",
        pins=[
            PinDef("A", "input", "Float"),
            PinDef("B", "input", "Float"),
            PinDef("ReturnValue", "output", "Bool"),
        ],
    ),
    "LessThan": NodePattern(
        type_name="LessThan",
        category="Math",
        description="Returns true if A < B",
        pins=[
            PinDef("A", "input", "Float"),
            PinDef("B", "input", "Float"),
            PinDef("ReturnValue", "output", "Bool"),
        ],
    ),
    "ClampFloat": NodePattern(
        type_name="ClampFloat",
        category="Math",
        description="Clamps a value between min and max",
        pins=[
            PinDef("Value", "input", "Float"),
            PinDef("Min", "input", "Float"),
            PinDef("Max", "input", "Float"),
            PinDef("ReturnValue", "output", "Float"),
        ],
    ),
    "MakeRotator": NodePattern(
        type_name="MakeRotator",
        category="Math",
        description="Creates a Rotator from Roll, Pitch, Yaw values",
        pins=[
            PinDef("Roll", "input", "Float"),
            PinDef("Pitch", "input", "Float"),
            PinDef("Yaw", "input", "Float"),
            PinDef("ReturnValue", "output", "Rotator"),
        ],
    ),
    "MakeVector": NodePattern(
        type_name="MakeVector",
        category="Math",
        description="Creates a Vector from X, Y, Z values",
        pins=[
            PinDef("X", "input", "Float"),
            PinDef("Y", "input", "Float"),
            PinDef("Z", "input", "Float"),
            PinDef("ReturnValue", "output", "Vector"),
        ],
    ),

    # === ACTOR FUNCTIONS ===
    "DestroyActor": NodePattern(
        type_name="DestroyActor",
        category="Actor",
        description="Destroys this actor",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Then", "output", "exec"),
        ],
    ),
    "SetActorHiddenInGame": NodePattern(
        type_name="SetActorHiddenInGame",
        category="Actor",
        description="Shows or hides the actor in game",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("bNewHidden", "input", "Bool"),
            PinDef("Then", "output", "exec"),
        ],
    ),
    "AddActorLocalRotation": NodePattern(
        type_name="AddActorLocalRotation",
        category="Actor",
        description="Adds rotation to the actor in local space",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("DeltaRotation", "input", "Rotator"),
            PinDef("Then", "output", "exec"),
        ],
    ),
    "GetDistanceTo": NodePattern(
        type_name="GetDistanceTo",
        category="Actor",
        description="Gets the distance between this actor and another",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("OtherActor", "input", "Actor"),
            PinDef("Then", "output", "exec"),
            PinDef("ReturnValue", "output", "Float"),
        ],
    ),

    # === COMPONENTS ===
    "SetVisibility": NodePattern(
        type_name="SetVisibility",
        category="Components",
        description="Sets visibility of a component",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Target", "input", "Object"),
            PinDef("NewVisibility", "input", "Bool"),
            PinDef("Then", "output", "exec"),
        ],
    ),
    "SetRelativeRotation": NodePattern(
        type_name="SetRelativeRotation",
        category="Components",
        description="Sets the relative rotation of a component",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Target", "input", "Object"),
            PinDef("NewRotation", "input", "Rotator"),
            PinDef("Then", "output", "exec"),
        ],
    ),

    # === UTILITY ===
    "PrintString": NodePattern(
        type_name="PrintString",
        category="Utility",
        description="Prints a string to the screen and/or log",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("InString", "input", "String"),
            PinDef("Then", "output", "exec"),
        ],
    ),
    "SetTimerByFunctionName": NodePattern(
        type_name="SetTimerByFunctionName",
        category="Utility",
        description="Sets a timer that calls a function by name",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("FunctionName", "input", "String"),
            PinDef("Time", "input", "Float"),
            PinDef("bLooping", "input", "Bool"),
            PinDef("Then", "output", "exec"),
        ],
    ),
    "GetWorldDeltaSeconds": NodePattern(
        type_name="GetWorldDeltaSeconds",
        category="Utility",
        description="Returns the time elapsed since the last frame",
        pins=[
            PinDef("ReturnValue", "output", "Float"),
        ],
    ),

    # === VARIABLES ===
    "GetVar": NodePattern(
        type_name="GetVar",
        category="Variables",
        description="Gets the value of a Blueprint variable",
        pins=[
            PinDef("Value", "output", "Object"),  # Type depends on variable
        ],
    ),
    "SetVar": NodePattern(
        type_name="SetVar",
        category="Variables",
        description="Sets the value of a Blueprint variable",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Value", "input", "Object"),  # Type depends on variable
            PinDef("Then", "output", "exec"),
        ],
    ),

    # === CASTING ===
    "CastToCharacter": NodePattern(
        type_name="CastToCharacter",
        category="Casting",
        description="Attempts to cast an object to the Character class",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Object", "input", "Object"),
            PinDef("CastSucceeded", "output", "exec"),
            PinDef("CastFailed", "output", "exec"),
            PinDef("AsCharacter", "output", "Object"),
        ],
    ),

    # === AUDIO ===
    "PlaySound": NodePattern(
        type_name="PlaySound",
        category="Audio",
        description="Plays a sound effect",
        pins=[
            PinDef("Execute", "input", "exec"),
            PinDef("Sound", "input", "Object"),
            PinDef("Then", "output", "exec"),
        ],
    ),
}


def get_node_pattern(type_name: str) -> NodePattern | None:
    """Look up a node pattern by type name."""
    return NODE_CATALOG.get(type_name)


def get_nodes_by_category(category: str) -> list[NodePattern]:
    """Get all node patterns in a given category."""
    return [n for n in NODE_CATALOG.values() if n.category == category]


def get_all_categories() -> list[str]:
    """Get all unique node categories."""
    return sorted(set(n.category for n in NODE_CATALOG.values()))


def validate_connection(from_type: str, from_pin: str, to_type: str, to_pin: str) -> tuple[bool, str]:
    """
    Validate that a connection between two pins is valid.
    Returns (is_valid, error_message).
    """
    from_pattern = get_node_pattern(from_type)
    to_pattern = get_node_pattern(to_type)

    if not from_pattern:
        return True, ""  # Unknown type, can't validate
    if not to_pattern:
        return True, ""

    # Find source pin
    source_pin = None
    for pin in from_pattern.pins:
        if pin.name == from_pin and pin.direction == "output":
            source_pin = pin
            break

    # Find target pin
    target_pin = None
    for pin in to_pattern.pins:
        if pin.name == to_pin and pin.direction == "input":
            target_pin = pin
            break

    if source_pin and target_pin:
        # Type compatibility check
        if source_pin.pin_type == "exec" and target_pin.pin_type != "exec":
            return False, f"Cannot connect exec pin to {target_pin.pin_type} pin"
        if source_pin.pin_type != "exec" and target_pin.pin_type == "exec":
            return False, f"Cannot connect {source_pin.pin_type} pin to exec pin"

    return True, ""


# Quick summary when run directly
if __name__ == "__main__":
    print("Blueprint Node Catalog")
    print("=" * 50)
    for category in get_all_categories():
        nodes = get_nodes_by_category(category)
        print(f"\n{category} ({len(nodes)} nodes):")
        for node in nodes:
            pin_summary = ", ".join(
                f"{p.name}({'→' if p.direction == 'output' else '←'}{p.pin_type})"
                for p in node.pins
            )
            print(f"  {node.type_name}: {pin_summary}")

    print(f"\nTotal: {len(NODE_CATALOG)} node types across {len(get_all_categories())} categories")
