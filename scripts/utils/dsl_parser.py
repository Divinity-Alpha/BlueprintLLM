"""
utils/dsl_parser.py
-------------------
Parser and validator for the Blueprint DSL format.
Converts DSL text <-> structured Python objects.
Used by the training pipeline, the validator, and the UE5 compiler plugin.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class PinDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"


class ConnectionType(Enum):
    EXEC = "exec"
    DATA = "data"


@dataclass
class Variable:
    name: str
    type: str
    default_value: str = ""


@dataclass
class NodeDef:
    id: str
    node_type: str
    properties: dict = field(default_factory=dict)


@dataclass
class Connection:
    conn_type: ConnectionType
    from_node: str
    from_pin: str
    to_node: str
    to_pin: str
    data_type: str = ""  # Only for DATA connections


@dataclass
class BlueprintDSL:
    name: str = ""
    parent_class: str = "Actor"
    category: str = ""
    variables: list = field(default_factory=list)
    graphs: dict = field(default_factory=dict)  # graph_name -> {"nodes": [], "connections": []}

    def get_all_nodes(self) -> list[NodeDef]:
        nodes = []
        for graph in self.graphs.values():
            nodes.extend(graph["nodes"])
        return nodes

    def get_all_connections(self) -> list[Connection]:
        conns = []
        for graph in self.graphs.values():
            conns.extend(graph["connections"])
        return conns


class DSLParseError(Exception):
    def __init__(self, message: str, line_number: int = 0, line_text: str = ""):
        self.line_number = line_number
        self.line_text = line_text
        super().__init__(f"Line {line_number}: {message} | '{line_text}'")


def parse_dsl(text: str) -> BlueprintDSL:
    """Parse Blueprint DSL text into a structured BlueprintDSL object."""
    # Strip BOM (PowerShell's Out-File and some editors add this)
    text = text.lstrip("\ufeff")

    bp = BlueprintDSL()
    current_graph = None
    errors = []

    for line_num, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        try:
            if stripped.startswith("BLUEPRINT:"):
                bp.name = stripped.split(":", 1)[1].strip()

            elif stripped.startswith("PARENT:"):
                bp.parent_class = stripped.split(":", 1)[1].strip()

            elif stripped.startswith("CATEGORY:"):
                bp.category = stripped.split(":", 1)[1].strip()

            elif stripped.startswith("VAR "):
                var = _parse_variable(stripped, line_num)
                bp.variables.append(var)

            elif stripped.startswith("GRAPH:"):
                graph_name = stripped.split(":", 1)[1].strip()
                current_graph = graph_name
                bp.graphs[graph_name] = {"nodes": [], "connections": []}

            elif stripped.startswith("NODE "):
                if current_graph is None:
                    raise DSLParseError("NODE defined before GRAPH declaration", line_num, stripped)
                node = _parse_node(stripped, line_num)
                bp.graphs[current_graph]["nodes"].append(node)

            elif stripped.startswith("EXEC "):
                if current_graph is None:
                    raise DSLParseError("EXEC defined before GRAPH declaration", line_num, stripped)
                conn = _parse_connection(stripped, ConnectionType.EXEC, line_num)
                bp.graphs[current_graph]["connections"].append(conn)

            elif stripped.startswith("DATA "):
                if current_graph is None:
                    raise DSLParseError("DATA defined before GRAPH declaration", line_num, stripped)
                conn = _parse_connection(stripped, ConnectionType.DATA, line_num)
                bp.graphs[current_graph]["connections"].append(conn)

            else:
                errors.append(DSLParseError(f"Unknown directive: {stripped[:20]}...", line_num, stripped))

        except DSLParseError as e:
            errors.append(e)

    if errors:
        error_summary = "\n".join(str(e) for e in errors)
        raise DSLParseError(f"Found {len(errors)} parse errors:\n{error_summary}")

    return bp


def _parse_variable(line: str, line_num: int) -> Variable:
    """Parse: VAR IsOpen: Bool = false"""
    match = re.match(r'VAR\s+(\w+)\s*:\s*(\w+)\s*(?:=\s*(.+))?', line)
    if not match:
        raise DSLParseError(f"Invalid VAR syntax", line_num, line)
    return Variable(
        name=match.group(1),
        type=match.group(2),
        default_value=match.group(3).strip() if match.group(3) else "",
    )


def _parse_node(line: str, line_num: int) -> NodeDef:
    """Parse: NODE n1: Event_BeginPlay [Key=Value, Key2=Value2]"""
    match = re.match(r'NODE\s+(\w+)\s*:\s*(\w+)\s*(?:\[(.+)\])?', line)
    if not match:
        raise DSLParseError(f"Invalid NODE syntax", line_num, line)

    properties = {}
    if match.group(3):
        # Parse key=value pairs, handling quoted values
        prop_text = match.group(3)
        for prop_match in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,\]]+))', prop_text):
            key = prop_match.group(1)
            value = prop_match.group(2) if prop_match.group(2) is not None else prop_match.group(3).strip()
            properties[key] = value

    return NodeDef(
        id=match.group(1),
        node_type=match.group(2),
        properties=properties,
    )


def _parse_connection(line: str, conn_type: ConnectionType, line_num: int) -> Connection:
    """Parse: EXEC n1.Then -> n2.Execute  or  DATA n1.ReturnValue -> n2.A [Float]"""
    prefix = "EXEC" if conn_type == ConnectionType.EXEC else "DATA"
    pattern = rf'{prefix}\s+(\w+)\.(\w+)\s*->\s*(\w+)\.(\w+)\s*(?:\[(\w+)\])?'
    match = re.match(pattern, line)
    if not match:
        raise DSLParseError(f"Invalid {prefix} syntax", line_num, line)

    return Connection(
        conn_type=conn_type,
        from_node=match.group(1),
        from_pin=match.group(2),
        to_node=match.group(3),
        to_pin=match.group(4),
        data_type=match.group(5) or "",
    )


def to_dsl_text(bp: BlueprintDSL) -> str:
    """Convert a BlueprintDSL object back to DSL text."""
    lines = []

    lines.append(f"BLUEPRINT: {bp.name}")
    lines.append(f"PARENT: {bp.parent_class}")
    if bp.category:
        lines.append(f"CATEGORY: {bp.category}")
    lines.append("")

    # Variables
    if bp.variables:
        lines.append("# --- VARIABLES ---")
        for var in bp.variables:
            default = f" = {var.default_value}" if var.default_value else ""
            lines.append(f"VAR {var.name}: {var.type}{default}")
        lines.append("")

    # Graphs
    for graph_name, graph_data in bp.graphs.items():
        lines.append(f"# --- {graph_name.upper()} ---")
        lines.append(f"GRAPH: {graph_name}")
        lines.append("")

        # Nodes
        for node in graph_data["nodes"]:
            props = ""
            if node.properties:
                prop_parts = []
                for k, v in node.properties.items():
                    if " " in str(v) or "," in str(v):
                        prop_parts.append(f'{k}="{v}"')
                    else:
                        prop_parts.append(f"{k}={v}")
                props = f" [{', '.join(prop_parts)}]"
            lines.append(f"NODE {node.id}: {node.node_type}{props}")

        lines.append("")

        # Connections — exec first, then data
        exec_conns = [c for c in graph_data["connections"] if c.conn_type == ConnectionType.EXEC]
        data_conns = [c for c in graph_data["connections"] if c.conn_type == ConnectionType.DATA]

        if exec_conns:
            lines.append("# Execution flow")
            for conn in exec_conns:
                lines.append(f"EXEC {conn.from_node}.{conn.from_pin} -> {conn.to_node}.{conn.to_pin}")

        if data_conns:
            lines.append("# Data flow")
            for conn in data_conns:
                type_tag = f" [{conn.data_type}]" if conn.data_type else ""
                lines.append(
                    f"DATA {conn.from_node}.{conn.from_pin} -> {conn.to_node}.{conn.to_pin}{type_tag}"
                )

        lines.append("")

    return "\n".join(lines)


# --- Quick test ---
if __name__ == "__main__":
    sample = """
BLUEPRINT: BP_TestLight
PARENT: Actor
CATEGORY: Lighting

VAR IsOn: Bool = false
VAR Brightness: Float = 1.0

GRAPH: EventGraph

NODE n1: Event_BeginPlay
NODE n2: PrintString [InString="Light Ready"]
NODE n3: Event_InputAction [ActionName="ToggleLight"]
NODE n4: FlipFlop
NODE n5: SetVisibility [Target=LightComponent, NewVisibility=true]
NODE n6: SetVisibility [Target=LightComponent, NewVisibility=false]

EXEC n1.Then -> n2.Execute
EXEC n3.Pressed -> n4.Execute
EXEC n4.A -> n5.Execute
EXEC n4.B -> n6.Execute

DATA n4.IsA -> n5.NewVisibility [Bool]
"""

    print("Parsing sample DSL...")
    bp = parse_dsl(sample)

    print(f"Blueprint: {bp.name}")
    print(f"Parent: {bp.parent_class}")
    print(f"Variables: {len(bp.variables)}")
    for v in bp.variables:
        print(f"  {v.name}: {v.type} = {v.default_value}")

    for graph_name, graph in bp.graphs.items():
        print(f"\nGraph: {graph_name}")
        print(f"  Nodes: {len(graph['nodes'])}")
        print(f"  Connections: {len(graph['connections'])}")

    print("\n--- Round-trip ---")
    print(to_dsl_text(bp))
