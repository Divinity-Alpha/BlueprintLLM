"""
BlueprintLLM DSL Parser
Parses v6 model output into structured JSON IR for UE5.7 plugin.
"""

import re
import json
from node_map import resolve


def clean_dsl(raw: str) -> list:
    """Clean raw model output, return list of stripped lines."""
    lines = []
    for line in raw.split("\n"):
        s = line.rstrip().rstrip("}").strip()
        if s.startswith("Create a Blueprint"): break
        if s.startswith("END OUTPUT"): break
        s = re.sub(r'\(stypy.*$', '', s).strip()
        if s and s not in ("{", "}"):
            lines.append(s)
    return lines


def parse_params(param_str: str) -> dict:
    """Parse [Key=Value, Key2=Value2] parameter strings."""
    if not param_str:
        return {}
    params = {}
    key, val, in_key, depth = "", "", True, 0
    for c in param_str:
        if c == '(': depth += 1; val += c
        elif c == ')': depth -= 1; val += c
        elif c == '=' and in_key and depth == 0: in_key = False
        elif c == ',' and depth == 0:
            if key.strip(): params[key.strip()] = val.strip().strip('"')
            key, val, in_key = "", "", True
        elif in_key: key += c
        else: val += c
    if key.strip():
        params[key.strip()] = val.strip().strip('"')
    return params


def parse(raw: str) -> dict:
    """
    Parse DSL text into IR dict.
    Returns {"blueprint": {...}, "errors": [...], "warnings": [...]}.
    """
    lines = clean_dsl(raw)
    bp = {"name": "", "parent": "Actor", "category": None}
    variables = []
    nodes = []
    connections = []
    errors = []
    warnings = []
    node_ids = set()

    for line in lines:
        # BLUEPRINT: name
        if line.startswith("BLUEPRINT:"):
            bp["name"] = line.split(":", 1)[1].strip()

        # PARENT: class
        elif line.startswith("PARENT:"):
            bp["parent"] = line.split(":", 1)[1].strip()

        # CATEGORY: name
        elif line.startswith("CATEGORY:"):
            bp["category"] = line.split(":", 1)[1].strip()

        # GRAPH: name (skip, we only use EventGraph)
        elif line.startswith("GRAPH:"):
            pass

        # VAR Name: Type = Default
        elif line.startswith("VAR "):
            m = re.match(r'VAR\s+(\w+)\s*:\s*(\w+)\s*=\s*(.+)', line)
            if m:
                variables.append({"name": m.group(1), "type": m.group(2), "default": m.group(3).strip()})
            else:
                warnings.append(f"Bad VAR: {line}")

        # NODE n1: NodeType [params]
        elif line.startswith("NODE "):
            m = re.match(r'NODE\s+(n\d+)\s*:\s*(\S+)\s*(?:\[(.+?)\])?\s*(?:\{.*)?', line)
            if m:
                nid, ntype, pstr = m.group(1), m.group(2), m.group(3) or ""
                canonical, mapping = resolve(ntype)
                node = {"id": nid, "dsl_type": canonical, "params": parse_params(pstr)}
                if mapping:
                    node["ue_class"] = mapping.get("ue_class", "")
                    for k in ("ue_function", "ue_event", "cast_class", "param_key"):
                        if k in mapping:
                            node[k] = mapping[k]
                else:
                    node["ue_class"] = "UNMAPPED"
                    warnings.append(f"Unmapped: {ntype} ({nid})")
                nodes.append(node)
                node_ids.add(nid)
            else:
                errors.append(f"Bad NODE: {line}")

        # EXEC n1.Pin -> n2.Pin
        elif line.startswith("EXEC "):
            m = re.match(r'EXEC\s+(n\d+)\.(\S+)\s*->\s*(n\d+)\.(\S+)', line)
            if m:
                connections.append({"type": "exec",
                    "src_node": m.group(1), "src_pin": m.group(2),
                    "dst_node": m.group(3), "dst_pin": m.group(4)})
            else:
                warnings.append(f"Bad EXEC: {line}")

        # DATA n1.Pin -> n2.Pin [Type]
        elif line.startswith("DATA "):
            m = re.match(r'DATA\s+(n\d+)\.(\S+)\s*->\s*(n\d+)\.(\S+?)(?:\s*\[(\w+)\])?\s*', line)
            if m:
                connections.append({"type": "data",
                    "src_node": m.group(1), "src_pin": m.group(2),
                    "dst_node": m.group(3), "dst_pin": m.group(4),
                    "data_type": m.group(5)})
            else:
                # Try literal: DATA value -> n2.Pin [Type]
                m2 = re.match(r'DATA\s+(\S+)\s*->\s*(n\d+)\.(\S+?)(?:\s*\[(\w+)\])?\s*', line)
                if m2:
                    connections.append({"type": "data_literal",
                        "value": m2.group(1),
                        "dst_node": m2.group(2), "dst_pin": m2.group(3),
                        "data_type": m2.group(4)})
                else:
                    warnings.append(f"Bad DATA: {line}")

        elif line and not line.startswith("//"):
            warnings.append(f"Unknown: {line[:80]}")

    # Validate references
    for c in connections:
        if c["type"] != "data_literal":
            if c.get("src_node") not in node_ids:
                errors.append(f"Unknown src node: {c.get('src_node')}")
        if c.get("dst_node") not in node_ids:
            errors.append(f"Unknown dst node: {c.get('dst_node')}")

    # Check for events
    if not any(n["dsl_type"].startswith("Event_") for n in nodes):
        warnings.append("No event nodes found")

    # Auto-layout
    depth = {n["id"]: 0 for n in nodes}
    for _ in range(50):
        changed = False
        for c in connections:
            if c["type"] == "exec":
                s, d = c.get("src_node"), c.get("dst_node")
                if s in depth and d in depth:
                    nd = depth[s] + 1
                    if nd > depth[d]:
                        depth[d] = nd
                        changed = True
        if not changed: break

    y_at = {}
    for n in nodes:
        d = depth.get(n["id"], 0)
        if d not in y_at: y_at[d] = 0
        n["position"] = [d * 300, y_at[d] * 150]
        y_at[d] += 1

    return {
        "ir": {
            "metadata": {"name": bp["name"], "parent_class": bp["parent"], "category": bp.get("category")},
            "variables": variables,
            "nodes": nodes,
            "connections": connections,
        },
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "nodes": len(nodes),
            "connections": len(connections),
            "variables": len(variables),
            "mapped": sum(1 for n in nodes if n.get("ue_class") != "UNMAPPED"),
            "unmapped": sum(1 for n in nodes if n.get("ue_class") == "UNMAPPED"),
            "has_events": any(n["dsl_type"].startswith("Event_") for n in nodes),
        }
    }


def save_ir(result: dict, path: str):
    """Save IR to .blueprint.json file."""
    with open(path, "w") as f:
        json.dump(result["ir"], f, indent=2)
