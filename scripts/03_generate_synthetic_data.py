"""
03_generate_synthetic_data.py
-----------------------------
Generates synthetic training data by combining Blueprint patterns
with randomized parameters and natural language descriptions.

Usage:
    python scripts/03_generate_synthetic_data.py --count 1000 --output datasets/train.jsonl

This is your primary volume generator for Phase 3.
"""

import json
import random
import sys
import argparse
from pathlib import Path
from itertools import product as iter_product

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="1")


# ============================================================
# BLUEPRINT PATTERN TEMPLATES
# ============================================================
# Each pattern is a function that returns (description, dsl_text)
# with randomized parameters. Add more patterns to expand coverage.

def pattern_print_on_begin_play():
    messages = [
        "Hello World", "Game Started", "Actor Initialized",
        "Ready", "System Online", "Blueprint Active",
    ]
    msg = random.choice(messages)
    desc = f'Create a Blueprint that prints "{msg}" when the game starts.'
    dsl = f"""BLUEPRINT: BP_Print{msg.replace(' ', '')}
PARENT: Actor

GRAPH: EventGraph

NODE n1: Event_BeginPlay
NODE n2: PrintString [InString="{msg}"]

EXEC n1.Then -> n2.Execute"""
    return desc, dsl


def pattern_toggle_visibility():
    components = ["MeshComponent", "LightComponent", "ParticleComponent", "WidgetComponent"]
    inputs = ["ToggleVis", "Interact", "Activate", "Switch"]
    comp = random.choice(components)
    inp = random.choice(inputs)
    bp_name = f"BP_Toggle{comp.replace('Component', '')}"

    desc = f'Create a Blueprint with a {comp} that toggles visibility when the player presses the "{inp}" input action.'
    dsl = f"""BLUEPRINT: {bp_name}
PARENT: Actor

VAR bIsVisible: Bool = true

GRAPH: EventGraph

NODE n1: Event_InputAction [ActionName="{inp}"]
NODE n2: FlipFlop
NODE n3: SetVisibility [Target={comp}, NewVisibility=true]
NODE n4: SetVisibility [Target={comp}, NewVisibility=false]

EXEC n1.Pressed -> n2.Execute
EXEC n2.A -> n3.Execute
EXEC n2.B -> n4.Execute"""
    return desc, dsl


def pattern_overlap_trigger():
    actions = [
        ("PrintString", 'InString="Player Entered"', "prints a message"),
        ("PlaySound", 'Sound=OverlapSound', "plays a sound"),
        ("SetActorHiddenInGame", 'bNewHidden=true', "hides itself"),
        ("DestroyActor", '', "destroys itself"),
    ]
    action_node, action_props, action_desc = random.choice(actions)
    prop_str = f" [{action_props}]" if action_props else ""

    desc = f"Create a Blueprint that {action_desc} when a player overlaps with it."
    dsl = f"""BLUEPRINT: BP_OverlapTrigger
PARENT: Actor

GRAPH: EventGraph

NODE n1: Event_ActorBeginOverlap
NODE n2: CastToCharacter
NODE n3: {action_node}{prop_str}

EXEC n1.Then -> n2.Execute
EXEC n2.CastSucceeded -> n3.Execute

DATA n1.OtherActor -> n2.Object [Actor]"""
    return desc, dsl


def pattern_timer_loop():
    intervals = [0.5, 1.0, 2.0, 3.0, 5.0]
    actions = [
        ("PrintString", 'InString="Tick"', "prints 'Tick'"),
        ("AddActorLocalRotation", 'DeltaRotation=(Yaw=5.0)', "rotates slightly"),
        ("AddActorLocalOffset", 'DeltaLocation=(Z=1.0)', "moves upward"),
    ]
    interval = random.choice(intervals)
    action_node, action_props, action_desc = random.choice(actions)

    desc = f"Create a Blueprint that {action_desc} every {interval} seconds using a timer."
    dsl = f"""BLUEPRINT: BP_TimerLoop
PARENT: Actor

GRAPH: EventGraph

NODE n1: Event_BeginPlay
NODE n2: SetTimerByFunctionName [FunctionName="OnTimer", Time={interval}, bLooping=true]
NODE n3: Event_CustomEvent [EventName="OnTimer"]
NODE n4: {action_node} [{action_props}]

EXEC n1.Then -> n2.Execute
EXEC n3.Then -> n4.Execute"""
    return desc, dsl


def pattern_branch_condition():
    conditions = [
        ("Health", "Float", "50.0", "health is above 50"),
        ("Score", "Int", "100", "score exceeds 100"),
        ("IsAlive", "Bool", "true", "the actor is alive"),
        ("Ammo", "Int", "0", "ammo is greater than 0"),
    ]
    var_name, var_type, threshold, cond_desc = random.choice(conditions)

    true_actions = [
        ("PrintString", 'InString="Condition Met"', "prints 'Condition Met'"),
        ("SetActorHiddenInGame", 'bNewHidden=false', "shows the actor"),
    ]
    false_actions = [
        ("PrintString", 'InString="Condition Failed"', "prints 'Condition Failed'"),
        ("DestroyActor", '', "destroys the actor"),
    ]

    true_node, true_props, true_desc = random.choice(true_actions)
    false_node, false_props, false_desc = random.choice(false_actions)

    compare_node = "GreaterThan" if var_type in ("Float", "Int") else "BooleanAND"

    desc = (
        f"Create a Blueprint that checks if {cond_desc}. "
        f"If true, it {true_desc}. If false, it {false_desc}."
    )
    dsl = f"""BLUEPRINT: BP_Conditional{var_name}
PARENT: Actor

VAR {var_name}: {var_type} = {threshold}

GRAPH: EventGraph

NODE n1: Event_BeginPlay
NODE n2: GetVar [Variable={var_name}]
NODE n3: {compare_node} [B={threshold}]
NODE n4: Branch
NODE n5: {true_node} [{true_props}]
NODE n6: {false_node} [{false_props}]

EXEC n1.Then -> n4.Execute
EXEC n4.True -> n5.Execute
EXEC n4.False -> n6.Execute

DATA n2.Value -> n3.A [{var_type}]
DATA n3.ReturnValue -> n4.Condition [Bool]"""
    return desc, dsl


def pattern_key_press_action():
    keys = ["E", "F", "Q", "R", "T", "SpaceBar", "LeftShift"]
    actions = [
        ("Jump", "makes the character jump"),
        ("Sprint", "toggles sprinting"),
        ("Interact", "interacts with nearby objects"),
        ("Reload", "reloads the weapon"),
        ("UseAbility", "activates a special ability"),
    ]
    key = random.choice(keys)
    action_name, action_desc = random.choice(actions)

    desc = f'Create a Blueprint that {action_desc} when the player presses the {key} key.'
    dsl = f"""BLUEPRINT: BP_{action_name}
PARENT: Character

GRAPH: EventGraph

NODE n1: Event_InputAction [ActionName="{action_name}"]
NODE n2: PrintString [InString="{action_name} activated"]
NODE n3: CallFunction [FunctionName="Execute{action_name}"]

EXEC n1.Pressed -> n2.Execute
EXEC n2.Then -> n3.Execute"""
    return desc, dsl


def pattern_health_system():
    max_health_options = [100, 150, 200, 250]
    max_health = random.choice(max_health_options)
    damage_options = [10, 15, 20, 25, 50]
    damage = random.choice(damage_options)

    desc = (
        f"Create a health system Blueprint with {max_health} max health. "
        f"It should take {damage} damage when hit and destroy the actor when health reaches zero."
    )
    dsl = f"""BLUEPRINT: BP_HealthSystem
PARENT: Actor
CATEGORY: Gameplay

VAR CurrentHealth: Float = {max_health}
VAR MaxHealth: Float = {max_health}

GRAPH: EventGraph

NODE n1: Event_CustomEvent [EventName="TakeDamage"]
NODE n2: GetVar [Variable=CurrentHealth]
NODE n3: SubtractFloat [B={damage}]
NODE n4: ClampFloat [Min=0.0, Max={max_health}]
NODE n5: SetVar [Variable=CurrentHealth]
NODE n6: LessEqualFloat [B=0.0]
NODE n7: Branch
NODE n8: DestroyActor
NODE n9: PrintString [InString="Damage Taken"]

EXEC n1.Then -> n3.Execute
EXEC n3.Then -> n5.Execute
EXEC n5.Then -> n7.Execute
EXEC n7.True -> n8.Execute
EXEC n7.False -> n9.Execute

DATA n2.Value -> n3.A [Float]
DATA n3.ReturnValue -> n4.Value [Float]
DATA n4.ReturnValue -> n5.Value [Float]
DATA n5.Value -> n6.A [Float]
DATA n6.ReturnValue -> n7.Condition [Bool]"""
    return desc, dsl


def pattern_rotating_actor():
    axes = [
        ("Yaw", "on the Y axis"),
        ("Pitch", "on the X axis"),
        ("Roll", "on the Z axis"),
    ]
    speeds = [0.5, 1.0, 2.0, 5.0, 10.0]
    axis_name, axis_desc = random.choice(axes)
    speed = random.choice(speeds)

    desc = f"Create a Blueprint for an actor that continuously rotates {axis_desc} at a speed of {speed} degrees per tick."
    dsl = f"""BLUEPRINT: BP_Rotating{axis_name}
PARENT: Actor

GRAPH: EventGraph

NODE n1: Event_Tick
NODE n2: GetWorldDeltaSeconds
NODE n3: MultiplyFloat [B={speed}]
NODE n4: MakeRotator [{axis_name}=1.0]
NODE n5: AddActorLocalRotation

EXEC n1.Then -> n5.Execute

DATA n1.DeltaSeconds -> n3.A [Float]
DATA n3.ReturnValue -> n4.{axis_name} [Float]
DATA n4.ReturnValue -> n5.DeltaRotation [Rotator]"""
    return desc, dsl


def pattern_pickup_item():
    items = [
        ("HealthPack", "health pack", "CurrentHealth", "25"),
        ("AmmoCrate", "ammo crate", "AmmoCount", "30"),
        ("SpeedBoost", "speed boost", "MoveSpeed", "200"),
        ("Shield", "shield pickup", "ShieldAmount", "50"),
        ("Coin", "coin", "CoinCount", "1"),
    ]
    item_class, item_name, var_name, add_value = random.choice(items)

    desc = (
        f"Create a {item_name} Blueprint that adds {add_value} to the player's "
        f"{var_name} when overlapped, then destroys itself."
    )
    dsl = f"""BLUEPRINT: BP_{item_class}
PARENT: Actor
CATEGORY: Pickups

GRAPH: EventGraph

NODE n1: Event_ActorBeginOverlap
NODE n2: CastToCharacter
NODE n3: GetVar [Variable={var_name}]
NODE n4: AddFloat [B={add_value}]
NODE n5: SetVar [Variable={var_name}]
NODE n6: PlaySound [Sound=PickupSound]
NODE n7: DestroyActor

EXEC n1.Then -> n2.Execute
EXEC n2.CastSucceeded -> n3.Execute
EXEC n3.Then -> n4.Execute
EXEC n4.Then -> n5.Execute
EXEC n5.Then -> n6.Execute
EXEC n6.Then -> n7.Execute

DATA n1.OtherActor -> n2.Object [Actor]
DATA n3.Value -> n4.A [Float]
DATA n4.ReturnValue -> n5.Value [Float]"""
    return desc, dsl


def pattern_delay_sequence():
    delays = [0.5, 1.0, 1.5, 2.0, 3.0]
    num_steps = random.randint(2, 4)
    step_delays = [random.choice(delays) for _ in range(num_steps)]
    messages = [f"Step {i+1}" for i in range(num_steps)]

    desc_parts = [f"waits {d}s then prints '{m}'" for d, m in zip(step_delays, messages)]
    desc = "Create a Blueprint that on begin play " + ", ".join(desc_parts) + "."

    nodes = ["NODE n1: Event_BeginPlay"]
    execs = []
    prev = "n1"
    idx = 2

    for i, (delay, msg) in enumerate(zip(step_delays, messages)):
        delay_id = f"n{idx}"
        print_id = f"n{idx + 1}"
        nodes.append(f"NODE {delay_id}: Delay [Duration={delay}]")
        nodes.append(f'NODE {print_id}: PrintString [InString="{msg}"]')
        execs.append(f"EXEC {prev}.Then -> {delay_id}.Execute")
        execs.append(f"EXEC {delay_id}.Completed -> {print_id}.Execute")
        prev = print_id
        idx += 2

    node_block = "\n".join(nodes)
    exec_block = "\n".join(execs)

    dsl = f"""BLUEPRINT: BP_DelaySequence
PARENT: Actor

GRAPH: EventGraph

{node_block}

{exec_block}"""
    return desc, dsl


# ============================================================
# PATTERN REGISTRY
# ============================================================

PATTERNS = [
    (pattern_print_on_begin_play, 1.0),          # weight
    (pattern_toggle_visibility, 1.5),
    (pattern_overlap_trigger, 2.0),
    (pattern_timer_loop, 1.5),
    (pattern_branch_condition, 2.0),
    (pattern_key_press_action, 1.5),
    (pattern_health_system, 2.0),
    (pattern_rotating_actor, 1.0),
    (pattern_pickup_item, 2.0),
    (pattern_delay_sequence, 1.5),
]


def generate_dataset(count: int, seed: int = 42) -> list[dict]:
    """Generate training examples using weighted random pattern selection."""
    random.seed(seed)

    patterns, weights = zip(*PATTERNS)
    total_weight = sum(weights)
    normalized = [w / total_weight for w in weights]

    examples = []
    seen_descriptions = set()

    attempts = 0
    max_attempts = count * 5  # Avoid infinite loops

    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        pattern_fn = random.choices(patterns, weights=normalized, k=1)[0]

        try:
            description, dsl_text = pattern_fn()
        except Exception as e:
            print(f"Warning: Pattern {pattern_fn.__name__} failed: {e}")
            continue

        # Deduplicate
        if description in seen_descriptions:
            continue
        seen_descriptions.add(description)

        examples.append({
            "instruction": description,
            "input": "",
            "output": dsl_text.strip(),
            "pattern": pattern_fn.__name__,
        })

    print(f"Generated {len(examples)} unique examples from {attempts} attempts")
    return examples


def save_jsonl(examples: list[dict], output_path: Path):
    """Save examples as JSONL (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Blueprint training data")
    parser.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="datasets/train.jsonl", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=float, default=0.9, help="Train/validation split ratio")
    args = parser.parse_args()

    plog.start_step("1.3", "Generate synthetic data", f"count={args.count}")
    examples = generate_dataset(args.count, args.seed)

    # Split into train/validation
    split_idx = int(len(examples) * args.split)
    random.shuffle(examples)
    train = examples[:split_idx]
    val = examples[split_idx:]

    output = Path(args.output)
    save_jsonl(train, output)
    save_jsonl(val, output.with_name("validation.jsonl"))

    # Print stats
    pattern_counts = {}
    for ex in examples:
        p = ex["pattern"]
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    print("\nPattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count} ({count/len(examples)*100:.1f}%)")
    plog.complete_step("1.3", "Generate synthetic data", f"{len(examples)} examples")


if __name__ == "__main__":
    main()
