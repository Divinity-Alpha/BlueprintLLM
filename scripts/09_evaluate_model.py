"""
09_evaluate_model.py
--------------------
Evaluates a trained (or in-progress) Blueprint LLM model with real generation tests.
This goes far beyond loss numbers — it actually generates Blueprints from test prompts
and measures whether they're correct.

Usage:
    # Evaluate a finished model
    python scripts/09_evaluate_model.py --model models/blueprint-lora/final

    # Evaluate a mid-training checkpoint
    python scripts/09_evaluate_model.py --model models/blueprint-lora/checkpoint-100

    # Compare two models side by side
    python scripts/09_evaluate_model.py --model models/blueprint-lora-v1/final --compare models/blueprint-lora-v2/final

    # Quick smoke test (3 prompts instead of full suite)
    python scripts/09_evaluate_model.py --model models/blueprint-lora/final --quick

    # Save detailed report
    python scripts/09_evaluate_model.py --model models/blueprint-lora/final --report results/eval_v1.json

This is how you ACTUALLY know if the model is learning. Loss going down is necessary
but not sufficient — a model can have low loss and still produce broken Blueprints.
This script tests real generation quality.
"""

import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logger import get_logger as _get_pipeline_logger
from error_handler import per_prompt_timeout
plog = _get_pipeline_logger(step_prefix="7")

# Per-prompt generation timeout (seconds)
GENERATE_TIMEOUT = 300


# ============================================================
# TEST SUITE
# ============================================================
# Each test has: prompt, difficulty, and expected elements that
# MUST appear in a correct output. Organized by difficulty tier.

TEST_SUITE = [
    # ----------------------------------------------------------
    # TIER 1: Trivial (should pass after ~50 training examples)
    # These test basic DSL syntax and the simplest patterns.
    # ----------------------------------------------------------
    {
        "id": "T1_01",
        "tier": 1,
        "prompt": "Create a Blueprint that prints 'Hello World' when the game starts.",
        "description": "Simplest possible: one event, one action",
        "expected_nodes": ["Event_BeginPlay", "PrintString"],
        "expected_exec_count": 1,
        "expected_data_count": 0,
        "required_keywords": ["BLUEPRINT:", "GRAPH:", "NODE", "EXEC"],
        "forbidden_keywords": [],
    },
    {
        "id": "T1_02",
        "tier": 1,
        "prompt": "Create a Blueprint that destroys itself on begin play.",
        "description": "Event to single action, different function",
        "expected_nodes": ["Event_BeginPlay", "DestroyActor"],
        "expected_exec_count": 1,
        "expected_data_count": 0,
        "required_keywords": ["BLUEPRINT:", "EXEC"],
        "forbidden_keywords": [],
    },
    {
        "id": "T1_03",
        "tier": 1,
        "prompt": "Make an Actor Blueprint that prints 'Game Over' at the start.",
        "description": "Same as T1_01 with different wording and message",
        "expected_nodes": ["Event_BeginPlay", "PrintString"],
        "expected_exec_count": 1,
        "expected_data_count": 0,
        "required_keywords": ["PARENT: Actor", "Game Over"],
        "forbidden_keywords": [],
    },

    # ----------------------------------------------------------
    # TIER 2: Basic (should pass after ~200 training examples)
    # Introduces variables, data wires, and simple logic.
    # ----------------------------------------------------------
    {
        "id": "T2_01",
        "tier": 2,
        "prompt": "Create a Blueprint for an actor that rotates continuously on the yaw axis.",
        "description": "Event Tick + data wire for delta time + rotation",
        "expected_nodes": ["Event_Tick"],
        "expected_exec_count": 1,  # At minimum
        "expected_data_count": 1,  # At minimum: delta seconds feeds into something
        "required_keywords": ["BLUEPRINT:", "EXEC", "DATA"],
        "forbidden_keywords": [],
    },
    {
        "id": "T2_02",
        "tier": 2,
        "prompt": "Create a Blueprint that toggles a light on and off when the player presses E.",
        "description": "Input action + FlipFlop + two paths",
        "expected_nodes": ["Event_InputAction", "FlipFlop"],
        "expected_exec_count": 3,  # Input->FlipFlop, FlipFlop.A->On, FlipFlop.B->Off
        "expected_data_count": 0,
        "required_keywords": ["BLUEPRINT:", "EXEC"],
        "forbidden_keywords": [],
    },
    {
        "id": "T2_03",
        "tier": 2,
        "prompt": "Create a Blueprint that prints a message when a player overlaps with it.",
        "description": "Overlap event + cast + action",
        "expected_nodes": ["Event_ActorBeginOverlap"],
        "expected_exec_count": 2,  # Overlap->Cast, Cast->Print (at minimum)
        "expected_data_count": 1,  # OtherActor feeds into Cast
        "required_keywords": ["BLUEPRINT:", "EXEC", "DATA"],
        "forbidden_keywords": [],
    },

    # ----------------------------------------------------------
    # TIER 3: Intermediate (should pass after ~500 training examples)
    # Multi-step logic, branches, variables, math.
    # ----------------------------------------------------------
    {
        "id": "T3_01",
        "tier": 3,
        "prompt": "Create a health pickup that adds 25 health on overlap and then destroys itself.",
        "description": "Overlap -> Cast -> Get -> Add -> Set -> Destroy chain",
        "expected_nodes": ["Event_ActorBeginOverlap", "DestroyActor"],
        "expected_exec_count": 4,  # At minimum: several chained actions
        "expected_data_count": 2,  # OtherActor + math chain
        "required_keywords": ["BLUEPRINT:", "EXEC", "DATA", "25"],
        "forbidden_keywords": [],
    },
    {
        "id": "T3_02",
        "tier": 3,
        "prompt": "Create a health system with 100 max HP that checks if health reaches zero and destroys the actor.",
        "description": "Variables + math + comparison + branch + conditional destroy",
        "expected_nodes": ["Branch", "DestroyActor"],
        "expected_exec_count": 3,  # At minimum
        "expected_data_count": 3,  # Math chain feeds into branch condition
        "required_keywords": ["BLUEPRINT:", "VAR", "EXEC", "DATA", "Branch"],
        "forbidden_keywords": [],
    },
    {
        "id": "T3_03",
        "tier": 3,
        "prompt": "Create a Blueprint with a timer that prints 'Tick' every 2 seconds.",
        "description": "BeginPlay -> SetTimer, CustomEvent -> PrintString",
        "expected_nodes": ["Event_BeginPlay", "SetTimerByFunctionName"],
        "expected_exec_count": 2,  # BeginPlay->SetTimer, CustomEvent->Print
        "expected_data_count": 0,
        "required_keywords": ["BLUEPRINT:", "2"],
        "forbidden_keywords": [],
    },

    # ----------------------------------------------------------
    # TIER 4: Advanced (should pass after ~1000+ training examples)
    # Novel combinations, complex logic, multiple concepts.
    # ----------------------------------------------------------
    {
        "id": "T4_01",
        "tier": 4,
        "prompt": "Create a door that opens when a player presses E within 200 units. The door should rotate 90 degrees on the yaw axis.",
        "description": "Input + distance check + branch + rotation — multi-concept",
        "expected_nodes": ["Event_InputAction", "Branch"],
        "expected_exec_count": 3,
        "expected_data_count": 3,
        "required_keywords": ["BLUEPRINT:", "EXEC", "DATA", "200", "90"],
        "forbidden_keywords": [],
    },
    {
        "id": "T4_02",
        "tier": 4,
        "prompt": "Create a coin pickup Blueprint that adds 1 to a coin counter, plays a sound, and destroys itself when a player overlaps.",
        "description": "Overlap + cast + variable increment + sound + destroy sequence",
        "expected_nodes": ["Event_ActorBeginOverlap", "DestroyActor"],
        "expected_exec_count": 4,
        "expected_data_count": 2,
        "required_keywords": ["BLUEPRINT:", "VAR", "EXEC", "DATA"],
        "forbidden_keywords": [],
    },
]

QUICK_TESTS = ["T1_01", "T2_01", "T3_01"]  # Subset for --quick mode


# ============================================================
# SCORING
# ============================================================

@dataclass
class TestResult:
    test_id: str
    tier: int
    prompt: str
    passed: bool
    score: float  # 0.0 to 1.0
    output: str
    generation_time: float
    checks: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def score_output(output: str, test: dict) -> TestResult:
    """Score a generated output against expected criteria."""
    checks = {}
    errors = []
    points = 0
    max_points = 0

    # ---- CHECK 1: Basic DSL structure (worth 30%) ----
    max_points += 30
    structure_score = 0

    has_blueprint = "BLUEPRINT:" in output
    has_graph = "GRAPH:" in output
    has_node = "NODE " in output or "NODE\t" in output
    has_exec = "EXEC " in output

    if has_blueprint:
        structure_score += 8
    else:
        errors.append("Missing BLUEPRINT: declaration")
    if has_graph:
        structure_score += 7
    else:
        errors.append("Missing GRAPH: declaration")
    if has_node:
        structure_score += 8
    else:
        errors.append("Missing NODE definitions")
    if has_exec:
        structure_score += 7
    else:
        errors.append("Missing EXEC connections")

    checks["structure"] = {"score": structure_score, "max": 30}
    points += structure_score

    # ---- CHECK 2: Required keywords present (worth 20%) ----
    max_points += 20
    keyword_hits = 0
    for kw in test["required_keywords"]:
        if kw in output:
            keyword_hits += 1
        else:
            errors.append(f"Missing required keyword: '{kw}'")

    keyword_score = int(20 * keyword_hits / max(len(test["required_keywords"]), 1))
    checks["keywords"] = {"score": keyword_score, "max": 20,
                           "found": keyword_hits, "expected": len(test["required_keywords"])}
    points += keyword_score

    # ---- CHECK 3: Expected node types present (worth 20%) ----
    max_points += 20
    node_hits = 0
    for node_type in test["expected_nodes"]:
        if node_type in output:
            node_hits += 1
        else:
            errors.append(f"Missing expected node: {node_type}")

    node_score = int(20 * node_hits / max(len(test["expected_nodes"]), 1))
    checks["nodes"] = {"score": node_score, "max": 20,
                        "found": node_hits, "expected": len(test["expected_nodes"])}
    points += node_score

    # ---- CHECK 4: Connection counts reasonable (worth 15%) ----
    max_points += 15
    exec_count = output.count("EXEC ")
    data_count = output.count("DATA ")

    exec_ok = exec_count >= test["expected_exec_count"]
    data_ok = data_count >= test["expected_data_count"]

    conn_score = 0
    if exec_ok:
        conn_score += 8
    else:
        errors.append(f"Too few EXEC connections: got {exec_count}, expected >= {test['expected_exec_count']}")
    if data_ok:
        conn_score += 7
    else:
        errors.append(f"Too few DATA connections: got {data_count}, expected >= {test['expected_data_count']}")

    checks["connections"] = {"score": conn_score, "max": 15,
                              "exec_found": exec_count, "data_found": data_count}
    points += conn_score

    # ---- CHECK 5: DSL parseable (worth 15%) ----
    max_points += 15
    parse_score = 0
    try:
        from utils.dsl_parser import parse_dsl
        bp = parse_dsl(output)
        parse_score = 15
    except Exception as e:
        errors.append(f"Parse error: {str(e)[:100]}")

    checks["parseable"] = {"score": parse_score, "max": 15}
    points += parse_score

    # ---- CHECK 6: No forbidden keywords (pass/fail) ----
    for kw in test.get("forbidden_keywords", []):
        if kw in output:
            errors.append(f"Contains forbidden keyword: '{kw}'")
            points = max(0, points - 5)

    # ---- FINAL SCORE ----
    final_score = points / max(max_points, 1)
    passed = final_score >= 0.7  # 70% threshold to pass

    return TestResult(
        test_id=test["id"],
        tier=test["tier"],
        prompt=test["prompt"],
        passed=passed,
        score=round(final_score, 3),
        output=output,
        generation_time=0,  # Set by caller
        checks=checks,
        errors=errors,
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path: str, base_model: str = None):
    """Load a trained model for evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Auto-detect base model
    config_path = Path(model_path).parent / "training_config.json"
    if not config_path.exists():
        config_path = Path(model_path).parent.parent / "training_config.json"

    if base_model is None and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        base_model = config.get("base_model", "meta-llama/Meta-Llama-3.1-8B")
    elif base_model is None:
        base_model = "meta-llama/Meta-Llama-3.1-8B"

    # Load system prompt (use the one saved with the model)
    prompt_path = Path(model_path).parent / "system_prompt.txt"
    if not prompt_path.exists():
        prompt_path = Path(model_path).parent.parent / "system_prompt.txt"

    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"  Loaded system prompt from {prompt_path}")
    else:
        system_prompt = "You are a Blueprint programming assistant for Unreal Engine 5. Generate valid Blueprint DSL code."
        print(f"  WARNING: No system prompt found, using basic fallback")

    print(f"Loading base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Use 8-bit for 70B models (Blackwell compat), 4-bit for smaller.
    # low_cpu_mem_usage=True prevents OOM on 32GB system RAM.
    from transformers import BitsAndBytesConfig
    if "70b" in base_model.lower():
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    return model, tokenizer, system_prompt


def generate(model, tokenizer, system_prompt: str, prompt: str, max_tokens: int = 768) -> tuple[str, float]:
    """Generate and return (output, time_seconds)."""
    import torch

    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.1,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
        )
    elapsed = time.time() - start

    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip(), elapsed


# ============================================================
# EVALUATION RUNNER
# ============================================================

def run_evaluation(model, tokenizer, system_prompt: str, test_ids: list = None) -> list[TestResult]:
    """Run the full test suite and return scored results."""
    tests = TEST_SUITE
    if test_ids:
        tests = [t for t in TEST_SUITE if t["id"] in test_ids]

    results = []
    for i, test in enumerate(tests):
        print(f"  [{i+1}/{len(tests)}] {test['id']} (Tier {test['tier']}): {test['prompt'][:50]}...", end=" ", flush=True)

        # Wrap generate() in timeout to avoid hanging on bad prompts
        gen_result, timed_out = per_prompt_timeout(
            lambda t=test: generate(model, tokenizer, system_prompt, t["prompt"]),
            timeout_seconds=GENERATE_TIMEOUT,
        )

        if timed_out:
            result = TestResult(
                test_id=test["id"],
                tier=test["tier"],
                prompt=test["prompt"],
                passed=False,
                score=0.0,
                output="[TIMEOUT]",
                generation_time=GENERATE_TIMEOUT,
                checks={},
                errors=[f"Generation timed out after {GENERATE_TIMEOUT}s"],
            )
            print(f"TIMEOUT ({GENERATE_TIMEOUT}s)")
        else:
            output, gen_time = gen_result
            result = score_output(output, test)
            result.generation_time = round(gen_time, 2)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} ({result.score:.0%}, {gen_time:.1f}s)")

        results.append(result)
        detail = f"{test['id']} {'TIMEOUT' if timed_out else ('PASS' if result.passed else 'FAIL')}"
        plog.progress("7.2", i + 1, len(tests), detail)

    return results


def print_report(results: list[TestResult], model_name: str = ""):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT{f' — {model_name}' if model_name else ''}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Overall stats
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / max(total, 1)
    avg_time = sum(r.generation_time for r in results) / max(total, 1)

    print(f"\nOverall: {passed}/{total} passed ({passed/max(total,1)*100:.0f}%)")
    print(f"Average score: {avg_score:.1%}")
    print(f"Average generation time: {avg_time:.1f}s")

    # Per-tier breakdown
    tiers = sorted(set(r.tier for r in results))
    print(f"\n{'Tier':<8} {'Passed':<12} {'Avg Score':<12} {'Status'}")
    print("-" * 50)
    for tier in tiers:
        tier_results = [r for r in results if r.tier == tier]
        tier_passed = sum(1 for r in tier_results if r.passed)
        tier_total = len(tier_results)
        tier_avg = sum(r.score for r in tier_results) / max(tier_total, 1)

        if tier_avg >= 0.9:
            status = "EXCELLENT"
        elif tier_avg >= 0.7:
            status = "GOOD"
        elif tier_avg >= 0.4:
            status = "LEARNING"
        else:
            status = "NOT YET"

        print(f"Tier {tier:<4} {tier_passed}/{tier_total:<10} {tier_avg:<12.1%} {status}")

    # Individual results
    print(f"\n{'ID':<8} {'Tier':<6} {'Score':<8} {'Time':<8} {'Result':<8} Errors")
    print("-" * 70)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        error_summary = "; ".join(r.errors[:2]) if r.errors else "—"
        if len(error_summary) > 40:
            error_summary = error_summary[:37] + "..."
        print(f"{r.test_id:<8} {r.tier:<6} {r.score:<8.0%} {r.generation_time:<8.1f} {status:<8} {error_summary}")

    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")

    if avg_score < 0.4:
        print("  The model is in early stages. This is normal for <100 training examples.")
        print("  -> Focus on adding more Tier 1 and Tier 2 examples to your training data.")
        print("  -> Make sure training loss is actually decreasing (check logs).")
    elif avg_score < 0.7:
        print("  The model is learning! Basic patterns work but complex ones need more data.")
        print("  -> Add more training examples for the specific patterns that are failing.")
        print("  -> Check the errors above for common failure modes.")
    elif avg_score < 0.9:
        print("  The model is performing well. Focus on edge cases and variety.")
        print("  -> Add more diverse phrasings for the same Blueprint patterns.")
        print("  -> Add examples that combine multiple concepts (Tier 4 style).")
    else:
        print("  Excellent performance! The model has learned the core patterns well.")
        print("  -> Expand the node vocabulary and add more advanced patterns.")
        print("  -> Consider increasing training data complexity.")

    # Show worst failures for targeted improvement
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  Worst failures (fix these first):")
        for r in sorted(failures, key=lambda x: x.score)[:3]:
            print(f"    {r.test_id}: {r.prompt[:60]}")
            for err in r.errors[:2]:
                print(f"      -> {err}")

    print("=" * 70)


# ============================================================
# COMPARISON MODE
# ============================================================

def compare_models(results_a: list[TestResult], results_b: list[TestResult],
                   name_a: str, name_b: str):
    """Print a side-by-side comparison of two model evaluations."""
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON: {name_a} vs {name_b}")
    print("=" * 70)

    print(f"\n{'Test':<8} {'Tier':<6} {name_a:<20} {name_b:<20} {'Delta'}")
    print("-" * 70)

    improvements = 0
    regressions = 0

    for ra, rb in zip(results_a, results_b):
        delta = rb.score - ra.score
        delta_str = f"{delta:+.0%}"
        if delta > 0.05:
            delta_str += " [UP]"
            improvements += 1
        elif delta < -0.05:
            delta_str += " [DN]"
            regressions += 1

        status_a = "PASS" if ra.passed else "FAIL"
        status_b = "PASS" if rb.passed else "FAIL"

        print(f"{ra.test_id:<8} {ra.tier:<6} {ra.score:.0%} {status_a:<13} {rb.score:.0%} {status_b:<13} {delta_str}")

    avg_a = sum(r.score for r in results_a) / max(len(results_a), 1)
    avg_b = sum(r.score for r in results_b) / max(len(results_b), 1)
    delta_avg = avg_b - avg_a

    print("-" * 70)
    print(f"{'AVERAGE':<14} {avg_a:.1%}{'':<15} {avg_b:.1%}{'':<15} {delta_avg:+.1%}")
    print(f"\nImprovements: {improvements}  |  Regressions: {regressions}  |  No change: {len(results_a) - improvements - regressions}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Blueprint LLM model quality")
    parser.add_argument("--model", type=str, required=True, help="Path to LoRA model")
    parser.add_argument("--base_model", type=str, help="Base model (auto-detected)")
    parser.add_argument("--compare", type=str, help="Second model path for comparison")
    parser.add_argument("--quick", action="store_true", help="Run only 3 smoke tests")
    parser.add_argument("--tier", type=int, help="Run only tests from this tier (1-4)")
    parser.add_argument("--report", type=str, help="Save detailed JSON report to file")
    args = parser.parse_args()

    # Determine which tests to run
    test_ids = None
    if args.quick:
        test_ids = QUICK_TESTS
        print("Running quick smoke test (3 prompts)...")
    elif args.tier:
        test_ids = [t["id"] for t in TEST_SUITE if t["tier"] == args.tier]
        print(f"Running Tier {args.tier} tests ({len(test_ids)} prompts)...")
    else:
        print(f"Running full test suite ({len(TEST_SUITE)} prompts)...")

    # Load and evaluate primary model
    plog.start_step("7.1", "Load model", f"Model: {args.model}")
    print(f"\nLoading model: {args.model}")
    model, tokenizer, system_prompt = load_model(args.model, args.base_model)
    plog.complete_step("7.1", "Load model")

    plog.start_step("7.2", "Run test suite", f"{len(test_ids) if test_ids else len(TEST_SUITE)} tests")
    print("\nRunning evaluation...\n")
    results = run_evaluation(model, tokenizer, system_prompt, test_ids)
    plog.complete_step("7.2", "Run test suite",
                        f"{sum(1 for r in results if r.passed)}/{len(results)} passed")

    plog.start_step("7.3", "Score results")
    print_report(results, args.model)
    plog.complete_step("7.3", "Score results")

    # Comparison mode
    if args.compare:
        print(f"\nLoading comparison model: {args.compare}")
        model_b, tokenizer_b, prompt_b = load_model(args.compare, args.base_model)
        print("\nRunning comparison evaluation...\n")
        results_b = run_evaluation(model_b, tokenizer_b, prompt_b, test_ids)
        print_report(results_b, args.compare)
        compare_models(results, results_b, args.model, args.compare)

    # Save report
    if args.report:
        plog.start_step("7.4", "Generate report")
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model": args.model,
            "date": datetime.now().isoformat(),
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "avg_score": round(sum(r.score for r in results) / max(len(results), 1), 3),
                "avg_time": round(sum(r.generation_time for r in results) / max(len(results), 1), 2),
            },
            "results": [asdict(r) for r in results],
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        plog.complete_step("7.4", "Generate report", str(report_path))


if __name__ == "__main__":
    main()
