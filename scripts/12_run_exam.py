"""
12_run_exam.py
--------------
Runs a lesson file through the current model, compares output to expected DSL,
and saves detailed results for grading.

Part of the Claude Teaching Loop:
  1. Claude creates lesson (lesson_XX.json)
  2. This script runs the exam (exam_results_XX.jsonl)
  3. Claude grades the results and creates corrections
  4. Corrections feed into next training run

Usage:
    python scripts/12_run_exam.py --lesson lessons/lesson_01.json --model models/blueprint-lora-v2/final
    python scripts/12_run_exam.py --lesson lessons/lesson_01.json --model models/blueprint-lora-v2/final --base_model meta-llama/Meta-Llama-3.1-8B
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from stop_signal_utils import is_stop_requested, clear_signal
from backup_utils import auto_backup
from pipeline_logger import get_logger as _get_pipeline_logger
plog = _get_pipeline_logger(step_prefix="6")

# Must match TRAINING_SYSTEM_PROMPT in 04_train_blueprint_lora.py
SYSTEM_PROMPT = """You are a Blueprint DSL generator for Unreal Engine 5.
Given a description, output ONLY valid Blueprint DSL code.
Use this exact format:

BLUEPRINT: <n>
PARENT: <ParentClass>
GRAPH: EventGraph
NODE n1: <NodeType> [Property=Value]
EXEC n1.Then -> n2.Execute
DATA n1.Pin -> n2.Pin [Type]

Output ONLY the DSL. No explanations."""


def load_model(model_path, base_model=None):
    """Load the fine-tuned model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Auto-detect base model
    if base_model is None:
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config) as f:
                ac = json.load(f)
            base_model = ac.get("base_model_name_or_path", "meta-llama/Llama-3.2-3B")
        else:
            base_model = "meta-llama/Llama-3.2-3B"

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )

    print(f"Loading LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model, tokenizer, base_model


def generate(model, tokenizer, instruction, max_tokens=512, temperature=0.1):
    """Generate DSL from an instruction."""
    import torch
    import re

    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    stop_ids = [tokenizer.eos_token_id]
    for tok in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=temperature > 0, top_p=0.9, repetition_penalty=1.1,
            eos_token_id=stop_ids,
        )

    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Extract DSL from potentially messy output
    # Try code fence extraction first
    fence = re.search(r'```(?:\w*\n)?(.*?)```', raw, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        if "BLUEPRINT:" in candidate or "NODE" in candidate:
            return candidate, raw

    # Find BLUEPRINT: line and capture until junk starts
    lines = raw.split('\n')
    dsl_lines = []
    in_dsl = False
    for line in lines:
        s = line.strip()
        if s.startswith("BLUEPRINT:"):
            in_dsl = True
        if in_dsl and s and any(s.startswith(m) for m in [
            "## ", "Valid node", "Rules for", "---", "Line |",
            "**", "### ", "IN:", "OUT:", "Your output",
        ]):
            break
        if in_dsl:
            dsl_lines.append(line)

    if dsl_lines:
        return '\n'.join(dsl_lines).strip(), raw

    return raw, raw


def validate_dsl(dsl_text):
    """Validate DSL and return structured result."""
    try:
        from utils.dsl_parser import parse_dsl
        bp = parse_dsl(dsl_text)
        nodes = sum(len(g["nodes"]) for g in bp.graphs.values())
        conns = sum(len(g["connections"]) for g in bp.graphs.values())
        return {
            "valid": True,
            "name": bp.name,
            "parent": bp.parent_class,
            "nodes": nodes,
            "connections": conns,
            "variables": len(bp.variables),
            "error": None,
        }
    except Exception as e:
        return {
            "valid": False,
            "name": None,
            "nodes": 0,
            "connections": 0,
            "variables": 0,
            "error": str(e),
        }


def compare_outputs(expected_dsl, actual_dsl):
    """Compare expected vs actual DSL and produce a diff report."""
    expected_lines = set(l.strip() for l in expected_dsl.splitlines() if l.strip() and not l.strip().startswith("#"))
    actual_lines = set(l.strip() for l in actual_dsl.splitlines() if l.strip() and not l.strip().startswith("#"))

    missing = expected_lines - actual_lines   # In expected but not actual
    extra = actual_lines - expected_lines      # In actual but not expected
    correct = expected_lines & actual_lines    # In both

    # Categorize
    missing_nodes = [l for l in missing if l.startswith("NODE")]
    missing_exec = [l for l in missing if l.startswith("EXEC")]
    missing_data = [l for l in missing if l.startswith("DATA")]
    extra_nodes = [l for l in extra if l.startswith("NODE")]
    extra_exec = [l for l in extra if l.startswith("EXEC")]
    extra_data = [l for l in extra if l.startswith("DATA")]

    total = len(expected_lines)
    score = len(correct) / max(total, 1)

    return {
        "score": round(score, 3),
        "correct_lines": len(correct),
        "total_expected_lines": total,
        "missing_lines": sorted(list(missing)),
        "extra_lines": sorted(list(extra)),
        "missing_nodes": missing_nodes,
        "missing_exec": missing_exec,
        "missing_data": missing_data,
        "extra_nodes": extra_nodes,
        "extra_exec": extra_exec,
        "extra_data": extra_data,
    }


def run_exam(lesson_path, model_path, base_model, output_dir):
    """Run all prompts from a lesson through the model."""

    plog.start_step("6.1", "Load lesson", str(lesson_path))
    with open(lesson_path, encoding="utf-8") as f:
        lesson = json.load(f)

    print(f"\n{'='*60}")
    print(f"  EXAM: {lesson['lesson_name']}")
    print(f"  {len(lesson['prompts'])} prompts")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")
    plog.complete_step("6.1", "Load lesson", f"{len(lesson['prompts'])} prompts")

    plog.start_step("6.2", "Load model for exam", str(model_path))
    model, tokenizer, detected_base = load_model(model_path, base_model)
    plog.complete_step("6.2", "Load model for exam")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"exam_{lesson['lesson_id']}_{ts}.jsonl"
    summary_file = output_dir / f"exam_{lesson['lesson_id']}_{ts}_summary.json"

    results = []
    total_score = 0
    valid_count = 0

    plog.start_step("6.3", "Run prompts", f"{len(lesson['prompts'])} prompts")
    for i, prompt in enumerate(lesson["prompts"]):
        print(f"[{i+1}/{len(lesson['prompts'])}] {prompt['id']}: {prompt['instruction'][:60]}...")

        start = time.time()
        cleaned_dsl, raw_output = generate(model, tokenizer, prompt["instruction"])
        elapsed = time.time() - start

        # Validate
        validation = validate_dsl(cleaned_dsl)

        # Compare to expected
        comparison = compare_outputs(prompt["expected_dsl"], cleaned_dsl)

        status = "[OK]" if validation["valid"] else "[X]"
        print(f"  {status} Score: {comparison['score']:.0%} | "
              f"Nodes: {validation['nodes']} | "
              f"Connections: {validation['connections']} | "
              f"{elapsed:.1f}s")

        if not validation["valid"]:
            # Show first error line
            err = validation["error"]
            if err:
                first_line = err.split('\n')[0][:80]
                print(f"  Error: {first_line}")

        result = {
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "instruction": prompt["instruction"],
            "expected_dsl": prompt["expected_dsl"],
            "actual_dsl": cleaned_dsl,
            "raw_output": raw_output,
            "validation": validation,
            "comparison": comparison,
            "time_seconds": round(elapsed, 1),
        }
        results.append(result)

        if validation["valid"]:
            valid_count += 1
        total_score += comparison["score"]

        # Write incrementally
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        plog.progress("6.3", i + 1, len(lesson["prompts"]),
                      f"{prompt['id']} {'OK' if validation['valid'] else 'FAIL'}")

        # Check for graceful stop between prompts
        if is_stop_requested():
            print(f"\n  GRACEFUL STOP REQUESTED after {i+1}/{len(lesson['prompts'])} prompts.")
            print(f"  Partial results saved to: {results_file}")
            clear_signal()
            break
    plog.complete_step("6.3", "Run prompts",
                        f"{valid_count}/{len(results)} valid")

    # Summary
    avg_score = total_score / max(len(results), 1)
    summary = {
        "lesson_id": lesson["lesson_id"],
        "lesson_name": lesson["lesson_name"],
        "model": str(model_path),
        "base_model": detected_base,
        "timestamp": ts,
        "total_prompts": len(results),
        "valid_syntax": valid_count,
        "valid_syntax_pct": round(valid_count / max(len(results), 1) * 100, 1),
        "avg_similarity_score": round(avg_score * 100, 1),
        "per_category": {},
    }

    # Category breakdown
    for r in results:
        cat = r["category"]
        if cat not in summary["per_category"]:
            summary["per_category"][cat] = {"count": 0, "valid": 0, "avg_score": 0}
        summary["per_category"][cat]["count"] += 1
        if r["validation"]["valid"]:
            summary["per_category"][cat]["valid"] += 1
        summary["per_category"][cat]["avg_score"] += r["comparison"]["score"]

    for cat in summary["per_category"]:
        s = summary["per_category"][cat]
        s["avg_score"] = round(s["avg_score"] / max(s["count"], 1) * 100, 1)

    plog.start_step("6.6", "Save exam results")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  EXAM RESULTS")
    print(f"{'='*60}")
    print(f"  Valid syntax:     {valid_count}/{len(results)} ({summary['valid_syntax_pct']}%)")
    print(f"  Avg similarity:   {summary['avg_similarity_score']}%")
    print(f"  Results:          {results_file}")
    print(f"  Summary:          {summary_file}")
    print(f"{'='*60}\n")
    plog.complete_step("6.6", "Save exam results",
                        f"Avg similarity: {summary['avg_similarity_score']}%")

    # Post-exam backup
    plog.start_step("6.7", "Post-exam backup")
    import re as _re
    _ver_match = _re.search(r'v(\d+)', str(model_path))
    _version = f"v{_ver_match.group(1)}" if _ver_match else None
    _lesson_id = lesson.get("lesson_id")
    try:
        auto_backup(trigger="exam_complete", version=_version, lesson=_lesson_id)
    except Exception as e:
        print(f"[Backup] Post-exam backup failed (non-fatal): {e}")
    plog.complete_step("6.7", "Post-exam backup")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a lesson exam through the Blueprint LLM")
    parser.add_argument("--lesson", required=True, help="Path to lesson JSON file")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model (e.g. models/blueprint-lora-v2/final)")
    parser.add_argument("--base_model", default=None, help="Base model override")
    parser.add_argument("--output", default="results/exams", help="Output directory for results")
    args = parser.parse_args()

    run_exam(args.lesson, args.model, args.base_model, args.output)


if __name__ == "__main__":
    main()
