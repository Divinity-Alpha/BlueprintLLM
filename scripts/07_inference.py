"""
07_inference.py
---------------
Run inference with your fine-tuned Blueprint LLM.
Supports both Hugging Face LoRA models and Ollama-served models.

Usage:
    # Using fine-tuned LoRA model
    python scripts/07_inference.py --model models/blueprint-lora/final --prompt "Create a rotating cube"

    # Using Ollama
    python scripts/07_inference.py --ollama blueprint-llm --prompt "Create a rotating cube"

    # Interactive mode
    python scripts/07_inference.py --model models/blueprint-lora/final --interactive

    # Batch mode (process multiple prompts)
    python scripts/07_inference.py --model models/blueprint-lora/final --batch prompts.txt --output results/
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))



# Fallback system prompt — used only when model directory has no system_prompt.txt.
# Prefer loading the saved prompt from the model so inference matches training.
DEFAULT_SYSTEM_PROMPT = """You are a Blueprint DSL generator for Unreal Engine 5.
Given a description, output ONLY valid Blueprint DSL code.
Use this exact format:

BLUEPRINT: <n>
PARENT: <ParentClass>
GRAPH: EventGraph
NODE n1: <NodeType> [Property=Value]
EXEC n1.Then -> n2.Execute
DATA n1.Pin -> n2.Pin [Type]

Output ONLY the DSL. No explanations."""


# ============================================================
# HUGGING FACE INFERENCE
# ============================================================

def load_hf_model(model_path: str, base_model: str = None):
    """Load a fine-tuned LoRA model for inference.

    Returns (model, tokenizer, system_prompt).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Determine base model from config if not specified
    model_dir = Path(model_path).parent
    config_path = model_dir / "training_config.json"
    if base_model is None and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        base_model = config.get("base_model", "meta-llama/Llama-3.2-3B")
    elif base_model is None:
        # Try to detect from adapter_config.json
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config) as f:
                ac = json.load(f)
            base_model = ac.get("base_model_name_or_path", "meta-llama/Llama-3.2-3B")
        else:
            base_model = "meta-llama/Llama-3.2-3B"

    # Load saved system prompt from model directory (matches training)
    system_prompt = DEFAULT_SYSTEM_PROMPT
    prompt_path = model_dir / "system_prompt.txt"
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"Loaded system prompt from {prompt_path} ({len(system_prompt):,} chars)")
    else:
        print(f"No system_prompt.txt in {model_dir}, using default prompt")

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Use 8-bit for 70B models (Blackwell compat), 4-bit for smaller
    try:
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
    except Exception:
        # Fallback: load without quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    print(f"Loading LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    return model, tokenizer, system_prompt


def generate_hf(model, tokenizer, prompt: str, max_tokens: int = 512,
                temperature: float = 0.1, system_prompt: str = None) -> str:
    """Generate Blueprint DSL using a Hugging Face model."""
    import torch

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Build stop token IDs
    stop_token_ids = [tokenizer.eos_token_id]
    for token_str in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_token_ids.append(tid)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=stop_token_ids,
        )

    # Decode only the new tokens
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = response.strip()

    # Extract the actual DSL from the model's output.
    # The model may wrap it in code fences, prefix it with headers, or
    # append the system prompt / node reference after the DSL.
    return extract_dsl(response)


def extract_dsl(raw: str) -> str:
    """Extract clean DSL from potentially messy model output."""

    # Strategy 1: If there's a code fence, extract its contents
    import re
    fence_match = re.search(r'```(?:\w*\n)?(.*?)```', raw, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        # Verify it looks like DSL (has BLUEPRINT: or NODE lines)
        if "BLUEPRINT:" in candidate or "NODE" in candidate:
            return candidate

    # Strategy 2: Find the DSL block by looking for BLUEPRINT: line
    lines = raw.split('\n')
    dsl_lines = []
    in_dsl = False
    for line in lines:
        stripped = line.strip()
        # Start capturing at BLUEPRINT:
        if stripped.startswith("BLUEPRINT:"):
            in_dsl = True
        # Stop capturing at system prompt / reference leakage
        if in_dsl and stripped and any(stripped.startswith(m) for m in [
            "## ", "Valid node", "Rules for", "---", "Line |",
            "**", "### ", "IN:", "OUT:", "Your output",
        ]):
            break
        if in_dsl:
            dsl_lines.append(line)

    if dsl_lines:
        return '\n'.join(dsl_lines).strip()

    # Strategy 3: Just return everything, stripping obvious junk
    # Remove leading markdown headers that aren't DSL
    cleaned = raw
    while cleaned and cleaned.split('\n')[0].strip().startswith('##'):
        cleaned = '\n'.join(cleaned.split('\n')[1:])

    return cleaned.strip()


# ============================================================
# OLLAMA INFERENCE
# ============================================================

def generate_ollama(model_name: str, prompt: str) -> str:
    """Generate Blueprint DSL using Ollama."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model_name,
        "system": DEFAULT_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"Error: Cannot connect to Ollama at localhost:11434")
        print(f"Make sure Ollama is running: 'ollama serve'")
        raise


# ============================================================
# VALIDATION INTEGRATION
# ============================================================

def validate_output(dsl_text: str) -> dict:
    """Validate the generated DSL and return results."""
    try:
        from utils.dsl_parser import parse_dsl
        bp = parse_dsl(dsl_text)
        return {
            "valid_syntax": True,
            "name": bp.name,
            "nodes": sum(len(g["nodes"]) for g in bp.graphs.values()),
            "connections": sum(len(g["connections"]) for g in bp.graphs.values()),
        }
    except Exception as e:
        return {"valid_syntax": False, "error": str(e)}


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode(generate_fn):
    """Run an interactive REPL for Blueprint generation."""
    print("\n" + "=" * 60)
    print("BLUEPRINT LLM — Interactive Mode")
    print("=" * 60)
    print("Type a description to generate a Blueprint.")
    print("Commands: /quit, /validate, /save <filename>")
    print("=" * 60 + "\n")

    last_output = None

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if prompt.startswith("/validate") and last_output:
            result = validate_output(last_output)
            if result["valid_syntax"]:
                print(f"[OK] Valid! {result['nodes']} nodes, {result['connections']} connections")
            else:
                print(f"[X] Invalid: {result['error']}")
            continue

        if prompt.startswith("/save ") and last_output:
            filename = prompt[6:].strip()
            Path(filename).write_text(last_output)
            print(f"Saved to {filename}")
            continue

        print("\nGenerating...", end="", flush=True)
        start = time.time()
        output = generate_fn(prompt)
        elapsed = time.time() - start
        print(f" ({elapsed:.1f}s)\n")

        print("--- Generated Blueprint DSL ---")
        print(output)
        print("--- End ---\n")

        # Auto-validate
        result = validate_output(output)
        if result["valid_syntax"]:
            print(f"[OK] Syntax valid ({result['nodes']} nodes, {result['connections']} connections)")
        else:
            print(f"[!] Syntax issues: {result.get('error', 'unknown')}")

        last_output = output
        print()


# ============================================================
# BATCH MODE
# ============================================================

def batch_mode(generate_fn, prompts_file: str, output_dir: str):
    """Process multiple prompts from a file."""
    prompts = Path(prompts_file).read_text().strip().splitlines()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if not prompt or prompt.startswith("#"):
            continue

        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...", end=" ", flush=True)
        start = time.time()
        output = generate_fn(prompt)
        elapsed = time.time() - start

        validation = validate_output(output)
        status = "[OK]" if validation["valid_syntax"] else "[X]"
        print(f"{status} ({elapsed:.1f}s)")

        # Save individual DSL file
        dsl_file = output_path / f"blueprint_{i+1:04d}.dsl"
        dsl_file.write_text(output)

        results.append({
            "prompt": prompt,
            "output": output,
            "validation": validation,
            "time": elapsed,
        })

    # Save summary
    summary_path = output_path / "batch_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    valid_count = sum(1 for r in results if r["validation"]["valid_syntax"])
    print(f"\nResults: {valid_count}/{len(results)} valid ({valid_count/max(len(results),1)*100:.0f}%)")
    print(f"Saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Blueprint LLM Inference")
    parser.add_argument("--model", type=str, help="Path to fine-tuned LoRA model")
    parser.add_argument("--base_model", type=str, help="Base model name (auto-detected from config)")
    parser.add_argument("--ollama", type=str, help="Ollama model name")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--batch", type=str, help="File with prompts (one per line)")
    parser.add_argument("--output", type=str, default="results/", help="Output directory for batch mode")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    # Setup generation function
    if args.ollama:
        print(f"Using Ollama model: {args.ollama}")
        generate_fn = lambda prompt: generate_ollama(args.ollama, prompt)
    elif args.model:
        model, tokenizer, system_prompt = load_hf_model(args.model, args.base_model)
        generate_fn = lambda prompt: generate_hf(
            model, tokenizer, prompt, args.max_tokens, args.temperature, system_prompt
        )
    else:
        print("Error: Specify --model (HuggingFace) or --ollama (Ollama)")
        sys.exit(1)

    # Execute mode
    if args.interactive:
        interactive_mode(generate_fn)
    elif args.batch:
        batch_mode(generate_fn, args.batch, args.output)
    elif args.prompt:
        output = generate_fn(args.prompt)
        print(output)
        print()
        result = validate_output(output)
        if result["valid_syntax"]:
            print(f"[OK] Valid ({result['nodes']} nodes, {result['connections']} connections)")
        else:
            print(f"[X] {result['error']}")
    else:
        # Default to interactive
        interactive_mode(generate_fn)


if __name__ == "__main__":
    main()
