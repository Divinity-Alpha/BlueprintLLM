"""
04_train_blueprint_lora.py (v2 — Enhanced System Prompt)
---------------------------------------------------------
Fine-tunes a LoRA adapter on top of LLaMA 3.1 8B (or similar) using your
Blueprint DSL training data. Uses QLoRA (4-bit quantization) to fit in ~6 GB VRAM.

KEY CHANGE FROM v1: Now uses the enhanced system prompt that includes the full
node vocabulary reference. The model sees the node "cheat sheet" during training,
so it learns to use it as a lookup table rather than memorizing everything.

Usage:
    # Step 1: Generate the enhanced system prompt (do this once, and again whenever
    #          you add new nodes to blueprint_patterns.py)
    python scripts/08_generate_system_prompt.py --output scripts/system_prompt.txt

    # Step 2: Train with the enhanced prompt
    python scripts/04_train_blueprint_lora.py \
        --base_model meta-llama/Meta-Llama-3.1-8B \
        --dataset datasets/train.jsonl \
        --output models/blueprint-lora \
        --epochs 3

    # OR: Train with the basic prompt (no node reference — not recommended)
    python scripts/04_train_blueprint_lora.py --basic-prompt ...

Prerequisites:
    - Activated venv with all packages from your setup guide
    - Hugging Face authentication completed (Step 10 of your guide)
    - Meta LLaMA license accepted on Hugging Face
    - CUDA-capable GPU with 8+ GB VRAM
"""

import os
import json
import sys
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Add parent to path so we can import the prompt generator
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    # Model
    "base_model": "meta-llama/Meta-Llama-3.1-8B",
    "max_seq_length": 4096,  # Increased from 2048 to fit enhanced prompt

    # LoRA
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Training
    "epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "max_grad_norm": 0.3,

    # Quantization
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "use_double_quant": True,

    # Output
    "output_dir": "models/blueprint-lora",
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
}


# ============================================================
# SYSTEM PROMPT — TWO VERSIONS
# ============================================================

BASIC_SYSTEM_PROMPT = """You are a Blueprint programming assistant for Unreal Engine 5. \
Given a natural language description of desired game behavior, you generate \
valid Blueprint DSL code that implements that behavior.

Your output must follow the Blueprint DSL format:
- BLUEPRINT: <n>
- PARENT: <parent class>
- VAR <n>: <type> = <default>
- GRAPH: <graph name>
- NODE <id>: <type> [properties]
- EXEC <from>.<pin> -> <to>.<pin>
- DATA <from>.<pin> -> <to>.<pin> [<type>]

Generate only the DSL code, no explanations."""


def load_enhanced_system_prompt() -> str:
    """
    Load the enhanced system prompt with the full node reference.
    Falls back to generating it on the fly from the node catalog.
    """
    # Check for pre-generated prompt file
    prompt_path = Path(__file__).parent / "system_prompt.txt"
    if prompt_path.exists():
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"  Loaded enhanced system prompt from {prompt_path}")
        print(f"  ({len(prompt):,} chars, ~{len(prompt)//4:,} tokens)")
        return prompt

    # Generate directly from the patterns module
    try:
        from utils.blueprint_patterns import NODE_CATALOG, get_all_categories, get_nodes_by_category

        lines = [BASIC_SYSTEM_PROMPT, "", "## NODE REFERENCE",
                 "Valid node types with their pins. Use ONLY these exact names.", ""]

        for category in get_all_categories():
            nodes = get_nodes_by_category(category)
            lines.append(f"### {category}")
            for node in nodes:
                in_pins = [f"{p.name}:{p.pin_type}" for p in node.pins if p.direction == "input"]
                out_pins = [f"{p.name}:{p.pin_type}" for p in node.pins if p.direction == "output"]
                lines.append(f"**{node.type_name}** — {node.description}")
                if in_pins:
                    lines.append(f"  IN: {', '.join(in_pins)}")
                if out_pins:
                    lines.append(f"  OUT: {', '.join(out_pins)}")
                lines.append("")

        lines.extend([
            "## CONNECTION RULES",
            "- exec pins connect ONLY to exec pins (EXEC lines)",
            "- Data pins connect ONLY to matching types (DATA lines)",
            "- Bool->Bool, Float->Float, Int->Float (implicit cast), Actor->Object (upcast OK)",
            "",
            "## DATA TYPES",
            "Bool, Int, Float, String, Vector, Rotator, Transform, Actor, Object, Class, Array",
            "",
            "## PARENT CLASSES",
            "Actor, Pawn, Character, PlayerController, GameModeBase, ActorComponent, UserWidget",
            "",
            "Generate ONLY the DSL code, no explanations.",
        ])

        prompt = "\n".join(lines)
        print(f"  Built enhanced prompt from node catalog ({len(NODE_CATALOG)} nodes)")
        print(f"  ({len(prompt):,} chars, ~{len(prompt)//4:,} tokens)")

        # Save for next time
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"  Saved to {prompt_path} for future runs")
        return prompt

    except ImportError:
        print("  WARNING: Could not load node catalog. Using basic prompt.")
        return BASIC_SYSTEM_PROMPT


# Global — set once based on command-line flag
ACTIVE_SYSTEM_PROMPT = None


# ============================================================
# PROMPT FORMATTING
# ============================================================

## Training uses a SHORT system prompt — the model learns the DSL patterns
# from the examples themselves, not from a huge reference table.
# The enhanced prompt is only used at INFERENCE time as context.

TRAINING_SYSTEM_PROMPT = """You are a Blueprint DSL generator for Unreal Engine 5.
Given a description, output ONLY valid Blueprint DSL code.
Use this exact format:

BLUEPRINT: <Name>
PARENT: <ParentClass>
GRAPH: EventGraph
NODE n1: <NodeType> [Property=Value]
EXEC n1.Then -> n2.Execute
DATA n1.Pin -> n2.Pin [Type]

Output ONLY the DSL. No explanations."""


def format_training_example(example: dict) -> str:
    """Format a training example as a chat-style prompt.
    
    Uses a SHORT system prompt so the model focuses on learning
    the instruction->DSL mapping, not memorizing a huge reference.
    """
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{TRAINING_SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['output']}<|eot_id|>"
    )


def format_inference_prompt(instruction: str) -> str:
    """Format a prompt for inference (no expected output).
    
    Uses the SAME short system prompt that was used during training.
    The enhanced node reference can optionally be prepended to the
    user instruction for extra context.
    """
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{TRAINING_SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path: str) -> Dataset:
    """Load JSONL dataset and format for training."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {path}")

    # Format as text
    formatted = [{"text": format_training_example(ex)} for ex in examples]

    # Show a sample to verify formatting
    sample = formatted[0]["text"]
    print(f"\n--- Sample formatted example (first 300 chars) ---")
    print(sample[:300])
    print("--- End sample ---\n")

    return Dataset.from_list(formatted)


# ============================================================
# MODEL SETUP
# ============================================================

def setup_model_and_tokenizer(config: dict):
    """Load base model with quantization and attach LoRA."""

    print(f"Loading base model: {config['base_model']}")

    # Quantization config
    bnb_config = None
    if config["use_4bit"]:
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config["use_double_quant"],
        )
        print("Using 4-bit quantization (QLoRA)")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Prepare for k-bit training
    if config["use_4bit"]:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Attach LoRA
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, lora_config


# ============================================================
# TRAINING
# ============================================================

def train(config: dict):
    """Run the full training pipeline."""

    train_dataset = load_dataset(config["dataset"])
    val_dataset = None
    val_path = Path(config["dataset"]).with_name("validation.jsonl")
    if val_path.exists():
        val_dataset = load_dataset(str(val_path))

    model, tokenizer, lora_config = setup_model_and_tokenizer(config)

    # Newer trl versions use SFTConfig instead of TrainingArguments.
    # SFTConfig extends TrainingArguments and adds max_seq_length, dataset_text_field, packing.
    use_fp16 = config.get("use_fp16", False)
    use_bf16 = config.get("use_bf16", False)

    try:
        from trl import SFTConfig
        training_args = SFTConfig(
            output_dir=config["output_dir"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            weight_decay=config["weight_decay"],
            max_grad_norm=config["max_grad_norm"],
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=3,
            max_seq_length=config["max_seq_length"],
            dataset_text_field="text",
            packing=False,
        )
        print("Using SFTConfig (newer trl API)")
    except (ImportError, TypeError):
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            weight_decay=config["weight_decay"],
            max_grad_norm=config["max_grad_norm"],
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=3,
        )
        print("Using TrainingArguments (older trl API)")

    # These args vary by library version — set them safely
    try:
        training_args.eval_strategy = "steps" if val_dataset else "no"
        if val_dataset:
            training_args.eval_steps = config["eval_steps"]
            training_args.load_best_model_at_end = True
    except Exception:
        pass  # Older versions may not support these

    # SFTTrainer API has changed significantly across trl versions.
    # Detect which parameters are accepted and adapt accordingly.
    import inspect
    sft_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": training_args,
    }

    # tokenizer vs processing_class
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # max_seq_length (moved to SFTConfig in newer versions)
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = config["max_seq_length"]

    # dataset_text_field
    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"

    # packing
    if "packing" in sft_params:
        trainer_kwargs["packing"] = False

    print(f"SFTTrainer params detected: {[k for k in trainer_kwargs.keys() if k != 'args']}")
    trainer = SFTTrainer(**trainer_kwargs)

    prompt_type = "ENHANCED (with node reference)" if ACTIVE_SYSTEM_PROMPT != BASIC_SYSTEM_PROMPT else "BASIC (no node reference)"
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"  Model: {config['base_model']}")
    print(f"  System prompt: {prompt_type}")
    print(f"  System prompt size: ~{len(ACTIVE_SYSTEM_PROMPT)//4:,} tokens")
    print(f"  Dataset: {len(train_dataset)} examples")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Max sequence length: {config['max_seq_length']}")
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  LoRA rank: {config['lora_r']}")
    print(f"  Output: {config['output_dir']}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    final_path = os.path.join(config["output_dir"], "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModel saved to: {final_path}")

    # Save config for reproducibility
    config["system_prompt_type"] = prompt_type
    config["system_prompt_length"] = len(ACTIVE_SYSTEM_PROMPT)
    config_path = os.path.join(config["output_dir"], "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    # Save the system prompt alongside the model so inference uses the same one
    prompt_save_path = os.path.join(config["output_dir"], "system_prompt.txt")
    with open(prompt_save_path, "w", encoding="utf-8") as f:
        f.write(ACTIVE_SYSTEM_PROMPT)
    print(f"System prompt saved to: {prompt_save_path}")

    return final_path


# ============================================================
# QUICK TEST
# ============================================================

def quick_test(model_path: str, base_model: str):
    """Run a quick inference test on the trained model."""
    from peft import PeftModel

    print("\n--- Quick inference test ---")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, model_path)

    test_prompts = [
        "Create a Blueprint that prints 'Hello' when the game starts.",
        "Create a Blueprint for an actor that rotates continuously.",
        "Create a pickup Blueprint that gives the player 50 health on overlap.",
    ]

    for prompt in test_prompts:
        input_text = format_inference_prompt(prompt)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=512, temperature=0.1,
                do_sample=True, top_p=0.9, repetition_penalty=1.1,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print(f"\nPROMPT: {prompt}")
        print(f"OUTPUT:\n{response[:500]}")
        print("-" * 40)


# ============================================================
# MAIN
# ============================================================

def main():
    global ACTIVE_SYSTEM_PROMPT

    parser = argparse.ArgumentParser(description="Fine-tune LLM for Blueprint generation")
    parser.add_argument("--base_model", type=str, default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--dataset", type=str, default="datasets/train.jsonl")
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--lora_r", type=int, default=DEFAULT_CONFIG["lora_r"])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"])
    parser.add_argument("--basic-prompt", action="store_true",
                        help="Use basic system prompt WITHOUT node reference (not recommended)")
    parser.add_argument("--test", action="store_true", help="Run quick test after training")
    args = parser.parse_args()

    # Select system prompt
    if args.basic_prompt:
        ACTIVE_SYSTEM_PROMPT = BASIC_SYSTEM_PROMPT
        print("Using BASIC system prompt (no node reference)")
    else:
        print("Loading enhanced system prompt with node reference...")
        ACTIVE_SYSTEM_PROMPT = load_enhanced_system_prompt()

    config = {**DEFAULT_CONFIG}
    config.update({
        "base_model": args.base_model,
        "dataset": args.dataset,
        "output_dir": args.output,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "max_seq_length": args.max_seq_length,
    })

    # Verify CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Cannot train. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_arch = torch.cuda.get_device_capability(0)  # e.g. (7, 5) for Turing, (8, 6) for Ampere

    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram:.1f} GB")
    print(f"Compute capability: {gpu_arch[0]}.{gpu_arch[1]}")

    # Determine best precision for this GPU
    # bf16 requires compute capability >= 8.0 (Ampere: RTX 3000+, A100, etc.)
    # fp16 requires compute capability >= 7.0 (Volta/Turing: GTX 1600+, RTX 2000+)
    # Pascal (GTX 1070/1080, compute 6.x) has fp16 but with issues on some ops
    if gpu_arch[0] >= 8:
        use_bf16 = True
        use_fp16 = False
        compute_dtype_str = "bfloat16"
        print(f"  Precision: bf16 (Ampere+ GPU detected)")
    elif gpu_arch[0] >= 7:
        use_bf16 = False
        use_fp16 = True
        compute_dtype_str = "float16"
        print(f"  Precision: fp16 (Turing/Volta GPU detected)")
    else:
        # Pascal (6.x) and older — fp16 grad scaling can fail with bf16 model weights
        use_bf16 = False
        use_fp16 = False
        compute_dtype_str = "float16"
        print(f"  Precision: float32 training (Pascal/older GPU — fp16/bf16 grad scaling unreliable)")

    # Apply precision to config so QLoRA uses the right compute dtype
    config["bnb_4bit_compute_dtype"] = compute_dtype_str
    config["use_bf16"] = use_bf16
    config["use_fp16"] = use_fp16

    # Auto-adjust for low-VRAM GPUs
    if vram < 10:
        print(f"NOTE: {vram:.0f} GB VRAM detected. Adjusting config for low-VRAM GPU.")

        # Switch to a smaller model if using the default 8B
        if config["base_model"] == "meta-llama/Meta-Llama-3.1-8B":
            config["base_model"] = "meta-llama/Llama-3.2-3B"
            print(f"  Switched model to: {config['base_model']} (8B too large for {vram:.0f} GB)")
            print(f"  (Override with --base_model if you want to try a different model)")

        if config["lora_r"] > 32:
            config["lora_r"] = 32
            config["lora_alpha"] = 64
            print(f"  Reduced LoRA rank to {config['lora_r']} (alpha={config['lora_alpha']})")
        if config["max_seq_length"] > 2048:
            config["max_seq_length"] = 2048
            print(f"  Reduced max_seq_length to {config['max_seq_length']}")
        config["gradient_checkpointing"] = True
        print(f"  Gradient checkpointing enabled")

    elif vram < 16:
        # 12 GB range (RTX 4070, RTX 3060 12GB, etc.)
        # Can handle 8B with 4-bit, but needs tight settings
        print(f"NOTE: {vram:.0f} GB VRAM detected. 8B model fits with optimized settings.")
        if config["lora_r"] > 32:
            config["lora_r"] = 32
            config["lora_alpha"] = 64
            print(f"  LoRA rank: {config['lora_r']} (alpha={config['lora_alpha']})")
        if config["max_seq_length"] > 2048:
            config["max_seq_length"] = 2048
            print(f"  Max sequence length: {config['max_seq_length']}")
        config["gradient_checkpointing"] = True
        print(f"  Gradient checkpointing: enabled")
    elif vram >= 16:
        # Plenty of VRAM — can use larger model and full settings
        print(f"NOTE: {vram:.0f} GB VRAM detected. Using full config.")
        # Keep defaults (8B model, LoRA rank 64, seq len 4096)

    model_path = train(config)

    if args.test:
        quick_test(os.path.join(model_path), args.base_model)


if __name__ == "__main__":
    main()
