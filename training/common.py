from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import EXTRACTION_SYSTEM_PROMPT  # noqa: E402

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "add_common_arguments",
    "initialize_wandb",
    "latest_evaluation_metrics",
    "load_lora_model",
]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, help="Validated JSONL training data")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--wandb-project")


def load_lora_model(
    base_model: str,
    rank: int,
    alpha: int,
    adapter_path: str | None = None,
) -> tuple[Any, Any, bool, bool]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not use_cuda and not use_mps:
        raise RuntimeError("Training requires either CUDA or Apple Silicon MPS.")

    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    compute_dtype = (
        torch.bfloat16 if use_bf16
        else torch.float16 if use_cuda or use_mps
        else torch.float32
    )
    model_options: dict[str, Any] = {"torch_dtype": compute_dtype}
    if use_cuda:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_options["device_map"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_options)
    if use_mps:
        model.to("mps")
    model.config.use_cache = False
    if use_cuda:
        model = prepare_model_for_kbit_training(model)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora)
    return model, tokenizer, use_bf16, use_cuda


def initialize_wandb(args: argparse.Namespace, stage: str) -> Any:
    if not args.wandb_project:
        return None
    import wandb

    run_name = (
        f"{stage}-r{args.rank}-a{args.alpha}-lr{args.learning_rate:g}"
    )
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )


def latest_evaluation_metrics(trainer: Any) -> dict[str, Any]:
    for entry in reversed(trainer.state.log_history):
        metrics = {
            key: value for key, value in entry.items()
            if key.startswith("eval_")
        }
        if metrics:
            return metrics
    return {}
