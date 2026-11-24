#!/usr/bin/env python3
"""LoRA direct preference optimization on generated invoice preference pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from common import (
    EXTRACTION_SYSTEM_PROMPT,
    add_common_arguments,
    initialize_wandb,
    latest_evaluation_metrics,
    load_lora_model,
)


def load_pairs(
    path: Path,
    tokenizer: Any,
    max_samples: int | None = None,
) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            record = json.loads(line)
            source = record.get("source")
            chosen = record.get("chosen")
            rejected = record.get("rejected")
            if not all(isinstance(value, str) and value for value in (source, chosen, rejected)):
                raise ValueError(f"line {line_number} must have source, chosen, and rejected text")
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": source},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError("Preference dataset is empty.")
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.set_defaults(learning_rate=5e-5)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    run = initialize_wandb(args, "dpo")
    try:
        model, tokenizer, use_bf16, use_cuda = load_lora_model(
            args.base_model,
            args.rank,
            args.alpha,
            adapter_path=args.sft_adapter,
        )
        dataset_path = Path(args.dataset)
        dataset = load_pairs(dataset_path, tokenizer, args.max_samples)
        validation_path = dataset_path.with_name("dpo_valid.jsonl")
        eval_dataset = (
            load_pairs(validation_path, tokenizer, args.max_samples)
            if validation_path.is_file() else None
        )
        training_args = DPOConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_seq_length,
            max_prompt_length=min(1024, args.max_seq_length - 1),
            beta=args.beta,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
            bf16=use_bf16,
            fp16=use_cuda and not use_bf16,
            precompute_ref_log_probs=True,
            report_to="wandb" if run else "none",
            run_name=run.name if run else None,
            remove_unused_columns=False,
        )
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        train_metrics = trainer.train().metrics
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        trainer.save_state()
        evaluation_metrics = latest_evaluation_metrics(trainer)
        report = {
            "stage": "dpo",
            "base_model": args.base_model,
            "sft_adapter": args.sft_adapter,
            "dataset": str(dataset_path),
            "train_samples": len(dataset),
            "validation_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "rank": args.rank,
            "alpha": args.alpha,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "beta": args.beta,
            "device": "cuda" if use_cuda else "mps",
            "quantization": "4-bit NF4 QLoRA" if use_cuda else "none (MPS LoRA)",
            "train_metrics": train_metrics,
            "validation_metrics": evaluation_metrics,
        }
        Path(args.output_dir, "training_metrics.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    finally:
        if run:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
