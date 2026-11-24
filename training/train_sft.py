#!/usr/bin/env python3
"""LoRA supervised fine-tuning for accepted invoice extraction pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from common import (
    EXTRACTION_SYSTEM_PROMPT,
    add_common_arguments,
    initialize_wandb,
    latest_evaluation_metrics,
    load_lora_model,
)


def load_examples(path: Path, tokenizer: Any, max_samples: int | None = None) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            record = json.loads(line)
            source = record.get("source")
            completion = record.get("completion")
            if not isinstance(source, str) or not isinstance(completion, dict):
                raise ValueError(f"line {line_number} must have source and completion")
            conversation = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": source},
                {
                    "role": "assistant",
                    "content": json.dumps(completion, ensure_ascii=False, sort_keys=True),
                },
            ]
            rows.append({
                "text": tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            })
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError("Training dataset is empty.")
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    run = initialize_wandb(args, "sft")
    try:
        model, tokenizer, use_bf16, use_cuda = load_lora_model(
            args.base_model, args.rank, args.alpha
        )
        dataset_path = Path(args.dataset)
        dataset = load_examples(dataset_path, tokenizer, args.max_samples)
        validation_path = dataset_path.with_name("valid.jsonl")
        eval_dataset = (
            load_examples(validation_path, tokenizer, args.max_samples)
            if validation_path.is_file() else None
        )
        training_args = SFTConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
            bf16=use_bf16,
            fp16=use_cuda and not use_bf16,
            report_to="wandb" if run else "none",
            run_name=run.name if run else None,
            dataset_text_field="text",
            remove_unused_columns=True,
        )
        trainer = SFTTrainer(
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
            "stage": "sft",
            "base_model": args.base_model,
            "dataset": str(dataset_path),
            "train_samples": len(dataset),
            "validation_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "rank": args.rank,
            "alpha": args.alpha,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
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
