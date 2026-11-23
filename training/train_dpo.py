#!/usr/bin/env python3
"""QLoRA direct preference optimization on generated invoice preference pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from common import add_common_arguments, initialize_wandb, load_qlora_model


def load_pairs(path: Path, tokenizer: Any) -> Dataset:
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
                [{"role": "user", "content": "Extract invoice fields from this source:\n" + source}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not rows:
        raise ValueError("Preference dataset is empty.")
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    run = initialize_wandb(args, "dpo")
    model, tokenizer, use_bf16 = load_qlora_model(
        args.base_model, args.rank, args.alpha
    )
    dataset = load_pairs(Path(args.dataset), tokenizer)
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
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="wandb",
        run_name=run.name,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
