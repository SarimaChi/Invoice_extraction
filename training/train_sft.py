#!/usr/bin/env python3
"""QLoRA supervised fine-tuning for accepted invoice extraction pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from common import add_common_arguments, initialize_wandb, load_qlora_model


def load_examples(path: Path, tokenizer: Any) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            record = json.loads(line)
            source = record.get("source")
            completion = record.get("completion")
            if not isinstance(source, str) or not isinstance(completion, dict):
                raise ValueError(f"line {line_number} must have source and completion")
            conversation = [
                {"role": "user", "content": "Extract invoice fields from this source:\n" + source},
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
    if not rows:
        raise ValueError("Training dataset is empty.")
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    run = initialize_wandb(args, "sft")
    model, tokenizer, use_bf16 = load_qlora_model(
        args.base_model, args.rank, args.alpha
    )
    dataset = load_examples(Path(args.dataset), tokenizer)
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
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="wandb",
        run_name=run.name,
        dataset_text_field="text",
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
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
