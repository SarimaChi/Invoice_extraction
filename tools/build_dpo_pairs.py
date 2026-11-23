#!/usr/bin/env python3
"""Create controlled DPO negatives from source-grounded invoice annotations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import FIELDS, is_source_grounded  # noqa: E402


def create_pair(record: Any, strategy: str, record_index: int) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    source = record.get("source")
    chosen = record.get("completion")
    if not isinstance(source, str) or not source.strip() or not isinstance(chosen, dict):
        raise ValueError("record must contain non-empty source and completion object")
    if set(chosen) != set(FIELDS):
        raise ValueError("completion must contain exactly the invoice schema fields")
    for field, value in chosen.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
        if value and not is_source_grounded(value, source):
            raise ValueError(f"{field} is not present in the source")
    populated = [
        field for field in FIELDS
        if isinstance(chosen[field], str) and chosen[field]
    ]
    if not populated:
        return None

    field = populated[record_index % len(populated)]
    rejected = dict(chosen)
    if strategy == "missing" or (strategy == "mixed" and record_index % 2 == 0):
        rejected[field] = None
    else:
        hallucination = f"UNSUPPORTED_VALUE_{field.upper()}"
        suffix = 1
        while is_source_grounded(hallucination, source):
            hallucination = f"UNSUPPORTED_VALUE_{field.upper()}_{suffix}"
            suffix += 1
        rejected[field] = hallucination

    return {
        "source": source,
        "chosen": json.dumps(chosen, ensure_ascii=False, sort_keys=True),
        "rejected": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
        "corrupted_field": field,
        "corruption": "missing" if rejected[field] is None else "hallucinated",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Accepted training JSONL")
    parser.add_argument("--output", type=Path, required=True, help="DPO pairs JSONL")
    parser.add_argument(
        "--strategy", choices=("missing", "hallucinated", "mixed"), default="mixed"
    )
    args = parser.parse_args()

    written = skipped = 0
    with args.input.open(encoding="utf-8") as input_file, \
            args.output.open("w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(input_file):
            try:
                record = json.loads(line)
                pair = create_pair(record, args.strategy, line_number)
            except (json.JSONDecodeError, ValueError) as error:
                parser.error(f"line {line_number + 1}: {error}")
            if pair is None:
                skipped += 1
                continue
            output_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} pairs; skipped {skipped} empty annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
