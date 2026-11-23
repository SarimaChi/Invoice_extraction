#!/usr/bin/env python3
"""Filter synthetic extraction pairs for parseability, schema, and source support."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import FIELDS, is_source_grounded  # noqa: E402


def parse_completion(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("completion must be a JSON object")
    if set(value) != set(FIELDS):
        raise ValueError("completion must contain exactly the invoice schema fields")
    for field, item in value.items():
        if item is not None and not isinstance(item, str):
            raise ValueError(f"{field} must be a string or null")
    return value


def filter_record(record: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(record, dict):
        return None, "record must be a JSON object"
    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        return None, "source must be non-empty text"
    try:
        completion = parse_completion(record.get("completion"))
    except (ValueError, json.JSONDecodeError) as error:
        return None, str(error)
    for field, value in completion.items():
        if value is not None and value and not is_source_grounded(value, source):
            return None, f"{field} is not present in the source"
    return {"source": source, "completion": completion}, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Synthetic JSONL input")
    parser.add_argument("--accepted", type=Path, required=True, help="Accepted JSONL output")
    parser.add_argument("--rejected", type=Path, required=True, help="Rejected JSONL output")
    args = parser.parse_args()

    kept = dropped = 0
    with args.input.open(encoding="utf-8") as source_file, \
            args.accepted.open("w", encoding="utf-8") as accepted_file, \
            args.rejected.open("w", encoding="utf-8") as rejected_file:
        for line_number, line in enumerate(source_file, 1):
            try:
                record = json.loads(line)
                clean, reason = filter_record(record)
            except json.JSONDecodeError as error:
                clean, reason = None, f"invalid JSON: {error.msg}"
            if clean is not None:
                accepted_file.write(json.dumps(clean, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected_file.write(json.dumps({
                    "line": line_number,
                    "reason": reason,
                    "record": line.rstrip("\n"),
                }, ensure_ascii=False) + "\n")
                dropped += 1
    print(f"Accepted {kept}; rejected {dropped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
