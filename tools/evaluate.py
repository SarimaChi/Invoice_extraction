#!/usr/bin/env python3
"""Evaluate invoice JSONL predictions for field F1, validity, and hallucinations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import FIELDS, is_source_grounded  # noqa: E402


def load_prediction(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    return value


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    non_null_predictions = hallucinations = valid_json = 0
    field_correct = field_total = 0

    for record in records:
        source = record.get("source", "")
        gold = record.get("gold", {})
        prediction = load_prediction(record.get("prediction"))
        if prediction is not None:
            valid_json += 1

        for field in FIELDS:
            expected = gold.get(field) if isinstance(gold, dict) else None
            actual = prediction.get(field) if prediction is not None else None
            expected_present = expected is not None and expected != ""
            actual_present = actual is not None and actual != ""
            field_total += 1
            field_correct += int(expected == actual)
            true_positive += int(expected_present and actual_present and expected == actual)
            false_positive += int(actual_present and (not expected_present or expected != actual))
            false_negative += int(expected_present and (not actual_present or expected != actual))
            if actual_present:
                non_null_predictions += 1
                if not isinstance(actual, str) or not isinstance(source, str) \
                        or not is_source_grounded(actual, source):
                    hallucinations += 1

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count = len(records)
    return {
        "documents": count,
        "field_level": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_accuracy": field_correct / field_total if field_total else 0.0,
            "fields": field_total,
        },
        "json_validity": valid_json / count if count else 0.0,
        "hallucination_rate": hallucinations / non_null_predictions if non_null_predictions else 0.0,
        "hallucinated_values": hallucinations,
        "non_null_predictions": non_null_predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSONL with source, gold, prediction")
    parser.add_argument("--label", default="model", help="Model/version name in output")
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    with args.predictions.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                parser.error(f"line {line_number} is not valid JSON: {error.msg}")
            if not isinstance(record, dict):
                parser.error(f"line {line_number} must be a JSON object")
            records.append(record)
    print(json.dumps({"model": args.label, **evaluate(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
