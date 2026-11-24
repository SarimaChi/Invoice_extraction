#!/usr/bin/env python3
"""Evaluate invoice JSONL predictions for field F1, validity, and hallucinations."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import FIELDS, is_source_grounded, values_equivalent  # noqa: E402


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
    non_null_predictions = hallucinations = syntax_valid = schema_valid = 0
    exact_correct = normalized_correct = field_total = 0
    per_field: dict[str, Counter[str]] = {
        field: Counter() for field in FIELDS
    }

    for record in records:
        source = record.get("source", "")
        gold = record.get("gold", {})
        prediction = load_prediction(record.get("prediction"))
        if prediction is not None:
            syntax_valid += 1
        is_schema_valid = (
            prediction is not None
            and set(prediction) == set(FIELDS)
            and all(value is None or isinstance(value, str) for value in prediction.values())
        )
        schema_valid += int(is_schema_valid)

        for field in FIELDS:
            expected = gold.get(field) if isinstance(gold, dict) else None
            actual = prediction.get(field) if prediction is not None else None
            expected_present = expected is not None and expected != ""
            actual_present = actual is not None and actual != ""
            field_total += 1
            exact = actual == expected
            equivalent = values_equivalent(field, actual, expected)
            exact_correct += int(exact)
            normalized_correct += int(equivalent)
            true_positive += int(expected_present and actual_present and equivalent)
            false_positive += int(actual_present and (not expected_present or not equivalent))
            false_negative += int(expected_present and (not actual_present or not equivalent))
            per_field[field]["tp"] += int(expected_present and actual_present and equivalent)
            per_field[field]["fp"] += int(actual_present and (not expected_present or not equivalent))
            per_field[field]["fn"] += int(expected_present and (not actual_present or not equivalent))
            per_field[field]["exact_correct"] += int(exact)
            per_field[field]["count"] += 1
            if actual_present:
                non_null_predictions += 1
                if not isinstance(actual, str) or not isinstance(source, str) \
                        or not is_source_grounded(actual, source):
                    hallucinations += 1

        if prediction is not None:
            for field, actual in prediction.items():
                if field not in FIELDS and actual is not None and actual != "":
                    non_null_predictions += 1
                    if not isinstance(actual, str) or not isinstance(source, str) \
                            or not is_source_grounded(actual, source):
                        hallucinations += 1

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count = len(records)
    by_field = {}
    for field, metrics in per_field.items():
        field_precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) \
            if metrics["tp"] + metrics["fp"] else 0.0
        field_recall = metrics["tp"] / (metrics["tp"] + metrics["fn"]) \
            if metrics["tp"] + metrics["fn"] else 0.0
        by_field[field] = {
            "precision": field_precision,
            "recall": field_recall,
            "f1": (
                2 * field_precision * field_recall / (field_precision + field_recall)
                if field_precision + field_recall else 0.0
            ),
            "exact_accuracy": metrics["exact_correct"] / metrics["count"]
            if metrics["count"] else 0.0,
            "support": metrics["tp"] + metrics["fn"],
        }
    return {
        "documents": count,
        "field_level": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_accuracy": exact_correct / field_total if field_total else 0.0,
            "normalized_accuracy": normalized_correct / field_total if field_total else 0.0,
            "fields": field_total,
        },
        "by_field": by_field,
        "json_validity": schema_valid / count if count else 0.0,
        "json_syntax_validity": syntax_valid / count if count else 0.0,
        "schema_validity": schema_valid / count if count else 0.0,
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
