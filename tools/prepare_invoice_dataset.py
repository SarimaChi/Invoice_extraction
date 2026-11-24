#!/usr/bin/env python3
"""Convert the authorized invoice image/XML/word-box corpus into leak-safe JSONL."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ExtractionError, FIELDS, extract_document_text, is_source_grounded


XML_FIELDS = {
    "invoice_number": "invoice_number",
    "supplier": "supplier",
    "customer": "customer",
    "issue_date": "invoice_date",
    "due_date": "invoice_date_due",
    "currency": "currency_id",
    "subtotal": "total_untaxed",
    "tax": "tax_amount",
    "total": "total_amount",
}


def read_annotation_text(path: Path) -> str:
    with path.open(encoding="utf-8-sig", newline="") as annotation_file:
        header = annotation_file.readline()
        if not header:
            raise ValueError(f"{path} is empty")
        delimiter = "\t" if header.count("\t") >= header.count(",") else ","
        annotation_file.seek(0)
        reader = csv.DictReader(annotation_file, delimiter=delimiter)
        required = {"left", "top", "height", "text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing word-box columns: {sorted(required)}")

        words: list[tuple[float, float, float, str]] = []
        for line_number, row in enumerate(reader, 2):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                left = float(row["left"])
                top = float(row["top"])
                height = float(row["height"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number} has invalid word coordinates") from error
            words.append((top + height / 2, left, height, text))

    if not words:
        return ""
    words.sort(key=lambda word: (word[0], word[1]))
    tolerance = max(4.0, statistics.median(word[2] for word in words) * 0.6)
    lines: list[list[tuple[float, float, float, str]]] = []
    for word in words:
        if lines:
            current_center = statistics.median(item[0] for item in lines[-1])
            if abs(word[0] - current_center) <= tolerance:
                lines[-1].append(word)
                continue
        lines.append([word])
    return "\n".join(
        " ".join(word[3] for word in sorted(line, key=lambda item: item[1]))
        for line in lines
    )


def read_gold(path: Path) -> dict[str, str | None]:
    root = ET.parse(path).getroot()
    values: dict[str, str | None] = {}
    for field in FIELDS:
        xml_field = XML_FIELDS[field]
        value = (root.findtext(xml_field) or "").strip()
        values[field] = value or None
    return values


def assign_splits(families: list[str]) -> dict[str, str]:
    ordered = sorted(set(families))
    if len(ordered) < 3:
        raise ValueError("At least three invoice template families are required.")
    test_count = max(1, round(len(ordered) * 0.22))
    valid_count = max(1, round(len(ordered) * 0.11))
    train_count = len(ordered) - test_count - valid_count
    if train_count < 1:
        raise ValueError("Could not allocate at least one family to each split.")
    return {
        **{family: "train" for family in ordered[:train_count]},
        **{family: "valid" for family in ordered[train_count:train_count + valid_count]},
        **{family: "test" for family in ordered[train_count + valid_count:]},
    }


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    min_core_support: float = 0.5,
    source_mode: str = "annotations",
) -> dict[str, Any]:
    if not 0 <= min_core_support <= 1:
        raise ValueError("min_core_support must be between 0 and 1.")
    if source_mode not in ("annotations", "tesseract"):
        raise ValueError("source_mode must be 'annotations' or 'tesseract'.")
    xml_paths = sorted(source_dir.glob("invoice_dataset_model_*/xml/*.xml"))
    if not xml_paths:
        raise ValueError(f"No invoice XML files found below {source_dir}")

    raw_records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    ocr_failures: list[dict[str, str]] = []
    ocr_documents = 0
    for xml_path in xml_paths:
        family = xml_path.parent.parent.name
        annotation_path = xml_path.parent.parent / "annotations" / (xml_path.stem + ".tsv")
        image_path = xml_path.parent.parent / "images" / (xml_path.stem + ".jpg")
        if not annotation_path.is_file() or not image_path.is_file():
            skipped["missing_image_or_annotation"] += 1
            continue
        if source_mode == "tesseract":
            try:
                source, _ = extract_document_text(image_path.name, image_path.read_bytes())
                ocr_documents += 1
            except ExtractionError as error:
                skipped["tesseract_failure"] += 1
                ocr_failures.append({
                    "document_id": f"{family}/{xml_path.stem}",
                    "reason": str(error)[:200],
                })
                continue
        else:
            source = read_annotation_text(annotation_path)
        if not source:
            skipped["empty_annotation_text"] += 1
            continue
        raw_gold = read_gold(xml_path)
        supported_gold = {
            field: value if value and is_source_grounded(value, source) else None
            for field, value in raw_gold.items()
        }
        raw_records.append({
            "document_id": f"{family}/{xml_path.stem}",
            "template_family": family,
            "source": source,
            "raw_gold": raw_gold,
            "gold": supported_gold,
        })

    core_fields = ("invoice_number", "supplier", "total")
    family_quality: dict[str, dict[str, Any]] = {}
    for family in sorted({record["template_family"] for record in raw_records}):
        family_records = [
            record for record in raw_records if record["template_family"] == family
        ]
        field_quality = {}
        for field in FIELDS:
            labeled = sum(record["raw_gold"][field] is not None for record in family_records)
            grounded = sum(record["gold"][field] is not None for record in family_records)
            field_quality[field] = {
                "non_null_xml_labels": labeled,
                "grounded_in_source_text": grounded,
                "support_rate": grounded / labeled if labeled else None,
            }
        family_quality[family] = {
            "documents": len(family_records),
            "fields": field_quality,
        }
    excluded_families = [
        family for family, quality in family_quality.items()
        if any(
            (quality["fields"][field]["support_rate"] or 0) < min_core_support
            for field in core_fields
        )
    ]
    included_families = [
        family for family in family_quality if family not in excluded_families
    ]
    split_by_family = assign_splits(included_families)
    records = [
        {
            "document_id": record["document_id"],
            "template_family": record["template_family"],
            "split": split_by_family[record["template_family"]],
            "source": record["source"],
            "gold": record["gold"],
            "completion": record["gold"],
        }
        for record in raw_records
        if record["template_family"] in split_by_family
    ]

    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        digest = hashlib.sha256(_normalize_for_hash(record["source"]).encode("utf-8")).hexdigest()
        source_splits[digest].add(record["split"])
    leaking_sources = sum(len(splits) > 1 for splits in source_splits.values())
    if leaking_sources:
        raise ValueError(
            f"Found {leaking_sources} identical source texts across dataset splits; "
            "merge those template families or remove duplicates before training."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, list[dict[str, Any]]] = {
        name: [record for record in records if record["split"] == name]
        for name in ("train", "valid", "test")
    }
    for split, split_items in split_records.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as output_file:
            for item in split_items:
                output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    support_counts = Counter(
        field for record in raw_records
        if record["template_family"] in split_by_family
        for field, value in record["gold"].items() if value
    )
    gold_counts = Counter(
        field for record in raw_records
        if record["template_family"] in split_by_family
        for field, value in record["raw_gold"].items() if value
    )
    family_splits = {
        split: sorted(family for family, assigned in split_by_family.items() if assigned == split)
        for split in ("train", "valid", "test")
    }
    report = {
        "source_directory": str(source_dir),
        "source_mode": source_mode,
        "rights_note": (
            "Use is based on the user's confirmation of permission; the source repository "
            "does not declare a license. Do not redistribute source documents."
        ),
        "documents": len(records),
        "raw_documents": len(raw_records),
        "excluded_documents": len(raw_records) - len(records),
        "skipped": dict(skipped),
        "ocr_documents": ocr_documents,
        "ocr_failures": ocr_failures,
        "min_core_support": min_core_support,
        "core_fields_for_template_quality": list(core_fields),
        "excluded_template_families": {
            family: family_quality[family] for family in excluded_families
        },
        "template_field_quality": family_quality,
        "split_documents": {split: len(items) for split, items in split_records.items()},
        "template_families": family_splits,
        "field_support": {
            field: {
                "non_null_xml_labels": gold_counts[field],
                "grounded_in_source_text": support_counts[field],
                "unsupported": gold_counts[field] - support_counts[field],
            }
            for field in FIELDS
        },
        "identical_source_texts_across_splits": leaking_sources,
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _normalize_for_hash(value: str) -> str:
    return " ".join(value.casefold().split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/invoice_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/invoice_dataset"))
    parser.add_argument(
        "--source-mode",
        choices=("tesseract", "annotations"),
        default="tesseract",
        help="Use the same OCR pipeline as the app or the dataset word-box transcription.",
    )
    parser.add_argument("--min-core-support", type=float, default=0.5)
    args = parser.parse_args()
    report = prepare_dataset(
        args.source_dir,
        args.output_dir,
        min_core_support=args.min_core_support,
        source_mode=args.source_mode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
