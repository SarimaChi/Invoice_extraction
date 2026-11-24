import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import (
    FIELDS,
    ExtractionError,
    InputError,
    _detect_document_format,
    _parse_extract_request,
    extract_document_text,
    is_source_grounded,
    validate_extraction,
)
from tools.build_dpo_pairs import create_pair
from tools.benchmark_models import benchmark_api
from tools.evaluate import evaluate
from tools.filter_synthetic import filter_record
from tools.prepare_invoice_dataset import prepare_dataset
from tools.serve_model import ModelServer
from training.common import latest_evaluation_metrics


class GroundingTests(unittest.TestCase):
    def test_numeric_formatting_is_grounded_but_unseen_values_are_not(self):
        source = "Total due: $1,240.00"
        self.assertTrue(is_source_grounded("1240.00", source))
        self.assertFalse(is_source_grounded("1420.00", source))
        self.assertFalse(is_source_grounded("12", "Invoice year: 2024"))

    def test_numeric_and_date_format_variants_are_grounded(self):
        self.assertTrue(is_source_grounded("75974.0", "TOTAL: 75 974,00 EUR"))
        self.assertTrue(is_source_grounded("2015-02-02", "Date: 02/02/2015"))

    def test_unseen_values_are_withheld(self):
        result = validate_extraction({"total": "999.00"}, "Total: 125.00")
        self.assertIsNone(result["fields"]["total"])
        self.assertEqual(result["rejected"][0]["field"], "total")

    def test_unknown_model_keys_fail_closed(self):
        with self.assertRaises(ExtractionError):
            validate_extraction({"made_up": "value"}, "value")

    def test_file_type_uses_extension_and_signature(self):
        self.assertEqual(
            _detect_document_format("invoice.PNG", b"\x89PNG\r\n\x1a\nimage"),
            ".png",
        )
        with self.assertRaises(InputError):
            _detect_document_format("invoice.png", b"%PDF-1.7")
        with self.assertRaises(InputError):
            _detect_document_format("invoice.docx", b"not an invoice")

    def test_multipart_parser_returns_uploaded_file(self):
        boundary = "upload-boundary"
        image = b"\x89PNG\r\n\x1a\nfake"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="invoice.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
        source, uploaded_file = _parse_extract_request(
            f"multipart/form-data; boundary={boundary}", body
        )
        self.assertIsNone(source)
        self.assertEqual(uploaded_file, ("invoice.png", image))

    def test_multipart_parser_rejects_text_and_file_together(self):
        boundary = "upload-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="source"\r\n\r\n'
            "invoice text\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="invoice.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
            "\x89PNG\r\n\x1a\nfake\r\n"
            f"--{boundary}--\r\n"
        ).encode("latin-1")
        with self.assertRaises(InputError):
            _parse_extract_request(f"multipart/form-data; boundary={boundary}", body)

    def test_searchable_pdf_text_is_extracted_without_ocr(self):
        import fitz

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Invoice No: INV-42\nTotal: 123.45", fontsize=18)
        text, used_ocr = extract_document_text("invoice.pdf", document.tobytes())
        self.assertIn("INV-42", text)
        self.assertIn("123.45", text)
        self.assertFalse(used_ocr)
        document.close()

    def test_dataset_preparation_parses_comma_delimited_tsv_and_splits_by_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            output = Path(temp_dir) / "processed"
            for index in range(1, 4):
                family = root / f"invoice_dataset_model_{index}"
                (family / "xml").mkdir(parents=True)
                (family / "annotations").mkdir()
                (family / "images").mkdir()
                xml = (
                    "<document>"
                    f"<supplier>Vendor {index}</supplier>"
                    f"<invoice_number>INV-{index}</invoice_number>"
                    "<invoice_date>2024-01-02</invoice_date>"
                    "<total_amount>100.00</total_amount>"
                    "</document>"
                )
                (family / "xml" / f"doc{index}.xml").write_text(xml, encoding="utf-8")
                with (family / "annotations" / f"doc{index}.tsv").open(
                    "w", encoding="utf-8", newline=""
                ) as annotation_file:
                    writer = csv.writer(annotation_file)
                    writer.writerow(["left", "top", "width", "height", "text", "label"])
                    writer.writerow([0, 0, 40, 12, "Invoice", "O"])
                    writer.writerow([45, 0, 60, 12, f"INV-{index}", "NUMBER"])
                    writer.writerow([0, 30, 55, 12, "Supplier", "O"])
                    writer.writerow([60, 30, 55, 12, f"Vendor {index}", "SUPPLIER"])
                    writer.writerow([0, 60, 40, 12, "Date", "O"])
                    writer.writerow([45, 60, 80, 12, "2024-01-02", "INVOICE_DATE"])
                    writer.writerow([0, 90, 50, 12, "Total", "O"])
                    writer.writerow([55, 90, 60, 12, "$100.00", "TOTAL_AMOUNT"])
                (family / "images" / f"doc{index}.jpg").write_bytes(b"fixture")

            report = prepare_dataset(root, output)
            self.assertEqual(report["documents"], 3)
            self.assertEqual(report["split_documents"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(
                report["template_field_quality"]["invoice_dataset_model_1"]["fields"]
                ["invoice_number"]["grounded_in_source_text"],
                1,
            )
            train_record = json.loads((output / "train.jsonl").read_text().splitlines()[0])
            self.assertTrue(
                is_source_grounded(
                    train_record["completion"]["supplier"], train_record["source"]
                )
            )
            self.assertEqual(report["identical_source_texts_across_splits"], 0)

    def test_missing_fields_are_null(self):
        result = validate_extraction({"total": "125.00"}, "Total: 125.00")
        self.assertEqual(set(result["fields"]), set(FIELDS))
        self.assertIsNone(result["fields"]["supplier"])


class DataToolTests(unittest.TestCase):
    def test_latest_training_evaluation_metrics_come_from_final_eval_log(self):
        class FakeTrainer:
            class State:
                log_history = [
                    {"eval_loss": 2.0, "step": 1},
                    {"loss": 1.0, "step": 2},
                    {"eval_loss": 1.5, "eval_accuracy": 0.8, "step": 2},
                ]

            state = State()

        self.assertEqual(
            latest_evaluation_metrics(FakeTrainer()),
            {"eval_loss": 1.5, "eval_accuracy": 0.8},
        )

    def test_api_reference_benchmark_records_metrics_without_network(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"completion_tokens": 2},
                }).encode()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tools.benchmark_models.urllib.request.urlopen", return_value=FakeResponse()):
                metrics = benchmark_api(
                    "reference",
                    [{"source": "invoice", "gold": {}}],
                    "http://localhost/v1/chat/completions",
                    "test-model",
                    None,
                    32,
                    Path(temp_dir),
                )
            self.assertEqual(metrics["json_validity"], 0.0)
            self.assertEqual(metrics["generated_tokens_per_second"] > 0, True)
            self.assertTrue(metrics["reference_endpoint_configured"])

    def test_synthetic_filter_rejects_unsupported_field(self):
        completion = {field: None for field in FIELDS}
        completion["supplier"] = "Invented Ltd"
        clean, reason = filter_record({
            "source": "Invoice total: 40.00",
            "completion": completion,
        })
        self.assertIsNone(clean)
        self.assertIn("supplier", reason)

    def test_dpo_pair_corrupts_one_present_field(self):
        completion = {field: None for field in FIELDS}
        completion["total"] = "40.00"
        pair = create_pair(
            {"source": "Invoice total: 40.00", "completion": completion},
            "hallucinated",
            0,
        )
        self.assertEqual(pair["corrupted_field"], "total")
        self.assertNotEqual(pair["chosen"], pair["rejected"])
        self.assertIn("hallucinated", pair["corruption"])

    def test_evaluator_counts_invalid_json_and_hallucination(self):
        prediction = {field: None for field in FIELDS}
        prediction["total"] = "999.00"
        record = {
            "source": "Total 100.00",
            "gold": {"total": "100.00"},
            "prediction": json.dumps(prediction),
        }
        metrics = evaluate([record])
        self.assertEqual(metrics["json_validity"], 1.0)
        self.assertEqual(metrics["hallucinated_values"], 1)
        self.assertEqual(metrics["field_level"]["f1"], 0.0)
        record["prediction"] = "{not json}"
        self.assertEqual(evaluate([record])["json_validity"], 0.0)

    def test_evaluator_uses_numeric_equivalence_and_requires_schema(self):
        gold = {field: None for field in FIELDS}
        gold["total"] = "75974.0"
        prediction = {field: None for field in FIELDS}
        prediction["total"] = "75 974,00"
        metrics = evaluate([{
            "source": "Total 75 974,00 EUR",
            "gold": gold,
            "prediction": prediction,
        }])
        self.assertEqual(metrics["field_level"]["f1"], 1.0)
        self.assertEqual(metrics["field_level"]["exact_accuracy"], 8 / 9)
        self.assertEqual(metrics["field_level"]["normalized_accuracy"], 1.0)
        self.assertEqual(metrics["hallucinated_values"], 0)
        prediction["extra"] = "invented"
        schema_metrics = evaluate([{
            "source": "Total 75 974,00 EUR",
            "gold": gold,
            "prediction": prediction,
        }])
        self.assertEqual(schema_metrics["json_syntax_validity"], 1.0)
        self.assertEqual(schema_metrics["json_validity"], 0.0)
        self.assertEqual(schema_metrics["hallucinated_values"], 1)

    def test_local_model_server_returns_openai_compatible_completion(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, prompt):
                return {"input_ids": [1, 2, 3]}

        model_server = ModelServer(object(), FakeTokenizer(), "demo", "mps", 64)
        with patch("tools.serve_model.generate", return_value=("{}", 0.25, 2)):
            response = model_server.complete(
                [{"role": "user", "content": "Invoice source"}], 8
            )
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "{}")
        self.assertEqual(response["usage"]["prompt_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
