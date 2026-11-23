import json
import tempfile
import unittest
from pathlib import Path

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
from tools.evaluate import evaluate
from tools.filter_synthetic import filter_record


class GroundingTests(unittest.TestCase):
    def test_numeric_formatting_is_grounded_but_unseen_values_are_not(self):
        source = "Total due: $1,240.00"
        self.assertTrue(is_source_grounded("1240.00", source))
        self.assertFalse(is_source_grounded("1420.00", source))

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

    def test_missing_fields_are_null(self):
        result = validate_extraction({"total": "125.00"}, "Total: 125.00")
        self.assertEqual(set(result["fields"]), set(FIELDS))
        self.assertIsNone(result["fields"]["supplier"])


class DataToolTests(unittest.TestCase):
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
        record = {
            "source": "Total 100.00",
            "gold": {"total": "100.00"},
            "prediction": json.dumps({"total": "999.00"}),
        }
        metrics = evaluate([record])
        self.assertEqual(metrics["json_validity"], 1.0)
        self.assertEqual(metrics["hallucinated_values"], 1)
        self.assertEqual(metrics["field_level"]["f1"], 0.0)
        record["prediction"] = "{not json}"
        self.assertEqual(evaluate([record])["json_validity"], 0.0)


if __name__ == "__main__":
    unittest.main()
