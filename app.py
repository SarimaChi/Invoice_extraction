from __future__ import annotations

import json
import os
import re
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
MAX_REQUEST_BYTES = 10_100_000
MAX_JSON_REQUEST_BYTES = 1_000_000
MAX_UPLOAD_BYTES = 10_000_000
MAX_IMAGE_PIXELS = 25_000_000
MAX_PDF_PAGES = 20
MAX_SOURCE_CHARACTERS = 100_000
FIELDS = (
    "invoice_number",
    "supplier",
    "customer",
    "issue_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
)


class ExtractionError(Exception):
    pass


class InputError(ExtractionError):
    pass


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def is_source_grounded(value: str, source: str) -> bool:
    normalized_value = _normalize(value)
    return bool(normalized_value) and normalized_value in _normalize(source)


def _detect_document_format(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    signatures = {
        ".pdf": content.startswith(b"%PDF-"),
        ".png": content.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": content.startswith(b"\xff\xd8\xff"),
        ".jpeg": content.startswith(b"\xff\xd8\xff"),
    }
    if suffix not in signatures:
        raise InputError("Unsupported file type. Upload a PDF, PNG, JPG, or JPEG.")
    if not signatures[suffix]:
        raise InputError("The uploaded file content does not match its file extension.")
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise InputError("Files must be between 1 byte and 10 MB.")
    return suffix


def _ocr_image(image: Any) -> str:
    try:
        import pytesseract
    except ImportError as error:
        raise ExtractionError(
            "Image OCR dependencies are missing. Install requirements.txt and Tesseract."
        ) from error
    try:
        return pytesseract.image_to_string(image, timeout=30)
    except pytesseract.TesseractNotFoundError as error:
        raise ExtractionError(
            "Tesseract OCR is not installed. Install it to read image and scanned PDF files."
        ) from error
    except pytesseract.TesseractError as error:
        raise ExtractionError(f"OCR failed: {error}") from error


def extract_document_text(filename: str, content: bytes) -> tuple[str, bool]:
    file_format = _detect_document_format(filename, content)
    if file_format in (".png", ".jpg", ".jpeg"):
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise ExtractionError(
                "Image dependencies are missing. Install the packages in requirements.txt."
            ) from error
        try:
            with Image.open(BytesIO(content)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InputError("Image dimensions exceed the 25-megapixel limit.")
                image.load()
                text = _ocr_image(image)
        except InputError:
            raise
        except UnidentifiedImageError as error:
            raise InputError("The uploaded image could not be decoded.") from error
        except OSError as error:
            raise InputError(f"Could not read the uploaded image: {error}") from error
        if not text.strip():
            raise InputError("No readable text was found in the image.")
        return text.strip(), True

    try:
        import fitz
        from PIL import Image
    except ImportError as error:
        raise ExtractionError(
            "PDF/image dependencies are missing. Install the packages in requirements.txt."
        ) from error

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as error:
        raise InputError("The uploaded PDF could not be opened.") from error

    with document:
        if document.needs_pass:
            raise InputError("Password-protected PDFs are not supported.")
        if document.page_count == 0:
            raise InputError("The PDF has no pages.")
        if document.page_count > MAX_PDF_PAGES:
            raise InputError(f"PDFs are limited to {MAX_PDF_PAGES} pages.")

        pages: list[str] = []
        used_ocr = False
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if len(page_text) < 20:
                used_ocr = True
                width, height = page.rect.width, page.rect.height
                if width <= 0 or height <= 0:
                    raise InputError(f"PDF page {page_number} has invalid dimensions.")
                scale = min(2.0, (MAX_IMAGE_PIXELS / (width * height)) ** 0.5)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = Image.open(BytesIO(pixmap.tobytes("png")))
                page_text = _ocr_image(image).strip()
            if page_text:
                pages.append(f"--- Page {page_number} ---\n{page_text}")
            if sum(len(page_content) for page_content in pages) > MAX_SOURCE_CHARACTERS:
                raise InputError(
                    f"Extracted document text cannot exceed {MAX_SOURCE_CHARACTERS:,} characters."
                )

    text = "\n\n".join(pages).strip()
    if not text:
        raise InputError("No readable text was found in the PDF.")
    return text, used_ocr


def _parse_extract_request(content_type: str, raw_body: bytes) -> tuple[str | None, tuple[str, bytes] | None]:
    if content_type.split(";", 1)[0].strip().casefold() == "application/json":
        if len(raw_body) > MAX_JSON_REQUEST_BYTES:
            raise InputError("Text requests cannot exceed 1 MB.")
        try:
            request = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InputError("Request body must be valid JSON.") from error
        source = request.get("source") if isinstance(request, dict) else None
        if not isinstance(source, str) or not source.strip():
            raise InputError("Provide non-empty invoice text in 'source'.")
        return source, None

    if not content_type.casefold().startswith("multipart/form-data"):
        raise InputError("Send invoice text as JSON or upload a supported file.")
    try:
        header = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
        )
    except UnicodeEncodeError as error:
        raise InputError("Invalid multipart content type.") from error
    message = BytesParser(policy=default).parsebytes(header + raw_body)
    if not message.is_multipart():
        raise InputError("Multipart upload is missing a valid boundary.")

    source: str | None = None
    uploaded_file: tuple[str, bytes] | None = None
    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if field_name == "source":
            payload = part.get_payload(decode=True) or b""
            try:
                source = payload.decode(part.get_content_charset() or "utf-8")
            except (LookupError, UnicodeDecodeError) as error:
                raise InputError("The source text must be UTF-8 encoded.") from error
        elif field_name == "file" and part.get_filename():
            if uploaded_file is not None:
                raise InputError("Upload only one file at a time.")
            payload = part.get_payload(decode=True)
            if payload is None:
                raise InputError("The uploaded file is empty.")
            uploaded_file = (Path(part.get_filename()).name, payload)

    if source and source.strip() and uploaded_file:
        raise InputError("Choose either pasted text or a file, not both.")
    if uploaded_file:
        return None, uploaded_file
    if not isinstance(source, str) or not source.strip():
        raise InputError("Paste invoice text or choose a PDF, PNG, JPG, or JPEG file.")
    return source, None


def validate_extraction(candidate: Any, source: str) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ExtractionError("The model response must be a JSON object.")

    unexpected = set(candidate) - set(FIELDS)
    if unexpected:
        raise ExtractionError(
            "The model returned unexpected fields: " + ", ".join(sorted(unexpected))
        )

    accepted: dict[str, str | None] = {}
    rejected: list[dict[str, str]] = []
    for field in FIELDS:
        value = candidate.get(field)
        if value is None or value == "":
            accepted[field] = None
        elif not isinstance(value, str):
            rejected.append(
                {"field": field, "value": str(value), "reason": "Value must be a string."}
            )
            accepted[field] = None
        elif is_source_grounded(value, source):
            accepted[field] = value
        else:
            rejected.append(
                {
                    "field": field,
                    "value": value,
                    "reason": "Value does not appear in the source text.",
                }
            )
            accepted[field] = None

    return {"fields": accepted, "rejected": rejected}


def _extract_with_model(source: str) -> dict[str, Any]:
    endpoint = os.environ.get(
        "MODEL_API_URL", "http://localhost:8001/v1/chat/completions"
    )
    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    api_key = os.environ.get("MODEL_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    schema = {field: "string or null" for field in FIELDS}
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract invoice fields from the supplied source. Never infer or "
                    "invent values. Return only a JSON object with exactly these keys: "
                    + json.dumps(schema)
                    + ". Use null when a value is absent. Preserve the exact text "
                    "used for each value in the source."
                ),
            },
            {"role": "user", "content": source},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise ExtractionError(f"Model server returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise ExtractionError(
            "Could not reach the model server. Start a vLLM-compatible endpoint or "
            "set MODEL_API_URL."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExtractionError("Model server returned an invalid response.") from error

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ExtractionError("Model server response is missing chat completion content.") from error

    if not isinstance(content, str):
        raise ExtractionError("Model completion content must be text.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ExtractionError("Model did not return valid JSON.") from error


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path not in ("/", "/index.html"):
            self._send_json(404, {"error": "Not found."})
            return
        page = (ROOT / "static" / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:
        if self.path != "/api/extract":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise InputError("Invalid request content length.") from error
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise InputError("Request body must be between 1 byte and 10 MB.")
            source, uploaded_file = _parse_extract_request(
                self.headers.get("Content-Type", ""), self.rfile.read(length)
            )
            result: dict[str, Any]
            if uploaded_file:
                filename, file_content = uploaded_file
                source, used_ocr = extract_document_text(filename, file_content)
                result = {"document": {
                    "filename": filename,
                    "type": Path(filename).suffix.casefold().lstrip("."),
                    "ocr_used": used_ocr,
                }}
            else:
                result = {}
            if not source or not source.strip():
                raise InputError("No readable invoice text was provided.")
            if len(source) > MAX_SOURCE_CHARACTERS:
                raise InputError(
                    f"Invoice text cannot exceed {MAX_SOURCE_CHARACTERS:,} characters."
                )
            candidate = _extract_with_model(source)
            result.update(validate_extraction(candidate, source))
            self._send_json(200, result)
        except InputError as error:
            self._send_json(400, {"error": str(error)})
        except ExtractionError as error:
            self._send_json(502, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args))


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Invoice extraction app listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
