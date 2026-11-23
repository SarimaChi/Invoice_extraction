# Groundwork — evidence-first invoice extraction

A small, local-first app for extracting invoice fields into JSON, checking every
non-null value against the source text, filtering synthetic training pairs, and
measuring extraction quality. It is intentionally an invoice MVP rather than a
claim that a model has already been fine-tuned or benchmarked.

## Run the app

Requires Python 3.9 or newer and an OpenAI-compatible chat-completions server.
For local inference, start vLLM separately, for example:

```sh
vllm serve Qwen/Qwen2.5-3B-Instruct --host 127.0.0.1 --port 8001
```

Install the file-reading dependencies:

```sh
python -m pip install -r requirements.txt
```

For PNG/JPG and scanned PDF OCR, Tesseract must also be installed. On macOS:

```sh
brew install tesseract
```

Then in another terminal:

```sh
cp .env.example .env
set -a
. ./.env
set +a
python app.py
```

Open <http://127.0.0.1:8080>, paste invoice text, load the sample, or upload a
PDF, PNG, JPG, or JPEG (up to 10 MB). PDFs are limited to 20 pages. The app
extracts embedded PDF text directly and uses local Tesseract OCR for images and
scanned PDF pages. It uses `MODEL_API_URL`, `MODEL_NAME`, and optional
`MODEL_API_KEY` from the process environment. The model endpoint is not bundled;
failed requests are reported rather than replaced with fake extraction output.

File parsing and OCR happen locally, but the extracted text is sent to the
configured model endpoint. Use a local model endpoint for private documents.

The JSON response is `{ "fields": { ... }, "rejected": [ ... ] }`. Values that
fail a normalized source-substring check are set to `null` and reported in
`rejected`. This check catches unsupported strings; it does not prove that a
supported value has the correct semantic meaning. The browser app only accepts
plain invoice text or supported image/PDF uploads.

## Data quality

Generate candidate pairs as JSONL with a `source` string and `completion` value
(a JSON object or a JSON-encoded string). Filter malformed, schema-incompatible,
or ungrounded pairs:

```sh
python tools/filter_synthetic.py candidates.jsonl \
  --accepted data/accepted.jsonl --rejected data/rejected.jsonl
```

Accepted rows have `source` and a `completion` object containing exactly these
keys, each a string or `null`:

`invoice_number`, `supplier`, `customer`, `issue_date`, `due_date`, `currency`,
`subtotal`, `tax`, `total`.

Treat source documents and annotations as sensitive data. Keep private datasets
out of version control; `.gitignore` excludes common local data directories.

## Evaluation

Prepare JSONL with one document per line:

```json
{"source":"Invoice INV-12 total $100.00","gold":{"invoice_number":"INV-12","total":"100.00"},"prediction":"{\"invoice_number\":\"INV-12\",\"total\":\"100.00\"}"}
```

`prediction` may be a JSON object or JSON string; an invalid JSON string counts
as invalid and as an empty prediction. Run:

```sh
python tools/evaluate.py predictions.jsonl --label "SFT+DPO AWQ"
```

The report includes field-level micro precision/recall/F1, exact field accuracy,
JSON validity, and unsupported-value rate. Empty predictions have a zero
hallucination rate when no non-null values were emitted; interpret the count
alongside the rate.

## Training

The optional training scripts use 4-bit NF4 QLoRA with PEFT/TRL and initialize
a W&B run for each experiment. They are intended for a CUDA/Linux machine; they
are not part of the lightweight app dependencies and have not been run as part
of this scaffold's tests.

```sh
pip install -r requirements-training.txt
python tools/build_dpo_pairs.py data/accepted.jsonl \
  --output data/preferences.jsonl --strategy mixed
python training/train_sft.py --dataset data/accepted.jsonl \
  --output-dir outputs/qwen-invoice-sft --rank 16 --alpha 32 \
  --learning-rate 0.0002
python training/train_dpo.py --dataset data/preferences.jsonl \
  --base-model outputs/qwen-invoice-sft \
  --output-dir outputs/qwen-invoice-dpo --rank 16 --alpha 32 \
  --learning-rate 0.00005
```

The preference-pair builder creates one controlled negative per non-empty
annotation by removing a populated field or replacing it with an unsupported
sentinel. Review generated pairs before training, keep DPO examples separate
from the held-out gold set, and run independent validation during any model
sweep. The scripts log rank, alpha, learning rate, and other run arguments to
W&B; authenticate or set `WANDB_MODE=offline` before starting training.

## Tests

```sh
python -m unittest discover -s tests -v
```

## Training and deployment roadmap

This repository contains app, pair filtering, offline evaluation, controlled
DPO-pair generation, and optional SFT/DPO training entry points. It does **not**
contain a trained checkpoint, hand-labelled gold set, quantization benchmark,
or measured results. Add those only after preparing a properly licensed,
privacy-reviewed dataset and selecting a CUDA training environment. Do not
report portfolio accuracy or throughput numbers until evaluation has been run
on a held-out, human-labelled set. Quantization/serving options such as GGUF or
AWQ should be tested separately; a vLLM-compatible endpoint can serve a base,
fine-tuned, or quantized model without changing the app.
