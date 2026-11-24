# Groundwork — evidence-first invoice extraction

A small, local-first app for extracting invoice fields into JSON, checking every
non-null value against the source text, filtering synthetic training pairs, and
measuring extraction quality.

## Run the app

Requires Python 3.9 or newer and an OpenAI-compatible chat-completions server.
For local inference, start vLLM separately:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-training-mps.txt
brew install tesseract
```

## 2. Prepare the dataset

[`invoice_dataset`](https://github.com/mouadhamri/invoice_dataset). 

```sh
python tools/prepare_invoice_dataset.py \
  --source-dir data/raw/invoice_dataset \
  --output-dir data/processed/invoice_dataset \
  --source-mode tesseract

python tools/build_dpo_pairs.py data/processed/invoice_dataset/train.jsonl \
  --output data/processed/invoice_dataset/dpo_train.jsonl --strategy mixed
python tools/build_dpo_pairs.py data/processed/invoice_dataset/valid.jsonl \
  --output data/processed/invoice_dataset/dpo_valid.jsonl --strategy mixed
```

The measured data preparation retained 700 of 900 documents: 400 train, 100
validation, and 200 test invoices, split by template family. 

## 3. Train SFT and DPO

```sh
python training/train_sft.py \
  --dataset data/processed/invoice_dataset/train.jsonl \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --output-dir outputs/qwen-invoice-sft \
  --rank 16 --alpha 32 --learning-rate 0.0002 --epochs 1

python training/train_dpo.py \
  --dataset data/processed/invoice_dataset/dpo_train.jsonl \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --sft-adapter outputs/qwen-invoice-sft \
  --output-dir outputs/qwen-invoice-dpo \
  --rank 16 --alpha 32 --learning-rate 0.00005 --epochs 1 \
  --max-samples 100 --beta 0.1
```

Training metrics are saved as `training_metrics.json` in each output directory.

## 4. Reproduce the benchmark

```sh
python tools/benchmark_models.py \
  --data data/processed/invoice_dataset/test.jsonl \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --sft-adapter outputs/qwen-invoice-sft \
  --dpo-adapter outputs/qwen-invoice-dpo \
  --device mps --limit 20 --max-new-tokens 192 \
  --output-dir outputs/evaluation-20
```

## 5. Run the API and app

In one terminal, serve the trained SFT adapter:

```sh
python tools/serve_model.py \
  --adapter outputs/qwen-invoice-sft --device mps \
  --host 127.0.0.1 --port 8001
```

In another terminal:

```sh
MODEL_API_URL=http://127.0.0.1:8001/v1/chat/completions \
MODEL_NAME=groundwork-invoice python app.py
```


## 6. Run tests

```sh
python -m unittest discover -s tests -v
```

## Results

| Model | Field micro-F1 | JSON valid | Unsupported-value (hallucination) rate | Mean latency |
|---|---:|---:|---:|---:|
| Qwen2.5-3B base | 0.442 | 65% | 15.4% | 15.32 s |
| SFT | **0.945** | **100%** | **1.0%** | **8.16 s** |
| SFT+DPO | 0.639 | 95% | 12.4% | 10.22 s |

SFT outperformed the base model and DPO in this pilot. DPO training metrics
did not predict better extraction quality. The measured SFT run reported train
loss 0.9693, validation loss 1.2256, and validation mean token accuracy 0.7649.
DPO reported validation loss 0.0242 and preference accuracy 1.0 on 100 pairs.
