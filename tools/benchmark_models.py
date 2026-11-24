#!/usr/bin/env python3
"""Compare base/SFT/DPO adapters on the held-out split and record latency."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import EXTRACTION_SYSTEM_PROMPT  # noqa: E402
from evaluate import evaluate


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(record, dict) or not isinstance(record.get("source"), str):
                raise ValueError(f"{path}:{line_number} needs a source string")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was selected, but PyTorch reports it unavailable.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but PyTorch reports it unavailable.")
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(base_model: str, adapter: str | None, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def make_prompt(tokenizer: Any, source: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(
    model: Any,
    tokenizer: Any,
    source: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, float, int]:
    import torch

    prompt = make_prompt(tokenizer, source)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        if device == "mps":
            generated_ids = []
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values
            for index in range(max_new_tokens):
                generated_ids.append(next_token)
                if tokenizer.eos_token_id is not None \
                        and int(next_token.item()) == tokenizer.eos_token_id:
                    break
                if index + 1 == max_new_tokens:
                    break
                attention_mask = torch.cat((
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
                ), dim=-1)
                output = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = output.past_key_values
            generated = torch.cat(generated_ids, dim=-1)[0].to("cpu")
        else:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            generated = output[0, input_ids.shape[1]:]
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return tokenizer.decode(generated, skip_special_tokens=True), elapsed, len(generated)


def benchmark_model(
    label: str,
    adapter: str | None,
    records: list[dict[str, Any]],
    base_model: str,
    device: str,
    max_new_tokens: int,
    warmup_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    load_start = time.perf_counter()
    model, tokenizer = load_model(base_model, adapter, device)
    load_seconds = time.perf_counter() - load_start

    warmup_start = time.perf_counter()
    generate(model, tokenizer, records[0]["source"], device, warmup_tokens)
    warmup_seconds = time.perf_counter() - warmup_start

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    generated_tokens = 0
    inference_start = time.perf_counter()
    for index, record in enumerate(records, 1):
        prediction, latency, token_count = generate(
            model, tokenizer, record["source"], device, max_new_tokens
        )
        latencies.append(latency)
        generated_tokens += token_count
        predictions.append({
            "document_id": record.get("document_id", str(index)),
            "source": record["source"],
            "gold": record.get("gold", {}),
            "prediction": prediction,
            "latency_seconds": latency,
            "generated_tokens": token_count,
        })
        if index % 25 == 0 or index == len(records):
            print(f"{label}: evaluated {index}/{len(records)} documents")
    inference_seconds = time.perf_counter() - inference_start

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{label}_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as prediction_file:
        for record in predictions:
            prediction_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = evaluate(predictions)
    total_latency = sum(latencies)
    metrics.update({
        "model": label,
        "base_model": base_model,
        "adapter": adapter,
        "device": device,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "inference_seconds": inference_seconds,
        "mean_latency_seconds": statistics.mean(latencies),
        "p50_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": sorted(latencies)[
            min(len(latencies) - 1, max(0, math.ceil(0.95 * len(latencies)) - 1))
        ],
        "documents_per_second": len(records) / inference_seconds if inference_seconds else 0,
        "generated_tokens_per_second": (
            generated_tokens / total_latency if total_latency else 0
        ),
        "prediction_file": str(prediction_path),
    })

    del model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return metrics


def benchmark_api(
    label: str,
    records: list[dict[str, Any]],
    endpoint: str,
    model_name: str,
    api_key: str | None,
    max_new_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    def request_completion(source: str) -> tuple[str, float, int | None]:
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": source},
            ],
            "temperature": 0,
            "max_tokens": max_new_tokens,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(endpoint, data=payload, headers=headers)
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read(1024).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Reference endpoint returned HTTP {error.code}: {detail}"
            ) from error
        elapsed = time.perf_counter() - start
        if not isinstance(body, dict):
            raise ValueError("Reference endpoint returned a non-object response.")
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Reference endpoint returned an invalid chat completion.") from error
        if not isinstance(content, str):
            raise ValueError("Reference endpoint completion content must be text.")
        usage = body.get("usage")
        completion_tokens = (
            usage.get("completion_tokens")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int)
            else None
        )
        return content, elapsed, completion_tokens

    warmup_start = time.perf_counter()
    request_completion(records[0]["source"])
    warmup_seconds = time.perf_counter() - warmup_start
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    generated_tokens = 0
    token_counts_available = True
    inference_start = time.perf_counter()
    for index, record in enumerate(records, 1):
        prediction, latency, token_count = request_completion(record["source"])
        latencies.append(latency)
        if token_count is None:
            token_counts_available = False
        else:
            generated_tokens += token_count
        predictions.append({
            "document_id": record.get("document_id", str(index)),
            "source": record["source"],
            "gold": record.get("gold", {}),
            "prediction": prediction,
            "latency_seconds": latency,
            "generated_tokens": token_count,
        })
        if index % 25 == 0 or index == len(records):
            print(f"{label}: evaluated {index}/{len(records)} documents")

    inference_seconds = time.perf_counter() - inference_start
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{label}_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as prediction_file:
        for record in predictions:
            prediction_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = evaluate(predictions)
    metrics.update({
        "model": label,
        "base_model": model_name,
        "reference_endpoint_configured": True,
        "warmup_seconds": warmup_seconds,
        "inference_seconds": inference_seconds,
        "mean_latency_seconds": statistics.mean(latencies),
        "p50_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": sorted(latencies)[
            min(len(latencies) - 1, max(0, math.ceil(0.95 * len(latencies)) - 1))
        ],
        "documents_per_second": len(records) / inference_seconds if inference_seconds else 0,
        "generated_tokens_per_second": (
            generated_tokens / sum(latencies)
            if token_counts_available and sum(latencies) else None
        ),
        "prediction_file": str(prediction_path),
    })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/invoice_dataset/test.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sft-adapter")
    parser.add_argument("--dpo-adapter")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument(
        "--reference-api-url",
        help="Optional OpenAI-compatible /chat/completions endpoint for an API reference.",
    )
    parser.add_argument("--reference-model", help="Model name required with --reference-api-url.")
    parser.add_argument(
        "--reference-api-key-env",
        default="MODEL_API_KEY",
        help="Environment variable containing the optional reference API key.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1 or args.warmup_tokens < 1:
        parser.error("--max-new-tokens and --warmup-tokens must be positive")
    if bool(args.reference_api_url) != bool(args.reference_model):
        parser.error("--reference-api-url and --reference-model must be supplied together")

    import torch

    device = choose_device(torch, args.device)
    records = load_records(args.data, args.limit)
    models = [("base", None)]
    if args.sft_adapter:
        models.append(("sft", args.sft_adapter))
    if args.dpo_adapter:
        models.append(("sft_dpo", args.dpo_adapter))

    results = [
        benchmark_model(
            label,
            adapter,
            records,
            args.base_model,
            device,
            args.max_new_tokens,
            args.warmup_tokens,
            args.output_dir,
        )
        for label, adapter in models
    ]
    if args.reference_api_url:
        import os

        results.append(benchmark_api(
            "frontier_reference",
            records,
            args.reference_api_url,
            args.reference_model,
            os.environ.get(args.reference_api_key_env),
            args.max_new_tokens,
            args.output_dir,
        ))
    report_path = args.output_dir / "benchmark_results.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved benchmark report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
