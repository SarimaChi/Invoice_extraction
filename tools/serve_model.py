#!/usr/bin/env python3
"""Serve a local base model or PEFT adapter through an OpenAI-compatible API."""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_models import generate, load_model, make_prompt


MAX_BODY_BYTES = 1_000_000
generation_lock = threading.Lock()


class ModelServer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        device: str,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

    def complete(self, messages: Any, requested_tokens: Any) -> dict[str, Any]:
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list.")
        user_messages = [
            item.get("content") for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        if not user_messages or not isinstance(user_messages[-1], str):
            raise ValueError("Provide a user message with text content.")
        source = user_messages[-1]
        if not source.strip():
            raise ValueError("The user message cannot be empty.")
        limit = self.max_new_tokens
        if requested_tokens is not None:
            if not isinstance(requested_tokens, int) or requested_tokens < 1:
                raise ValueError("'max_tokens' must be a positive integer.")
            limit = min(requested_tokens, self.max_new_tokens)

        prompt = make_prompt(self.tokenizer, source)
        prompt_tokens = len(self.tokenizer(prompt)["input_ids"])
        prediction, latency, generated_tokens = generate(
            self.model,
            self.tokenizer,
            source,
            self.device,
            limit,
        )
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": prediction},
                "finish_reason": "length" if generated_tokens >= limit else "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            },
            "latency_seconds": latency,
        }


class ModelHTTPServer(ThreadingHTTPServer):
    model_server: ModelServer


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "GroundworkModel/1.0"

    @property
    def model_server(self) -> ModelServer:
        return self.server.model_server

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": self.model_server.model_name, "object": "model"}],
            })
            return
        self._send_json(404, {"error": {"message": "Not found."}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "Not found."}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": {"message": "Invalid content length."}})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {"error": {"message": "Request body must be under 1 MB."}})
            return
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object.")
            with generation_lock:
                response = self.model_server.complete(
                    request.get("messages"),
                    request.get("max_tokens", request.get("max_completion_tokens")),
                )
            self._send_json(200, response)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send_json(400, {"error": {"message": str(error)}})
        except (RuntimeError, MemoryError) as error:
            self._send_json(500, {"error": {"message": f"Model inference failed: {error}"}})

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--model-name", default="groundwork-invoice")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512:
        parser.error("--max-new-tokens must be between 1 and 512")

    import torch

    device = args.device
    if device == "auto":
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    model, tokenizer = load_model(args.base_model, args.adapter, device)
    server = ModelHTTPServer((args.host, args.port), RequestHandler)
    server.model_server = ModelServer(
        model, tokenizer, args.model_name, device, args.max_new_tokens
    )
    print(f"Serving {args.model_name} on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
