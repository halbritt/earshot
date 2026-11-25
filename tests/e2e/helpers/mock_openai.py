"""Local OpenAI-compatible mock HTTP server for generator and classifier tests."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


@dataclass
class MockCompletionSpec:
    """Scripted response specification for /v1/chat/completions and /v1/completions."""

    status_code: int = 200
    content: str | None = None
    raw_body: bytes | None = None
    finish_reason: str = "stop"
    delay_s: float = 0.0
    prompt_tokens: int = 120
    completion_tokens: int = 20
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    top_logprobs: list[dict[str, float]] | None = None
    model: str | None = None
    error_message: str = "upstream failure"

    @classmethod
    def valid_question(
        cls,
        question: str,
        evidence_ids: list[str],
        *,
        cached_tokens: int = 0,
        delay_s: float = 0.0,
    ) -> MockCompletionSpec:
        payload = {
            "question": question,
            "evidence_segment_ids": list(evidence_ids),
            "decline_reason": None,
        }
        return cls(
            content=json.dumps(payload),
            cached_tokens=cached_tokens,
            delay_s=delay_s,
        )

    @classmethod
    def decline(
        cls,
        reason: str = "already answered",
        *,
        cached_tokens: int = 0,
    ) -> MockCompletionSpec:
        payload: dict[str, Any] = {
            "question": None,
            "evidence_segment_ids": [],
            "decline_reason": reason,
        }
        return cls(content=json.dumps(payload), cached_tokens=cached_tokens)

    @classmethod
    def malformed_truncated(
        cls,
        partial_text: str = '{"question": "Is this',
    ) -> MockCompletionSpec:
        return cls(content=partial_text, finish_reason="length")

    @classmethod
    def ungrounded(
        cls,
        question: str = "Did Alice approve $42,000 on May 19?",
        evidence_ids: list[str] | None = None,
    ) -> MockCompletionSpec:
        payload = {
            "question": question,
            "evidence_segment_ids": evidence_ids or ["segment-9999"],
            "decline_reason": None,
        }
        return cls(content=json.dumps(payload))

    @classmethod
    def http_error(
        cls,
        status_code: int = 500,
        message: str = "upstream failure",
    ) -> MockCompletionSpec:
        return cls(status_code=status_code, error_message=message)

    @classmethod
    def timeout(cls, delay_s: float = 5.0) -> MockCompletionSpec:
        body = json.dumps({"question": None, "decline_reason": "late"})
        return cls(delay_s=delay_s, content=body)


@dataclass(frozen=True)
class RecordedOpenAIRequest:
    """Captured request sent to MockOpenAIServer."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any]
    received_at_monotonic: float

    def prompt_blocks(self) -> list[str]:
        """Extract ordered text blocks from messages[] or prompt for prompt inspection."""
        blocks: list[str] = []
        messages = self.body.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    blocks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            blocks.append(part["text"])
        prompt = self.body.get("prompt")
        if isinstance(prompt, str):
            blocks.append(prompt)
        return blocks

    def joined_prompt(self) -> str:
        return "\n".join(self.prompt_blocks())


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/1.0"

    def __init__(self, *args: Any, mock_owner: MockOpenAIServer, **kwargs: Any) -> None:
        self._owner = mock_owner
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        loaded = json.loads(raw.decode("utf-8"))
        return loaded if isinstance(loaded, dict) else {"_raw": loaded}

    def _write_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError, ConnectionResetError:
            return

    def _write_json(self, status_code: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._write_bytes(status_code, body, "application/json")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        headers = {k.lower(): v for k, v in self.headers.items()}
        self._owner._record_request("GET", path, headers, {})
        if path in {"/v1/models", "/models"}:
            models = self._owner.get_models()
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": m, "object": "model", "owned_by": "mock-openai"} for m in models
                    ],
                },
            )
            return
        self._write_json(
            404,
            {"error": {"message": f"Not found: {path}", "type": "invalid_request_error"}},
        )

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        headers = {k.lower(): v for k, v in self.headers.items()}
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._write_json(
                400,
                {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            )
            return

        self._owner._record_request("POST", path, headers, body)
        if path not in {
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/completions",
            "/completions",
        }:
            self._write_json(
                404,
                {"error": {"message": f"Not found: {path}", "type": "invalid_request_error"}},
            )
            return

        spec = self._owner._select_spec(body)
        if spec.delay_s > 0.0:
            time.sleep(spec.delay_s)
        if spec.raw_body is not None:
            self._write_bytes(spec.status_code, spec.raw_body, "application/json")
            return
        if spec.status_code >= 400:
            self._write_json(
                spec.status_code,
                {
                    "error": {
                        "message": spec.error_message,
                        "type": "server_error",
                        "code": str(spec.status_code),
                    }
                },
            )
            return

        model_name = spec.model or str(body.get("model") or self._owner.get_models()[0])
        usage = {
            "prompt_tokens": spec.prompt_tokens,
            "completion_tokens": spec.completion_tokens,
            "total_tokens": spec.prompt_tokens + spec.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": spec.cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": spec.reasoning_tokens},
        }
        is_chat = path.endswith("/chat/completions")
        if is_chat:
            payload = self._build_chat_payload(spec, model_name, body, usage)
        else:
            payload = self._build_text_payload(spec, model_name, body, usage)
        self._write_json(spec.status_code, payload)

    @staticmethod
    def _build_chat_payload(
        spec: MockCompletionSpec,
        model_name: str,
        body: dict[str, Any],
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        content = spec.content if spec.content is not None else "{}"
        choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": spec.finish_reason,
        }
        if spec.top_logprobs is not None or body.get("logprobs"):
            lp_map = (
                spec.top_logprobs[0] if spec.top_logprobs else {"CONFIRM_OWNER": -0.2, "NONE": -1.8}
            )
            top_tok = max(lp_map, key=lambda k: lp_map[k])
            choice["logprobs"] = {
                "content": [
                    {
                        "token": top_tok,
                        "logprob": lp_map[top_tok],
                        "top_logprobs": [
                            {"token": tok, "logprob": score} for tok, score in lp_map.items()
                        ],
                    }
                ]
            }
        return {
            "id": "chatcmpl-mock-001",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": usage,
        }

    @staticmethod
    def _build_text_payload(
        spec: MockCompletionSpec,
        model_name: str,
        body: dict[str, Any],
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        text = spec.content if spec.content is not None else "CONFIRM_OWNER"
        choice: dict[str, Any] = {
            "index": 0,
            "text": text,
            "finish_reason": spec.finish_reason,
        }
        if spec.top_logprobs is not None or body.get("logprobs") is not None:
            lp_list = spec.top_logprobs or [{"CONFIRM_OWNER": -0.15, "NONE": -2.1}]
            choice["logprobs"] = {
                "tokens": [max(m, key=lambda k: m[k]) for m in lp_list],
                "token_logprobs": [max(m.values()) for m in lp_list],
                "top_logprobs": lp_list,
            }
        return {
            "id": "cmpl-mock-001",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": usage,
        }


class MockOpenAIServer:
    """Thread-hosted OpenAI-compatible HTTP mock server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque[MockCompletionSpec] = deque()
        self._matchers: list[tuple[Callable[[dict[str, Any]], bool], MockCompletionSpec]] = []
        self._default = MockCompletionSpec.decline("default mock decline")
        self._models: list[str] = ["mock-generator-v1", "mock-classifier-v1"]
        self.requests: list[RecordedOpenAIRequest] = []
        self._httpd: _QuietThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port = 0

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    def __enter__(self) -> MockOpenAIServer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._httpd is not None:
            return
        handler_cls = partial(_MockOpenAIHandler, mock_owner=self)
        httpd = _QuietThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self._httpd = httpd
        self._port = int(httpd.server_address[1])
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def enqueue(self, *specs: MockCompletionSpec) -> None:
        with self._lock:
            self._queue.extend(specs)

    def add_matcher(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        spec: MockCompletionSpec,
    ) -> None:
        with self._lock:
            self._matchers.append((predicate, spec))

    def set_default(self, spec: MockCompletionSpec) -> None:
        with self._lock:
            self._default = spec

    def set_models(self, models: list[str]) -> None:
        with self._lock:
            self._models = list(models)

    def get_models(self) -> list[str]:
        with self._lock:
            return list(self._models)

    def clear_requests(self) -> None:
        with self._lock:
            self.requests.clear()

    def _record_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> None:
        with self._lock:
            self.requests.append(
                RecordedOpenAIRequest(
                    method=method,
                    path=path,
                    headers=headers,
                    body=body,
                    received_at_monotonic=time.monotonic(),
                )
            )

    def _select_spec(self, body: dict[str, Any]) -> MockCompletionSpec:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            for pred, spec in self._matchers:
                if pred(body):
                    return spec
            return self._default
