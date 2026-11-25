"""OpenAI-compatible provider adapter, capability table, egress log, and circuit breaker (§9.3, §14.3, §14.6, B13)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import httpx

from coach.contracts import EgressRecord, HealthState, Origin

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_FORBIDDEN_SECRET_KEYS = frozenset({"api_key", "secret", "token", "password"})


class ProviderError(RuntimeError):
    """Base error for provider configuration, transport, or policy failures."""


class UnsupportedModelConfigurationError(ProviderError, ValueError):
    """Raised at startup when (model_id, reasoning_setting) is unsupported (§9.3)."""


class MissingProviderSecretError(ProviderError, ValueError):
    """Raised when a hosted provider lacks its configured environment variable."""


class LocalOnlyBlockedError(ProviderError):
    """Raised when local_only=True blocks a hosted provider call (§14.3)."""


class BreakerOpenError(ProviderError):
    """Raised when the generator circuit breaker is open (B13)."""


class ProviderTimeoutError(ProviderError, TimeoutError):
    """Raised when a provider request exceeds its deadline or HTTP timeout."""


class BreakerState(StrEnum):
    """Circuit breaker states (§8.6, §16, B13)."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Declared model capabilities for startup validation (§9.3)."""

    supported_reasoning: frozenset[str | None]
    supports_prefix_cache_marker: bool = False
    reasoning_tokens_share_output_cap: bool = True
    default_max_output_tokens: int = 256


MODEL_CAPABILITY_TABLE: dict[str, ModelCapabilities] = {
    "qwen3.8-27b": ModelCapabilities(
        supported_reasoning=frozenset({None, "none", "off", "low", "medium", "high"}),
        supports_prefix_cache_marker=False,
        reasoning_tokens_share_output_cap=True,
        default_max_output_tokens=256,
    ),
    "qwen3:14b": ModelCapabilities(
        supported_reasoning=frozenset({None, "none", "off", "low", "medium", "high"}),
        supports_prefix_cache_marker=False,
        reasoning_tokens_share_output_cap=True,
        default_max_output_tokens=256,
    ),
    "gpt-6-luna": ModelCapabilities(
        supported_reasoning=frozenset({None, "none", "low", "medium", "high"}),
        supports_prefix_cache_marker=True,
        reasoning_tokens_share_output_cap=True,
        default_max_output_tokens=256,
    ),
    "gpt-6.2-sol": ModelCapabilities(
        supported_reasoning=frozenset({"low", "medium", "high"}),
        supports_prefix_cache_marker=True,
        reasoning_tokens_share_output_cap=True,
        default_max_output_tokens=512,
    ),
    "gpt-6.2-astra": ModelCapabilities(
        supported_reasoning=frozenset({"low", "medium", "high"}),
        supports_prefix_cache_marker=True,
        reasoning_tokens_share_output_cap=True,
        default_max_output_tokens=512,
    ),
    "gpt-4o-mini": ModelCapabilities(
        supported_reasoning=frozenset({None}),
        supports_prefix_cache_marker=True,
        reasoning_tokens_share_output_cap=False,
        default_max_output_tokens=256,
    ),
    "gpt-4.1-mini": ModelCapabilities(
        supported_reasoning=frozenset({None}),
        supports_prefix_cache_marker=True,
        reasoning_tokens_share_output_cap=False,
        default_max_output_tokens=256,
    ),
}


def _infer_is_hosted(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host not in _LOCAL_HOSTS


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Provider configuration without raw secrets (§9.3, §14.6)."""

    provider: str
    base_url: str
    model_id: str
    api_key_env: str | None = None
    reasoning_setting: str | None = None
    is_hosted: bool | None = None
    local_only: bool = False
    output_token_cap: int = 160
    breaker_consecutive_failures: int = 3
    breaker_open_ms: int = 60_000
    use_streaming: bool = True

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("ProviderConfig.provider must be non-empty")
        if not self.base_url.strip():
            raise ValueError("ProviderConfig.base_url must be non-empty")
        if not self.model_id.strip():
            raise ValueError("ProviderConfig.model_id must be non-empty")
        if self.api_key_env is not None:
            env_name = self.api_key_env.strip()
            if not env_name or any(ch.isspace() for ch in env_name) or env_name.startswith("sk-"):
                raise ValueError(
                    "ProviderConfig.api_key_env must be an environment variable name, not a secret"
                )
        if self.output_token_cap < 1:
            raise ValueError("ProviderConfig.output_token_cap must be >= 1")
        if self.breaker_consecutive_failures < 1:
            raise ValueError("ProviderConfig.breaker_consecutive_failures must be >= 1")
        if self.breaker_open_ms < 1:
            raise ValueError("ProviderConfig.breaker_open_ms must be >= 1")

    @property
    def effective_is_hosted(self) -> bool:
        if self.is_hosted is not None:
            return self.is_hosted
        return _infer_is_hosted(self.base_url)

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(provider={self.provider!r}, base_url={self.base_url!r}, "
            f"model_id={self.model_id!r}, api_key_env={self.api_key_env!r}, "
            f"reasoning_setting={self.reasoning_setting!r}, local_only={self.local_only!r})"
        )


def load_provider_config(data: Mapping[str, Any]) -> ProviderConfig:
    """Parse provider config mapping while rejecting raw secret fields (§14.6)."""
    bad_keys = _FORBIDDEN_SECRET_KEYS & set(data)
    if bad_keys:
        raise ValueError(
            f"Raw secrets {sorted(bad_keys)} are forbidden in config; use 'api_key_env'"
        )
    return ProviderConfig(
        provider=str(data["provider"]),
        base_url=str(data["base_url"]),
        model_id=str(data["model_id"]),
        api_key_env=str(data["api_key_env"]) if data.get("api_key_env") is not None else None,
        reasoning_setting=(
            str(data["reasoning_setting"]) if data.get("reasoning_setting") is not None else None
        ),
        is_hosted=bool(data["is_hosted"]) if data.get("is_hosted") is not None else None,
        local_only=bool(data.get("local_only", False)),
        output_token_cap=int(data.get("output_token_cap", 160)),
        breaker_consecutive_failures=int(data.get("breaker_consecutive_failures", 3)),
        breaker_open_ms=int(data.get("breaker_open_ms", 60_000)),
        use_streaming=bool(data.get("use_streaming", True)),
    )


def resolve_api_key(config: ProviderConfig) -> str | None:
    """Read API key from configured environment variable without logging it."""
    if config.api_key_env is None:
        return None
    val = os.environ.get(config.api_key_env)
    return val if val else None


def validate_startup_capabilities(
    config: ProviderConfig,
    extra_capabilities: Mapping[str, ModelCapabilities] | None = None,
) -> ModelCapabilities:
    """Validate model/reasoning support and env secret presence at startup (§9.3, §14.6)."""
    table = dict(MODEL_CAPABILITY_TABLE)
    if extra_capabilities:
        table.update(extra_capabilities)

    caps = table.get(config.model_id)
    if caps is None:
        if config.reasoning_setting is not None:
            raise UnsupportedModelConfigurationError(
                f"Unknown model {config.model_id!r} cannot validate reasoning_setting="
                f"{config.reasoning_setting!r}"
            )
        caps = ModelCapabilities(supported_reasoning=frozenset({None}))
    elif config.reasoning_setting not in caps.supported_reasoning:
        raise UnsupportedModelConfigurationError(
            f"Model {config.model_id!r} does not support reasoning_setting="
            f"{config.reasoning_setting!r}; supported={sorted(str(x) for x in caps.supported_reasoning)}"
        )

    if config.effective_is_hosted and not config.local_only:
        if config.api_key_env is None:
            raise MissingProviderSecretError(
                f"Hosted provider {config.provider!r} requires api_key_env"
            )
        if not resolve_api_key(config):
            raise MissingProviderSecretError(
                f"Environment variable {config.api_key_env!r} is missing or empty"
            )
    return caps


@dataclass(frozen=True, slots=True)
class PromptBlocks:
    """§9.2 4-block prompt ordered from least to most volatile."""

    policy_block: str
    brief_block: str
    transcript_block: str
    opportunity_block: str
    cache_breakpoint_after_block: int = 2

    def cache_prefix_sha256(self) -> str:
        raw = f"{self.policy_block}\n\n{self.brief_block}".encode()
        return hashlib.sha256(raw).hexdigest()

    def full_sha256(self) -> str:
        raw = (
            f"{self.policy_block}\n\n{self.brief_block}\n\n"
            f"{self.transcript_block}\n\n{self.opportunity_block}"
        ).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Provider-independent generation request (§9.3)."""

    session_id: str
    candidate_id: str | None
    lane: Origin
    prompt_blocks: PromptBlocks
    segment_ids: tuple[str, ...]
    output_token_cap: int
    deadline_ms: int


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """Normalized provider generation output, timings, and usage (§9.3)."""

    egress_id: str
    text: str
    finish_reason: str | None
    usage: dict[str, int]
    request_sent_ms: int
    first_byte_ms: int | None
    complete_ms: int


def _zero_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }


def normalize_usage(raw_usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Normalize OpenAI/llama.cpp/Anthropic usage into M1 keys (never empty)."""
    if not raw_usage:
        return _zero_usage()
    prompt_details = raw_usage.get("prompt_tokens_details")
    comp_details = raw_usage.get("completion_tokens_details")
    cached = 0
    if isinstance(prompt_details, Mapping):
        cached = int(prompt_details.get("cached_tokens") or 0)
    if cached == 0:
        cached = int(
            raw_usage.get("cache_read_input_tokens") or raw_usage.get("cached_input_tokens") or 0
        )
    reasoning = 0
    if isinstance(comp_details, Mapping):
        reasoning = int(comp_details.get("reasoning_tokens") or 0)
    if reasoning == 0:
        reasoning = int(raw_usage.get("reasoning_tokens") or 0)
    return {
        "input_tokens": max(
            0, int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
        ),
        "cached_input_tokens": max(0, cached),
        "output_tokens": max(
            0, int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
        ),
        "reasoning_tokens": max(0, reasoning),
    }


def build_openai_payload(
    config: ProviderConfig,
    caps: ModelCapabilities,
    req: GenerationRequest,
    *,
    stream: bool | None = None,
) -> dict[str, Any]:
    """Build OpenAI-compatible chat/completions payload preserving §9.2 cache prefix."""
    blocks = req.prompt_blocks
    use_stream = config.use_streaming if stream is None else stream
    if caps.supports_prefix_cache_marker and blocks.cache_breakpoint_after_block >= 2:
        system_content: str | list[dict[str, Any]] = [
            {"type": "text", "text": blocks.policy_block},
            {
                "type": "text",
                "text": blocks.brief_block,
                "cache_control": {"type": "ephemeral"},
            },
        ]
    else:
        system_content = f"{blocks.policy_block}\n\n{blocks.brief_block}"

    user_content = f"{blocks.transcript_block}\n\n{blocks.opportunity_block}"
    payload: dict[str, Any] = {
        "model": config.model_id,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": min(req.output_token_cap, config.output_token_cap),
        "temperature": 0.0,
        "stream": use_stream,
    }
    if use_stream:
        payload["stream_options"] = {"include_usage": True}
    if caps.supports_prefix_cache_marker:
        payload["prompt_cache_key"] = blocks.cache_prefix_sha256()[:32]
    if config.reasoning_setting is not None:
        if config.model_id.startswith("qwen3") and config.reasoning_setting in {"none", "off"}:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["reasoning_effort"] = config.reasoning_setting
    return payload


@dataclass(slots=True)
class ProviderAdapter:
    """Async OpenAI-compatible provider adapter with egress logging & circuit breaker."""

    config: ProviderConfig
    transport: httpx.AsyncBaseTransport | None = None
    client: httpx.AsyncClient | None = None
    egress_sink: Callable[[EgressRecord], None] | None = None
    on_health_change: Callable[[HealthState], None] | None = None
    clock_ms: Callable[[], int] = field(default=lambda: int(time.monotonic() * 1000))
    extra_capabilities: Mapping[str, ModelCapabilities] | None = None
    validate_on_init: bool = True

    _caps: ModelCapabilities = field(init=False)
    _breaker_state: BreakerState = field(init=False, default=BreakerState.CLOSED)
    _consecutive_failures: int = field(init=False, default=0)
    _opened_at_ms: int | None = field(init=False, default=None)
    _half_open_in_flight: bool = field(init=False, default=False)
    _health_state: HealthState = field(init=False, default=HealthState.OK)
    _recovery_task: asyncio.Task[None] | None = field(init=False, default=None)
    _manual_waiting: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.validate_on_init:
            self._caps = validate_startup_capabilities(self.config, self.extra_capabilities)
        else:
            table = dict(MODEL_CAPABILITY_TABLE)
            if self.extra_capabilities:
                table.update(self.extra_capabilities)
            self._caps = table.get(
                self.config.model_id,
                ModelCapabilities(supported_reasoning=frozenset({None})),
            )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._caps

    @property
    def breaker_state(self) -> BreakerState:
        return self._breaker_state

    @property
    def health_state(self) -> HealthState:
        return self._health_state

    @property
    def manual_waiting(self) -> int:
        return self._manual_waiting

    def advance_clock(self, now_ms: int) -> BreakerState:
        """Evaluate scheduled OPEN -> HALF_OPEN recovery at now_ms (B13)."""
        if (
            self._breaker_state is BreakerState.OPEN
            and self._opened_at_ms is not None
            and now_ms >= self._opened_at_ms + self.config.breaker_open_ms
        ):
            self._breaker_state = BreakerState.HALF_OPEN
            self._half_open_in_flight = False
        return self._breaker_state

    def _set_health(self, state: HealthState) -> None:
        if self._health_state is not state:
            self._health_state = state
            if self.on_health_change is not None:
                self.on_health_change(state)

    def _cancel_recovery_task(self) -> None:
        if self._recovery_task is not None and not self._recovery_task.done():
            self._recovery_task.cancel()
        self._recovery_task = None

    def schedule_recovery(self) -> None:
        """Schedule an asyncio timer to move OPEN -> HALF_OPEN after breaker_open_ms."""
        self._cancel_recovery_task()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay_s = self.config.breaker_open_ms / 1000.0

        async def _wait_and_half_open() -> None:
            await asyncio.sleep(delay_s)
            self.advance_clock(self.clock_ms())

        self._recovery_task = loop.create_task(_wait_and_half_open())

    def _record_failure(self, now_ms: int) -> None:
        self._consecutive_failures += 1
        self._half_open_in_flight = False
        if (
            self._breaker_state is BreakerState.HALF_OPEN
            or self._consecutive_failures >= self.config.breaker_consecutive_failures
        ):
            self._breaker_state = BreakerState.OPEN
            self._opened_at_ms = now_ms
            self._set_health(HealthState.TEMPLATE_ONLY)
            self.schedule_recovery()

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at_ms = None
        self._half_open_in_flight = False
        if self._breaker_state is not BreakerState.CLOSED:
            self._breaker_state = BreakerState.CLOSED
            self._cancel_recovery_task()
        self._set_health(HealthState.OK)

    async def _parse_sse(
        self,
        resp: httpx.Response,
        on_first_byte: Callable[[], None],
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        chunks: list[str] = []
        finish_reason: str | None = None
        raw_usage: dict[str, Any] | None = None
        saw_first = False
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            if not saw_first:
                saw_first = True
                on_first_byte()
            obj = json.loads(data_str)
            if not isinstance(obj, Mapping):
                raise ValueError("Provider SSE chunk JSON must be an object")
            usage_obj = obj.get("usage")
            if isinstance(usage_obj, Mapping):
                raw_usage = dict(usage_obj)
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, Mapping):
                    fr = c0.get("finish_reason")
                    if isinstance(fr, str):
                        finish_reason = fr
                    delta = c0.get("delta")
                    if isinstance(delta, Mapping):
                        content = delta.get("content")
                        if isinstance(content, str):
                            chunks.append(content)
        return ("".join(chunks), finish_reason, raw_usage)

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        """Execute one chat/completions call, enforcing local_only, breaker, deadline, & egress_log."""
        if self.config.local_only and self.config.effective_is_hosted:
            raise LocalOnlyBlockedError(
                f"local_only=True blocks hosted provider {self.config.provider!r}"
            )

        now_ms = self.clock_ms()
        state = self.advance_clock(now_ms)
        if state is BreakerState.OPEN:
            raise BreakerOpenError("Generator circuit breaker is OPEN")
        probe_acquired = False
        if state is BreakerState.HALF_OPEN:
            if self._half_open_in_flight:
                raise BreakerOpenError("Generator half-open probe already in flight")
            self._half_open_in_flight = True
            probe_acquired = True

        remaining_ms = req.deadline_ms - now_ms
        if remaining_ms <= 0:
            if probe_acquired:
                self._half_open_in_flight = False
            if self.egress_sink is not None:
                self.egress_sink(
                    EgressRecord(
                        schema_version=1,
                        egress_id=f"eg-{uuid.uuid4().hex[:16]}",
                        session_id=req.session_id,
                        candidate_id=req.candidate_id,
                        lane=req.lane,
                        provider=self.config.provider,
                        model_id=self.config.model_id,
                        segment_ids=req.segment_ids,
                        sent_at_ms=now_ms,
                        first_byte_ms=None,
                        complete_ms=now_ms,
                        finish_reason="timeout",
                        usage=_zero_usage(),
                        discarded=True,
                    )
                )
            raise ProviderTimeoutError("Generation deadline already expired")

        if req.lane is Origin.MANUAL:
            self._manual_waiting += 1

        egress_id = f"eg-{uuid.uuid4().hex[:16]}"
        sent_ms = self.clock_ms()
        first_byte_ms: int | None = None
        complete_ms: int | None = None
        finish_reason: str | None = None
        usage = _zero_usage()
        discarded = False
        owns_client = False
        client: httpx.AsyncClient | None = None

        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = resolve_api_key(self.config)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        timeout_s = max(0.001, remaining_ms / 1000.0)
        try:
            payload = build_openai_payload(self.config, self._caps, req)
            owns_client = self.client is None
            client = (
                self.client
                if self.client is not None
                else httpx.AsyncClient(transport=self.transport, timeout=timeout_s)
            )
            async with asyncio.timeout(timeout_s):
                if payload.get("stream"):
                    async with client.stream(
                        "POST", url, json=payload, headers=headers, timeout=timeout_s
                    ) as resp:
                        if resp.status_code >= 400:
                            await resp.aread()
                            discarded = True
                            finish_reason = f"http_{resp.status_code}"
                            if resp.status_code >= 500 or resp.status_code == 429:
                                self._record_failure(self.clock_ms())
                            else:
                                self._half_open_in_flight = False
                            raise ProviderError(f"Provider returned HTTP {resp.status_code}")

                        def _mark_first() -> None:
                            nonlocal first_byte_ms
                            if first_byte_ms is None:
                                first_byte_ms = self.clock_ms()

                        text, finish_reason, raw_usage = await self._parse_sse(resp, _mark_first)
                        complete_ms = self.clock_ms()
                        if first_byte_ms is None:
                            first_byte_ms = complete_ms
                        usage = normalize_usage(raw_usage)
                else:
                    resp = await client.post(url, json=payload, headers=headers, timeout=timeout_s)
                    first_byte_ms = self.clock_ms()
                    if resp.status_code >= 400:
                        discarded = True
                        finish_reason = f"http_{resp.status_code}"
                        if resp.status_code >= 500 or resp.status_code == 429:
                            self._record_failure(self.clock_ms())
                        else:
                            self._half_open_in_flight = False
                        raise ProviderError(f"Provider returned HTTP {resp.status_code}")
                    data = resp.json()
                    if not isinstance(data, Mapping):
                        raise ValueError("Provider JSON response must be an object")
                    complete_ms = self.clock_ms()
                    choices = data.get("choices") or []
                    if not isinstance(choices, list):
                        raise ValueError("Provider JSON 'choices' must be a list")
                    c0 = choices[0] if choices else {}
                    if not isinstance(c0, Mapping):
                        raise ValueError("Provider JSON choice entry must be an object")
                    msg = c0.get("message") or {}
                    if not isinstance(msg, Mapping):
                        raise ValueError("Provider JSON 'message' must be an object")
                    text = str(msg.get("content") or "")
                    fr = c0.get("finish_reason")
                    finish_reason = str(fr) if fr is not None else None
                    usage = normalize_usage(data.get("usage"))
            self._record_success()
            return GenerationResponse(
                egress_id=egress_id,
                text=text,
                finish_reason=finish_reason,
                usage=usage,
                request_sent_ms=sent_ms,
                first_byte_ms=first_byte_ms,
                complete_ms=complete_ms,
            )
        except asyncio.CancelledError:
            discarded = True
            finish_reason = "cancelled"
            complete_ms = self.clock_ms()
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            discarded = True
            finish_reason = "timeout"
            complete_ms = self.clock_ms()
            self._record_failure(complete_ms)
            raise ProviderTimeoutError("Provider request timed out") from exc
        except httpx.TransportError as exc:
            discarded = True
            finish_reason = "transport_error"
            complete_ms = self.clock_ms()
            self._record_failure(complete_ms)
            raise ProviderError(f"Provider transport error: {exc}") from exc
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            httpx.HTTPError,
        ) as exc:
            discarded = True
            finish_reason = "invalid_response"
            complete_ms = self.clock_ms()
            self._record_failure(complete_ms)
            raise ProviderError(f"Invalid provider response: {exc}") from exc
        finally:
            if probe_acquired:
                self._half_open_in_flight = False
            if req.lane is Origin.MANUAL:
                self._manual_waiting = max(0, self._manual_waiting - 1)
            if self.egress_sink is not None:
                rec = EgressRecord(
                    schema_version=1,
                    egress_id=egress_id,
                    session_id=req.session_id,
                    candidate_id=req.candidate_id,
                    lane=req.lane,
                    provider=self.config.provider,
                    model_id=self.config.model_id,
                    segment_ids=req.segment_ids,
                    sent_at_ms=sent_ms,
                    first_byte_ms=first_byte_ms,
                    complete_ms=complete_ms,
                    finish_reason=finish_reason,
                    usage=usage,
                    discarded=discarded,
                )
                self.egress_sink(rec)
            if owns_client and client is not None:
                await client.aclose()

    async def aclose(self) -> None:
        self._cancel_recovery_task()
        if self.client is not None:
            await self.client.aclose()
