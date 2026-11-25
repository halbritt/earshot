"""Source adapters, Nemotron segmenter, recipient binder, and replay player (§6.1, §6.7)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import signal
import sys
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from coach.contracts import (
    SCHEMA_VERSION,
    CanonicalEvent,
    ControlSubtype,
    EventKind,
    HealthState,
    Stability,
    TelemetryLevel,
    TelemetryRecord,
)

__all__ = [
    "AdapterError",
    "AsrWord",
    "BinderConfig",
    "Clock",
    "DeepgramNormalizer",
    "DirectionSample",
    "EarshotGateSample",
    "HealthCallback",
    "ManualClock",
    "MonotonicSessionClock",
    "NemotronSegmenter",
    "NumpySpectralVerifier",
    "PeeceeConfig",
    "PeeceeLeaseUnavailableError",
    "PeeceeProcessError",
    "PeeceeProtocolError",
    "PeeceeRealtimeAdapter",
    "PeeceeSpeechProcess",
    "RawEventSink",
    "RecipientBinder",
    "ReplayPlayer",
    "ReplayTimingMode",
    "SegmenterConfig",
    "SpeakerBinding",
    "SpeakerVerifier",
    "TelemetrySink",
    "parse_earshot_direction_jsonl",
    "parse_nemo_offline_json",
]


DEFAULT_PEECEE_SCRIPT = (
    Path.home() / "git" / "infra" / "hosts" / "peecee" / "config" / "nemo-speech" / "peecee-speech"
)
DEFAULT_WS_URL = "ws://127.0.0.1:18921/v1/audio/transcriptions/realtime"
_PUNCTUATION_ATTACH = frozenset({",", ".", "?", "!", ";", ":"})


class AdapterError(RuntimeError):
    """Base error raised by transcript adapters."""


class PeeceeLeaseUnavailableError(AdapterError):
    """Raised when gpu-fleet cannot provide or retain the peecee marker lease (exit 75)."""


class PeeceeProcessError(AdapterError):
    """Raised when the peecee-speech subprocess fails unexpectedly."""


class PeeceeProtocolError(AdapterError):
    """Raised on malformed handshake or error frames from NeMo-Speech.cpp."""


class ReplayTimingMode(StrEnum):
    """Replay timing modes (§12.6)."""

    RECORDED = "recorded"
    SCALED = "scaled"
    STEPPED = "stepped"


class Clock(Protocol):
    """Monotonic millisecond session clock protocol (§6.4)."""

    def now_ms(self) -> int: ...


class TelemetrySink(Protocol):
    """Structural sink matching BoundedRecorderBuffer.emit."""

    def emit(self, record: TelemetryRecord) -> bool: ...


class RawEventSink(Protocol):
    """Structural sink for raw provider payloads (§6.6)."""

    def record_raw(
        self,
        *,
        session_id: str,
        source_id: str,
        received_ms: int,
        payload: Mapping[str, Any] | str,
    ) -> Any: ...


class SpeakerVerifier(Protocol):
    """Pluggable voice-embedding enrollment and scoring interface (§6.7)."""

    def enroll(self, speaker_id: str, audio_pcm16: bytes, sample_rate: int = 16000) -> None: ...

    def score(self, audio_pcm16: bytes, sample_rate: int = 16000) -> dict[str, float]: ...


HealthCallback = Callable[[HealthState, str, dict[str, Any]], Awaitable[None] | None]


class MonotonicSessionClock:
    """Monotonic clock returning elapsed milliseconds since construction."""

    def __init__(self, start_ms: int = 0) -> None:
        self._origin_ns = time.monotonic_ns()
        self._start_ms = max(0, start_ms)

    def now_ms(self) -> int:
        elapsed_ms = (time.monotonic_ns() - self._origin_ns) // 1_000_000
        return self._start_ms + max(0, int(elapsed_ms))


class ManualClock:
    """Deterministic virtual clock for tests and stepped replay."""

    def __init__(self, initial_ms: int = 0) -> None:
        self._now_ms = max(0, initial_ms)

    def now_ms(self) -> int:
        return self._now_ms

    def set_ms(self, value_ms: int) -> int:
        if value_ms < self._now_ms:
            raise ValueError(f"ManualClock cannot move backwards: {value_ms} < {self._now_ms}")
        self._now_ms = value_ms
        return self._now_ms

    def advance(self, delta_ms: int) -> int:
        if delta_ms < 0:
            raise ValueError(f"delta_ms must be >= 0, got {delta_ms}")
        self._now_ms += delta_ms
        return self._now_ms


@dataclass(frozen=True, slots=True)
class AsrWord:
    """Single word emitted by NeMo-Speech / Nemotron ASR."""

    word: str
    start_ms: int
    end_ms: int
    speaker: str | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError(f"AsrWord.start_ms must be >= 0, got {self.start_ms}")
        if self.end_ms < self.start_ms:
            raise ValueError(f"AsrWord.end_ms ({self.end_ms}) < start_ms ({self.start_ms})")


@dataclass(frozen=True, slots=True)
class EarshotGateSample:
    """Speech and 9 dB echo-residual gate sample from earshot front end (§6.7)."""

    at_ms: int
    speech_active: bool
    echo_residual_ok: bool = True
    playback_active: bool = False
    azimuth_deg: float | None = None
    contrast: float | None = None
    level_db: float | None = None

    @property
    def is_quality_ok(self) -> bool:
        return self.speech_active and self.echo_residual_ok


@dataclass(frozen=True, slots=True)
class DirectionSample:
    """Azimuth sample from `earshot direction` JSONL (§6.7)."""

    t_ms: int
    azimuth_deg: float
    contrast: float
    level_db: float


@dataclass(frozen=True, slots=True)
class SpeakerBinding:
    """Resolved canonical speaker binding for a Sortformer cluster."""

    speaker_id: str
    speaker_confidence: float
    speaker_uncertain: bool


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    """Configuration for punctuation-free Nemotron word segmentation."""

    channel: int = 0
    inter_word_gap_ms: int = 600
    boundary_gap_ms: int = 600
    max_segment_words: int = 25
    max_segment_ms: int = 8000
    asr_confidence_floor: float | None = None


@dataclass(frozen=True, slots=True)
class BinderConfig:
    """Configuration for Sortformer cluster -> ME / S1..Sn / S? recipient binding."""

    recipient_id: str = "ME"
    recipient_azimuth_deg: float | None = None
    seat_azimuths_deg: Mapping[str, float] = field(default_factory=dict)
    azimuth_tolerance_deg: float = 25.0
    min_confident_duration_ms: int = 1000
    min_confidence: float = 0.65
    margin_threshold: float = 0.10
    min_direction_contrast: float = 1.5
    auto_enroll_unknown: bool = False


@dataclass(frozen=True, slots=True)
class PeeceeConfig:
    """Configuration for peecee-speech subprocess and realtime WebSocket client."""

    ws_url: str = DEFAULT_WS_URL
    executable: Path | str = DEFAULT_PEECEE_SCRIPT
    language: str = "en-US"
    sample_rate: int = 16000
    word_timestamps: bool = True
    speaker_diarization: bool = True
    endpointing_ms: int | None = None
    api_key: str | None = None
    frame_duration_ms: int = 160
    auto_commit_silence_ms: int | None = 640
    max_uncommitted_audio_ms: int = 10_000
    ready_timeout_s: float = 45.0
    ssh_stop_grace_s: float = 2.0
    kill_stop_grace_s: float = 3.0
    extra_args: tuple[str, ...] = ()

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * 2 * self.frame_duration_ms // 1000)


def parse_earshot_direction_jsonl(raw: str | Iterable[str]) -> list[DirectionSample]:
    """Parse `earshot direction` JSONL lines or JSON array into DirectionSample objects."""
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            parsed = json.loads(s)
            if not isinstance(parsed, list):
                raise ValueError(
                    f"Expected JSON array in earshot direction output, got {type(parsed)}"
                )
            return [
                DirectionSample(
                    t_ms=max(0, round(float(obj["t"]) * 1000.0)),
                    azimuth_deg=float(obj["azimuth"]) % 360.0,
                    contrast=float(obj.get("contrast", 1.0)),
                    level_db=float(obj.get("level_db", -60.0)),
                )
                for obj in parsed
                if isinstance(obj, Mapping)
            ]
        lines: Iterable[str] = raw.splitlines()
    else:
        lines = raw
    samples: list[DirectionSample] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if not isinstance(obj, Mapping):
            raise ValueError(f"Expected JSON object in earshot direction output, got {type(obj)}")
        samples.append(
            DirectionSample(
                t_ms=max(0, round(float(obj["t"]) * 1000.0)),
                azimuth_deg=float(obj["azimuth"]) % 360.0,
                contrast=float(obj.get("contrast", 1.0)),
                level_db=float(obj.get("level_db", -60.0)),
            )
        )
    return samples


class NumpySpectralVerifier:
    """Zero-dependency spectral cosine-similarity voice verifier using numpy."""

    def __init__(self, n_bins: int = 24, n_fft: int = 512, hop: int = 256) -> None:
        if n_bins < 4:
            raise ValueError("n_bins must be >= 4")
        self._n_bins = n_bins
        self._n_fft = n_fft
        self._hop = hop
        self._profiles: dict[str, np.ndarray] = {}

    def _extract_embedding(self, audio_pcm16: bytes, sample_rate: int) -> np.ndarray | None:
        if len(audio_pcm16) < 4:
            return None
        usable = len(audio_pcm16) - (len(audio_pcm16) % 2)
        pcm = np.frombuffer(audio_pcm16[:usable], dtype="<i2").astype(np.float64) / 32768.0
        if pcm.size < 32:
            return None
        rms = float(np.sqrt(np.mean(pcm * pcm)))
        if rms < 1e-4:
            return None
        # Pre-emphasis + framed log-power spectrum averaged over log-spaced frequency bands.
        emphasized = np.empty_like(pcm)
        emphasized[0] = pcm[0]
        emphasized[1:] = pcm[1:] - 0.97 * pcm[:-1]
        win_len = min(self._n_fft, emphasized.size)
        window = np.hanning(win_len)
        frames: list[np.ndarray] = []
        step = max(64, min(self._hop, win_len // 2))
        for start in range(0, emphasized.size - win_len + 1, step):
            frames.append(emphasized[start : start + win_len] * window)
        if not frames:
            frames.append(emphasized * np.hanning(emphasized.size))
        stacked = np.vstack(frames)
        spec = np.abs(np.fft.rfft(stacked, n=self._n_fft, axis=1)) ** 2
        power = np.mean(spec, axis=0) + 1e-10
        freqs = np.fft.rfftfreq(self._n_fft, d=1.0 / max(1, sample_rate))
        lo_freq = 120.0
        hi_freq = min(7200.0, float(sample_rate) * 0.48)
        edges = np.geomspace(lo_freq, max(lo_freq + 50.0, hi_freq), self._n_bins + 1)
        feats = np.empty(self._n_bins, dtype=np.float64)
        for idx in range(self._n_bins):
            mask = (freqs >= edges[idx]) & (freqs <= edges[idx + 1])
            if np.any(mask):
                feats[idx] = math.log(float(np.mean(power[mask])))
            else:
                nearest = int(np.argmin(np.abs(freqs - 0.5 * (edges[idx] + edges[idx + 1]))))
                feats[idx] = math.log(float(power[nearest]))
        feats -= float(np.mean(feats))
        norm = float(np.linalg.norm(feats))
        if norm <= 1e-9:
            return None
        return feats / norm

    def enroll(self, speaker_id: str, audio_pcm16: bytes, sample_rate: int = 16000) -> None:
        vec = self._extract_embedding(audio_pcm16, sample_rate)
        if vec is None:
            raise ValueError(f"Audio too short or silent to enroll speaker {speaker_id!r}")
        if speaker_id in self._profiles:
            merged = self._profiles[speaker_id] + vec
            norm = float(np.linalg.norm(merged))
            self._profiles[speaker_id] = merged / norm if norm > 1e-9 else vec
        else:
            self._profiles[speaker_id] = vec

    def score(self, audio_pcm16: bytes, sample_rate: int = 16000) -> dict[str, float]:
        if not self._profiles:
            return {}
        vec = self._extract_embedding(audio_pcm16, sample_rate)
        if vec is None:
            return {}
        out: dict[str, float] = {}
        for spk_id, ref in self._profiles.items():
            cos = float(np.clip(np.dot(vec, ref), -1.0, 1.0))
            out[spk_id] = max(0.0, cos) ** 2
        return out


def _angular_distance_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


class RecipientBinder:
    """Binds Sortformer speaker clusters (1-4) to ME / S1..Sn / S? (§6.4, §6.7)."""

    def __init__(
        self,
        *,
        session_id: str,
        source_id: str = "peecee",
        config: BinderConfig | None = None,
        verifier: SpeakerVerifier | None = None,
        sequence_fn: Callable[[], int] | None = None,
        revision_tracker: dict[str, int] | None = None,
    ) -> None:
        self._session_id = session_id
        self._source_id = source_id
        self._config = config or BinderConfig()
        self._verifier = verifier
        self._seq_counter = 0
        self._has_custom_sequence_fn = sequence_fn is not None
        self._has_custom_revisions = revision_tracker is not None
        self._sequence_fn = sequence_fn
        self._revisions = revision_tracker if revision_tracker is not None else {}
        self._epoch = 0
        self._next_anon_idx = 1
        self._allocated_anon_labels: set[str] = set()
        self._cluster_bindings: dict[tuple[int, str], SpeakerBinding] = {}
        self._cluster_segments: dict[tuple[int, str], list[str]] = {}
        self._directions: list[DirectionSample] = []
        self._event_counter = 0

    def attach_trackers(
        self,
        *,
        sequence_fn: Callable[[], int],
        revision_tracker: dict[str, int],
    ) -> None:
        """Share sequence and revision trackers with a parent segmenter."""
        if not self._has_custom_sequence_fn:
            self._sequence_fn = sequence_fn
        if not self._has_custom_revisions:
            for seg_id, rev in self._revisions.items():
                revision_tracker[seg_id] = max(revision_tracker.get(seg_id, 0), rev)
            self._revisions = revision_tracker

    @property
    def epoch(self) -> int:
        return self._epoch

    def reset_diarizer_epoch(self) -> int:
        """Advance diarizer epoch after input_audio_buffer.commit resets Sortformer."""
        self._epoch += 1
        return self._epoch

    def feed_direction(self, sample: DirectionSample) -> None:
        self._directions.append(sample)

    def feed_direction_jsonl(self, raw: str | Iterable[str]) -> int:
        samples = parse_earshot_direction_jsonl(raw)
        self._directions.extend(samples)
        return len(samples)

    def _next_sequence(self) -> int:
        if self._sequence_fn is not None:
            return self._sequence_fn()
        self._seq_counter += 1
        return self._seq_counter

    def _next_event_id(self, seg_id: str) -> str:
        self._event_counter += 1
        return f"spk-{seg_id}-{self._event_counter}"

    def _next_revision(self, seg_id: str) -> int:
        rev = self._revisions.get(seg_id, 0) + 1
        self._revisions[seg_id] = rev
        return rev

    def _seat_table(self) -> dict[str, float]:
        seats = dict(self._config.seat_azimuths_deg)
        if (
            self._config.recipient_azimuth_deg is not None
            and self._config.recipient_id not in seats
        ):
            seats[self._config.recipient_id] = self._config.recipient_azimuth_deg
        return seats

    def _mean_azimuth(self, start_ms: int, end_ms: int) -> float | None:
        relevant = [
            d
            for d in self._directions
            if start_ms <= d.t_ms <= end_ms and d.contrast >= self._config.min_direction_contrast
        ]
        if not relevant:
            # Allow slight window expansion (+/- 500 ms) around segment bounds.
            relevant = [
                d
                for d in self._directions
                if (start_ms - 500) <= d.t_ms <= (end_ms + 500)
                and d.contrast >= self._config.min_direction_contrast
            ]
        if not relevant:
            return None
        sin_sum = sum(d.contrast * math.sin(math.radians(d.azimuth_deg)) for d in relevant)
        cos_sum = sum(d.contrast * math.cos(math.radians(d.azimuth_deg)) for d in relevant)
        if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
            return None
        return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0

    def _allocate_anon_label(self, epoch: int) -> str:
        del epoch
        used = {
            b.speaker_id
            for b in self._cluster_bindings.values()
            if b.speaker_id not in ("S?", self._config.recipient_id)
        } | self._allocated_anon_labels
        idx = self._next_anon_idx
        while f"S{idx}" in used:
            idx += 1
        label = f"S{idx}"
        self._allocated_anon_labels.add(label)
        self._next_anon_idx = idx + 1
        return label

    def _infer_binding(
        self,
        *,
        epoch: int,
        raw_cluster: str,
        start_ms: int,
        end_ms: int,
        azimuth_deg: float | None,
        audio_pcm16: bytes | None,
    ) -> SpeakerBinding:
        if not raw_cluster or raw_cluster == "0":
            return SpeakerBinding(speaker_id="S?", speaker_confidence=0.0, speaker_uncertain=True)

        duration_ms = max(0, end_ms - start_ms)
        short_span = duration_ms < self._config.min_confident_duration_ms
        key = (epoch, raw_cluster)
        prior = self._cluster_bindings.get(key)

        candidates: dict[str, float] = {}
        ambiguous = False

        if self._verifier is not None and audio_pcm16:
            scores = self._verifier.score(audio_pcm16)
            if scores:
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                top_id, top_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                if (top_score - second_score) < self._config.margin_threshold and len(ranked) > 1:
                    ambiguous = True
                if top_score >= self._config.min_confidence:
                    candidates[top_id] = top_score

        eff_az = azimuth_deg if azimuth_deg is not None else self._mean_azimuth(start_ms, end_ms)
        if eff_az is not None:
            seats = self._seat_table()
            seat_scores: list[tuple[str, float]] = []
            for spk_id, seat_az in seats.items():
                dist = _angular_distance_deg(eff_az, seat_az)
                if dist <= self._config.azimuth_tolerance_deg:
                    conf = max(
                        0.5, 1.0 - (dist / max(1.0, self._config.azimuth_tolerance_deg)) * 0.45
                    )
                    seat_scores.append((spk_id, conf))
            seat_scores.sort(key=lambda kv: kv[1], reverse=True)
            if seat_scores:
                best_seat, best_conf = seat_scores[0]
                if (
                    len(seat_scores) > 1
                    and (best_conf - seat_scores[1][1]) < self._config.margin_threshold
                ):
                    ambiguous = True
                if best_seat not in candidates or best_conf > candidates[best_seat]:
                    candidates[best_seat] = best_conf

        if candidates:
            ranked_cand = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
            chosen_id, chosen_conf = ranked_cand[0]
            if (
                len(ranked_cand) > 1
                and (chosen_conf - ranked_cand[1][1]) < self._config.margin_threshold
            ):
                ambiguous = True
            uncertain = short_span or ambiguous or chosen_conf < self._config.min_confidence
            return SpeakerBinding(
                speaker_id=chosen_id,
                speaker_confidence=round(chosen_conf, 4),
                speaker_uncertain=uncertain,
            )

        if prior is not None and prior.speaker_id != "S?":
            uncertain = short_span or prior.speaker_uncertain
            return SpeakerBinding(
                speaker_id=prior.speaker_id,
                speaker_confidence=prior.speaker_confidence,
                speaker_uncertain=uncertain,
            )

        if short_span:
            return SpeakerBinding(speaker_id="S?", speaker_confidence=0.3, speaker_uncertain=True)

        anon_id = self._allocate_anon_label(epoch)
        return SpeakerBinding(speaker_id=anon_id, speaker_confidence=0.6, speaker_uncertain=False)

    def bind_segment(
        self,
        segment_id: str,
        raw_cluster: str | None,
        start_ms: int,
        end_ms: int,
        *,
        azimuth_deg: float | None = None,
        audio_pcm16: bytes | None = None,
    ) -> SpeakerBinding:
        """Bind a segment to ME / S1..Sn / S? without emitting retrospective updates."""
        binding, _ = self.bind_segment_with_updates(
            segment_id,
            raw_cluster,
            start_ms,
            end_ms,
            received_ms=end_ms,
            azimuth_deg=azimuth_deg,
            audio_pcm16=audio_pcm16,
            emit_updates=False,
        )
        return binding

    def bind_segment_with_updates(
        self,
        segment_id: str,
        raw_cluster: str | None,
        start_ms: int,
        end_ms: int,
        *,
        received_ms: int,
        azimuth_deg: float | None = None,
        audio_pcm16: bytes | None = None,
        emit_updates: bool = True,
    ) -> tuple[SpeakerBinding, list[CanonicalEvent]]:
        """Bind segment and optionally emit SPEAKER_UPDATE for prior cluster segments."""
        cluster = (raw_cluster or "").strip()
        binding = self._infer_binding(
            epoch=self._epoch,
            raw_cluster=cluster,
            start_ms=start_ms,
            end_ms=end_ms,
            azimuth_deg=azimuth_deg,
            audio_pcm16=audio_pcm16,
        )
        if not cluster or cluster == "0":
            return binding, []

        key = (self._epoch, cluster)
        prior = self._cluster_bindings.get(key)
        prior_segs = list(self._cluster_segments.get(key, ()))
        updates: list[CanonicalEvent] = []

        should_store = prior is None or (
            binding.speaker_id != "S?"
            and (
                prior.speaker_id == "S?"
                or (not binding.speaker_uncertain and binding.speaker_id != prior.speaker_id)
                or (prior.speaker_uncertain and not binding.speaker_uncertain)
            )
        )
        if should_store:
            self._cluster_bindings[key] = binding
            if (
                emit_updates
                and prior is not None
                and prior_segs
                and (
                    binding.speaker_id != prior.speaker_id
                    or binding.speaker_uncertain != prior.speaker_uncertain
                )
            ):
                for prev_seg_id in prior_segs:
                    updates.append(
                        CanonicalEvent(
                            schema_version=SCHEMA_VERSION,
                            session_id=self._session_id,
                            event_id=self._next_event_id(prev_seg_id),
                            kind=EventKind.SPEAKER_UPDATE,
                            received_ms=received_ms,
                            source_sequence=self._next_sequence(),
                            source_id=self._source_id,
                            segment_id=prev_seg_id,
                            source_revision=self._next_revision(prev_seg_id),
                            speaker_id=binding.speaker_id,
                            speaker_confidence=binding.speaker_confidence,
                            speaker_uncertain=binding.speaker_uncertain,
                        )
                    )

        segs = self._cluster_segments.setdefault(key, [])
        if segment_id not in segs:
            segs.append(segment_id)
        if (
            self._config.auto_enroll_unknown
            and self._verifier is not None
            and audio_pcm16
            and not binding.speaker_uncertain
            and binding.speaker_id not in ("S?", self._config.recipient_id)
            and (end_ms - start_ms) >= self._config.min_confident_duration_ms
        ):
            with contextlib.suppress(ValueError):
                self._verifier.enroll(binding.speaker_id, audio_pcm16)
        return binding, updates

    def reconcile(
        self,
        raw_cluster: str,
        *,
        received_ms: int,
        new_speaker_id: str | None = None,
        confidence: float | None = None,
        uncertain: bool = False,
        azimuth_deg: float | None = None,
        audio_pcm16: bytes | None = None,
        epoch: int | None = None,
    ) -> list[CanonicalEvent]:
        """Reconcile a Sortformer cluster and emit SPEAKER_UPDATE for prior segments."""
        target_epoch = self._epoch if epoch is None else epoch
        cluster = raw_cluster.strip()
        key = (target_epoch, cluster)
        seg_ids = self._cluster_segments.get(key, [])
        if new_speaker_id is not None:
            resolved = SpeakerBinding(
                speaker_id=new_speaker_id,
                speaker_confidence=1.0 if confidence is None else confidence,
                speaker_uncertain=uncertain,
            )
        else:
            resolved = self._infer_binding(
                epoch=target_epoch,
                raw_cluster=cluster,
                start_ms=0,
                end_ms=self._config.min_confident_duration_ms,
                azimuth_deg=azimuth_deg,
                audio_pcm16=audio_pcm16,
            )
        prior = self._cluster_bindings.get(key)
        self._cluster_bindings[key] = resolved
        if prior is not None and (
            prior.speaker_id == resolved.speaker_id
            and prior.speaker_uncertain == resolved.speaker_uncertain
        ):
            return []
        events: list[CanonicalEvent] = []
        for seg_id in seg_ids:
            events.append(
                CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._session_id,
                    event_id=self._next_event_id(seg_id),
                    kind=EventKind.SPEAKER_UPDATE,
                    received_ms=received_ms,
                    source_sequence=self._next_sequence(),
                    source_id=self._source_id,
                    segment_id=seg_id,
                    source_revision=self._next_revision(seg_id),
                    speaker_id=resolved.speaker_id,
                    speaker_confidence=resolved.speaker_confidence,
                    speaker_uncertain=resolved.speaker_uncertain,
                )
            )
        return events


class NemotronSegmenter:
    """Punctuation-free word/interim segmenter for Nemotron + earshot gates (§6.7)."""

    def __init__(
        self,
        *,
        session_id: str,
        source_id: str = "peecee",
        config: SegmenterConfig | None = None,
        binder: RecipientBinder | None = None,
        sequence_fn: Callable[[], int] | None = None,
        revision_tracker: dict[str, int] | None = None,
    ) -> None:
        self._session_id = session_id
        self._source_id = source_id
        self._config = config or SegmenterConfig()
        self._revisions = revision_tracker if revision_tracker is not None else {}
        self._seq_counter = 0
        self._sequence_fn = sequence_fn
        if binder is not None:
            binder.attach_trackers(
                sequence_fn=self._next_sequence,
                revision_tracker=self._revisions,
            )
            self._binder = binder
        else:
            self._binder = RecipientBinder(
                session_id=session_id,
                source_id=source_id,
                sequence_fn=self._next_sequence,
                revision_tracker=self._revisions,
            )
        self._event_counter = 0
        self._interim_epoch = 0
        self._partial_buffer = ""
        self._active_interim_seg_id: str | None = None
        self._active_interim_start_ms = 0
        self._pending_words: list[AsrWord] = []
        self._gates: list[EarshotGateSample] = []
        self._last_gate: EarshotGateSample | None = None
        self._last_committed_end_ms = 0
        self._last_speaker_id: str | None = None
        self._used_seg_ids: set[str] = set()

    @property
    def binder(self) -> RecipientBinder:
        return self._binder

    def _next_sequence(self) -> int:
        if self._sequence_fn is not None:
            return self._sequence_fn()
        self._seq_counter += 1
        return self._seq_counter

    def _next_event_id(self, prefix: str) -> str:
        self._event_counter += 1
        return f"{prefix}-{self._event_counter}"

    def _next_revision(self, seg_id: str) -> int:
        rev = self._revisions.get(seg_id, 0) + 1
        self._revisions[seg_id] = rev
        return rev

    def _gate_is_low_quality(self, start_ms: int, end_ms: int) -> bool:
        if not self._gates:
            return False
        overlapping = [g for g in self._gates if start_ms <= g.at_ms <= end_ms]
        if overlapping:
            return any(not g.is_quality_ok for g in overlapping)
        prior = [g for g in self._gates if g.at_ms <= start_ms]
        if prior:
            return not prior[-1].is_quality_ok
        return False

    def _gate_closed_between(self, prev_end_ms: int, curr_start_ms: int) -> bool:
        if len(self._gates) < 2:
            return False
        for idx in range(1, len(self._gates)):
            prev_g = self._gates[idx - 1]
            curr_g = self._gates[idx]
            if (
                prev_g.speech_active
                and not curr_g.speech_active
                and prev_end_ms < curr_g.at_ms <= curr_start_ms
            ):
                return True
        return False

    def _make_segment_id(self, start_ms: int) -> str:
        base = f"seg-ch{self._config.channel}-{start_ms:08d}"
        if base not in self._used_seg_ids:
            self._used_seg_ids.add(base)
            return base
        suffix = 1
        while f"{base}-{suffix}" in self._used_seg_ids:
            suffix += 1
        seg_id = f"{base}-{suffix}"
        self._used_seg_ids.add(seg_id)
        return seg_id

    def reset_pending(self) -> None:
        """Clear buffered interim and pending words without emitting events."""
        self._partial_buffer = ""
        self._active_interim_seg_id = None
        self._pending_words.clear()

    def feed_partial(
        self,
        delta_text: str,
        *,
        audio_processed_ms: int,
        received_ms: int,
        speaker: str | None = None,
        is_replacement: bool = False,
    ) -> list[CanonicalEvent]:
        """Emit an INTERIM segment event for the trailing partial word in `.delta`."""
        if not delta_text:
            return []
        if not is_replacement and delta_text == self._partial_buffer:
            return []
        if is_replacement or (
            self._partial_buffer
            and delta_text.startswith(self._partial_buffer)
            and len(delta_text) > len(self._partial_buffer)
        ):
            self._partial_buffer = delta_text
        else:
            self._partial_buffer += delta_text

        stripped = self._partial_buffer.rstrip()
        if not stripped:
            return []
        trailing_word = stripped.rsplit(maxsplit=1)[-1]

        if self._active_interim_seg_id is None:
            self._interim_epoch += 1
            self._active_interim_start_ms = max(self._last_committed_end_ms, audio_processed_ms)
            self._active_interim_seg_id = f"seg-ch{self._config.channel}-interim"

        seg_id = self._active_interim_seg_id
        start_ms = self._active_interim_start_ms
        end_ms = max(start_ms, audio_processed_ms)
        low_gate = self._gate_is_low_quality(start_ms, end_ms)
        speaker_id = speaker if speaker and speaker != "0" else "S?"
        return [
            CanonicalEvent(
                schema_version=SCHEMA_VERSION,
                session_id=self._session_id,
                event_id=self._next_event_id("evt-int"),
                kind=EventKind.SEGMENT,
                received_ms=received_ms,
                source_sequence=self._next_sequence(),
                source_id=self._source_id,
                segment_id=seg_id,
                source_revision=self._next_revision(seg_id),
                channel=self._config.channel,
                speaker_id=speaker_id,
                speaker_confidence=0.3 if speaker_id == "S?" else 0.7,
                start_ms=start_ms,
                end_ms=end_ms,
                text=trailing_word,
                stability=Stability.INTERIM,
                low_quality_gate=low_gate,
                asr_low_confidence=low_gate,
                speaker_uncertain=speaker_id == "S?",
            )
        ]

    def _clear_active_interim(self, *, received_ms: int) -> list[CanonicalEvent]:
        self._partial_buffer = ""
        if self._active_interim_seg_id is None:
            return []
        seg_id = self._active_interim_seg_id
        self._active_interim_seg_id = None
        return [
            CanonicalEvent(
                schema_version=SCHEMA_VERSION,
                session_id=self._session_id,
                event_id=self._next_event_id("evt-clr"),
                kind=EventKind.SEGMENT,
                received_ms=received_ms,
                source_sequence=self._next_sequence(),
                source_id=self._source_id,
                segment_id=seg_id,
                source_revision=self._next_revision(seg_id),
                channel=self._config.channel,
                speaker_id="S?",
                start_ms=self._active_interim_start_ms,
                end_ms=self._active_interim_start_ms,
                text="",
                stability=Stability.INTERIM,
                speaker_uncertain=True,
            )
        ]

    def feed_gate(self, gate: EarshotGateSample, *, received_ms: int) -> list[CanonicalEvent]:
        """Record an earshot speech/echo gate sample and emit BOUNDARY on gate close."""
        self._gates.append(gate)
        events: list[CanonicalEvent] = []
        if self._last_gate is not None and self._last_gate.speech_active and not gate.speech_active:
            if self._pending_words:
                events.extend(
                    self._flush_pending(
                        received_ms=received_ms,
                        emit_boundary=True,
                        boundary_reason="gate_close",
                    )
                )
            else:
                events.append(
                    CanonicalEvent(
                        schema_version=SCHEMA_VERSION,
                        session_id=self._session_id,
                        event_id=self._next_event_id("evt-bnd"),
                        kind=EventKind.BOUNDARY,
                        received_ms=received_ms,
                        source_sequence=self._next_sequence(),
                        source_id=self._source_id,
                        channel=self._config.channel,
                        speaker_id=self._last_speaker_id,
                        end_ms=max(gate.at_ms, self._last_committed_end_ms),
                        boundary="gate_close",
                    )
                )
        self._last_gate = gate
        return events

    def _flush_pending(
        self,
        *,
        received_ms: int,
        emit_boundary: bool,
        boundary_reason: str | None = None,
        audio_pcm16_fn: Callable[[int, int], bytes | None] | None = None,
    ) -> list[CanonicalEvent]:
        if not self._pending_words:
            return []
        words = list(self._pending_words)
        self._pending_words.clear()
        start_ms = words[0].start_ms
        end_ms = max(start_ms, max(w.end_ms for w in words))
        self._last_committed_end_ms = max(self._last_committed_end_ms, end_ms)

        parts: list[str] = []
        for w in words:
            tok = w.word.strip()
            if not tok:
                continue
            if tok in _PUNCTUATION_ATTACH and parts:
                parts[-1] = f"{parts[-1]}{tok}"
            else:
                parts.append(tok)
        text = " ".join(parts)

        seg_id = self._make_segment_id(start_ms)
        seg_seq = self._next_sequence()
        speaker_counts = Counter(
            w.speaker for w in words if w.speaker is not None and w.speaker.strip() not in ("", "0")
        )
        raw_cluster = speaker_counts.most_common(1)[0][0] if speaker_counts else None
        audio_slice = audio_pcm16_fn(start_ms, end_ms) if audio_pcm16_fn is not None else None

        binding, spk_updates = self._binder.bind_segment_with_updates(
            seg_id,
            raw_cluster,
            start_ms,
            end_ms,
            received_ms=received_ms,
            audio_pcm16=audio_slice,
            emit_updates=True,
        )
        self._last_speaker_id = binding.speaker_id

        confs = [float(w.confidence) for w in words]
        mean_conf = sum(confs) / len(confs) if confs else 1.0
        low_gate = self._gate_is_low_quality(start_ms, end_ms)
        low_conf = low_gate or (
            self._config.asr_confidence_floor is not None
            and mean_conf < self._config.asr_confidence_floor
        )

        events: list[CanonicalEvent] = [
            CanonicalEvent(
                schema_version=SCHEMA_VERSION,
                session_id=self._session_id,
                event_id=self._next_event_id("evt-seg"),
                kind=EventKind.SEGMENT,
                received_ms=received_ms,
                source_sequence=seg_seq,
                source_id=self._source_id,
                segment_id=seg_id,
                source_revision=self._next_revision(seg_id),
                channel=self._config.channel,
                speaker_id=binding.speaker_id,
                speaker_confidence=binding.speaker_confidence,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                stability=Stability.COMMITTED,
                boundary=boundary_reason,
                asr_confidence=round(mean_conf, 4),
                low_quality_gate=low_gate,
                asr_low_confidence=low_conf,
                speaker_uncertain=binding.speaker_uncertain,
            )
        ]
        events.extend(spk_updates)
        if emit_boundary:
            events.append(
                CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._session_id,
                    event_id=self._next_event_id("evt-bnd"),
                    kind=EventKind.BOUNDARY,
                    received_ms=received_ms,
                    source_sequence=self._next_sequence(),
                    source_id=self._source_id,
                    channel=self._config.channel,
                    speaker_id=binding.speaker_id,
                    end_ms=end_ms,
                    boundary=boundary_reason or "segment",
                )
            )
        return events

    def feed_words(
        self,
        words: Sequence[AsrWord],
        *,
        received_ms: int,
        gates: Sequence[EarshotGateSample] = (),
        audio_pcm16_fn: Callable[[int, int], bytes | None] | None = None,
        flush_end: bool = False,
        emit_end_boundary: bool = True,
    ) -> list[CanonicalEvent]:
        """Ingest completed timestamped words and emit COMMITTED/BOUNDARY events."""
        self._gates.extend(gates)
        if gates:
            self._last_gate = gates[-1]
        events = self._clear_active_interim(received_ms=received_ms)

        for word in words:
            tok = word.word.strip()
            if not tok:
                continue
            if tok in _PUNCTUATION_ATTACH and self._pending_words:
                prev = self._pending_words[-1]
                self._pending_words[-1] = AsrWord(
                    word=f"{prev.word.strip()}{tok}",
                    start_ms=prev.start_ms,
                    end_ms=max(prev.end_ms, word.end_ms),
                    speaker=prev.speaker,
                    confidence=prev.confidence,
                )
                continue

            if self._pending_words:
                prev = self._pending_words[-1]
                prev_spk: str | None = None
                for pw in reversed(self._pending_words):
                    if pw.speaker not in (None, "", "0"):
                        prev_spk = pw.speaker
                        break
                curr_spk = word.speaker if word.speaker not in (None, "", "0") else None
                gap_ms = word.start_ms - prev.end_ms
                dur_ms = word.end_ms - self._pending_words[0].start_ms

                if prev_spk is not None and curr_spk is not None and curr_spk != prev_spk:
                    events.extend(
                        self._flush_pending(
                            received_ms=received_ms,
                            emit_boundary=True,
                            boundary_reason="speaker_change",
                            audio_pcm16_fn=audio_pcm16_fn,
                        )
                    )
                elif gap_ms >= self._config.inter_word_gap_ms:
                    events.extend(
                        self._flush_pending(
                            received_ms=received_ms,
                            emit_boundary=gap_ms >= self._config.boundary_gap_ms,
                            boundary_reason="silence",
                            audio_pcm16_fn=audio_pcm16_fn,
                        )
                    )
                elif self._gate_closed_between(prev.end_ms, word.start_ms):
                    events.extend(
                        self._flush_pending(
                            received_ms=received_ms,
                            emit_boundary=True,
                            boundary_reason="gate_close",
                            audio_pcm16_fn=audio_pcm16_fn,
                        )
                    )
                elif (
                    len(self._pending_words) >= self._config.max_segment_words
                    or dur_ms >= self._config.max_segment_ms
                ):
                    events.extend(
                        self._flush_pending(
                            received_ms=received_ms,
                            emit_boundary=False,
                            boundary_reason="segment_cap",
                            audio_pcm16_fn=audio_pcm16_fn,
                        )
                    )
            self._pending_words.append(word)

        if flush_end and self._pending_words:
            events.extend(
                self._flush_pending(
                    received_ms=received_ms,
                    emit_boundary=emit_end_boundary,
                    boundary_reason="commit" if emit_end_boundary else None,
                    audio_pcm16_fn=audio_pcm16_fn,
                )
            )
        return events

    def flush(
        self,
        *,
        received_ms: int,
        emit_boundary: bool = True,
        boundary_reason: str = "commit",
        audio_pcm16_fn: Callable[[int, int], bytes | None] | None = None,
    ) -> list[CanonicalEvent]:
        """Flush any buffered completed words as a COMMITTED segment."""
        events = self._clear_active_interim(received_ms=received_ms)
        events.extend(
            self._flush_pending(
                received_ms=received_ms,
                emit_boundary=emit_boundary,
                boundary_reason=boundary_reason,
                audio_pcm16_fn=audio_pcm16_fn,
            )
        )
        return events


def _extract_nemo_words(payload: Mapping[str, Any], offset_ms: int = 0) -> list[AsrWord]:
    raw_words = payload.get("words")
    if raw_words is None and isinstance(payload.get("item"), Mapping):
        raw_words = payload["item"].get("words")
    if not isinstance(raw_words, Sequence):
        return []
    out: list[AsrWord] = []
    for item in raw_words:
        if not isinstance(item, Mapping):
            continue
        word_text = str(item.get("word", ""))
        if not word_text.strip():
            continue
        start_raw = float(item.get("start", 0.0))
        end_raw = float(item.get("end", start_raw))
        start_ms = max(0, offset_ms + round(start_raw * 1000.0))
        end_ms = max(start_ms, offset_ms + round(end_raw * 1000.0))
        conf = float(item.get("confidence", 1.0))
        spk_raw = item.get("speaker")
        spk_str: str | None = None
        if spk_raw is not None:
            s = str(spk_raw).strip()
            if s and s != "0":
                spk_str = s
        out.append(
            AsrWord(
                word=word_text,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=spk_str,
                confidence=conf,
            )
        )
    return out


def parse_nemo_offline_json(
    payload: Mapping[str, Any] | str | Path,
    *,
    session_id: str,
    source_id: str = "peecee-offline",
    start_received_ms: int = 0,
    segmenter: NemotronSegmenter | None = None,
    binder: RecipientBinder | None = None,
    segmenter_config: SegmenterConfig | None = None,
    gates: Sequence[EarshotGateSample] = (),
    audio_pcm16_fn: Callable[[int, int], bytes | None] | None = None,
) -> list[CanonicalEvent]:
    """Ingest `peecee-speech RECORDING.wav` / `nemo-speech transcribe --stream --json` output."""
    if isinstance(payload, Path):
        data: Mapping[str, Any] = json.loads(payload.read_text(encoding="utf-8"))
    elif isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = payload

    seg = segmenter or NemotronSegmenter(
        session_id=session_id,
        source_id=source_id,
        config=segmenter_config,
        binder=binder,
    )
    words = _extract_nemo_words(data)
    if not words and isinstance(data.get("segments"), Sequence):
        for chunk in data["segments"]:
            if isinstance(chunk, Mapping):
                words.extend(_extract_nemo_words(chunk))
    if not words:
        return []
    events = seg.feed_words(
        words,
        received_ms=start_received_ms,
        gates=gates,
        audio_pcm16_fn=audio_pcm16_fn,
        flush_end=True,
        emit_end_boundary=True,
    )
    # Set realistic monotonic received_ms >= end_ms on offline events if start_received_ms == 0.
    if start_received_ms == 0:
        adjusted: list[CanonicalEvent] = []
        last_rx = 0
        for ev in events:
            if ev.end_ms is not None:
                last_rx = max(last_rx, ev.end_ms)
            rx = max(last_rx, ev.end_ms) if ev.end_ms is not None else last_rx
            if rx != ev.received_ms:
                d = ev.to_dict()
                d["received_ms"] = rx
                adjusted.append(CanonicalEvent.from_dict(d))
            else:
                adjusted.append(ev)
        return adjusted
    return events


def _find_descendant_pids(
    root_pid: int, target_names: frozenset[str] = frozenset({"ssh"})
) -> list[int]:
    """Walk /proc/<pid>/stat and return descendant PIDs matching target command names."""
    children: dict[int, list[int]] = {}
    comms: dict[int, str] = {}
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            rparen = stat_text.rfind(")")
            lparen = stat_text.find("(")
            if lparen < 0 or rparen <= lparen:
                continue
            comm = stat_text[lparen + 1 : rparen]
            rest = stat_text[rparen + 2 :].split()
            if len(rest) < 2:
                continue
            ppid = int(rest[1])
            children.setdefault(ppid, []).append(pid)
            comms[pid] = comm
        except OSError:
            continue

    matched: list[int] = []
    queue: deque[int] = deque(children.get(root_pid, []))
    seen: set[int] = {root_pid}
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        comm_name = comms.get(cur, "")
        if comm_name in target_names:
            matched.append(cur)
        else:
            try:
                cmd_raw = (proc_root / str(cur) / "cmdline").read_bytes()
                argv0 = cmd_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                if Path(argv0).name in target_names:
                    matched.append(cur)
            except OSError:
                pass
        queue.extend(children.get(cur, []))
    return matched


async def _invoke_health_cb(
    cb: HealthCallback | None,
    state: HealthState,
    reason: str,
    detail: dict[str, Any],
) -> None:
    if cb is None:
        return
    res = cb(state, reason, detail)
    if res is not None:
        await res


def _clean_subprocess_env() -> dict[str, str]:
    """Strip active virtualenv from env so external tools (gpu-fleet-run) use system python3."""
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    venv_bins = {str(Path(sys.prefix) / "bin")}
    if venv:
        venv_bins.add(str(Path(venv) / "bin"))
    if "PATH" in env:
        parts = [p for p in env["PATH"].split(os.pathsep) if p and p not in venv_bins]
        env["PATH"] = os.pathsep.join(parts)
    return env


class PeeceeSpeechProcess:
    """Subprocess lifecycle manager for `peecee-speech --serve` (§6.7, §8.6)."""

    def __init__(
        self,
        *,
        session_id: str,
        config: PeeceeConfig | None = None,
        clock: Clock | None = None,
        telemetry_sink: TelemetrySink | None = None,
        health_callback: HealthCallback | None = None,
        gap_callback: Callable[[CanonicalEvent], Awaitable[None] | None] | None = None,
    ) -> None:
        self._session_id = session_id
        self._config = config or PeeceeConfig()
        self._clock = clock or MonotonicSessionClock()
        self._telemetry = telemetry_sink
        self._health_cb = health_callback
        self._gap_cb = gap_callback
        self._proc: asyncio.subprocess.Process | None = None
        self._pgid: int | None = None
        self._stopping = False
        self._logs: deque[str] = deque(maxlen=200)
        self._drain_tasks: list[asyncio.Task[None]] = []
        self._monitor_task: asyncio.Task[None] | None = None
        self._tel_counter = 0
        self._gap_counter = 0
        self._health = HealthState.OK
        self._last_exit_code: int | None = None

    @property
    def health(self) -> HealthState:
        return self._health

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def logs(self) -> list[str]:
        return list(self._logs)

    def _emit_telemetry(
        self,
        *,
        level: TelemetryLevel,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        if self._telemetry is None:
            return
        self._tel_counter += 1
        self._telemetry.emit(
            TelemetryRecord(
                schema_version=SCHEMA_VERSION,
                record_id=f"tel-proc-{self._tel_counter}",
                session_id=self._session_id,
                at_ms=self._clock.now_ms(),
                level=level,
                kind=kind,
                stage="transcript_adapter",
                payload=payload,
            )
        )

    async def _drain_stream(self, stream: asyncio.StreamReader | None, label: str) -> None:
        if stream is None:
            return
        try:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._logs.append(f"[{label}] {line}")
        except asyncio.CancelledError:
            raise

    async def _handle_unexpected_exit(self, exit_code: int) -> None:
        self._last_exit_code = exit_code
        self._health = HealthState.NO_TRANSCRIPT
        reason = "peecee_lease_unavailable" if exit_code == 75 else "peecee_process_exited"
        detail = {
            "health": HealthState.NO_TRANSCRIPT.value,
            "reason": reason,
            "exit_code": exit_code,
        }
        self._emit_telemetry(level=TelemetryLevel.LIFECYCLE, kind="adapter_health", payload=detail)
        await _invoke_health_cb(self._health_cb, HealthState.NO_TRANSCRIPT, reason, detail)
        if self._gap_cb is not None:
            self._gap_counter += 1
            now_ms = self._clock.now_ms()
            gap_ev = CanonicalEvent(
                schema_version=SCHEMA_VERSION,
                session_id=self._session_id,
                event_id=f"ctrl-proc-gap-{self._gap_counter}",
                kind=EventKind.CONTROL,
                received_ms=now_ms,
                source_sequence=self._gap_counter,
                source_id="peecee",
                subtype=ControlSubtype.GAP,
                start_ms=now_ms,
                end_ms=now_ms,
                reason=reason,
            )
            res = self._gap_cb(gap_ev)
            if res is not None:
                await res

    async def _monitor_exit(self) -> None:
        if self._proc is None:
            return
        try:
            code = await self._proc.wait()
            if not self._stopping:
                await self._handle_unexpected_exit(code)
        except asyncio.CancelledError:
            raise

    async def start(self, *, wait_for_ready: bool = True) -> None:
        """Spawn `peecee-speech --serve <language>` in its own session and await readiness."""
        if self._proc is not None and self._proc.returncode is None:
            return
        self._stopping = False
        self._health = HealthState.OK
        cmd = [
            str(self._config.executable),
            "--serve",
            self._config.language,
            *self._config.extra_args,
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env=_clean_subprocess_env(),
        )
        try:
            self._pgid = os.getpgid(self._proc.pid)
        except OSError:
            self._pgid = self._proc.pid
        self._drain_tasks = [
            asyncio.create_task(self._drain_stream(self._proc.stdout, "stdout")),
            asyncio.create_task(self._drain_stream(self._proc.stderr, "stderr")),
        ]
        self._monitor_task = asyncio.create_task(self._monitor_exit())
        if wait_for_ready:
            await self.wait_ready(timeout_s=self._config.ready_timeout_s)

    async def wait_ready(self, *, timeout_s: float = 45.0) -> None:
        """Wait until the forwarded WebSocket endpoint responds or subprocess fails."""
        parsed = urlparse(self._config.ws_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "wss" else 18921)
        deadline = time.monotonic() + timeout_s

        while True:
            if self._proc is not None and self._proc.returncode is not None:
                code = self._proc.returncode
                if self._monitor_task is not None:
                    await self._monitor_task
                if code == 75:
                    raise PeeceeLeaseUnavailableError(
                        "peecee marker GPU lease unavailable or lost (exit 75)"
                    )
                raise PeeceeProcessError(f"peecee-speech exited early with code {code}")

            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=min(1.0, max(0.05, deadline - time.monotonic())),
                )
                try:
                    req = (
                        f"GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
                    ).encode("ascii")
                    writer.write(req)
                    await writer.drain()
                    chunk = await asyncio.wait_for(reader.read(64), timeout=1.0)
                    if chunk:
                        return
                finally:
                    writer.close()
                    with contextlib.suppress(OSError):
                        await writer.wait_closed()
            except OSError, TimeoutError:
                pass

            if time.monotonic() >= deadline:
                await self.stop()
                self._health = HealthState.NO_TRANSCRIPT
                reason = "peecee_ready_timeout"
                detail = {
                    "health": HealthState.NO_TRANSCRIPT.value,
                    "reason": reason,
                    "timeout_s": timeout_s,
                    "host": host,
                    "port": port,
                }
                self._emit_telemetry(
                    level=TelemetryLevel.LIFECYCLE,
                    kind="adapter_health",
                    payload=detail,
                )
                await _invoke_health_cb(self._health_cb, HealthState.NO_TRANSCRIPT, reason, detail)
                raise PeeceeProcessError(
                    f"Timed out waiting {timeout_s:.1f}s for peecee-speech at {host}:{port}"
                )
            await asyncio.sleep(0.15)

    async def stop(self) -> None:
        """Send SIGTERM to ssh child(ren) first, then launcher/PGID, SIGKILL after grace."""
        if self._proc is None:
            return
        self._stopping = True
        proc = self._proc
        if proc.returncode is None:
            ssh_pids = _find_descendant_pids(proc.pid)
            for spid in ssh_pids:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(spid, signal.SIGTERM)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=self._config.ssh_stop_grace_s)

        if proc.returncode is None:
            if self._pgid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._pgid, signal.SIGTERM)
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=self._config.kill_stop_grace_s)
            except TimeoutError:
                if self._pgid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self._pgid, signal.SIGKILL)
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()

        for task in [*self._drain_tasks, self._monitor_task]:
            if task is not None and not task.done():
                task.cancel()
        for task in [*self._drain_tasks, self._monitor_task]:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, OSError):
                    await task
        self._drain_tasks.clear()
        self._monitor_task = None


class PeeceeRealtimeAdapter:
    """Realtime WebSocket client for NeMo-Speech.cpp v0.3.0 / peecee-speech (§6.7)."""

    def __init__(
        self,
        *,
        session_id: str,
        source_id: str = "peecee",
        config: PeeceeConfig | None = None,
        segmenter: NemotronSegmenter | None = None,
        binder: RecipientBinder | None = None,
        clock: Clock | None = None,
        telemetry_sink: TelemetrySink | None = None,
        raw_sink: RawEventSink | None = None,
        health_callback: HealthCallback | None = None,
        process: PeeceeSpeechProcess | None = None,
    ) -> None:
        self._session_id = session_id
        self._source_id = source_id
        self._config = config or PeeceeConfig()
        self._clock = clock or MonotonicSessionClock()
        self._telemetry = telemetry_sink
        self._raw_sink = raw_sink
        self._health_cb = health_callback
        self._process = process
        self._segmenter = segmenter or NemotronSegmenter(
            session_id=session_id,
            source_id=source_id,
            binder=binder,
        )
        self._ws: ClientConnection | None = None
        self._health = HealthState.OK
        self._paused = False
        self._off_record = False
        self._closed_cleanly = False
        self._commit_base_ms = 0
        self._last_utterance_audio_ms = 0
        self._last_epoch_word_end_ms = 0
        self._last_committed_end_ms = 0
        self._utterance_sent_ms = 0
        self._uncommitted_bytes = 0
        self._total_sent_ms = 0
        self._utterance_audio = bytearray()
        self._utterance_audio_base_ms = 0
        self._pending_commits: deque[tuple[int, int]] = deque()
        self._pending_events: asyncio.Queue[CanonicalEvent] = asyncio.Queue()
        self._audio_origin_ms: int | None = None
        self._halted_since_ms: int | None = None
        self._total_halted_ms = 0
        self._off_record_start_ms: int | None = None
        self._silence_accum_ms = 0
        self._had_speech_since_commit = False
        self._tel_counter = 0
        self._ctrl_counter = 0

    @property
    def health(self) -> HealthState:
        return self._health

    @property
    def commit_base_ms(self) -> int:
        return self._commit_base_ms

    @property
    def segmenter(self) -> NemotronSegmenter:
        return self._segmenter

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_off_record(self) -> bool:
        return self._off_record

    def _emit_tel(self, *, level: TelemetryLevel, kind: str, payload: dict[str, Any]) -> None:
        if self._telemetry is None:
            return
        self._tel_counter += 1
        self._telemetry.emit(
            TelemetryRecord(
                schema_version=SCHEMA_VERSION,
                record_id=f"tel-ws-{self._tel_counter}",
                session_id=self._session_id,
                at_ms=self._clock.now_ms(),
                level=level,
                kind=kind,
                stage="transcript_adapter",
                payload=payload,
            )
        )

    async def _set_health(self, state: HealthState, reason: str, detail: dict[str, Any]) -> None:
        self._health = state
        payload = {"health": state.value, "reason": reason, **detail}
        self._emit_tel(level=TelemetryLevel.LIFECYCLE, kind="adapter_health", payload=payload)
        await _invoke_health_cb(self._health_cb, state, reason, payload)

    def _slice_utterance_audio(self, start_ms: int, end_ms: int) -> bytes | None:
        if not self._utterance_audio or end_ms <= start_ms:
            return None
        rel_start = max(0, start_ms - self._utterance_audio_base_ms)
        rel_end = max(rel_start, end_ms - self._utterance_audio_base_ms)
        bps = self._config.sample_rate * 2
        byte_start = (rel_start * bps // 1000) & ~1
        byte_end = min(len(self._utterance_audio), (rel_end * bps // 1000) & ~1)
        if byte_end <= byte_start:
            return None
        return bytes(self._utterance_audio[byte_start:byte_end])

    def _next_ctrl_event(
        self,
        subtype: ControlSubtype,
        *,
        reason: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> CanonicalEvent:
        self._ctrl_counter += 1
        now_ms = self._clock.now_ms()
        seq = self._segmenter._next_sequence()
        return CanonicalEvent(
            schema_version=SCHEMA_VERSION,
            session_id=self._session_id,
            event_id=f"ctrl-ws-{self._ctrl_counter}",
            kind=EventKind.CONTROL,
            received_ms=now_ms,
            source_sequence=seq,
            source_id=self._source_id,
            subtype=subtype,
            start_ms=start_ms,
            end_ms=end_ms,
            reason=reason,
        )

    async def connect(self) -> None:
        """Optionally start subprocess, open WS, and complete session.update handshake."""
        if self._process is not None and self._process.pid is None:
            try:
                await self._process.start(wait_for_ready=True)
            except (PeeceeLeaseUnavailableError, PeeceeProcessError, OSError) as exc:
                reason = (
                    "peecee_lease_unavailable"
                    if isinstance(exc, PeeceeLeaseUnavailableError)
                    else "peecee_process_failed"
                )
                await self._set_health(
                    HealthState.NO_TRANSCRIPT,
                    reason,
                    {"err": str(exc)},
                )
                raise
        headers: dict[str, str] = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        self._closed_cleanly = False
        try:
            self._ws = await ws_connect(
                self._config.ws_url,
                additional_headers=headers or None,
                max_size=16 * 1024 * 1024,
            )
            created_raw = await self._ws.recv()
            created_msg = json.loads(created_raw)
            rx_ms = self._clock.now_ms()
            if self._raw_sink is not None:
                self._raw_sink.record_raw(
                    session_id=self._session_id,
                    source_id=self._source_id,
                    received_ms=rx_ms,
                    payload=created_msg,
                )
            if created_msg.get("type") != "session.created":
                raise PeeceeProtocolError(
                    f"Expected session.created, got {created_msg.get('type')!r}"
                )

            session_cfg: dict[str, Any] = {
                "sample_rate": self._config.sample_rate,
                "language": self._config.language,
                "word_timestamps": self._config.word_timestamps,
                "speaker_diarization": self._config.speaker_diarization,
            }
            if self._config.endpointing_ms is not None:
                session_cfg["endpointing_ms"] = self._config.endpointing_ms
            await self._ws.send(json.dumps({"type": "session.update", "session": session_cfg}))

            updated_raw = await self._ws.recv()
            updated_msg = json.loads(updated_raw)
            rx2_ms = self._clock.now_ms()
            if self._raw_sink is not None:
                self._raw_sink.record_raw(
                    session_id=self._session_id,
                    source_id=self._source_id,
                    received_ms=rx2_ms,
                    payload=updated_msg,
                )
            if updated_msg.get("type") == "error":
                raise PeeceeProtocolError(f"session.update rejected: {updated_msg}")
            if updated_msg.get("type") != "session.updated":
                raise PeeceeProtocolError(
                    f"Expected session.updated, got {updated_msg.get('type')!r}"
                )
            await self._set_health(HealthState.OK, "connected", {"ws_url": self._config.ws_url})
        except (OSError, ConnectionClosed, PeeceeProtocolError, json.JSONDecodeError) as exc:
            await self._set_health(
                HealthState.NO_TRANSCRIPT, "peecee_connect_failed", {"err": str(exc)}
            )
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
            raise

    async def close(self) -> None:
        self._closed_cleanly = True
        if self._ws is not None:
            with contextlib.suppress(OSError):
                await self._ws.close()
            self._ws = None
        if self._process is not None:
            await self._process.stop()

    def _update_halt_clock(self) -> None:
        halted = self._paused or self._off_record
        if halted and self._halted_since_ms is None:
            if self._audio_origin_ms is not None:
                self._halted_since_ms = self._clock.now_ms()
        elif not halted and self._halted_since_ms is not None:
            self._total_halted_ms += max(0, self._clock.now_ms() - self._halted_since_ms)
            self._halted_since_ms = None

    async def pause(self) -> None:
        self._paused = True
        self._update_halt_clock()
        await self._set_health(HealthState.PAUSED, "user_paused", {})

    async def resume(self) -> None:
        self._paused = False
        self._update_halt_clock()
        if not self._off_record:
            await self._set_health(HealthState.OK, "resumed", {})

    def set_off_record(self, enabled: bool) -> list[CanonicalEvent]:
        """Toggle off-the-record ingestion halt and emit OFF_RECORD_START/END (B18)."""
        if enabled == self._off_record:
            return []
        now_ms = self._clock.now_ms()
        self._off_record = enabled
        self._update_halt_clock()
        if enabled:
            self._off_record_start_ms = now_ms
            self._silence_accum_ms = 0
            self._had_speech_since_commit = False
            self._segmenter.reset_pending()
            return [
                self._next_ctrl_event(
                    ControlSubtype.OFF_RECORD_START,
                    reason="off_record",
                    start_ms=now_ms,
                )
            ]
        start_ms = self._off_record_start_ms if self._off_record_start_ms is not None else now_ms
        self._off_record_start_ms = None
        return [
            self._next_ctrl_event(
                ControlSubtype.OFF_RECORD_END,
                reason="off_record",
                start_ms=start_ms,
                end_ms=max(start_ms, now_ms),
            )
        ]

    async def commit_audio(self) -> None:
        """Send `input_audio_buffer.commit` to flush `.completed` words + speaker tags."""
        if self._ws is None:
            raise AdapterError("WebSocket is not connected")
        entry = (self._uncommitted_bytes, self._utterance_sent_ms)
        self._pending_commits.append(entry)
        self._uncommitted_bytes = 0
        self._utterance_sent_ms = 0
        self._silence_accum_ms = 0
        self._had_speech_since_commit = False
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except BaseException:
            if self._pending_commits and self._pending_commits[-1] == entry:
                self._pending_commits.pop()
            raise

    async def send_audio_pcm16(
        self,
        pcm16_bytes: bytes,
        *,
        gate: EarshotGateSample | None = None,
        pace_realtime: bool = False,
    ) -> int:
        """Stream 160 ms PCM16 frames (5120 bytes @ 16 kHz). Halts if paused/off-record."""
        if self._off_record or self._paused:
            return 0
        if self._ws is None:
            raise AdapterError("WebSocket is not connected")
        if gate is not None:
            for ev in self._segmenter.feed_gate(gate, received_ms=self._clock.now_ms()):
                self._pending_events.put_nowait(ev)

        frame_bytes = self._config.frame_bytes
        usable = len(pcm16_bytes) - (len(pcm16_bytes) % 2)
        pcm16_even = pcm16_bytes[:usable]
        sent = 0
        idx = 0
        while idx < usable:
            chunk = pcm16_even[idx : idx + frame_bytes]
            await self._ws.send(chunk)
            sent += len(chunk)
            self._utterance_audio.extend(chunk)
            self._uncommitted_bytes += len(chunk)
            chunk_ms = max(1, round(len(chunk) * 1000.0 / (2 * self._config.sample_rate)))
            if self._audio_origin_ms is None:
                self._audio_origin_ms = max(0, self._clock.now_ms() - chunk_ms)
            self._utterance_sent_ms += chunk_ms
            self._total_sent_ms += chunk_ms
            idx += frame_bytes

            if gate is not None:
                if gate.speech_active:
                    self._had_speech_since_commit = True
                    self._silence_accum_ms = 0
                else:
                    self._silence_accum_ms += chunk_ms
            else:
                self._had_speech_since_commit = True

            auto_ms = self._config.auto_commit_silence_ms
            if self._had_speech_since_commit and (
                (auto_ms is not None and self._silence_accum_ms >= auto_ms)
                or self._utterance_sent_ms >= self._config.max_uncommitted_audio_ms
            ):
                await self.commit_audio()

            if pace_realtime:
                await asyncio.sleep(chunk_ms / 1000.0)
        return sent

    def _handle_buffer_committed(self) -> None:
        if self._pending_commits:
            committed_bytes_len, committed_audio_ms = self._pending_commits.popleft()
        else:
            committed_bytes_len = self._uncommitted_bytes
            committed_audio_ms = self._utterance_sent_ms
            self._uncommitted_bytes = 0
            self._utterance_sent_ms = 0
        if committed_bytes_len > 0:
            del self._utterance_audio[:committed_bytes_len]
        advance_ms = max(
            committed_audio_ms,
            self._last_utterance_audio_ms,
            self._last_epoch_word_end_ms,
        )
        self._commit_base_ms = max(
            self._commit_base_ms + advance_ms,
            self._last_committed_end_ms,
        )
        self._last_utterance_audio_ms = 0
        self._last_epoch_word_end_ms = 0
        self._utterance_audio_base_ms = self._commit_base_ms
        self._segmenter.binder.reset_diarizer_epoch()

    def process_server_message(
        self,
        payload: Mapping[str, Any],
        received_ms: int,
    ) -> list[CanonicalEvent]:
        """Normalize one JSON frame from NeMo-Speech.cpp v0.3.0 into CanonicalEvents."""
        msg_type = str(payload.get("type", ""))
        if self._off_record:
            if msg_type.endswith(("transcription.delta", "transcription.completed")):
                rel_ms = max(0, round(float(payload.get("audio_processed", 0.0)) * 1000.0))
                self._last_utterance_audio_ms = max(self._last_utterance_audio_ms, rel_ms)
            elif msg_type == "input_audio_buffer.committed":
                self._handle_buffer_committed()
            return []
        if self._raw_sink is not None:
            self._raw_sink.record_raw(
                session_id=self._session_id,
                source_id=self._source_id,
                received_ms=received_ms,
                payload=payload,
            )
        if msg_type.endswith("transcription.delta"):
            rel_ms = max(0, round(float(payload.get("audio_processed", 0.0)) * 1000.0))
            self._last_utterance_audio_ms = max(self._last_utterance_audio_ms, rel_ms)
            abs_ms = self._commit_base_ms + rel_ms
            delta_text = str(payload.get("delta", ""))
            return self._segmenter.feed_partial(
                delta_text,
                audio_processed_ms=abs_ms,
                received_ms=received_ms,
            )

        if msg_type.endswith("transcription.completed"):
            rel_ms = max(0, round(float(payload.get("audio_processed", 0.0)) * 1000.0))
            self._last_utterance_audio_ms = max(self._last_utterance_audio_ms, rel_ms)
            words = _extract_nemo_words(payload, offset_ms=self._commit_base_ms)
            if not words:
                transcript = str(payload.get("transcript", "")).strip()
                if not transcript:
                    return []
                start_ms = self._commit_base_ms
                end_ms = max(start_ms, self._commit_base_ms + rel_ms)
                words = [AsrWord(word=transcript, start_ms=start_ms, end_ms=end_ms)]
            for w in words:
                self._last_epoch_word_end_ms = max(
                    self._last_epoch_word_end_ms,
                    w.end_ms - self._commit_base_ms,
                )
                self._last_committed_end_ms = max(self._last_committed_end_ms, w.end_ms)
            events = self._segmenter.feed_words(
                words,
                received_ms=received_ms,
                audio_pcm16_fn=self._slice_utterance_audio,
                flush_end=True,
                emit_end_boundary=True,
            )
            for ev in events:
                if (
                    ev.kind is EventKind.SEGMENT
                    and ev.stability is Stability.COMMITTED
                    and ev.end_ms is not None
                ):
                    self._last_committed_end_ms = max(self._last_committed_end_ms, ev.end_ms)
                    origin = self._audio_origin_ms or 0
                    halted = self._total_halted_ms
                    if self._halted_since_ms is not None:
                        halted += max(0, received_ms - self._halted_since_ms)
                    audio_clock_ms = max(0, received_ms - origin - halted)
                    self._emit_tel(
                        level=TelemetryLevel.DEBUG,
                        kind="upstream_delay",
                        payload={
                            "lag_ms": audio_clock_ms - ev.end_ms,
                            "raw_lag_ms": received_ms - ev.end_ms,
                            "audio_origin_ms": origin,
                            "halted_ms": halted,
                            "segment_id": ev.segment_id,
                            "end_ms": ev.end_ms,
                            "received_ms": received_ms,
                        },
                    )
            return events

        if msg_type == "input_audio_buffer.committed":
            self._handle_buffer_committed()
            return []

        if msg_type == "error":
            self._emit_tel(
                level=TelemetryLevel.LIFECYCLE,
                kind="adapter_error",
                payload={"error": dict(payload.get("error", {}))},
            )
        return []

    async def events(self) -> AsyncIterator[CanonicalEvent]:
        """Yield CanonicalEvents from the WebSocket until clean close or tunnel loss."""
        ws = self._ws
        if ws is None:
            raise AdapterError("WebSocket is not connected")

        recv_task: asyncio.Task[Any] | None = None
        queue_task: asyncio.Task[CanonicalEvent] | None = None
        try:
            while True:
                while not self._pending_events.empty():
                    yield self._pending_events.get_nowait()

                cur_ws = self._ws
                if cur_ws is None or self._closed_cleanly:
                    break

                if recv_task is None:
                    recv_task = asyncio.create_task(cur_ws.recv())
                queue_task = asyncio.create_task(self._pending_events.get())
                done, _ = await asyncio.wait(
                    {recv_task, queue_task}, return_when=asyncio.FIRST_COMPLETED
                )

                q_item: CanonicalEvent | None = None
                if queue_task in done:
                    q_item = queue_task.result()
                else:
                    if not queue_task.done():
                        queue_task.cancel()
                    try:
                        await queue_task
                    except asyncio.CancelledError:
                        cur = asyncio.current_task()
                        if (cur is not None and cur.cancelling() > 0) or not queue_task.cancelled():
                            raise
                    if (
                        queue_task.done()
                        and not queue_task.cancelled()
                        and queue_task.exception() is None
                    ):
                        q_item = queue_task.result()
                queue_task = None

                if q_item is not None:
                    yield q_item
                    while not self._pending_events.empty():
                        yield self._pending_events.get_nowait()

                if recv_task in done:
                    try:
                        raw_frame = recv_task.result()
                    except (ConnectionClosed, OSError) as exc:
                        recv_task = None
                        while not self._pending_events.empty():
                            yield self._pending_events.get_nowait()
                        if not self._closed_cleanly:
                            await self._set_health(
                                HealthState.NO_TRANSCRIPT,
                                "peecee_tunnel_lost",
                                {"err": str(exc)},
                            )
                            while not self._pending_events.empty():
                                yield self._pending_events.get_nowait()
                            yield self._next_ctrl_event(
                                ControlSubtype.GAP,
                                reason="peecee_tunnel_lost",
                                start_ms=self._segmenter._last_committed_end_ms,
                                end_ms=max(
                                    self._segmenter._last_committed_end_ms,
                                    self._commit_base_ms + self._last_utterance_audio_ms,
                                ),
                            )
                        break
                    recv_task = None
                    if isinstance(raw_frame, bytes):
                        continue
                    try:
                        msg = json.loads(raw_frame)
                    except json.JSONDecodeError as exc:
                        while not self._pending_events.empty():
                            yield self._pending_events.get_nowait()
                        if not self._closed_cleanly:
                            await self._set_health(
                                HealthState.NO_TRANSCRIPT,
                                "peecee_malformed_message",
                                {"err": str(exc)},
                            )
                            while not self._pending_events.empty():
                                yield self._pending_events.get_nowait()
                            yield self._next_ctrl_event(
                                ControlSubtype.GAP,
                                reason="peecee_malformed_message",
                                start_ms=self._segmenter._last_committed_end_ms,
                                end_ms=max(
                                    self._segmenter._last_committed_end_ms,
                                    self._commit_base_ms + self._last_utterance_audio_ms,
                                ),
                            )
                        break
                    if not isinstance(msg, Mapping):
                        continue
                    rx_ms = self._clock.now_ms()
                    for ev in self.process_server_message(msg, rx_ms):
                        yield ev
        finally:
            for task in (recv_task, queue_task):
                if task is not None:
                    if not task.done():
                        task.cancel()
                    with contextlib.suppress(BaseException):
                        await task


class DeepgramNormalizer:
    """Normalizes Deepgram-style streaming JSON messages into CanonicalEvents (§6.1)."""

    def __init__(
        self,
        *,
        session_id: str,
        source_id: str = "deepgram",
        speaker_map: Mapping[int | str, str] | None = None,
        confidence_floor: float | None = None,
    ) -> None:
        self._session_id = session_id
        self._source_id = source_id
        self._speaker_map: dict[str, str] = (
            {str(k): v for k, v in speaker_map.items()} if speaker_map else {}
        )
        self._confidence_floor = confidence_floor
        self._seq_counter = 0
        self._event_counter = 0
        self._revisions: dict[str, int] = {}
        self._active_utt_start_ms: dict[int, int] = {}
        self._last_speaker_by_channel: dict[int, str] = {}
        self._last_end_ms_by_channel: dict[int, int] = {}

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def _next_event_id(self, prefix: str) -> str:
        self._event_counter += 1
        return f"{prefix}-{self._event_counter}"

    def _next_revision(self, seg_id: str) -> int:
        rev = self._revisions.get(seg_id, 0) + 1
        self._revisions[seg_id] = rev
        return rev

    def _resolve_speaker(self, words: Sequence[Any]) -> tuple[str, bool]:
        speakers: list[str] = []
        for w in words:
            if isinstance(w, Mapping) and w.get("speaker") is not None:
                speakers.append(str(w["speaker"]))
        if not speakers:
            return "S?", True
        raw_spk = Counter(speakers).most_common(1)[0][0]
        if raw_spk in self._speaker_map:
            return self._speaker_map[raw_spk], False
        if raw_spk.isdigit():
            return f"S{int(raw_spk) + 1}", False
        return raw_spk, False

    def normalize(
        self,
        payload: Mapping[str, Any] | str,
        received_ms: int,
    ) -> list[CanonicalEvent]:
        """Normalize a Deepgram `Results` or `UtteranceEnd` JSON payload."""
        data: Mapping[str, Any] = json.loads(payload) if isinstance(payload, str) else payload
        msg_type = str(data.get("type", "Results"))

        if msg_type == "UtteranceEnd":
            ch_raw = data.get("channel", [0])
            ch = int(ch_raw[0]) if isinstance(ch_raw, Sequence) and ch_raw else int(ch_raw or 0)
            end_ms = max(0, round(float(data.get("last_word_end", 0.0)) * 1000.0))
            self._active_utt_start_ms.pop(ch, None)
            return [
                CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._session_id,
                    event_id=self._next_event_id("dg-bnd"),
                    kind=EventKind.BOUNDARY,
                    received_ms=received_ms,
                    source_sequence=self._next_seq(),
                    source_id=self._source_id,
                    channel=ch,
                    speaker_id=self._last_speaker_by_channel.get(ch),
                    end_ms=end_ms,
                    boundary="utterance_end",
                )
            ]

        if msg_type != "Results" and "channel" not in data:
            return []

        ch_idx_raw = data.get("channel_index", [0])
        ch = (
            int(ch_idx_raw[0])
            if isinstance(ch_idx_raw, Sequence) and ch_idx_raw
            else int(ch_idx_raw or 0)
        )
        channel_obj = data.get("channel")
        if not isinstance(channel_obj, Mapping):
            return []
        alts = channel_obj.get("alternatives")
        if not isinstance(alts, Sequence) or not alts or not isinstance(alts[0], Mapping):
            return []
        alt = alts[0]
        transcript = str(alt.get("transcript", "")).strip()
        words_raw = alt.get("words")
        words: Sequence[Any] = words_raw if isinstance(words_raw, Sequence) else []
        is_final = bool(data.get("is_final", False))
        speech_final = bool(data.get("speech_final", False))

        has_explicit_start = ("start" in data) or bool(
            words and isinstance(words[0], Mapping) and "start" in words[0]
        )
        duration_ms = max(0, round(float(data.get("duration", 0.0)) * 1000.0))
        if has_explicit_start:
            start_ms = max(0, round(float(data.get("start", 0.0)) * 1000.0))
            if words and isinstance(words[0], Mapping) and "start" in words[0]:
                start_ms = max(0, round(float(words[0]["start"]) * 1000.0))
        elif ch in self._active_utt_start_ms:
            start_ms = self._active_utt_start_ms[ch]
        else:
            prev_end = self._last_end_ms_by_channel.get(ch)
            start_ms = 0 if prev_end is None else prev_end + 1
        end_ms = start_ms + duration_ms
        if words and isinstance(words[-1], Mapping) and "end" in words[-1]:
            end_ms = max(start_ms, round(float(words[-1]["end"]) * 1000.0))

        speaker_id, spk_uncertain = self._resolve_speaker(words)
        conf_raw = alt.get("confidence")
        asr_conf = float(conf_raw) if conf_raw is not None else None
        low_conf = (
            asr_conf is not None
            and self._confidence_floor is not None
            and asr_conf < self._confidence_floor
        )

        out: list[CanonicalEvent] = []
        if transcript:
            utt_start = self._active_utt_start_ms.setdefault(ch, start_ms)
            seg_id = f"dg-ch{ch}-{utt_start:08d}"
            self._last_speaker_by_channel[ch] = speaker_id
            self._last_end_ms_by_channel[ch] = max(
                self._last_end_ms_by_channel.get(ch, 0),
                end_ms,
            )
            stability = Stability.COMMITTED if is_final else Stability.INTERIM
            out.append(
                CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._session_id,
                    event_id=self._next_event_id("dg-seg"),
                    kind=EventKind.SEGMENT,
                    received_ms=received_ms,
                    source_sequence=self._next_seq(),
                    source_id=self._source_id,
                    segment_id=seg_id,
                    source_revision=self._next_revision(seg_id),
                    channel=ch,
                    speaker_id=speaker_id,
                    speaker_confidence=0.3 if spk_uncertain else 0.85,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=transcript,
                    stability=stability,
                    boundary="speech_final" if (is_final and speech_final) else None,
                    asr_confidence=asr_conf,
                    asr_low_confidence=low_conf,
                    speaker_uncertain=spk_uncertain,
                )
            )
        if is_final:
            self._active_utt_start_ms.pop(ch, None)
        if speech_final:
            self._active_utt_start_ms.pop(ch, None)
            bnd_speaker = speaker_id if words else self._last_speaker_by_channel.get(ch)
            out.append(
                CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._session_id,
                    event_id=self._next_event_id("dg-bnd"),
                    kind=EventKind.BOUNDARY,
                    received_ms=received_ms,
                    source_sequence=self._next_seq(),
                    source_id=self._source_id,
                    channel=ch,
                    speaker_id=bnd_speaker,
                    end_ms=end_ms,
                    boundary="speech_final",
                )
            )
        return out


class ReplayPlayer:
    """Deterministic replay player for CanonicalEvents (§12.6, §15)."""

    def __init__(
        self,
        events: Sequence[CanonicalEvent],
        *,
        preserve_order: bool = False,
    ) -> None:
        if preserve_order:
            self._events = list(events)
        else:
            self._events = sorted(
                events,
                key=lambda ev: (ev.received_ms, ev.source_sequence, ev.event_id),
            )
        self._cursor = 0
        self._budget = 0
        self._target_ms: int | None = None
        self._cond = asyncio.Condition()

    @property
    def events(self) -> tuple[CanonicalEvent, ...]:
        return tuple(self._events)

    @property
    def remaining(self) -> int:
        return max(0, len(self._events) - self._cursor)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        preserve_order: bool = False,
    ) -> ReplayPlayer:
        """Load CanonicalEvents from JSONL (skipping optional SessionBrief header)."""
        source = Path(path)
        loaded: list[CanonicalEvent] = []
        with source.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not isinstance(obj, Mapping):
                    raise ValueError(f"JSONL line {line_no} must be a JSON object")
                if "objective" in obj and "recipient" in obj and "kind" not in obj:
                    continue
                loaded.append(CanonicalEvent.from_dict(obj))
        return cls(loaded, preserve_order=preserve_order)

    @classmethod
    def from_raw_messages(
        cls,
        raw_items: Sequence[tuple[int, Mapping[str, Any]]],
        *,
        normalizer: Callable[[Mapping[str, Any], int], Sequence[CanonicalEvent]],
        preserve_order: bool = False,
    ) -> ReplayPlayer:
        """Re-normalize raw `(received_ms, payload)` records into a ReplayPlayer."""
        loaded: list[CanonicalEvent] = []
        for rx_ms, payload in raw_items:
            loaded.extend(normalizer(payload, rx_ms))
        return cls(loaded, preserve_order=preserve_order)

    def reset(self) -> None:
        self._cursor = 0
        self._budget = 0
        self._target_ms = None

    async def step(self, count: int = 1) -> None:
        """Release `count` events to a running `play(mode=STEPPED)` iterator."""
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        async with self._cond:
            self._budget += count
            self._cond.notify_all()

    async def step_to(self, target_ms: int) -> None:
        """Release all events with `received_ms <= target_ms` in stepped mode."""
        async with self._cond:
            self._target_ms = (
                target_ms if self._target_ms is None else max(self._target_ms, target_ms)
            )
            self._cond.notify_all()

    def next_step(self, *, clock: ManualClock | None = None) -> CanonicalEvent | None:
        """Synchronously pop the next event and advance virtual clock."""
        if self._cursor >= len(self._events):
            return None
        ev = self._events[self._cursor]
        self._cursor += 1
        if clock is not None and ev.received_ms > clock.now_ms():
            clock.set_ms(ev.received_ms)
        return ev

    async def play(
        self,
        *,
        mode: ReplayTimingMode = ReplayTimingMode.RECORDED,
        speed: float = 1.0,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: ManualClock | None = None,
    ) -> AsyncIterator[CanonicalEvent]:
        """Yield events under recorded, scaled, or stepped timing."""
        if speed <= 0.0:
            raise ValueError(f"speed must be > 0, got {speed}")
        eff_speed = 1.0 if mode is ReplayTimingMode.RECORDED else speed
        prev_rx: int | None = None

        while self._cursor < len(self._events):
            ev = self._events[self._cursor]
            if mode is ReplayTimingMode.STEPPED:
                async with self._cond:
                    while not (
                        self._budget > 0
                        or (self._target_ms is not None and ev.received_ms <= self._target_ms)
                    ):
                        await self._cond.wait()
                    if self._budget > 0:
                        self._budget -= 1
            else:
                if prev_rx is not None and ev.received_ms > prev_rx:
                    delay_s = (ev.received_ms - prev_rx) / (1000.0 * eff_speed)
                    if delay_s > 0.0:
                        await sleep_fn(delay_s)
                prev_rx = ev.received_ms

            self._cursor += 1
            if clock is not None and ev.received_ms > clock.now_ms():
                clock.set_ms(ev.received_ms)
            yield ev
