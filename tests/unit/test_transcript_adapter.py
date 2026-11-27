"""Unit and integration tests for coach/transcript_adapter.py (Milestone M2)."""

from __future__ import annotations

import ast
import asyncio
import json
import math
import signal
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coach.canonical_state import CanonicalState
from coach.contracts import (
    SCHEMA_VERSION,
    CanonicalEvent,
    ClassifierResult,
    ConsentBasis,
    ConsentConfig,
    ControlSubtype,
    DecisionContext,
    EventKind,
    HealthState,
    Origin,
    Outcome,
    Participant,
    ReasonCode,
    RecipientConfig,
    RetentionConfig,
    RoutingDecision,
    ScoreSet,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    TelemetryLevel,
)
from coach.storage import SQLiteStore, export_events_jsonl
from coach.telemetry import (
    BoundedRecorderBuffer,
    build_session_report,
    check_conservation_invariants,
)
from coach.transcript_adapter import (
    AsrWord,
    BinderConfig,
    DeepgramNormalizer,
    DirectionSample,
    EarshotGateSample,
    ManualClock,
    NemotronSegmenter,
    NumpySpectralVerifier,
    PeeceeConfig,
    PeeceeLeaseUnavailableError,
    PeeceeProcessError,
    PeeceeProtocolError,
    PeeceeRealtimeAdapter,
    PeeceeSpeechProcess,
    RecipientBinder,
    ReplayPlayer,
    ReplayTimingMode,
    SegmenterConfig,
    parse_earshot_direction_jsonl,
    parse_nemo_offline_json,
)

COACH_DIR = Path(__file__).resolve().parents[2] / "coach"


def _coach_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "coach" or alias.name.startswith("coach."):
                    imported.add(alias.name.removeprefix("coach.").split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                if mod:
                    imported.add(mod.split(".")[0])
                else:
                    for alias in node.names:
                        imported.add(alias.name)
            elif mod == "coach":
                for alias in node.names:
                    imported.add(alias.name)
            elif mod.startswith("coach."):
                imported.add(mod.removeprefix("coach.").split(".")[0])
    return imported


def test_transcript_adapter_upward_only_boundary() -> None:
    imports = _coach_imports(COACH_DIR / "transcript_adapter.py")
    assert imports <= {"contracts"}, f"Forbidden coach imports in transcript_adapter.py: {imports}"


def _make_brief(session_id: str = "sess-m2") -> SessionBrief:
    return SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Finalize rollout plan",
        recipient=RecipientConfig(participant_id="ME", role="tech_lead"),
        participants=(
            Participant(participant_id="ME", label="ME", role="tech_lead"),
            Participant(participant_id="S1", label="Alex", role="eng"),
            Participant(participant_id="S2", label="Priya", role="pm"),
        ),
        retention=RetentionConfig(store_transcript=True, store_raw_provider_events=True),
        consent=ConsentConfig(basis=ConsentBasis.STATED),
    )


def _make_harmonic_pcm16(
    f0_hz: float,
    duration_ms: int,
    *,
    sample_rate: int = 16000,
    amp: float = 0.35,
    tilt: float = 0.6,
) -> bytes:
    n = int(sample_rate * duration_ms / 1000)
    t = np.arange(n, dtype=np.float64) / sample_rate
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 7):
        sig += (tilt ** (k - 1)) * np.sin(2.0 * math.pi * f0_hz * k * t)
    peak = float(np.max(np.abs(sig))) or 1.0
    pcm = np.clip((sig / peak) * amp * 32767.0, -32768.0, 32767.0).astype("<i2")
    return pcm.tobytes()


class _StoreRawAdapter:
    def __init__(self, store: SQLiteStore, run_id: str, retention: RetentionConfig) -> None:
        self._store = store
        self._run_id = run_id
        self._retention = retention

    def record_raw(
        self,
        *,
        session_id: str,
        source_id: str,
        received_ms: int,
        payload: Mapping[str, Any] | str,
    ) -> bool:
        return self._store.record_raw_event(
            session_id=session_id,
            run_id=self._run_id,
            source_id=source_id,
            received_ms=received_ms,
            payload=payload,
            retention=self._retention,
        )


@dataclass
class MockUtteranceScript:
    deltas: list[str]
    words: list[dict[str, Any]]


@dataclass
class MockNeMoSpeechServer:
    """Mock NeMo-Speech.cpp v0.3.0 WebSocket server matching §6.7 protocol."""

    utterances: list[MockUtteranceScript] = field(default_factory=list)
    eou_mode: bool = False
    close_mid_stream: bool = False
    send_error_on_update: bool = False
    omit_speaker_on_short_word: bool = False
    required_api_key: str | None = None
    received_session_updates: list[dict[str, Any]] = field(default_factory=list)
    received_audio_frames: list[int] = field(default_factory=list)
    commit_count: int = 0
    _server: Server | None = None
    url: str = ""

    async def __aenter__(self) -> MockNeMoSpeechServer:
        self._server = await serve(self._handle, "127.0.0.1", 0)
        sock = next(iter(self._server.sockets))
        port = int(sock.getsockname()[1])
        self.url = f"ws://127.0.0.1:{port}/v1/audio/transcriptions/realtime"
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: ServerConnection) -> None:
        if self.required_api_key is not None:
            auth = ws.request.headers.get("Authorization", "") if ws.request else ""
            if auth != f"Bearer {self.required_api_key}":
                await ws.close(code=1008, reason="unauthorized")
                return

        ev_id = 0

        def next_eid() -> str:
            nonlocal ev_id
            ev_id += 1
            return f"event_{ev_id}"

        await ws.send(
            json.dumps(
                {
                    "type": "session.created",
                    "event_id": next_eid(),
                    "session": {
                        "input_audio_format": "pcm16",
                        "sample_rate": 16000,
                        "model": "nemotron-speech-streaming-en-0.6b",
                    },
                }
            )
        )

        audio_started = False
        utt_idx = 0
        total_bytes = 0
        delta_cursor = 0
        sample_rate = 16000
        eou_emitted_for_utt = False

        async for raw in ws:
            if isinstance(raw, bytes):
                audio_started = True
                self.received_audio_frames.append(len(raw))
                total_bytes += len(raw)
                audio_proc = total_bytes / (2.0 * sample_rate)
                script = (
                    self.utterances[utt_idx]
                    if utt_idx < len(self.utterances)
                    else MockUtteranceScript(deltas=[], words=[])
                )
                if delta_cursor < len(script.deltas):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.input_audio_transcription.delta",
                                "event_id": next_eid(),
                                "delta": script.deltas[delta_cursor],
                                "audio_processed": round(audio_proc, 3),
                            }
                        )
                    )
                    delta_cursor += 1
                if self.close_mid_stream and delta_cursor >= 1:
                    await ws.close(code=1011, reason="tunnel_lost")
                    return
                if self.eou_mode and not eou_emitted_for_utt and delta_cursor >= len(script.deltas):
                    eou_emitted_for_utt = True
                    await self._send_completed(ws, next_eid(), script, audio_proc)
                    utt_idx += 1
                    delta_cursor = 0
                    eou_emitted_for_utt = False
                continue

            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "session.update":
                if audio_started:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "event_id": next_eid(),
                                "error": {
                                    "message": "session configuration cannot change after audio starts",
                                    "type": "invalid_request_error",
                                    "code": "invalid_argument",
                                },
                            }
                        )
                    )
                    continue
                if self.send_error_on_update:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "event_id": next_eid(),
                                "error": {"message": "bad config", "type": "invalid_request_error"},
                            }
                        )
                    )
                    continue
                session_obj = dict(msg.get("session", {}))
                sample_rate = int(session_obj.get("sample_rate", 16000))
                self.received_session_updates.append(session_obj)
                await ws.send(
                    json.dumps(
                        {"type": "session.updated", "event_id": next_eid(), "session": session_obj}
                    )
                )
            elif mtype == "input_audio_buffer.commit":
                self.commit_count += 1
                audio_proc = total_bytes / (2.0 * sample_rate)
                script = (
                    self.utterances[utt_idx]
                    if utt_idx < len(self.utterances)
                    else MockUtteranceScript(deltas=[], words=[])
                )
                if script.words:
                    await self._send_completed(ws, next_eid(), script, audio_proc)
                total_bytes = 0
                delta_cursor = 0
                utt_idx += 1
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.committed", "event_id": next_eid()})
                )

    async def _send_completed(
        self,
        ws: ServerConnection,
        event_id: str,
        script: MockUtteranceScript,
        audio_proc: float,
    ) -> None:
        words_out: list[dict[str, Any]] = []
        for w in script.words:
            item = dict(w)
            if (
                self.omit_speaker_on_short_word
                and (float(item["end"]) - float(item["start"])) < 0.35
            ):
                item.pop("speaker", None)
            words_out.append(item)
        transcript = " ".join(str(w["word"]) for w in words_out)
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "event_id": event_id,
                    "transcript": transcript,
                    "audio_processed": round(audio_proc, 3),
                    "words": words_out,
                }
            )
        )


async def test_realtime_ws_mode_a_handshake_commit_reset_and_state_integration(
    tmp_path: Path,
) -> None:
    utt1 = MockUtteranceScript(
        deltas=["Someone ", "should ", "own ", "the roll"],
        words=[
            {"word": "Someone", "start": 0.10, "end": 0.45, "confidence": 1.0, "speaker": 1},
            {"word": "should", "start": 0.48, "end": 0.72, "confidence": 1.0, "speaker": 1},
            {"word": "own", "start": 0.75, "end": 0.95, "confidence": 1.0, "speaker": 1},
            {"word": "the", "start": 0.96, "end": 1.08, "confidence": 1.0, "speaker": 1},
            {"word": "rollback", "start": 1.10, "end": 1.48, "confidence": 1.0, "speaker": 1},
        ],
    )
    utt2 = MockUtteranceScript(
        deltas=["I can ", "handle ", "that"],
        words=[
            {"word": "I", "start": 0.12, "end": 0.25, "confidence": 1.0, "speaker": 2},
            {"word": "can", "start": 0.28, "end": 0.45, "confidence": 1.0, "speaker": 2},
            {"word": "handle", "start": 0.48, "end": 0.85, "confidence": 1.0, "speaker": 2},
            {"word": "that", "start": 0.88, "end": 1.25, "confidence": 1.0, "speaker": 2},
        ],
    )
    brief = _make_brief("sess-mode-a")
    store = SQLiteStore(tmp_path / "coach.db")
    store.upsert_session(brief=brief, run_id="run-1", started_ms=0)
    telemetry = BoundedRecorderBuffer(maxsize=128)
    clock = ManualClock(100)
    health_events: list[tuple[HealthState, str]] = []

    async with MockNeMoSpeechServer(utterances=[utt1, utt2], required_api_key="tok123") as srv:
        cfg = PeeceeConfig(ws_url=srv.url, api_key="tok123", auto_commit_silence_ms=320)
        adapter = PeeceeRealtimeAdapter(
            session_id="sess-mode-a",
            config=cfg,
            clock=clock,
            telemetry_sink=telemetry,
            raw_sink=_StoreRawAdapter(store, "run-1", brief.retention),
            health_callback=lambda st, rsn, _det: health_events.append((st, rsn)),
        )
        await adapter.connect()
        assert srv.received_session_updates == [
            {
                "sample_rate": 16000,
                "language": "en-US",
                "word_timestamps": True,
                "speaker_diarization": True,
            }
        ]
        collected: list[CanonicalEvent] = []

        async def _consume() -> None:
            async for ev in adapter.events():
                collected.append(ev)

        consumer = asyncio.create_task(_consume())
        # Utterance 1: 10 active 160ms frames (1600ms) then 2 silent gate frames (320ms -> auto-commit)
        pcm_frame = _make_harmonic_pcm16(150.0, 160)
        for _ in range(10):
            clock.advance(160)
            await adapter.send_audio_pcm16(
                pcm_frame,
                gate=EarshotGateSample(at_ms=clock.now_ms(), speech_active=True),
            )
            await asyncio.sleep(0.01)
        for _ in range(2):
            clock.advance(160)
            await adapter.send_audio_pcm16(
                b"\x00" * 5120,
                gate=EarshotGateSample(at_ms=clock.now_ms(), speech_active=False),
            )
            await asyncio.sleep(0.01)

        # Utterance 2: 3 active frames + explicit commit
        for _ in range(3):
            clock.advance(160)
            await adapter.send_audio_pcm16(
                _make_harmonic_pcm16(240.0, 160),
                gate=EarshotGateSample(at_ms=clock.now_ms(), speech_active=True),
            )
            await asyncio.sleep(0.01)
        clock.advance(500)
        await adapter.commit_audio()
        await asyncio.sleep(0.05)
        await adapter.close()
        await consumer

    assert srv.commit_count == 2
    assert all(size == 5120 for size in srv.received_audio_frames)
    assert health_events[0] == (HealthState.OK, "connected")

    state = CanonicalState(brief)
    for ev in collected:
        res = state.apply(ev)
        assert res.accepted, f"Rejected event {ev}: {res.drop_reason}"
        store.record_canonical_event(run_id="run-1", event=ev, retention=brief.retention)

    snap = state.snapshot()
    assert snap.counters["late_interim"] == 0
    assert snap.counters["stale_revision"] == 0
    assert all(s.text == "" for s in snap.interim_segments.values())
    assert len(snap.segments) == 2
    s1, s2 = snap.segments
    assert s1.text == "Someone should own the rollback"
    assert s2.text == "I can handle that"
    # Utterance 2 timestamps shifted past Utterance 1's 1920 ms commit epoch
    assert s2.start_ms >= 1920 + 120

    raw_rows = store.get_raw_events("sess-mode-a", "run-1")
    assert len(raw_rows) >= 6
    recs = telemetry.drain()
    delays = [r for r in recs if r.kind == "upstream_delay"]
    assert len(delays) == 2
    assert all("lag_ms" in d.payload for d in delays)
    store.close()


async def test_realtime_ws_eou_mode_b_does_not_reset_clock() -> None:
    utt1 = MockUtteranceScript(
        deltas=["First turn"],
        words=[{"word": "First", "start": 0.1, "end": 0.4, "confidence": 1.0, "speaker": 1}],
    )
    utt2 = MockUtteranceScript(
        deltas=["Second turn"],
        words=[{"word": "Second", "start": 1.5, "end": 1.9, "confidence": 1.0, "speaker": 2}],
    )
    clock = ManualClock(0)
    async with MockNeMoSpeechServer(utterances=[utt1, utt2], eou_mode=True) as srv:
        adapter = PeeceeRealtimeAdapter(
            session_id="sess-eou",
            config=PeeceeConfig(ws_url=srv.url, auto_commit_silence_ms=None),
            clock=clock,
        )
        await adapter.connect()
        events: list[CanonicalEvent] = []

        async def _consume() -> None:
            async for ev in adapter.events():
                events.append(ev)

        task = asyncio.create_task(_consume())
        for _ in range(2):
            clock.advance(160)
            await adapter.send_audio_pcm16(b"\x01\x00" * 2560)
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        await adapter.close()
        await task

    assert adapter.commit_base_ms == 0
    committed = [
        ev for ev in events if ev.kind is EventKind.SEGMENT and ev.stability is Stability.COMMITTED
    ]
    assert [(ev.text, ev.start_ms) for ev in committed] == [("First", 100), ("Second", 1500)]


async def test_realtime_ws_tunnel_drop_and_off_record_controls() -> None:
    utt = MockUtteranceScript(deltas=["hello"], words=[])
    telemetry = BoundedRecorderBuffer(maxsize=64)
    health_log: list[HealthState] = []
    clock = ManualClock(100)

    async with MockNeMoSpeechServer(utterances=[utt], close_mid_stream=True) as srv:
        adapter = PeeceeRealtimeAdapter(
            session_id="sess-drop",
            config=PeeceeConfig(ws_url=srv.url),
            clock=clock,
            telemetry_sink=telemetry,
            health_callback=lambda st, _r, _d: health_log.append(st),
        )
        await adapter.connect()

        # Verify off-the-record halts audio send and emits OFF_RECORD_START / END (B18)
        off_ev = adapter.set_off_record(True)
        assert len(off_ev) == 1 and off_ev[0].subtype is ControlSubtype.OFF_RECORD_START
        assert await adapter.send_audio_pcm16(b"\x01\x00" * 2560) == 0
        on_ev = adapter.set_off_record(False)
        assert len(on_ev) == 1 and on_ev[0].subtype is ControlSubtype.OFF_RECORD_END

        await adapter.pause()
        assert await adapter.send_audio_pcm16(b"\x01\x00" * 2560) == 0
        await adapter.resume()
        assert not adapter.is_paused
        assert health_log[:3] == [HealthState.OK, HealthState.PAUSED, HealthState.OK]

        received: list[CanonicalEvent] = []
        await adapter.send_audio_pcm16(b"\x01\x00" * 2560)
        async for ev in adapter.events():
            received.append(ev)

    assert adapter.health is HealthState.NO_TRANSCRIPT
    assert HealthState.NO_TRANSCRIPT in health_log
    gaps = [
        ev for ev in received if ev.kind is EventKind.CONTROL and ev.subtype is ControlSubtype.GAP
    ]
    assert len(gaps) == 1
    assert gaps[0].reason == "peecee_tunnel_lost"
    assert any(
        r.kind == "adapter_health" and r.payload.get("health") == "no_transcript"
        for r in telemetry.drain()
    )


async def test_realtime_ws_protocol_error_on_rejected_session_update() -> None:
    async with MockNeMoSpeechServer(send_error_on_update=True) as srv:
        adapter = PeeceeRealtimeAdapter(
            session_id="sess-err",
            config=PeeceeConfig(ws_url=srv.url),
        )
        with pytest.raises(PeeceeProtocolError):
            await adapter.connect()
        assert adapter.health is HealthState.NO_TRANSCRIPT


async def test_peecee_process_exit_75_maps_to_no_transcript_and_gap(tmp_path: Path) -> None:
    script = tmp_path / "peecee-speech-mock"
    script.write_text("#!/usr/bin/env bash\nexit 75\n", encoding="utf-8")
    script.chmod(0o755)

    telemetry = BoundedRecorderBuffer(maxsize=32)
    health_calls: list[tuple[HealthState, str, dict[str, Any]]] = []
    gaps: list[CanonicalEvent] = []

    proc = PeeceeSpeechProcess(
        session_id="sess-75",
        config=PeeceeConfig(
            executable=script, ws_url="ws://127.0.0.1:19975/ws", ready_timeout_s=3.0
        ),
        clock=ManualClock(250),
        telemetry_sink=telemetry,
        health_callback=lambda st, rsn, det: health_calls.append((st, rsn, det)),
        gap_callback=lambda ev: gaps.append(ev),
    )
    with pytest.raises(PeeceeLeaseUnavailableError):
        await proc.start(wait_for_ready=True)

    assert proc.health is HealthState.NO_TRANSCRIPT
    assert health_calls[-1][0] is HealthState.NO_TRANSCRIPT
    assert health_calls[-1][1] == "peecee_lease_unavailable"
    assert health_calls[-1][2]["exit_code"] == 75
    assert len(gaps) == 1 and gaps[0].subtype is ControlSubtype.GAP
    recs = telemetry.drain()
    assert any(
        r.level is TelemetryLevel.LIFECYCLE
        and r.kind == "adapter_health"
        and r.payload.get("exit_code") == 75
        for r in recs
    )


async def test_peecee_process_stop_sends_sigterm_to_ssh_child(tmp_path: Path) -> None:
    sig_log = tmp_path / "ssh_signal.txt"
    ssh_proxy = tmp_path / "ssh"
    ssh_proxy.write_text(
        f"#!{sys.executable}\n"
        "import signal, sys, time, http.server, threading\n"
        f"log = open({str(sig_log)!r}, 'w')\n"
        "def on_sig(signum, _):\n"
        "    log.write(f'{signum}\\n')\n"
        "    log.flush()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, on_sig)\n"
        "signal.signal(signal.SIGINT, on_sig)\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.end_headers()\n"
        "    def log_message(self, *a): pass\n"
        "srv = http.server.HTTPServer(('127.0.0.1', 0), H)\n"
        f"open({str(tmp_path / 'port.txt')!r}, 'w').write(str(srv.server_port))\n"
        "threading.Thread(target=srv.serve_forever, daemon=True).start()\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    ssh_proxy.chmod(0o755)

    launcher = tmp_path / "peecee-speech-mock"
    launcher.write_text(
        f"#!/usr/bin/env bash\n{ssh_proxy} &\nchild=$!\nwait $child\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    proc = PeeceeSpeechProcess(
        session_id="sess-stop",
        config=PeeceeConfig(executable=launcher, ws_url="ws://127.0.0.1:1/ws", ready_timeout_s=5.0),
    )
    await proc.start(wait_for_ready=False)
    port_file = tmp_path / "port.txt"
    for _ in range(50):
        if port_file.exists() and port_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    port = int(port_file.read_text().strip())
    proc._config = PeeceeConfig(executable=launcher, ws_url=f"ws://127.0.0.1:{port}/ws")
    await proc.wait_ready(timeout_s=5.0)

    await proc.stop()
    assert sig_log.read_text(encoding="utf-8").strip() == str(int(signal.SIGTERM))


def test_nemotron_segmenter_offline_json_splits_gaps_gates_and_caps() -> None:
    payload = {
        "file": "/tmp/example.wav",
        "text": "Hello, world okay next speaker",
        "duration": 5.0,
        "languages": ["en-US"],
        "words": [
            {"word": "Hello", "start": 0.20, "end": 0.45, "confidence": 1.0, "speaker": 1},
            {"word": ",", "start": 0.45, "end": 0.46, "confidence": 1.0, "speaker": 1},
            {"word": "world", "start": 0.48, "end": 0.80, "confidence": 1.0, "speaker": 1},
            # Silence gap = 750 ms (>= 600 ms) -> split + boundary
            {"word": "okay", "start": 1.55, "end": 1.85, "confidence": 1.0, "speaker": 1},
            # Speaker change 1 -> 2 -> split + boundary
            {"word": "next", "start": 2.00, "end": 2.30, "confidence": 1.0, "speaker": 2},
            {"word": "speaker", "start": 2.35, "end": 2.95, "confidence": 1.0, "speaker": 2},
        ],
    }
    gates = [
        EarshotGateSample(at_ms=200, speech_active=True, echo_residual_ok=True),
        EarshotGateSample(
            at_ms=1600, speech_active=False, echo_residual_ok=False
        ),  # B21 echo residual
        EarshotGateSample(at_ms=2000, speech_active=True, echo_residual_ok=True),
    ]
    events = parse_nemo_offline_json(
        payload,
        session_id="sess-offline",
        gates=gates,
        segmenter_config=SegmenterConfig(inter_word_gap_ms=600, boundary_gap_ms=600),
    )
    segs = [e for e in events if e.kind is EventKind.SEGMENT and e.stability is Stability.COMMITTED]
    bnds = [e for e in events if e.kind is EventKind.BOUNDARY]

    assert [s.text for s in segs] == ["Hello, world", "okay", "next speaker"]
    assert segs[0].low_quality_gate is False
    assert segs[1].low_quality_gate is True and segs[1].asr_low_confidence is True  # B21 preserved
    assert segs[1].speaker_uncertain is True  # 300ms span < 1000ms
    assert segs[2].low_quality_gate is False
    assert len(bnds) == 3

    # Verify max_segment_words cap (< 60 tokens)
    seg_cap = NemotronSegmenter(
        session_id="sess-cap",
        config=SegmenterConfig(max_segment_words=5, max_segment_ms=8000),
    )
    many_words = [
        {"word": f"w{i}", "start": i * 0.2, "end": i * 0.2 + 0.15, "confidence": 0.5, "speaker": 1}
        for i in range(12)
    ]
    cap_events = parse_nemo_offline_json(
        {"words": many_words},
        session_id="sess-cap",
        segmenter=seg_cap,
    )
    cap_segs = [e for e in cap_events if e.kind is EventKind.SEGMENT]
    assert [len((s.text or "").split()) for s in cap_segs] == [5, 5, 2]

    # Verify asr_confidence_floor flag
    floor_seg = NemotronSegmenter(
        session_id="sess-floor",
        config=SegmenterConfig(asr_confidence_floor=0.7),
    )
    floor_evs = parse_nemo_offline_json(
        {
            "words": [
                {"word": "unclear", "start": 0.1, "end": 1.3, "confidence": 0.45, "speaker": 1}
            ]
        },
        session_id="sess-floor",
        segmenter=floor_seg,
    )
    assert floor_evs[0].asr_low_confidence is True


def test_recipient_binder_spectral_azimuth_reconcile_and_cross_epoch() -> None:
    me_audio = _make_harmonic_pcm16(125.0, 1400, tilt=0.75)
    s1_audio = _make_harmonic_pcm16(245.0, 1400, tilt=0.45)

    verifier = NumpySpectralVerifier()
    verifier.enroll("ME", me_audio)
    verifier.enroll("S1", s1_audio)
    assert verifier.score(b"\x00" * 3200) == {}

    dir_jsonl = (
        '{"t": 0.50, "azimuth": 355.0, "contrast": 4.0, "level_db": -25.0}\n'
        '{"t": 1.00, "azimuth": 5.0, "contrast": 4.0, "level_db": -25.0}\n'
        '{"t": 2.50, "azimuth": 180.0, "contrast": 3.5, "level_db": -26.0}\n'
    )
    dir_samples = parse_earshot_direction_jsonl(dir_jsonl)
    assert len(dir_samples) == 3

    binder = RecipientBinder(
        session_id="sess-bind",
        config=BinderConfig(
            seat_azimuths_deg={"ME": 0.0, "S1": 180.0},
            azimuth_tolerance_deg=25.0,
            min_confident_duration_ms=1000,
        ),
        verifier=verifier,
    )
    for d in dir_samples:
        binder.feed_direction(d)

    # Short span < 1000 ms is flagged speaker_uncertain=True
    short_b = binder.bind_segment("seg-short", "1", 100, 600, audio_pcm16=me_audio[:16000])
    assert short_b.speaker_uncertain is True

    # Unknown cluster 0 -> S?
    unk_b = binder.bind_segment("seg-unk", "0", 100, 1500)
    assert unk_b.speaker_id == "S?" and unk_b.speaker_uncertain is True

    # Full span with ME voice + circular wraparound azimuth ~0 deg -> ME confident
    me_b = binder.bind_segment("seg-me", "1", 200, 1500, audio_pcm16=me_audio)
    assert me_b.speaker_id == "ME" and me_b.speaker_uncertain is False

    # Reconcile an earlier cluster from S1 -> ME (B05 / B22) and verify CanonicalState invalidation
    b_s1 = binder.bind_segment("seg-relabel", "3", 3000, 4200)
    assert b_s1.speaker_id == "S1"
    updates = binder.reconcile("3", received_ms=4300, new_speaker_id="ME", confidence=0.95)
    assert len(updates) == 1
    assert updates[0].kind is EventKind.SPEAKER_UPDATE
    assert updates[0].speaker_id == "ME"

    # Cross-epoch Sortformer cluster flip: epoch 1 swaps cluster 1 and 2, verifier still finds S1 vs ME
    binder.reset_diarizer_epoch()
    flipped_s1 = binder.bind_segment("seg-ep1-a", "1", 5000, 6400, audio_pcm16=s1_audio)
    flipped_me = binder.bind_segment("seg-ep1-b", "2", 6500, 7900, audio_pcm16=me_audio)
    assert flipped_s1.speaker_id == "S1"
    assert flipped_me.speaker_id == "ME"


def test_deepgram_normalizer_interim_committed_utterance_end() -> None:
    norm = DeepgramNormalizer(
        session_id="sess-dg",
        speaker_map={0: "ME", 1: "S1"},
        confidence_floor=0.75,
    )
    interim = norm.normalize(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "start": 1.2,
            "duration": 0.6,
            "is_final": False,
            "speech_final": False,
            "channel": {
                "alternatives": [
                    {
                        "transcript": "Someone should",
                        "confidence": 0.91,
                        "words": [{"word": "Someone", "start": 1.2, "end": 1.5, "speaker": 1}],
                    }
                ]
            },
        },
        received_ms=1900,
    )
    final = norm.normalize(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "start": 1.2,
            "duration": 1.1,
            "is_final": True,
            "speech_final": True,
            "channel": {
                "alternatives": [
                    {
                        "transcript": "Someone should own this.",
                        "confidence": 0.96,
                        "words": [
                            {"word": "Someone", "start": 1.2, "end": 1.5, "speaker": 1},
                            {"word": "this.", "start": 2.0, "end": 2.3, "speaker": 1},
                        ],
                    }
                ]
            },
        },
        received_ms=2450,
    )
    utt_end = norm.normalize(
        {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 2.9},
        received_ms=3500,
    )

    assert len(interim) == 1
    assert interim[0].stability is Stability.INTERIM
    assert interim[0].source_revision == 1
    assert len(final) == 2
    assert final[0].stability is Stability.COMMITTED
    assert final[0].segment_id == interim[0].segment_id
    assert final[0].source_revision == 2
    assert final[0].speaker_id == "S1"
    assert final[1].kind is EventKind.BOUNDARY and final[1].boundary == "speech_final"
    assert utt_end[0].kind is EventKind.BOUNDARY and utt_end[0].end_ms == 2900


async def test_replay_player_recorded_scaled_and_stepped_modes(tmp_path: Path) -> None:
    brief = _make_brief("sess-rep")
    evs = [
        CanonicalEvent(
            schema_version=SCHEMA_VERSION,
            session_id="sess-rep",
            event_id=f"ev-{i}",
            kind=EventKind.SEGMENT,
            received_ms=1000 + i * 500,
            source_sequence=i + 1,
            source_id="replay",
            segment_id=f"seg-{i}",
            source_revision=1,
            channel=0,
            speaker_id="S1",
            start_ms=800 + i * 500,
            end_ms=950 + i * 500,
            text=f"Turn {i}",
            stability=Stability.COMMITTED,
        )
        for i in range(3)
    ]
    jsonl_path = tmp_path / "events.jsonl"
    export_events_jsonl(evs, jsonl_path, brief=brief)

    player = ReplayPlayer.from_jsonl(jsonl_path)
    assert len(player.events) == 3

    sleeps_rec: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps_rec.append(sec)

    rec_out = [
        e.event_id async for e in player.play(mode=ReplayTimingMode.RECORDED, sleep_fn=fake_sleep)
    ]
    assert rec_out == ["ev-0", "ev-1", "ev-2"]
    assert sleeps_rec == [0.5, 0.5]

    player.reset()
    sleeps_scaled: list[float] = []

    async def fake_sleep_scaled(sec: float) -> None:
        sleeps_scaled.append(sec)

    await _collect_async(
        player.play(mode=ReplayTimingMode.SCALED, speed=2.0, sleep_fn=fake_sleep_scaled)
    )
    assert sleeps_scaled == [0.25, 0.25]

    player.reset()
    clock = ManualClock(0)
    stepped_ids: list[str] = []

    async def _run_stepped() -> None:
        async for ev in player.play(mode=ReplayTimingMode.STEPPED, clock=clock):
            stepped_ids.append(ev.event_id)

    task = asyncio.create_task(_run_stepped())
    await asyncio.sleep(0.01)
    assert stepped_ids == []
    await player.step(1)
    await asyncio.sleep(0.01)
    assert stepped_ids == ["ev-0"] and clock.now_ms() == 1000
    await player.step_to(2000)
    await task
    assert stepped_ids == ["ev-0", "ev-1", "ev-2"] and clock.now_ms() == 2000


async def _collect_async(it: AsyncIterator[CanonicalEvent]) -> list[CanonicalEvent]:
    return [item async for item in it]


def test_adapter_events_satisfy_canonical_state_invalidation_and_conservation_invariants(
    tmp_path: Path,
) -> None:
    brief = _make_brief("sess-inv")
    store = SQLiteStore(tmp_path / "inv.db")
    store.upsert_session(brief=brief, run_id="run-inv", started_ms=0)
    state = CanonicalState(brief)
    telemetry = BoundedRecorderBuffer(maxsize=128)

    verifier = NumpySpectralVerifier()
    s1_audio = _make_harmonic_pcm16(140.0, 1300, tilt=0.65)
    s2_audio = _make_harmonic_pcm16(260.0, 1300, tilt=0.40)

    binder = RecipientBinder(
        session_id="sess-inv",
        config=BinderConfig(min_confident_duration_ms=1000, auto_enroll_unknown=True),
        verifier=verifier,
    )
    seg = NemotronSegmenter(session_id="sess-inv", binder=binder)

    # Epoch 0: cluster "1" gets S1 and auto-enrolls
    evs1 = seg.feed_words(
        [
            AsrWord(word="We", start_ms=100, end_ms=450, speaker="1"),
            AsrWord(word="need", start_ms=480, end_ms=850, speaker="1"),
            AsrWord(word="owners", start_ms=880, end_ms=1350, speaker="1"),
        ],
        received_ms=1650,
        gates=[EarshotGateSample(at_ms=200, speech_active=True, echo_residual_ok=True)],
        audio_pcm16_fn=lambda _a, _b: s1_audio,
        flush_end=True,
    )
    # Epoch 1: Sortformer resets and labels the same voice as cluster "2", plus B21 echo-gated span
    binder.reset_diarizer_epoch()
    evs2 = seg.feed_words(
        [
            AsrWord(word="Okay", start_ms=2000, end_ms=2500, speaker="2"),
            AsrWord(word="understood", start_ms=2550, end_ms=3250, speaker="2"),
        ],
        received_ms=3500,
        gates=[EarshotGateSample(at_ms=2100, speech_active=False, echo_residual_ok=False)],
        audio_pcm16_fn=lambda _a, _b: s1_audio,
        flush_end=True,
    )
    # Also verify speaker_update invalidation (B22: S1 -> ME)
    reconcile_evs = binder.reconcile("2", received_ms=3600, new_speaker_id="ME", confidence=0.98)

    all_evs = [*evs1, *evs2, *reconcile_evs]
    committed_seg_ids: list[str] = []
    for ev in all_evs:
        res = state.apply(ev)
        assert res.accepted, f"Failed apply: {res.drop_reason}"
        store.record_canonical_event(run_id="run-inv", event=ev, retention=brief.retention)
        if ev.kind is EventKind.SEGMENT and ev.stability is Stability.COMMITTED and ev.segment_id:
            committed_seg_ids.append(ev.segment_id)

    # First segment stays S1, second segment was reconciled to ME via SPEAKER_UPDATE
    snap = state.snapshot()
    by_id = {s.segment_id: s for s in snap.segments}
    assert by_id[committed_seg_ids[0]].speaker_id == "S1"
    assert by_id[committed_seg_ids[1]].speaker_id == "ME"
    assert by_id[committed_seg_ids[1]].low_quality_gate is True
    assert snap.counters["material_revisions_after_commit"] == 1

    # Account for the non-recipient committed segment (S1) in decisions so Invariant 1 passes
    s1_rec = by_id[committed_seg_ids[0]]
    ctx = DecisionContext(
        schema_version=SCHEMA_VERSION,
        candidate_id="cand-1",
        session_id="sess-inv",
        origin=Origin.PROACTIVE,
        based_on_state_version=snap.state_version,
        builder_version="m2-test",
        prefilter_version="m2-test",
        anchor_refs=(SegmentRef(segment_id=s1_rec.segment_id, revision=s1_rec.revision),),
        context_refs=(),
        anchor_received_ms=1650,
        display_by_ms=9650,
        token_counts={"total": 8},
        omitted={},
        flags={},
        serialized_sha256="0" * 64,
    )
    store.record_decision(
        run_id="run-inv",
        context=ctx,
        classifier=ClassifierResult(
            schema_version=SCHEMA_VERSION,
            candidate_id="cand-1",
            based_on_state_version=snap.state_version,
            classifier_version="rules_v0",
            raw_scores=ScoreSet(opportunity=0.45, categories={"CONFIRM_OWNER": 0.45}),
            calibration_version="cal_v0",
            calibrated_scores=ScoreSet(opportunity=0.45, categories={"CONFIRM_OWNER": 0.45}),
            latency_ms=2.5,
            input_refs=ctx.anchor_refs,
        ),
        routing=RoutingDecision(
            schema_version=SCHEMA_VERSION,
            candidate_id="cand-1",
            policy_version="policy-v0-a",
            policy_sha256="0" * 64,
            based_on_state_version=snap.state_version,
            outcome=Outcome.ABSTAIN,
            route=None,
            category=None,
            reason_codes=(ReasonCode.SCORE_IN_BAND,),
            abstain_reasons=(ReasonCode.SCORE_IN_BAND,),
            thresholds=None,
            decided_ms=1670,
        ),
        terminal_state="abstained",
        terminal_reason=ReasonCode.SCORE_IN_BAND,
        serialized_text="[ANCHOR] S1: We need owners",
    )
    store.upsert_session(
        brief=brief,
        run_id="run-inv",
        started_ms=0,
        ended_ms=4000,
        revision_history=snap.revision_history,
        reducer_counters=snap.counters,
    )
    telemetry.flush_to_store(store, "sess-inv", "run-inv")

    inv_report = check_conservation_invariants(store, "sess-inv", "run-inv")
    assert inv_report.passed, f"Invariant violations: {inv_report.violations}"

    report = build_session_report(store, "sess-inv", "run-inv")
    p50 = report.upstream_delay_ms["p50"]
    assert p50 is not None and p50 >= 250.0
    assert s2_audio != s1_audio
    store.close()


def test_off_record_suppresses_raw_events_preserves_commit_bookkeeping_and_sets_gap_bounds(
    tmp_path: Path,
) -> None:
    brief = _make_brief("sess-otr")
    store = SQLiteStore(tmp_path / "otr.db")
    store.upsert_session(brief=brief, run_id="run-otr", started_ms=0)
    clock = ManualClock(500)
    adapter = PeeceeRealtimeAdapter(
        session_id="sess-otr",
        clock=clock,
        raw_sink=_StoreRawAdapter(store, "run-otr", brief.retention),
    )
    # Pre-off-record partial + delta arrives
    pre = adapter.process_server_message(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": "partial secret",
            "audio_processed": 0.4,
        },
        received_ms=600,
    )
    assert len(pre) == 1
    assert len(store.get_raw_events("sess-otr", "run-otr")) == 1

    clock.advance(500)
    start_ctrls = adapter.set_off_record(True)
    assert len(start_ctrls) == 1
    assert start_ctrls[0].subtype is ControlSubtype.OFF_RECORD_START
    assert adapter.segmenter._partial_buffer == ""
    assert adapter.segmenter._active_interim_seg_id is None

    # Off-the-record server payloads must NOT hit raw_events or emit canonical events
    assert (
        adapter.process_server_message(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "secret_off_record_text",
                "audio_processed": 0.9,
                "words": [
                    {"word": "secret_off_record_text", "start": 0.2, "end": 0.9, "speaker": 1}
                ],
            },
            received_ms=1100,
        )
        == []
    )
    # In-flight commit still advances commit_base_ms and diarizer epoch silently
    assert (
        adapter.process_server_message({"type": "input_audio_buffer.committed"}, received_ms=1150)
        == []
    )
    assert adapter.commit_base_ms >= 900
    assert adapter.segmenter.binder.epoch == 1
    raw_rows = store.get_raw_events("sess-otr", "run-otr")
    assert len(raw_rows) == 1

    clock.advance(1000)
    end_ctrls = adapter.set_off_record(False)
    assert len(end_ctrls) == 1
    assert end_ctrls[0].subtype is ControlSubtype.OFF_RECORD_END
    assert end_ctrls[0].start_ms == 1000
    assert end_ctrls[0].end_ms == 2000
    store.close()


async def test_async_commit_pipelining_and_max_uncommitted_no_spam() -> None:
    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[bytes | str] = []

        async def send(self, data: bytes | str) -> None:
            self.sent.append(data)

    clock = ManualClock(0)
    adapter = PeeceeRealtimeAdapter(session_id="sess-pipe", clock=clock)
    adapter._ws = _FakeWS()  # type: ignore[assignment]

    utt1_pcm = _make_harmonic_pcm16(130.0, 160)
    utt2_pcm = _make_harmonic_pcm16(260.0, 160)
    for _ in range(4):
        clock.advance(160)
        await adapter.send_audio_pcm16(utt1_pcm)
    await adapter.commit_audio()

    # Send 3 frames of Utt 2 BEFORE Utt 1's input_audio_buffer.committed arrives
    for _ in range(3):
        clock.advance(160)
        await adapter.send_audio_pcm16(utt2_pcm)

    adapter.process_server_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "audio_processed": 0.64,
            "words": [{"word": "utt1", "start": 0.1, "end": 0.5, "speaker": 1}],
        },
        received_ms=700,
    )
    adapter.process_server_message({"type": "input_audio_buffer.committed"}, received_ms=720)
    assert adapter.commit_base_ms == 640
    assert len(adapter._utterance_audio) == 3 * 5120
    assert adapter._slice_utterance_audio(740, 1040) == (utt2_pcm * 3)[3200:12800]

    # Mode B max_uncommitted_audio_ms must fire once per window, not per frame
    async with MockNeMoSpeechServer(eou_mode=True) as srv:
        ad2 = PeeceeRealtimeAdapter(
            session_id="sess-nospam",
            config=PeeceeConfig(
                ws_url=srv.url,
                auto_commit_silence_ms=None,
                max_uncommitted_audio_ms=480,
            ),
        )
        await ad2.connect()
        for _ in range(8):
            await ad2.send_audio_pcm16(utt1_pcm)
        await asyncio.sleep(0.03)
        await ad2.close()
    assert srv.commit_count == 2


async def test_cross_commit_monotonic_timestamps_and_upstream_delay_origin() -> None:
    telemetry = BoundedRecorderBuffer(maxsize=32)
    clock = ManualClock(0)
    adapter = PeeceeRealtimeAdapter(
        session_id="sess-mono",
        clock=clock,
        telemetry_sink=telemetry,
    )
    # Simulate 15s slow startup before first audio frame + 5s pause
    clock.advance(15_000)

    class _NoopWS:
        async def send(self, _data: bytes | str) -> None:
            return None

    adapter._ws = _NoopWS()  # type: ignore[assignment]
    frame = _make_harmonic_pcm16(140.0, 160)
    for _ in range(5):
        clock.advance(160)
        await adapter.send_audio_pcm16(frame)
    await adapter.pause()
    clock.advance(5_000)
    await adapter.resume()
    for _ in range(5):
        clock.advance(160)
        await adapter.send_audio_pcm16(frame)

    evs0 = adapter.process_server_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "audio_processed": 1.0,
            "words": [{"word": "first", "start": 0.8, "end": 1.4, "speaker": 1}],
        },
        received_ms=clock.now_ms(),
    )
    adapter.process_server_message(
        {"type": "input_audio_buffer.committed"}, received_ms=clock.now_ms()
    )
    evs1 = adapter.process_server_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "audio_processed": 0.5,
            "words": [{"word": "second", "start": 0.1, "end": 0.4, "speaker": 2}],
        },
        received_ms=clock.now_ms() + 100,
    )
    seg0 = next(e for e in evs0 if e.kind is EventKind.SEGMENT)
    seg1 = next(e for e in evs1 if e.kind is EventKind.SEGMENT)
    assert seg0.end_ms == 1400
    assert seg1.start_ms is not None and seg1.start_ms >= seg0.end_ms

    delays = [r for r in telemetry.drain() if r.kind == "upstream_delay"]
    assert len(delays) == 2
    assert delays[0].payload["lag_ms"] < 1000
    assert delays[0].payload["raw_lag_ms"] > 15_000


async def test_realtime_ws_send_audio_emits_gate_close_boundary_immediately() -> None:
    clock = ManualClock(0)
    async with MockNeMoSpeechServer(utterances=[], eou_mode=True) as srv:
        adapter = PeeceeRealtimeAdapter(
            session_id="sess-gate",
            config=PeeceeConfig(ws_url=srv.url, auto_commit_silence_ms=None),
            clock=clock,
        )
        await adapter.connect()
        emitted: list[CanonicalEvent] = []

        async def _consume() -> None:
            async for ev in adapter.events():
                emitted.append(ev)

        consumer = asyncio.create_task(_consume())
        clock.advance(160)
        await adapter.send_audio_pcm16(
            b"\x01\x00" * 2560,
            gate=EarshotGateSample(at_ms=clock.now_ms(), speech_active=True),
        )
        clock.advance(160)
        await adapter.send_audio_pcm16(
            b"\x00" * 5120,
            gate=EarshotGateSample(at_ms=clock.now_ms(), speech_active=False),
        )
        for _ in range(25):
            if any(e.kind is EventKind.BOUNDARY and e.boundary == "gate_close" for e in emitted):
                break
            await asyncio.sleep(0.01)
        assert any(e.kind is EventKind.BOUNDARY and e.boundary == "gate_close" for e in emitted)
        await adapter.close()
        await consumer


def test_cross_epoch_anon_label_isolation_shared_trackers_uncertain_and_spectral_contrast() -> None:
    # 1. Cross-epoch anonymous speaker label isolation (Challenger #1)
    voice_a = _make_harmonic_pcm16(125.0, 1400, tilt=0.75)
    voice_b = _make_harmonic_pcm16(245.0, 1400, tilt=0.45)
    voice_c = _make_harmonic_pcm16(480.0, 1400, tilt=0.35)
    verifier = NumpySpectralVerifier()
    binder = RecipientBinder(
        session_id="sess-ep",
        config=BinderConfig(min_confident_duration_ms=1000, auto_enroll_unknown=True),
        verifier=verifier,
    )
    b0_a = binder.bind_segment("s0a", "1", 0, 1400, audio_pcm16=voice_a)
    b0_b = binder.bind_segment("s0b", "2", 1500, 2900, audio_pcm16=voice_b)
    assert (b0_a.speaker_id, b0_b.speaker_id) == ("S1", "S2")

    binder.reset_diarizer_epoch()
    b1_b = binder.bind_segment("s1b", "1", 3000, 4400, audio_pcm16=voice_b)
    b1_c = binder.bind_segment("s1c", "2", 4500, 5900, audio_pcm16=voice_c)
    assert b1_b.speaker_id == "S2"
    assert b1_c.speaker_id == "S3"

    # 2. Single-speaker NumpySpectralVerifier avoids false-confident 0.85+ on unrelated tones (Challenger #10)
    single_v = NumpySpectralVerifier()
    single_v.enroll("ME", _make_harmonic_pcm16(120.0, 1400))
    scores = single_v.score(_make_harmonic_pcm16(240.0, 1400))
    assert scores["ME"] < 0.65

    # 3. Explicit binder shares sequence/revision trackers + emits SPEAKER_UPDATE on speaker_uncertain True -> False
    ext_binder = RecipientBinder(
        session_id="sess-share",
        config=BinderConfig(recipient_azimuth_deg=0.0, min_confident_duration_ms=1000),
    )
    ext_binder.feed_direction(
        DirectionSample(t_ms=300, azimuth_deg=0.0, contrast=4.0, level_db=-20.0)
    )
    ext_binder.feed_direction(
        DirectionSample(t_ms=1800, azimuth_deg=0.0, contrast=4.0, level_db=-20.0)
    )
    seg = NemotronSegmenter(session_id="sess-share", binder=ext_binder)
    evs1 = seg.feed_words([AsrWord("hi", 200, 500, "1")], received_ms=600, flush_end=True)
    seg_ev1 = next(e for e in evs1 if e.kind is EventKind.SEGMENT)
    assert seg_ev1.speaker_id == "ME" and seg_ev1.speaker_uncertain is True

    evs2 = seg.feed_words(
        [AsrWord("confirming", 1500, 2800, "1")], received_ms=2900, flush_end=True
    )
    upd = next(e for e in evs2 if e.kind is EventKind.SPEAKER_UPDATE)
    assert upd.segment_id == seg_ev1.segment_id
    assert upd.speaker_id == "ME"
    assert upd.speaker_uncertain is False
    assert upd.source_revision == 2
    seqs = [e.source_sequence for e in (*evs1, *evs2)]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


async def test_nemotron_split_across_unlabeled_word_idempotent_delta_and_bounded_interim() -> None:
    # 1. MockNeMoSpeechServer(omit_speaker_on_short_word=True) splits speaker 1 -> None -> speaker 2
    utt = MockUtteranceScript(
        deltas=["Alice uh Bob"],
        words=[
            {"word": "Alice", "start": 0.10, "end": 1.15, "confidence": 1.0, "speaker": 1},
            {"word": "uh", "start": 1.17, "end": 1.25, "confidence": 1.0, "speaker": 1},
            {"word": "Bob", "start": 1.27, "end": 2.40, "confidence": 1.0, "speaker": 2},
        ],
    )
    async with MockNeMoSpeechServer(utterances=[utt], omit_speaker_on_short_word=True) as srv:
        adapter = PeeceeRealtimeAdapter(
            session_id="sess-split", config=PeeceeConfig(ws_url=srv.url)
        )
        await adapter.connect()
        collected: list[CanonicalEvent] = []

        async def _consume() -> None:
            async for ev in adapter.events():
                collected.append(ev)

        task = asyncio.create_task(_consume())
        await adapter.send_audio_pcm16(b"\x01\x00" * 2560)
        await adapter.commit_audio()
        await asyncio.sleep(0.05)
        await adapter.close()
        await task

    segs = [
        e for e in collected if e.kind is EventKind.SEGMENT and e.stability is Stability.COMMITTED
    ]
    assert [(s.text, s.speaker_id) for s in segs] == [("Alice uh", "S1"), ("Bob", "S2")]

    # 2. Repeated identical cumulative delta is idempotent & interim buffer stays O(channels)
    seg = NemotronSegmenter(session_id="sess-int")
    state = CanonicalState(_make_brief("sess-int"))
    d1 = seg.feed_partial("hel", audio_processed_ms=200, received_ms=250)
    d2 = seg.feed_partial("hel", audio_processed_ms=250, received_ms=300)
    assert len(d1) == 1 and d1[0].text == "hel"
    assert d2 == []
    for ev in d1:
        assert state.apply(ev).accepted

    for idx in range(5):
        t0 = idx * 2000 + 100
        for ev in [
            *seg.feed_partial(f"word{idx}", audio_processed_ms=t0, received_ms=t0 + 20),
            *seg.feed_words(
                [AsrWord(f"word{idx}", t0, t0 + 1200, speaker="1")],
                received_ms=t0 + 1250,
                flush_end=True,
            ),
        ]:
            assert state.apply(ev).accepted
    assert len(state.snapshot().interim_segments) <= 1


def test_replay_from_raw_messages_offline_speaker_update_rx_and_deepgram_edge_cases() -> None:
    # 1. ReplayPlayer.from_raw_messages works with BOTH DeepgramNormalizer and PeeceeRealtimeAdapter
    dg = DeepgramNormalizer(session_id="sess-raw-dg")
    dg_raw: list[tuple[int, Mapping[str, Any]]] = [
        (
            1200,
            {
                "type": "Results",
                "channel_index": [0],
                "start": 0.5,
                "duration": 0.5,
                "is_final": True,
                "speech_final": True,
                "channel": {"alternatives": [{"transcript": "Hello there", "words": []}]},
            },
        )
    ]
    dg_player = ReplayPlayer.from_raw_messages(dg_raw, normalizer=dg.normalize)
    assert len(dg_player.events) == 2 and dg_player.events[0].text == "Hello there"

    peecee = PeeceeRealtimeAdapter(session_id="sess-raw-pc")
    pc_raw: list[tuple[int, Mapping[str, Any]]] = [
        (
            1500,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "audio_processed": 1.2,
                "words": [{"word": "Done", "start": 0.2, "end": 1.2, "speaker": 1}],
            },
        )
    ]
    pc_player = ReplayPlayer.from_raw_messages(pc_raw, normalizer=peecee.process_server_message)
    assert len(pc_player.events) == 2 and pc_player.events[0].text == "Done"

    # 2. parse_nemo_offline_json(..., start_received_ms=0) assigns monotonic received_ms to SPEAKER_UPDATE
    off_evs = parse_nemo_offline_json(
        {
            "words": [
                {"word": "short", "start": 0.5, "end": 0.8, "speaker": 1},
                {"word": "confirming", "start": 1.5, "end": 2.8, "speaker": 1},
            ]
        },
        session_id="sess-off-rx",
        binder=RecipientBinder(
            session_id="sess-off-rx",
            config=BinderConfig(recipient_azimuth_deg=0.0, min_confident_duration_ms=1000),
        ),
    )
    # Force an azimuth match on the second segment via a direct binder test with DirectionSample
    dir_binder = RecipientBinder(
        session_id="sess-off-rx2",
        config=BinderConfig(recipient_azimuth_deg=0.0, min_confident_duration_ms=1000),
    )
    dir_binder.feed_direction(
        DirectionSample(t_ms=1800, azimuth_deg=0.0, contrast=4.0, level_db=-20.0)
    )
    off_evs2 = parse_nemo_offline_json(
        {
            "words": [
                {"word": "short", "start": 0.5, "end": 0.8, "speaker": 1},
                {"word": "confirming", "start": 1.5, "end": 2.8, "speaker": 1},
            ]
        },
        session_id="sess-off-rx2",
        binder=dir_binder,
    )
    assert len(off_evs) >= 4
    upd = next(e for e in off_evs2 if e.kind is EventKind.SPEAKER_UPDATE)
    assert upd.received_ms >= 2800

    # 3. DeepgramNormalizer empty interim does not poison _active_utt_start_ms & omitted start is monotonic
    dg2 = DeepgramNormalizer(session_id="sess-dg-edge")
    assert (
        dg2.normalize(
            {
                "type": "Results",
                "channel_index": [0],
                "start": 1.0,
                "duration": 0.2,
                "is_final": False,
                "channel": {"alternatives": [{"transcript": ""}]},
            },
            1100,
        )
        == []
    )
    real = dg2.normalize(
        {
            "type": "Results",
            "channel_index": [0],
            "start": 5.0,
            "duration": 0.8,
            "is_final": True,
            "channel": {"alternatives": [{"transcript": "Real speech"}]},
        },
        5900,
    )
    assert real[0].segment_id == "dg-ch0-00005000"

    dg3 = DeepgramNormalizer(session_id="sess-dg-nostart")
    f1 = dg3.normalize(
        {
            "type": "Results",
            "channel_index": [0],
            "is_final": True,
            "channel": {"alternatives": [{"transcript": "First"}]},
        },
        1000,
    )
    f2 = dg3.normalize(
        {
            "type": "Results",
            "channel_index": [0],
            "is_final": True,
            "channel": {"alternatives": [{"transcript": "Second"}]},
        },
        2000,
    )
    assert f1[0].segment_id != f2[0].segment_id


async def test_malformed_ws_json_and_startup_timeout_lease_set_no_transcript(
    tmp_path: Path,
) -> None:
    # 1. Malformed WS text frame -> NO_TRANSCRIPT + GAP + LIFECYCLE telemetry, no unhandled crash
    async def _bad_ws_handler(ws: ServerConnection) -> None:
        await ws.send(json.dumps({"type": "session.created"}))
        await ws.recv()
        await ws.send(json.dumps({"type": "session.updated"}))
        await ws.send("<html>502 Bad Gateway</html>")
        await ws.close()

    telemetry = BoundedRecorderBuffer(maxsize=32)
    async with serve(_bad_ws_handler, "127.0.0.1", 0) as srv:
        port = int(next(iter(srv.sockets)).getsockname()[1])
        adapter = PeeceeRealtimeAdapter(
            session_id="sess-bad-json",
            config=PeeceeConfig(ws_url=f"ws://127.0.0.1:{port}/ws"),
            telemetry_sink=telemetry,
        )
        await adapter.connect()
        evs = [ev async for ev in adapter.events()]

    assert adapter.health is HealthState.NO_TRANSCRIPT
    assert len(evs) == 1
    assert evs[0].kind is EventKind.CONTROL and evs[0].subtype is ControlSubtype.GAP
    assert evs[0].reason == "peecee_malformed_message"
    assert any(
        r.level is TelemetryLevel.LIFECYCLE
        and r.kind == "adapter_health"
        and r.payload.get("reason") == "peecee_malformed_message"
        for r in telemetry.drain()
    )

    # 2. PeeceeSpeechProcess.wait_ready() timeout sets NO_TRANSCRIPT + telemetry + health_callback
    hang_script = tmp_path / "peecee-hang"
    hang_script.write_text("#!/usr/bin/env bash\nsleep 10\n", encoding="utf-8")
    hang_script.chmod(0o755)
    proc_tel = BoundedRecorderBuffer(maxsize=16)
    proc_health: list[tuple[HealthState, str]] = []
    proc = PeeceeSpeechProcess(
        session_id="sess-to",
        config=PeeceeConfig(
            executable=hang_script,
            ws_url="ws://127.0.0.1:19976/ws",
            ready_timeout_s=0.2,
            kill_stop_grace_s=0.2,
        ),
        telemetry_sink=proc_tel,
        health_callback=lambda st, rsn, _det: proc_health.append((st, rsn)),
    )
    with pytest.raises(PeeceeProcessError):
        await proc.start(wait_for_ready=True)
    assert proc.health is HealthState.NO_TRANSCRIPT
    assert proc_health[-1] == (HealthState.NO_TRANSCRIPT, "peecee_ready_timeout")
    assert any(r.kind == "adapter_health" for r in proc_tel.drain())

    # 3. PeeceeRealtimeAdapter.connect() sets NO_TRANSCRIPT when process fails (exit 75)
    exit75 = tmp_path / "peecee-75"
    exit75.write_text("#!/usr/bin/env bash\nexit 75\n", encoding="utf-8")
    exit75.chmod(0o755)
    proc75 = PeeceeSpeechProcess(
        session_id="sess-conn-75",
        config=PeeceeConfig(executable=exit75, ws_url="ws://127.0.0.1:19977/ws"),
    )
    ad_75 = PeeceeRealtimeAdapter(
        session_id="sess-conn-75",
        config=PeeceeConfig(ws_url="ws://127.0.0.1:19977/ws"),
        process=proc75,
    )
    with pytest.raises(PeeceeLeaseUnavailableError):
        await ad_75.connect()
    assert ad_75.health is HealthState.NO_TRANSCRIPT


async def test_events_queue_task_does_not_steal_on_close_same_tick_and_drain() -> None:
    class _StubWS:
        def __init__(self, frames: list[Any]) -> None:
            self._frames = list(frames)
            self.recv_calls = 0
            self.closed = False

        async def send(self, _data: Any) -> None:
            return None

        async def recv(self) -> Any:
            self.recv_calls += 1
            nxt = self._frames.pop(0)
            if isinstance(nxt, asyncio.Future):
                return await nxt
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

        async def close(self) -> None:
            self.closed = True

    loop = asyncio.get_running_loop()

    # 1. SAME_TICK: queue_task (E1) + recv_task (ConnectionClosed) finish together,
    #    and concurrent E2 is enqueued while suspended at `yield E1`.
    close_fut: asyncio.Future[Any] = loop.create_future()
    ws1 = _StubWS([close_fut])
    clock1 = ManualClock(100)
    ad1 = PeeceeRealtimeAdapter(session_id="sess-same-tick", clock=clock1)
    ad1._ws = ws1  # type: ignore[assignment]
    await ad1.send_audio_pcm16(
        b"\x01\x00" * 32, gate=EarshotGateSample(at_ms=100, speech_active=True)
    )
    collected1: list[CanonicalEvent] = []

    async def _consume1() -> None:
        async for ev in ad1.events():
            collected1.append(ev)
            if ev.kind is EventKind.BOUNDARY and ev.end_ms == 200:
                await ad1.send_audio_pcm16(
                    b"\x01\x00" * 32,
                    gate=EarshotGateSample(at_ms=300, speech_active=True),
                )
                await ad1.send_audio_pcm16(
                    b"", gate=EarshotGateSample(at_ms=400, speech_active=False)
                )

    t1 = asyncio.create_task(_consume1())
    await asyncio.sleep(0)
    # Trigger E1 and ConnectionClosed in the same tick
    await ad1.send_audio_pcm16(b"", gate=EarshotGateSample(at_ms=200, speech_active=False))
    close_fut.set_exception(ConnectionClosed(None, None))
    await t1

    bnds1 = [e for e in collected1 if e.kind is EventKind.BOUNDARY]
    gaps1 = [
        e for e in collected1 if e.kind is EventKind.CONTROL and e.subtype is ControlSubtype.GAP
    ]
    assert [b.end_ms for b in bnds1] == [200, 400]
    assert len(gaps1) == 1 and gaps1[0].reason == "peecee_tunnel_lost"
    assert ad1._pending_events.empty()

    # 2. ConnectionClosed drain loop + health_callback: E1 queued before close,
    #    E2 enqueued during `yield E1`, E3 enqueued during `_set_health`.
    ad2: PeeceeRealtimeAdapter | None = None

    def _on_health2(st: HealthState, _rsn: str, _det: Mapping[str, Any]) -> None:
        if ad2 is not None and st is HealthState.NO_TRANSCRIPT:
            for ev in ad2.segmenter.feed_gate(
                EarshotGateSample(at_ms=500, speech_active=True), received_ms=500
            ) + ad2.segmenter.feed_gate(
                EarshotGateSample(at_ms=600, speech_active=False), received_ms=600
            ):
                ad2._pending_events.put_nowait(ev)

    class _CloseDrainWS(_StubWS):
        async def recv(self) -> Any:
            self.recv_calls += 1
            assert ad2 is not None
            await ad2.send_audio_pcm16(b"", gate=EarshotGateSample(at_ms=200, speech_active=False))
            raise ConnectionClosed(None, None)

    ad2 = PeeceeRealtimeAdapter(session_id="sess-drain", health_callback=_on_health2)
    ad2._ws = _CloseDrainWS([])  # type: ignore[assignment]
    await ad2.send_audio_pcm16(
        b"\x01\x00" * 32, gate=EarshotGateSample(at_ms=100, speech_active=True)
    )
    collected2: list[CanonicalEvent] = []
    async for ev in ad2.events():
        collected2.append(ev)
        if ev.kind is EventKind.BOUNDARY and ev.end_ms == 200:
            await ad2.send_audio_pcm16(
                b"\x01\x00" * 32, gate=EarshotGateSample(at_ms=300, speech_active=True)
            )
            await ad2.send_audio_pcm16(b"", gate=EarshotGateSample(at_ms=400, speech_active=False))

    bnds2 = [e for e in collected2 if e.kind is EventKind.BOUNDARY]
    gaps2 = [
        e for e in collected2 if e.kind is EventKind.CONTROL and e.subtype is ControlSubtype.GAP
    ]
    assert [b.end_ms for b in bnds2] == [200, 400, 600]
    assert len(gaps2) == 1 and gaps2[0].reason == "peecee_tunnel_lost"
    assert ad2._pending_events.empty()

    # 3. JSONDecodeError drains _pending_events and does NOT schedule extra recv()
    class _MalformedWS(_StubWS):
        async def recv(self) -> Any:
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise AssertionError("recv() scheduled after malformed JSON")
            assert ad3 is not None
            await ad3.send_audio_pcm16(b"", gate=EarshotGateSample(at_ms=200, speech_active=False))
            return "<html>502 Bad Gateway</html>"

    ws3 = _MalformedWS([])
    ad3 = PeeceeRealtimeAdapter(session_id="sess-bad-drain")
    ad3._ws = ws3  # type: ignore[assignment]
    await ad3.send_audio_pcm16(
        b"\x01\x00" * 32, gate=EarshotGateSample(at_ms=100, speech_active=True)
    )
    collected3: list[CanonicalEvent] = []
    async for ev in ad3.events():
        collected3.append(ev)
        if ev.kind is EventKind.BOUNDARY and ev.end_ms == 200:
            await ad3.send_audio_pcm16(
                b"\x01\x00" * 32, gate=EarshotGateSample(at_ms=300, speech_active=True)
            )
            await ad3.send_audio_pcm16(b"", gate=EarshotGateSample(at_ms=400, speech_active=False))

    bnds3 = [e for e in collected3 if e.kind is EventKind.BOUNDARY]
    gaps3 = [
        e for e in collected3 if e.kind is EventKind.CONTROL and e.subtype is ControlSubtype.GAP
    ]
    assert [b.end_ms for b in bnds3] == [200, 400]
    assert len(gaps3) == 1 and gaps3[0].reason == "peecee_malformed_message"
    assert ad3.health is HealthState.NO_TRANSCRIPT
    assert ws3.recv_calls == 1
    assert ad3._pending_events.empty()


async def test_send_audio_pcm16_truncates_odd_trailing_byte() -> None:
    sent_frames: list[bytes] = []

    class _RecordWS:
        async def send(self, data: Any) -> None:
            if isinstance(data, bytes):
                sent_frames.append(data)

        async def recv(self) -> Any:
            raise ConnectionClosed(None, None)

        async def close(self) -> None:
            return None

    adapter = PeeceeRealtimeAdapter(
        session_id="sess-odd",
        config=PeeceeConfig(auto_commit_silence_ms=None),
    )
    adapter._ws = _RecordWS()  # type: ignore[assignment]

    odd_payload = (b"\x01\x02" * 3200) + b"\xff"  # 6401 bytes
    sent1 = await adapter.send_audio_pcm16(odd_payload)
    assert sent1 == 6400
    assert [len(f) for f in sent_frames] == [5120, 1280]
    assert all(len(f) % 2 == 0 for f in sent_frames)
    assert b"".join(sent_frames) == odd_payload[:6400]
    assert len(adapter._utterance_audio) == 6400
    assert bytes(adapter._utterance_audio) == odd_payload[:6400]

    sent2 = await adapter.send_audio_pcm16(b"\x03\x04\xee")
    assert sent2 == 2
    assert len(adapter._utterance_audio) == 6402
    assert bytes(adapter._utterance_audio[-2:]) == b"\x03\x04"

    # Single odd byte yields 0 sent and leaves alignment intact
    sent3 = await adapter.send_audio_pcm16(b"\xaa")
    assert sent3 == 0
    assert len(adapter._utterance_audio) == 6402


def test_gate_quality_allows_onset_boundary_and_blocks_silence_or_echo() -> None:
    payload = {
        "text": "Before launch we need someone to own it",
        "words": [
            {"word": "Before", "start": 1.00, "end": 1.25, "confidence": 1.0, "speaker": 1},
            {"word": "launch", "start": 1.25, "end": 1.55, "confidence": 1.0, "speaker": 1},
        ],
    }
    onset_gates = [
        EarshotGateSample(at_ms=1000, speech_active=False, echo_residual_ok=True),
        EarshotGateSample(at_ms=1160, speech_active=True, echo_residual_ok=True),
        EarshotGateSample(at_ms=1320, speech_active=True, echo_residual_ok=True),
        EarshotGateSample(at_ms=1480, speech_active=True, echo_residual_ok=True),
    ]
    evs_ok = parse_nemo_offline_json(payload, session_id="s-onset", gates=onset_gates)
    segs_ok = [
        e for e in evs_ok if e.kind is EventKind.SEGMENT and e.stability is Stability.COMMITTED
    ]
    assert len(segs_ok) == 1
    assert segs_ok[0].low_quality_gate is False

    silent_gates = [
        EarshotGateSample(at_ms=ms, speech_active=False, echo_residual_ok=True)
        for ms in (1000, 1160, 1320, 1480, 1640, 1800)
    ]
    evs_sil = parse_nemo_offline_json(payload, session_id="s-sil", gates=silent_gates)
    segs_sil = [
        e for e in evs_sil if e.kind is EventKind.SEGMENT and e.stability is Stability.COMMITTED
    ]
    assert len(segs_sil) == 1
    assert segs_sil[0].low_quality_gate is True

    echo_gates = [
        EarshotGateSample(at_ms=1000, speech_active=True, echo_residual_ok=True),
        EarshotGateSample(at_ms=1160, speech_active=True, echo_residual_ok=False),
        EarshotGateSample(at_ms=1320, speech_active=True, echo_residual_ok=True),
    ]
    evs_echo = parse_nemo_offline_json(payload, session_id="s-echo", gates=echo_gates)
    segs_echo = [
        e for e in evs_echo if e.kind is EventKind.SEGMENT and e.stability is Stability.COMMITTED
    ]
    assert len(segs_echo) == 1
    assert segs_echo[0].low_quality_gate is True
