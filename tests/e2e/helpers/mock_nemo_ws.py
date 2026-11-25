"""Local mock NeMo-Speech.cpp (20999cc) WebSocket server and mock peecee-speech binary."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

PCM16_FRAME_BYTES_160MS = 5120


@dataclass(frozen=True)
class MockWord:
    """Word timestamp + optional Sortformer speaker ID (1..4)."""

    word: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: int | None = None

    def to_dict(self, *, speaker_override: int | None = None) -> dict[str, Any]:
        spk = speaker_override if speaker_override is not None else self.speaker
        payload: dict[str, Any] = {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }
        if spk is not None and spk > 0:
            payload["speaker"] = spk
        return payload


@dataclass
class ScriptedTurn:
    """Scripted server response triggered after frame count or buffer commit."""

    trigger_after_frames: int | None = None
    trigger_on_commit: bool = False
    deltas: list[str] = field(default_factory=list)
    transcript: str = ""
    words: list[MockWord] = field(default_factory=list)
    drop_connection: bool = False
    error: tuple[str, str, str] | None = None
    delay_s: float = 0.0


@dataclass
class MockNemoConfig:
    """Behavior configuration for MockNemoServer."""

    sample_rate: int = 16000
    model: str = "nemotron-speech-streaming-en-0.6b.q8_0.gguf"
    script: list[ScriptedTurn] = field(default_factory=list)
    silence_mode: bool = False
    flip_speaker_on_commit: dict[int, int] | None = None
    reject_update_after_audio: bool = True
    fail_all_with: tuple[str, str, str] | None = None


class MockNemoServer:
    """Thread-hosted async WebSocket server matching NeMo-Speech.cpp 20999cc."""

    def __init__(self, config: MockNemoConfig | None = None) -> None:
        self._config = config or MockNemoConfig()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._server: Server | None = None
        self._thread: threading.Thread | None = None
        self._port = 0
        self._event_seq = 0
        self._active_clients: set[ServerConnection] = set()
        self.received_frames: list[bytes] = []
        self.received_events: list[dict[str, Any]] = []
        self.session_configs: list[dict[str, Any]] = []

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self._port}/v1/audio/transcriptions/realtime"

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/health"

    def __enter__(self) -> MockNemoServer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise TimeoutError("MockNemoServer failed to start within 5s")
        if self._startup_error is not None:
            raise self._startup_error

    def stop(self) -> None:
        loop = self._loop
        stop_ev = self._stop_event
        if loop is not None and stop_ev is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop_ev.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._loop = None
        self._stop_event = None
        self._server = None

    def set_config(self, config: MockNemoConfig) -> None:
        with self._lock:
            self._config = config

    def set_script(self, script: list[ScriptedTurn]) -> None:
        with self._lock:
            self._config.script = list(script)

    def reset_recordings(self) -> None:
        with self._lock:
            self.received_frames.clear()
            self.received_events.clear()
            self.session_configs.clear()

    def drop_active_clients(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            clients = list(self._active_clients)
        for ws in clients:
            loop.call_soon_threadsafe(self._schedule_close, ws)

    @staticmethod
    def _schedule_close(ws: ServerConnection) -> None:
        asyncio.create_task(ws.close(1011, "drop"))

    def _next_event_id(self) -> str:
        with self._lock:
            self._event_seq += 1
            return f"event_{self._event_seq}"

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve_forever())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            loop.close()

    async def _serve_forever(self) -> None:
        self._stop_event = asyncio.Event()
        async with serve(
            self._handle_client,
            "127.0.0.1",
            0,
            process_request=self._process_request,
        ) as server:
            self._server = server
            sock = next(iter(server.sockets))
            self._port = int(sock.getsockname()[1])
            self._ready.set()
            await self._stop_event.wait()

    @staticmethod
    def _process_request(_conn: ServerConnection, request: Request) -> Response | None:
        path = urlsplit(request.path).path
        if path == "/health":
            return Response(
                200,
                "OK",
                Headers([("Content-Type", "application/json")]),
                b'{"status": "ok"}',
            )
        if path not in {"/v1/audio/transcriptions/realtime", "/v1/realtime"}:
            return Response(
                404,
                "Not Found",
                Headers([("Content-Type", "application/json")]),
                b'{"error": {"message": "Not found", "type": "invalid_request_error"}}',
            )
        return None

    async def _send_error(
        self,
        ws: ServerConnection,
        *,
        err_type: str,
        code: str,
        message: str,
    ) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "error",
                    "event_id": self._next_event_id(),
                    "error": {
                        "message": message,
                        "type": err_type,
                        "code": code,
                    },
                }
            )
        )

    async def _emit_turn(
        self,
        ws: ServerConnection,
        turn: ScriptedTurn,
        *,
        audio_processed: float,
        include_words: bool,
        speaker_flip: dict[int, int] | None,
    ) -> bool:
        if turn.delay_s > 0.0:
            await asyncio.sleep(turn.delay_s)
        if turn.error is not None:
            err_type, code, message = turn.error
            await self._send_error(ws, err_type=err_type, code=code, message=message)
        if turn.drop_connection:
            await ws.close(1011, "scripted_drop")
            return False
        for delta in turn.deltas:
            await ws.send(
                json.dumps(
                    {
                        "event_id": self._next_event_id(),
                        "type": "conversation.item.input_audio_transcription.delta",
                        "delta": delta,
                        "audio_processed": round(audio_processed, 3),
                    }
                )
            )
        if turn.transcript or turn.words:
            payload: dict[str, Any] = {
                "event_id": self._next_event_id(),
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": turn.transcript,
                "audio_processed": round(audio_processed, 3),
            }
            if include_words:
                words_out: list[dict[str, Any]] = []
                for w in turn.words:
                    spk = w.speaker
                    if speaker_flip and spk is not None and spk in speaker_flip:
                        spk = speaker_flip[spk]
                    words_out.append(w.to_dict(speaker_override=spk))
                payload["words"] = words_out
            await ws.send(json.dumps(payload))
        return True

    async def _handle_client(self, ws: ServerConnection) -> None:
        with self._lock:
            self._active_clients.add(ws)
            cfg = self._config
        session_state: dict[str, Any] = {
            "input_audio_format": "pcm16",
            "sample_rate": cfg.sample_rate,
            "model": cfg.model,
            "word_timestamps": True,
            "speaker_diarization": True,
        }
        audio_started = False
        frame_count = 0
        audio_bytes = 0
        fired_indices: set[int] = set()
        try:
            await ws.send(
                json.dumps(
                    {
                        "event_id": self._next_event_id(),
                        "type": "session.created",
                        "session": dict(session_state),
                    }
                )
            )
            async for message in ws:
                with self._lock:
                    cfg = self._config
                if cfg.fail_all_with is not None:
                    err_type, code, err_msg = cfg.fail_all_with
                    await self._send_error(ws, err_type=err_type, code=code, message=err_msg)
                    continue
                if isinstance(message, bytes):
                    audio_started = True
                    frame_count += 1
                    audio_bytes += len(message)
                    with self._lock:
                        self.received_frames.append(message)
                    keep = await self._check_frame_turns(
                        ws, cfg, frame_count, audio_bytes, session_state, fired_indices
                    )
                    if not keep:
                        return
                else:
                    res = await self._handle_json_frame(
                        ws,
                        message,
                        cfg,
                        session_state,
                        audio_started,
                        frame_count,
                        audio_bytes,
                        fired_indices,
                    )
                    audio_started, frame_count, audio_bytes, keep = res
                    if not keep:
                        return
        except ConnectionClosed:
            return
        finally:
            with self._lock:
                self._active_clients.discard(ws)

    async def _check_frame_turns(
        self,
        ws: ServerConnection,
        cfg: MockNemoConfig,
        frame_count: int,
        audio_bytes: int,
        session_state: dict[str, Any],
        fired_indices: set[int],
    ) -> bool:
        if cfg.silence_mode:
            return True
        sr = int(session_state.get("sample_rate", cfg.sample_rate))
        audio_processed = audio_bytes / float(sr * 2)
        include_words = bool(
            session_state.get("word_timestamps", True)
            or session_state.get("speaker_diarization", True)
        )
        for idx, turn in enumerate(cfg.script):
            if (
                not turn.trigger_on_commit
                and turn.trigger_after_frames is not None
                and frame_count >= turn.trigger_after_frames
                and idx not in fired_indices
            ):
                fired_indices.add(idx)
                ok = await self._emit_turn(
                    ws,
                    turn,
                    audio_processed=audio_processed,
                    include_words=include_words,
                    speaker_flip=None,
                )
                if not ok:
                    return False
        return True

    async def _handle_json_frame(
        self,
        ws: ServerConnection,
        raw_text: str,
        cfg: MockNemoConfig,
        session_state: dict[str, Any],
        audio_started: bool,
        frame_count: int,
        audio_bytes: int,
        fired_indices: set[int],
    ) -> tuple[bool, int, int, bool]:
        try:
            event = json.loads(raw_text)
        except json.JSONDecodeError:
            await self._send_error(
                ws,
                err_type="invalid_request_error",
                code="invalid_argument",
                message="invalid JSON",
            )
            return audio_started, frame_count, audio_bytes, True
        if not isinstance(event, dict):
            await self._send_error(
                ws,
                err_type="invalid_request_error",
                code="invalid_argument",
                message="JSON message must be an object",
            )
            return audio_started, frame_count, audio_bytes, True

        with self._lock:
            self.received_events.append(event)

        msg_type = str(event.get("type", ""))
        if msg_type in {"session.update", "config"}:
            if audio_started and cfg.reject_update_after_audio:
                await self._send_error(
                    ws,
                    err_type="invalid_request_error",
                    code="invalid_argument",
                    message="session configuration cannot change after audio starts",
                )
                return audio_started, frame_count, audio_bytes, True
            sess = event.get("session", event)
            if not isinstance(sess, dict):
                await self._send_error(
                    ws,
                    err_type="invalid_request_error",
                    code="invalid_argument",
                    message="session must be an object",
                )
                return audio_started, frame_count, audio_bytes, True
            sr = sess.get("sample_rate", session_state["sample_rate"])
            if not isinstance(sr, int) or isinstance(sr, bool) or not (8000 <= sr <= 96000):
                await self._send_error(
                    ws,
                    err_type="invalid_request_error",
                    code="invalid_argument",
                    message="sample_rate must be integer in [8000, 96000]",
                )
                return audio_started, frame_count, audio_bytes, True
            max_spk = sess.get("max_speaker_count")
            if max_spk is not None and (
                not isinstance(max_spk, int) or isinstance(max_spk, bool) or max_spk <= 0
            ):
                await self._send_error(
                    ws,
                    err_type="invalid_request_error",
                    code="invalid_argument",
                    message="max_speaker_count must be positive integer",
                )
                return audio_started, frame_count, audio_bytes, True
            session_state.update(sess)
            with self._lock:
                self.session_configs.append(dict(session_state))
            await ws.send(
                json.dumps(
                    {
                        "event_id": self._next_event_id(),
                        "type": "session.updated",
                        "session": dict(session_state),
                    }
                )
            )
            return audio_started, frame_count, audio_bytes, True
        if msg_type == "input_audio_buffer.append":
            raw_b64 = event.get("audio")
            if not isinstance(raw_b64, str):
                await self._send_error(
                    ws,
                    err_type="invalid_request_error",
                    code="invalid_argument",
                    message="audio must be base64 string",
                )
                return audio_started, frame_count, audio_bytes, True
            try:
                pcm = base64.b64decode(raw_b64, validate=True)
            except binascii.Error:
                await self._send_error(
                    ws,
                    err_type="invalid_request_error",
                    code="invalid_argument",
                    message="invalid base64 audio",
                )
                return audio_started, frame_count, audio_bytes, True
            audio_started = True
            frame_count += 1
            audio_bytes += len(pcm)
            with self._lock:
                self.received_frames.append(pcm)
            keep = await self._check_frame_turns(
                ws, cfg, frame_count, audio_bytes, session_state, fired_indices
            )
            return audio_started, frame_count, audio_bytes, keep

        if msg_type == "input_audio_buffer.commit":
            sr = int(session_state.get("sample_rate", cfg.sample_rate))
            audio_processed = audio_bytes / float(sr * 2)
            include_words = bool(
                session_state.get("word_timestamps", True)
                or session_state.get("speaker_diarization", True)
            )
            if not cfg.silence_mode:
                for idx, turn in enumerate(cfg.script):
                    if turn.trigger_on_commit and idx not in fired_indices:
                        fired_indices.add(idx)
                        ok = await self._emit_turn(
                            ws,
                            turn,
                            audio_processed=audio_processed,
                            include_words=include_words,
                            speaker_flip=cfg.flip_speaker_on_commit,
                        )
                        if not ok:
                            return audio_started, frame_count, 0, False
            await ws.send(
                json.dumps(
                    {
                        "event_id": self._next_event_id(),
                        "type": "input_audio_buffer.committed",
                    }
                )
            )
            return audio_started, frame_count, 0, True

        if msg_type in {"input_audio_buffer.clear", "response.cancel"}:
            await ws.send(
                json.dumps(
                    {
                        "event_id": self._next_event_id(),
                        "type": "input_audio_buffer.cleared",
                    }
                )
            )
            return audio_started, frame_count, 0, True

        await self._send_error(
            ws,
            err_type="invalid_request_error",
            code="invalid_argument",
            message=f"unsupported realtime event type: {msg_type}",
        )
        return audio_started, frame_count, audio_bytes, True


_MOCK_PEECEE_SCRIPT = """#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

LANG_RE = re.compile(r"^[A-Za-z0-9-]+$")


def _usage() -> int:
    sys.stderr.write("Usage: peecee-speech [--serve [LANG|auto] | RECORDING.wav [LANG|auto]]\\n")
    return 2


def _serve(args: list[str]) -> int:
    if len(args) > 1:
        return _usage()
    lang = args[0] if args else "en-US"
    if not LANG_RE.match(lang):
        return _usage()

    forced_exit = os.environ.get("MOCK_PEECEE_EXIT_CODE", "").strip()
    if forced_exit:
        code = int(forced_exit)
        if code != 0:
            sys.stderr.write(f"peecee-speech: mock failure exit {code}\\n")
            return code

    delay_s = float(os.environ.get("MOCK_PEECEE_READY_DELAY_S", "0") or "0")
    if delay_s > 0:
        time.sleep(delay_s)

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; "
            "signal.signal(signal.SIGTERM, lambda *_: exit(0)); time.sleep(120)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_file = os.environ.get("MOCK_PEECEE_PID_FILE", "").strip()
    if pid_file:
        Path(pid_file).write_text(
            json.dumps({"parent_pid": os.getpid(), "child_pid": child.pid}),
            encoding="utf-8",
        )

    stopping = False

    def _on_term(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                child.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    sys.stderr.write(f"peecee-speech mock --serve ready ({lang})\\n")
    sys.stderr.flush()
    while not stopping:
        if child.poll() is not None:
            return int(child.returncode or 0)
        time.sleep(0.05)
    return 0


def _transcribe_wav(args: list[str]) -> int:
    positional = [a for a in args if a not in {"--stream", "--json"}]
    if not (1 <= len(positional) <= 2):
        return _usage()
    wav_path = Path(positional[0])
    lang = positional[1] if len(positional) == 2 else "auto"
    if not LANG_RE.match(lang):
        return _usage()
    if not wav_path.is_file() or not os.access(wav_path, os.R_OK):
        sys.stderr.write(f"peecee-speech: unreadable file: {wav_path}\\n")
        return 2

    forced_exit = os.environ.get("MOCK_PEECEE_EXIT_CODE", "").strip()
    if forced_exit:
        code = int(forced_exit)
        if code != 0:
            sys.stderr.write(f"peecee-speech: mock failure exit {code}\\n")
            return code

    json_path = os.environ.get("MOCK_PEECEE_JSON_PATH", "").strip()
    if json_path:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    elif os.environ.get("MOCK_PEECEE_JSON", "").strip():
        payload = json.loads(os.environ["MOCK_PEECEE_JSON"])
    else:
        duration = 1.6
        try:
            with wave.open(str(wav_path), "rb") as wf:
                duration = round(wf.getnframes() / float(wf.getframerate() or 16000), 3)
        except wave.Error:
            pass
        payload = {
            "file": str(wav_path),
            "text": "Someone should own the rollback plan",
            "confidence": 1.0,
            "duration": duration,
            "languages": ["en-US" if lang == "auto" else lang],
            "words": [
                {"word": "Someone", "start": 0.16, "end": 0.48, "confidence": 1.0, "speaker": 1},
                {"word": "should", "start": 0.48, "end": 0.72, "confidence": 1.0, "speaker": 1},
                {"word": "own", "start": 0.72, "end": 0.90, "confidence": 1.0, "speaker": 1},
                {"word": "the", "start": 0.90, "end": 1.02, "confidence": 1.0, "speaker": 1},
                {"word": "rollback", "start": 1.02, "end": 1.32, "confidence": 1.0, "speaker": 1},
                {"word": "plan", "start": 1.32, "end": 1.56, "confidence": 1.0, "speaker": 1},
            ],
        }
    sys.stderr.write("leased_ssh: connected to peecee (mock)\\n")
    sys.stdout.write(json.dumps(payload) + "\\n")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        return _usage()
    if argv[0] == "--serve":
        return _serve(argv[1:])
    if argv[0].startswith("-"):
        return _usage()
    return _transcribe_wav(argv)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def write_mock_peecee_speech(dest_dir: Path, filename: str = "peecee-speech") -> Path:
    """Write an executable mock peecee-speech script in dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    script_path = dest_dir / filename
    script_path.write_text(_MOCK_PEECEE_SCRIPT, encoding="utf-8")
    os.chmod(script_path, 0o755)
    return script_path
