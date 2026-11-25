"""Self-check tests for Milestone E1 opaque-box E2E infrastructure."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import yaml
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from tests.e2e.helpers.cli_runner import (
    ManagedProcess,
    build_coach_test_env,
    extract_ui_url_and_token,
    resolve_coach_cmd,
    resolve_earshot_cmd,
    run_cli,
    run_earshot_beam,
    run_earshot_direction,
)
from tests.e2e.helpers.db_assertions import (
    assert_egress_records_valid,
    assert_local_only_no_egress,
    assert_model_cache_has_entries,
    assert_session_text_deleted,
    assert_six_conservation_invariants,
    assert_wal_mode,
)
from tests.e2e.helpers.fixtures import (
    REASON_CODES,
    build_all_appendix_b_fixtures,
    build_appendix_b_fixture,
    build_boundary_event,
    build_brief_update_event,
    build_control_event,
    build_decision_context_row,
    build_deepgram_results,
    build_deepgram_utterance_end,
    build_label_record,
    build_ledger_event,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    build_speaker_update_event,
    build_template_dict,
    chunk_pcm16_frames,
    deep_merge_dict,
    generate_multichannel_pcm,
    generate_pcm16_silence,
    generate_pcm16_sine,
    read_jsonl,
    write_appendix_b_workspace,
    write_jsonl,
    write_policy_dict,
    write_wav_mono16,
)
from tests.e2e.helpers.mock_nemo_ws import (
    PCM16_FRAME_BYTES_160MS,
    MockNemoConfig,
    MockNemoServer,
    MockWord,
    ScriptedTurn,
)
from tests.e2e.helpers.mock_openai import MockCompletionSpec, MockOpenAIServer
from tests.e2e.helpers.ui_client import AnnotationClient, CardUIClient, assert_loopback_url


def test_mock_nemo_ws_handshake_frames_and_commit(mock_nemo_ws: MockNemoServer) -> None:
    mock_nemo_ws.set_config(
        MockNemoConfig(
            flip_speaker_on_commit={1: 2},
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    deltas=["Someone"],
                    transcript="Someone should own it",
                    words=[
                        MockWord("Someone", 0.10, 0.40, speaker=1),
                        MockWord("should", 0.40, 0.70, speaker=None),
                    ],
                ),
                ScriptedTurn(
                    trigger_on_commit=True,
                    transcript="Rollback plan",
                    words=[MockWord("Rollback", 0.80, 1.10, speaker=1)],
                ),
            ],
        )
    )

    health_resp = httpx.get(mock_nemo_ws.health_url, timeout=3.0)
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}
    bad_route = httpx.get(f"http://127.0.0.1:{mock_nemo_ws.port}/unknown", timeout=3.0)
    assert bad_route.status_code == 404

    frame = generate_pcm16_silence(duration_ms=160)
    assert len(frame) == PCM16_FRAME_BYTES_160MS

    async def _exercise() -> None:
        async with connect(mock_nemo_ws.url) as ws:
            created = json.loads(await ws.recv())
            assert created["type"] == "session.created"
            assert created["session"]["sample_rate"] == 16000

            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "sample_rate": 16000,
                            "language": "en-US",
                            "word_timestamps": True,
                            "speaker_diarization": True,
                        },
                    }
                )
            )
            updated = json.loads(await ws.recv())
            assert updated["type"] == "session.updated"

            await ws.send(frame)
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(frame).decode("ascii"),
                    }
                )
            )
            delta = json.loads(await ws.recv())
            assert delta["type"] == "conversation.item.input_audio_transcription.delta"
            assert delta["delta"] == "Someone"

            completed = json.loads(await ws.recv())
            assert completed["type"] == "conversation.item.input_audio_transcription.completed"
            assert completed["words"][0]["speaker"] == 1
            assert "speaker" not in completed["words"][1]

            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            commit_final = json.loads(await ws.recv())
            assert commit_final["words"][0]["speaker"] == 2
            committed = json.loads(await ws.recv())
            assert committed["type"] == "input_audio_buffer.committed"

            await ws.send(json.dumps({"type": "response.cancel"}))
            cleared = json.loads(await ws.recv())
            assert cleared["type"] == "input_audio_buffer.cleared"

    asyncio.run(_exercise())
    assert len(mock_nemo_ws.received_frames) == 2
    assert len(mock_nemo_ws.session_configs) == 1


def test_mock_nemo_ws_error_silence_and_drop(mock_nemo_ws: MockNemoServer) -> None:
    frame = generate_pcm16_silence(duration_ms=160)
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[ScriptedTurn(trigger_after_frames=2, drop_connection=True)],
        )
    )

    async def _exercise_error_and_drop() -> None:
        async with connect(mock_nemo_ws.url) as ws:
            _ = json.loads(await ws.recv())  # session.created
            await ws.send(frame)
            await ws.send(json.dumps({"type": "session.update", "session": {"sample_rate": 16000}}))
            err = json.loads(await ws.recv())
            assert err["type"] == "error"
            assert err["error"]["code"] == "invalid_argument"
            assert "cannot change" in err["error"]["message"]

            await ws.send(frame)
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=2.0)

    asyncio.run(_exercise_error_and_drop())


def test_mock_peecee_speech_exit75_stream_json_and_sigterm(
    mock_peecee_bin: Path,
    tmp_path: Path,
) -> None:
    # 1. Lease unavailable -> exit 75
    res75 = run_cli(
        [str(mock_peecee_bin), "--serve", "en-US"],
        env={"MOCK_PEECEE_EXIT_CODE": "75"},
    )
    assert res75.returncode == 75

    # 2. Invalid language -> exit 2
    res2 = run_cli([str(mock_peecee_bin), "--serve", "en/US"])
    assert res2.returncode == 2

    # 3. SIGTERM propagates to simulated ssh child (no orphan)
    pid_file = tmp_path / "peecee_pids.json"
    env = dict(os.environ)
    env["MOCK_PEECEE_PID_FILE"] = str(pid_file)
    proc = subprocess.Popen(
        [str(mock_peecee_bin), "--serve", "en-US"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.is_file()
        pids = json.loads(pid_file.read_text(encoding="utf-8"))
        child_pid = int(pids["child_pid"])
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5.0)
        time.sleep(0.1)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)

    # 4. Offline WAV transcription (strict args: RECORDING.wav [LANGUAGE|auto])
    wav_path = write_wav_mono16(tmp_path / "sample.wav", generate_pcm16_sine(duration_ms=800))
    res_bad_flags = run_cli([str(mock_peecee_bin), str(wav_path), "en-US", "--stream", "--json"])
    assert res_bad_flags.returncode == 2
    res_wav = run_cli([str(mock_peecee_bin), str(wav_path), "en-US"])
    assert res_wav.returncode == 0
    payload = res_wav.json()
    assert payload["file"] == str(wav_path)
    assert payload["duration"] == pytest.approx(0.8, abs=0.05)
    assert payload["languages"] == ["en-US"]
    assert len(payload["words"]) > 0
    assert "leased_ssh" in res_wav.stderr


def test_mock_openai_responses_and_prompt_inspection(mock_openai: MockOpenAIServer) -> None:
    models_resp = httpx.get(f"{mock_openai.base_url}/models", timeout=3.0)
    assert models_resp.status_code == 200
    assert "mock-generator-v1" in [m["id"] for m in models_resp.json()["data"]]

    mock_openai.enqueue(
        MockCompletionSpec.valid_question(
            "Who will own the rollback plan?", ["segment-0001"], cached_tokens=64
        ),
        MockCompletionSpec.decline("already answered"),
        MockCompletionSpec.malformed_truncated(),
        MockCompletionSpec.ungrounded(),
        MockCompletionSpec.http_error(500, "boom"),
        MockCompletionSpec.timeout(delay_s=0.6),
    )
    mock_openai.add_matcher(
        lambda b: "readout_probe" in str(b.get("prompt", "")),
        MockCompletionSpec(
            content="CONFIRM_OWNER",
            top_logprobs=[{"CONFIRM_OWNER": -0.1, "NONE": -2.4}],
        ),
    )

    with httpx.Client(timeout=2.0) as client:
        r1 = client.post(
            f"{mock_openai.base_url}/chat/completions",
            json={
                "model": "mock-generator-v1",
                "messages": [
                    {"role": "system", "content": "[POLICY] max_words=20\n[BRIEF] cutover"},
                    {
                        "role": "user",
                        "content": (
                            '[TRANSCRIPT QUOTED] "Someone should own it."\n'
                            "[OPPORTUNITY] CONFIRM_OWNER"
                        ),
                    },
                ],
            },
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert json.loads(d1["choices"][0]["message"]["content"])["question"].endswith("?")
        assert d1["usage"]["prompt_tokens_details"]["cached_tokens"] == 64

        r2 = client.post(f"{mock_openai.base_url}/chat/completions", json={"messages": []})
        assert json.loads(r2.json()["choices"][0]["message"]["content"])["decline_reason"] == (
            "already answered"
        )

        r3 = client.post(f"{mock_openai.base_url}/chat/completions", json={"messages": []})
        assert r3.json()["choices"][0]["finish_reason"] == "length"

        r4 = client.post(f"{mock_openai.base_url}/chat/completions", json={"messages": []})
        assert "$42,000" in r4.json()["choices"][0]["message"]["content"]

        r5 = client.post(f"{mock_openai.base_url}/chat/completions", json={"messages": []})
        assert r5.status_code == 500

        with pytest.raises(httpx.ReadTimeout):
            client.post(
                f"{mock_openai.base_url}/chat/completions",
                json={"messages": []},
                timeout=0.15,
            )

        r_readout = client.post(
            f"{mock_openai.base_url}/completions",
            json={"prompt": "readout_probe anchor", "logprobs": 2},
        )
        assert r_readout.status_code == 200
        assert (
            r_readout.json()["choices"][0]["logprobs"]["top_logprobs"][0]["CONFIRM_OWNER"] == -0.1
        )

    blocks = mock_openai.requests[1].prompt_blocks()
    assert len(blocks) == 2
    assert "[POLICY]" in blocks[0] and "[TRANSCRIPT QUOTED]" in blocks[1]


def test_fixture_builders_contracts_templates_wav_and_appendix_b(
    e2e_workspace: dict[str, Path],
) -> None:
    assert len(REASON_CODES) == 47
    brief = build_session_brief()
    assert brief["schema_version"] == 1
    assert brief["retention"]["store_transcript"] is True

    events = [
        build_segment_event(),
        build_speaker_update_event(),
        build_boundary_event(),
        build_brief_update_event(),
        build_ledger_event(),
        build_control_event(),
    ]
    assert {ev["kind"] for ev in events} == {
        "segment",
        "speaker_update",
        "boundary",
        "brief_update",
        "ledger",
        "control",
    }

    dg = build_deepgram_results("Someone should own it.", is_final=True, speech_final=True)
    assert dg["type"] == "Results"
    assert dg["channel"]["alternatives"][0]["transcript"] == "Someone should own it."
    dg_end = build_deepgram_utterance_end(1.5)
    assert dg_end["type"] == "UtteranceEnd"

    lbl = build_label_record()
    assert lbl["sample_source"] == "random"
    with pytest.raises(ValueError, match="sample_source"):
        build_label_record(sample_source="invalid_source")

    policy = build_policy_dict()
    assert policy["generator"]["breaker"]["consecutive_failures"] == 3
    assert policy["categories"]["CONFIRM_OWNER"]["required_lookback_s"] == 300
    assert policy["categories"]["CONFIRM_TIME"]["required_lookback_s"] == 300
    assert policy["categories"]["SUCCESS_CRITERION"]["required_lookback_s"] == 120
    assert "brief_patch" in events[3] and "changes" not in events[3]
    assert isinstance(events[4]["source_sequence"], int)
    assert isinstance(events[5]["source_sequence"], int)
    for tpl_name in (
        "confirm_owner_v1",
        "confirm_time_v1",
        "success_criterion_v1",
        "clarify_scope_v1",
        "request_evidence_v1",
    ):
        tpl = build_template_dict(tpl_name)
        assert tpl["id"] == tpl_name
        assert tpl["max_words"] == 12
        assert isinstance(tpl["grounded"]["text"], str)
        assert isinstance(tpl["grounded"]["slots"], dict)
        assert isinstance(tpl["generic"]["text"], str)
        assert isinstance(tpl["generic"]["allowed_when"], dict)
        assert (e2e_workspace["templates_dir"] / f"{tpl_name}.yaml").is_file()

    pcm = generate_pcm16_sine(duration_ms=320)
    frames = chunk_pcm16_frames(pcm, frame_ms=160)
    assert len(frames) == 2
    assert all(len(f) == PCM16_FRAME_BYTES_160MS for f in frames)
    wav_path = write_wav_mono16(e2e_workspace["root"] / "sine.wav", pcm)
    with wave.open(str(wav_path), "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)

    all_b = build_all_appendix_b_fixtures()
    assert len(all_b) == 24
    for num in range(1, 25):
        fid = f"B{num:02d}"
        assert fid in all_b
        assert set(all_b[fid].expected_reason_codes).issubset(REASON_CODES)
        assert "displayed" not in all_b[fid].expected_terminal_states
    for fid in ("B14", "B20", "B24"):
        assert "displayed" in all_b[fid].notes.get("expected_observed_states", ())
    for ev in all_b["B21"].events:
        assert "gate_silent" not in ev
        assert ev.get("low_quality_gate") is True


def _create_synthetic_coach_db(db_path: Path, *, journal_mode: str = "WAL") -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(f"PRAGMA journal_mode={journal_mode};")
        conn.executescript(
            """
            CREATE TABLE raw_events (session_id TEXT, event_id TEXT, payload_json TEXT);
            CREATE TABLE events (session_id TEXT, event_id TEXT, kind TEXT, payload_json TEXT);
            CREATE TABLE decisions (session_id TEXT, candidate_id TEXT, payload_json TEXT);
            CREATE TABLE suggestions (session_id TEXT, suggestion_id TEXT, payload_json TEXT);
            CREATE TABLE feedback (session_id TEXT, label_id TEXT, payload_json TEXT);
            CREATE TABLE model_cache (
                serialized_sha256 TEXT, model_version TEXT, payload_json TEXT
            );
            CREATE TABLE egress_log (session_id TEXT, payload_json TEXT);
            """
        )
        seg = build_segment_event(
            session_id="s1",
            event_id="e1",
            segment_id="seg-1",
            text="Someone should own the rollback plan.",
        )
        led1 = build_ledger_event(
            session_id="s1",
            event_id="e2",
            suggestion_id="sug-1",
            transition="displayed",
            cause="boundary_after_settle",
            at_ms=5500,
        )
        led2 = build_ledger_event(
            session_id="s1",
            event_id="e3",
            suggestion_id="sug-1",
            transition="used",
            cause="user_action",
            at_ms=7000,
        )
        conn.execute("INSERT INTO raw_events VALUES (?, ?, ?)", ("s1", "e1", json.dumps(seg)))
        for ev in (seg, led1, led2):
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                ("s1", ev["event_id"], ev["kind"], json.dumps(ev)),
            )
        dec_payload = {
            "candidate_id": "cand-1",
            "anchor_refs": [{"segment_id": "seg-1", "revision": 1}],
            "raw_scores": {"CONFIRM_OWNER": 0.9},
            "outcome": "ROUTE",
            "reason_code": "above_t_display",
            "route": "generator",
            "usage": {"model": "mock-gen", "prompt_tokens": 100, "latency_ms": 40},
            "serialized_input": "Someone should own the rollback plan.",
        }
        conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?)",
            ("s1", "cand-1", json.dumps(dec_payload)),
        )
        sug_payload = {
            "suggestion_id": "sug-1",
            "candidate_id": "cand-1",
            "category": "CONFIRM_OWNER",
            "route": "generator",
            "state": "used",
            "terminal_state": "used",
            "reason_code": "user_action",
            "question_text": "Who will own the rollback plan?",
            "evidence_refs": [{"segment_id": "seg-1", "revision": 1}],
            "usage": {"model": "mock-gen", "prompt_tokens": 90, "latency_ms": 35},
        }
        conn.execute(
            "INSERT INTO suggestions VALUES (?, ?, ?)",
            ("s1", "sug-1", json.dumps(sug_payload)),
        )
        conn.execute(
            "INSERT INTO feedback VALUES (?, ?, ?)",
            (
                "s1",
                "lbl-1",
                json.dumps(
                    {
                        "label_id": "lbl-1",
                        "evidence_refs": [{"segment_id": "seg-1", "revision": 1}],
                        "notes": "Sensitive note",
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO model_cache VALUES (?, ?, ?)",
            ("abc123", "rules_v0", json.dumps({"score": 0.9})),
        )
        conn.execute(
            "INSERT INTO egress_log VALUES (?, ?)",
            (
                "s1",
                json.dumps(
                    {
                        "segment_ids": ["seg-1"],
                        "provider": "mock",
                        "model": "mock-gen",
                        "timestamp_ms": 5000,
                        "lane": "proactive",
                    }
                ),
            ),
        )
        conn.commit()


def test_db_assertions_wal_invariants_and_deletion(tmp_path: Path) -> None:
    wal_db = tmp_path / "coach_wal.db"
    _create_synthetic_coach_db(wal_db, journal_mode="WAL")
    assert assert_wal_mode(wal_db) == "wal"

    del_db = tmp_path / "coach_delete.db"
    _create_synthetic_coach_db(del_db, journal_mode="DELETE")
    with pytest.raises(AssertionError, match="wal"):
        assert_wal_mode(del_db)

    summary = assert_six_conservation_invariants(wal_db, "s1")
    assert summary["candidates"] == 1
    assert summary["displayed_suggestions"] == 1
    assert_egress_records_valid(wal_db, "s1", expected_segment_ids={"seg-1"})
    assert_model_cache_has_entries(wal_db, min_count=1)

    with pytest.raises(AssertionError, match="local_only"):
        assert_local_only_no_egress(wal_db, "s1")

    # Negative invariant tests: Inv 3 unknown code & Inv 6 invalid revision
    bad_inv3 = tmp_path / "bad_inv3.db"
    _create_synthetic_coach_db(bad_inv3, journal_mode="WAL")
    with sqlite3.connect(str(bad_inv3)) as conn:
        conn.execute(
            "UPDATE suggestions SET payload_json = "
            "json_set(payload_json, '$.reason_code', 'made_up_code')"
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 3"):
        assert_six_conservation_invariants(bad_inv3, "s1")

    bad_inv6 = tmp_path / "bad_inv6.db"
    _create_synthetic_coach_db(bad_inv6, journal_mode="WAL")
    with sqlite3.connect(str(bad_inv6)) as conn:
        bad_refs = json.dumps([{"segment_id": "seg-1", "revision": 99}])
        conn.execute(
            "UPDATE suggestions SET payload_json = "
            "json_set(payload_json, '$.evidence_refs', json(?))",
            (bad_refs,),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 6"):
        assert_six_conservation_invariants(bad_inv6, "s1")

    # §14.2 deletion: wipe raw_events, text, serialized_input, question_text, notes; keep metadata
    with sqlite3.connect(str(wal_db)) as conn:
        conn.execute("DELETE FROM raw_events WHERE session_id = 's1'")
        conn.execute("UPDATE events SET payload_json = json_set(payload_json, '$.text', NULL)")
        conn.execute(
            "UPDATE decisions SET payload_json = json_set(payload_json, '$.serialized_input', NULL)"
        )
        conn.execute(
            "UPDATE suggestions SET payload_json = json_set(payload_json, '$.question_text', NULL)"
        )
        conn.execute("UPDATE feedback SET payload_json = json_set(payload_json, '$.notes', NULL)")
        conn.commit()

    assert_session_text_deleted(
        wal_db,
        "s1",
        forbidden_substrings=("rollback plan", "Sensitive note"),
        require_decisions_preserved=True,
        require_suggestions_preserved=True,
    )


@contextmanager
def _ephemeral_ui_stub(token: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    import threading

    received_actions: list[dict[str, Any]] = []
    ready = threading.Event()
    port_holder = [0]
    loop_holder: list[asyncio.AbstractEventLoop] = []
    stop_holder: list[asyncio.Event] = []

    def _check_auth(req: Request) -> bool:
        auth = req.headers.get("Authorization", "")
        if auth == f"Bearer {token}":
            return True
        qs = parse_qs(urlsplit(req.path).query)
        return qs.get("token", [None])[0] == token

    def _process_request(_conn: ServerConnection, req: Request) -> Response | None:
        path = urlsplit(req.path).path
        if not _check_auth(req):
            return Response(401, "Unauthorized", Headers(), b"Unauthorized")
        if path == "/":
            html = (
                b"<!doctype html><html><body><div id='card'></div>"
                b"<script>document.getElementById('card').textContent = 'ready';</script>"
                b"</body></html>"
            )
            return Response(200, "OK", Headers([("Content-Type", "text/html")]), html)
        if path == "/ws":
            return None
        return Response(404, "Not Found", Headers(), b"Not Found")

    async def _ws_handler(ws: ServerConnection) -> None:
        await ws.send(json.dumps({"type": "health", "health": "ok"}))
        await ws.send(
            json.dumps(
                {
                    "type": "card",
                    "suggestion_id": "sug-1",
                    "category": "CONFIRM_OWNER",
                    "question": "Who will own the rollback plan?",
                }
            )
        )
        try:
            async for raw in ws:
                msg = json.loads(raw)
                received_actions.append(msg)
                await ws.send(
                    json.dumps({"type": "ack", **{k: v for k, v in msg.items() if k != "type"}})
                )
        except ConnectionClosed:
            return

    async def _serve() -> None:
        stop_ev = asyncio.Event()
        stop_holder.append(stop_ev)
        async with serve(_ws_handler, "127.0.0.1", 0, process_request=_process_request) as srv:
            port_holder[0] = int(next(iter(srv.sockets)).getsockname()[1])
            ready.set()
            await stop_ev.wait()

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder.append(loop)
        loop.run_until_complete(_serve())
        loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    assert ready.wait(timeout=5.0)
    try:
        yield f"http://127.0.0.1:{port_holder[0]}", received_actions
    finally:
        if loop_holder and stop_holder:
            loop_holder[0].call_soon_threadsafe(stop_holder[0].set)
        thread.join(timeout=5.0)


@contextmanager
def _ephemeral_annotation_stub() -> Iterator[str]:
    import threading

    state: dict[str, Any] = {"step2_unlocked": set(), "step1_data": {}, "step2_data": {}}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, code: int, payload: Any) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/api/sessions/s1/prefix":
                at_ms = int(qs.get("at_ms", ["0"])[0])
                scroll_to = qs.get("scroll_to_ms", [None])[0]
                if scroll_to is not None and int(scroll_to) > at_ms:
                    self._send(409, {"error": "future scroll forbidden"})
                    return
                self._send(200, {"events": [{"event_id": "e1", "received_ms": min(at_ms, 4000)}]})
                return
            if parsed.path == "/api/candidates/cand-1/step1":
                self._send(200, {"candidate_id": "cand-1", "anchor": "Someone should own it."})
                return
            if parsed.path == "/api/retest/item-1":
                self._send(200, {"item_id": "item-1", "anchor": "Someone should own it."})
                return
            if parsed.path == "/api/retest/kappa":
                self._send(200, {"worthwhile_kappa": 0.72, "per_category": {"CONFIRM_OWNER": 0.8}})
                return
            if parsed.path == "/api/review-queue":
                self._send(200, {"items": [{"candidate_id": "cand-unlabeled", "origin": "replay"}]})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = json.loads(self.rfile.read(length).decode()) if length else {}
            if self.path == "/api/sessions/s1/marks":
                self._send(201, {"ok": True, **body})
                return
            if self.path == "/api/candidates/cand-1/step1":
                if "cand-1" in state["step2_unlocked"]:
                    self._send(409, {"error": "step1 locked after step2 unlock"})
                    return
                state["step1_data"]["cand-1"] = body
                self._send(200, {"locked": False})
                return
            if self.path == "/api/candidates/cand-1/step2/unlock":
                state["step2_unlocked"].add("cand-1")
                self._send(200, {"continuation_events": [{"event_id": "e2", "received_ms": 6000}]})
                return
            if self.path == "/api/candidates/cand-1/step2":
                state["step2_data"]["cand-1"] = body
                self._send(200, {"saved": True})
                return
            if self.path == "/api/retest/item-1":
                self._send(200, {"saved": True, "item_id": "item-1", **body})
                return
            self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5.0)


def test_ui_client_security_xss_ws_and_annotation_lock() -> None:
    with pytest.raises(AssertionError, match="loopback"):
        assert_loopback_url("http://0.0.0.0:8765")

    token = "secret-session-token-123"
    with _ephemeral_ui_stub(token) as (base_url, actions):
        client = CardUIClient(base_url, token)
        statuses = client.assert_auth_enforced()
        assert statuses["unauthenticated"] == 401
        assert statuses["ok_bearer"] == 200

        html = httpx.get(f"{base_url}/", headers={"Authorization": f"Bearer {token}"}).text
        client.inspect_html_xss_safety(html, forbidden_substrings=("<script>alert",))
        for bad_html, pat in (
            ("<script>el.innerHTML = x;</script>", "innerHTML"),
            ("<script>el['innerHTML'] = x;</script>", "innerHTML"),
            ("<script>el['inner' + 'HTML'] = x;</script>", "innerHTML"),
            ("<script>document.writeln('x');</script>", "document.write"),
            ("<script>new Notification('hi');</script>", "Notification"),
            ("<script>Notification.requestPermission();</script>", "Notification"),
            ("<script>Reflect.construct(Notification, ['hi']);</script>", "Notification"),
            ("<script>eval('1+1');</script>", "eval"),
            ("<script>new Function('return 1')();</script>", "Function"),
            ("<body><div><img src=x onerror=alert(1)></div></body>", "Unsafe HTML"),
            ("<body><div><svg onload=alert(1)></svg></div></body>", "Unsafe HTML"),
            ("<body><details open ontoggle=alert(1)></details></body>", "Unsafe HTML"),
            ("<body><a href='javascript:alert(1)'>x</a></body>", "Unsafe HTML"),
        ):
            with pytest.raises(AssertionError, match=pat):
                client.inspect_html_xss_safety(bad_html)
        for bad_card in (
            {"question": "<details open ontoggle=alert(1)>?"},
            {"question": "<a href='javascript:alert(1)'>click</a>?"},
            {"question": "Who owns onclick=alert(1)?"},
        ):
            with pytest.raises(AssertionError, match="markup/script"):
                client.assert_card_plain_text(bad_card)

        async def _exercise_ws() -> None:
            async with client.ws_session() as ws:
                health = await ws.wait_for_health("ok")
                assert health["health"] == "ok"
                card = await ws.wait_for_card()
                client.assert_card_plain_text(
                    card, raw_adversarial_text="<script>alert(1)</script>"
                )
                await ws.send_use("sug-1")
                await ws.send_dismiss("sug-1", key="2")
                await ws.send_pin("sug-1")
                await ws.send_manual_request()
                await ws.send_mode("Quiet")
                await ws.send_off_record(True)
                ack = await ws.wait_for_message(lambda m: m.get("type") == "ack")
                assert ack["action"] == "use"
                for _ in range(5):
                    next_ack = await ws.recv_json()
                    assert next_ack["type"] == "ack"

        asyncio.run(_exercise_ws())
        assert [a["action"] for a in actions] == [
            "use",
            "dismiss",
            "pin",
            "manual",
            "mode",
            "off_record",
        ]
        assert actions[1]["feedback"] == "already_answered"

    with _ephemeral_annotation_stub() as ann_url:
        ann = AnnotationClient(ann_url)
        prefix = ann.get_pass1_prefix("s1", 5000)
        assert len(prefix["events"]) == 1
        ann.assert_pass1_no_future_scroll("s1", at_ms=5000, requested_ms=10000)
        mark = ann.post_pass1_mark("s1", at_ms=5000, state_version=12, question="Who owns it?")
        assert mark["ok"] is True

        s1 = ann.get_step1_view("cand-1")
        assert s1["candidate_id"] == "cand-1"
        ann.submit_step1(
            "cand-1",
            worthwhile=True,
            categories=["CONFIRM_OWNER"],
            example_questions=["Who will own it?"],
        )
        step2_view = ann.unlock_step2("cand-1")
        assert len(step2_view["continuation_events"]) == 1
        lock_status = ann.assert_step1_locked("cand-1", {"worthwhile": False})
        assert lock_status == 409
        saved = ann.submit_step2("cand-1", close_ms=6000, closed_by="answered", ideal_ms=5200)
        assert saved["saved"] is True

        retest = ann.get_retest_item("item-1")
        assert retest["item_id"] == "item-1"
        submitted_retest = ann.submit_retest(
            "item-1", worthwhile=False, categories=(), none_reason="no_gap"
        )
        assert submitted_retest["saved"] is True
        queue = ann.get_review_queue()
        assert queue == [{"candidate_id": "cand-unlabeled", "origin": "replay"}]
        kappa = ann.get_kappa_report()
        assert kappa["worthwhile_kappa"] == 0.72


def test_cli_runner_earshot_beam_and_direction(repo_root: Path, tmp_path: Path) -> None:
    # Verify ManagedProcess clean SIGTERM shutdown on a generic python subprocess
    with ManagedProcess(
        [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(30)"],
        ready_regex="READY",
        startup_timeout=5.0,
    ) as proc:
        assert proc.pid is not None
        assert "READY" in proc.stdout_text

    # Verify earshot direction & beam against arrays/respeaker-core-v2.toml if binary exists
    try:
        resolve_earshot_cmd(repo_root)
    except FileNotFoundError:
        pytest.skip("earshot binary not built in target/ or PATH")

    array_toml = repo_root / "arrays" / "respeaker-core-v2.toml"
    raw_pcm = tmp_path / "array_8ch_32b.raw"
    raw_pcm.write_bytes(generate_multichannel_pcm(channels=8, duration_ms=1000, width_bits=32))

    dir_res = run_earshot_direction(array_toml, raw_pcm, every=0.5, repo_root=repo_root)
    assert dir_res.returncode == 0, dir_res.stderr
    lines = dir_res.jsonl()
    assert len(lines) >= 1
    assert {"t", "azimuth", "contrast", "level_db"}.issubset(lines[0].keys())

    out_wav = tmp_path / "beamed.wav"
    beam_res = run_earshot_beam(
        array_toml, raw_pcm, out_wav, steer="0", verbose=True, repo_root=repo_root
    )
    assert beam_res.returncode == 0, beam_res.stderr
    assert out_wav.is_file()
    assert "from 0.00 s the beam looks at 0 degrees" in beam_res.stderr
    with wave.open(str(out_wav), "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)


def _create_m1_style_db(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(
            """
            CREATE TABLE raw_events (session_id TEXT, run_id TEXT, payload_json TEXT);
            CREATE TABLE events (
                session_id TEXT, run_id TEXT, event_id TEXT, kind TEXT, segment_id TEXT,
                source_revision INTEGER, canonical_revision INTEGER, speaker_id TEXT,
                received_ms INTEGER, source_sequence INTEGER, stability TEXT,
                text TEXT, payload_json TEXT
            );
            CREATE TABLE decisions (
                session_id TEXT, run_id TEXT, candidate_id TEXT, anchor_refs_json TEXT,
                raw_scores_json TEXT, calibrated_scores_json TEXT, outcome TEXT, route TEXT,
                reason_codes_json TEXT, terminal_state TEXT, terminal_reason TEXT,
                serialized_text TEXT
            );
            CREATE TABLE suggestions (
                session_id TEXT, run_id TEXT, suggestion_id TEXT, candidate_id TEXT,
                route TEXT, current_state TEXT, terminal_reason TEXT, transitions_json TEXT,
                evidence_refs_json TEXT, text TEXT, grounded_slots TEXT
            );
            CREATE TABLE feedback (
                session_id TEXT, run_id TEXT, label_id TEXT, label_json TEXT
            );
            CREATE TABLE model_cache (
                serialized_sha256 TEXT, model_version TEXT, output_json TEXT
            );
            CREATE TABLE egress_log (
                egress_id TEXT, session_id TEXT, run_id TEXT, candidate_id TEXT,
                lane TEXT, provider TEXT, model_id TEXT, segment_ids_json TEXT,
                sent_at_ms INTEGER, usage_json TEXT
            );
            """
        )
        for seg_id, src_rev in (("seg-1", 3), ("seg-2", 1), ("seg-3", 1)):
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "s1",
                    "r1",
                    f"ev-{seg_id}",
                    "segment",
                    seg_id,
                    src_rev,
                    1,
                    "participant-b",
                    1000,
                    1,
                    "committed",
                    None,
                    None,
                ),
            )
        conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "r1",
                "cand-drop",
                json.dumps([{"segment_id": "seg-2", "revision": 1}]),
                None,
                None,
                None,
                None,
                "[]",
                "dropped",
                "too_short",
                None,
            ),
        )
        conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "r1",
                "cand-ret",
                json.dumps([{"segment_id": "seg-3", "revision": 1}]),
                None,
                None,
                None,
                None,
                "[]",
                "retired",
                "expired_unevaluated",
                None,
            ),
        )
        conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "r1",
                "cand-1",
                json.dumps([{"segment_id": "seg-1", "revision": 1}]),
                json.dumps({"opportunity": 0.9}),
                json.dumps({"opportunity": 0.88}),
                "ROUTE",
                "generator",
                json.dumps(["above_t_display"]),
                "routed",
                None,
                None,
            ),
        )
        transitions = [
            {"event_id": "l1", "transition": "displayed", "cause": "settle_timeout", "at_ms": 2000},
            {"event_id": "l2", "transition": "used", "cause": "user_action", "at_ms": 3000},
            {"event_id": "l3", "transition": "resolved", "cause": "resolution_cue", "at_ms": 4000},
        ]
        conn.execute(
            "INSERT INTO suggestions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "r1",
                "sug-1",
                "cand-1",
                "generator",
                "resolved",
                "resolution_cue",
                json.dumps(transitions),
                json.dumps([{"segment_id": "seg-1", "revision": 1}]),
                None,
                None,
            ),
        )
        conn.execute(
            "INSERT INTO egress_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "eg-1",
                "s1",
                "r1",
                "cand-1",
                "proactive",
                "mock",
                "mock-gen",
                json.dumps(["seg-1"]),
                1500,
                json.dumps({"prompt_tokens": 50}),
            ),
        )
        conn.commit()


def test_db_assertions_adversarial_false_negatives(tmp_path: Path) -> None:
    # 1. Empty DB with 0 tables must fail
    empty_db = tmp_path / "empty.db"
    with sqlite3.connect(str(empty_db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
    with pytest.raises(AssertionError, match="Missing required SQLite tables"):
        assert_six_conservation_invariants(empty_db, "s1")

    # 2. M1-style SQLiteStore schema passes (dropped/retired, canonical_revision, used->resolved)
    m1_db = tmp_path / "m1.db"
    _create_m1_style_db(m1_db)
    summary = assert_six_conservation_invariants(m1_db, "s1", run_id="r1")
    assert summary["candidates"] == 3
    assert_egress_records_valid(m1_db, "s1", run_id="r1", expected_segment_ids={"seg-1"})

    # 3. Inv 1: duplicate anchor across two candidates
    dup_anchor_db = tmp_path / "dup_anchor.db"
    _create_m1_style_db(dup_anchor_db)
    with sqlite3.connect(str(dup_anchor_db)) as conn:
        conn.execute(
            "UPDATE decisions SET anchor_refs_json = ? WHERE candidate_id = 'cand-drop'",
            (json.dumps([{"segment_id": "seg-1", "revision": 1}]),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 1"):
        assert_six_conservation_invariants(dup_anchor_db, "s1", run_id="r1")

    # 4. Inv 1: dropped candidate illegally holding raw_scores
    drop_scores_db = tmp_path / "drop_scores.db"
    _create_m1_style_db(drop_scores_db)
    with sqlite3.connect(str(drop_scores_db)) as conn:
        conn.execute(
            "UPDATE decisions SET raw_scores_json = ? WHERE candidate_id = 'cand-drop'",
            (json.dumps({"opportunity": 0.5}),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 1"):
        assert_six_conservation_invariants(drop_scores_db, "s1", run_id="r1")

    # 5. Inv 1: null/empty candidate_id
    null_cid_db = tmp_path / "null_cid.db"
    _create_m1_style_db(null_cid_db)
    with sqlite3.connect(str(null_cid_db)) as conn:
        conn.execute("UPDATE decisions SET candidate_id = '' WHERE candidate_id = 'cand-drop'")
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 1"):
        assert_six_conservation_invariants(null_cid_db, "s1", run_id="r1")

    # 6. Inv 3: non-terminal code on terminal candidate & orphan suggestion
    non_term_db = tmp_path / "non_term.db"
    _create_m1_style_db(non_term_db)
    with sqlite3.connect(str(non_term_db)) as conn:
        conn.execute(
            "UPDATE suggestions SET terminal_reason = 'settling' WHERE suggestion_id = 'sug-1'"
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 3"):
        assert_six_conservation_invariants(non_term_db, "s1", run_id="r1")

    orphan_db = tmp_path / "orphan.db"
    _create_m1_style_db(orphan_db)
    with sqlite3.connect(str(orphan_db)) as conn:
        conn.execute(
            "INSERT INTO suggestions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "r1",
                "sug-orphan",
                "cand-missing",
                "template",
                "unrendered",
                "bogus_code",
                "[]",
                "[]",
                None,
                None,
            ),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 3"):
        assert_six_conservation_invariants(orphan_db, "s1", run_id="r1")

    # 7. Inv 4: double-closing (used + dismissed) & unclosed transitions_json
    double_close_db = tmp_path / "double_close.db"
    _create_m1_style_db(double_close_db)
    with sqlite3.connect(str(double_close_db)) as conn:
        bad_trans = [
            {"event_id": "l1", "transition": "displayed", "cause": "settle_timeout", "at_ms": 2000},
            {"event_id": "l2", "transition": "used", "cause": "user_action", "at_ms": 3000},
            {"event_id": "l3", "transition": "dismissed", "cause": "user_action", "at_ms": 4000},
        ]
        conn.execute(
            "UPDATE suggestions SET transitions_json = ? WHERE suggestion_id = 'sug-1'",
            (json.dumps(bad_trans),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 4"):
        assert_six_conservation_invariants(double_close_db, "s1", run_id="r1")

    unclosed_db = tmp_path / "unclosed.db"
    _create_m1_style_db(unclosed_db)
    with sqlite3.connect(str(unclosed_db)) as conn:
        conn.execute(
            "UPDATE suggestions SET current_state = 'displayed', terminal_reason = NULL, "
            "transitions_json = ? WHERE suggestion_id = 'sug-1'",
            (json.dumps([{"event_id": "l1", "transition": "displayed", "at_ms": 2000}]),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 4|Inv 3"):
        assert_six_conservation_invariants(unclosed_db, "s1", run_id="r1")

    # 8. Inv 5: unrelated egress_log row must not mask missing usage on another generator candidate
    inv5_db = tmp_path / "inv5.db"
    _create_m1_style_db(inv5_db)
    with sqlite3.connect(str(inv5_db)) as conn:
        conn.execute(
            "UPDATE decisions SET outcome = 'ABSTAIN', route = 'generator', "
            "raw_scores_json = ?, terminal_state = 'ABSTAIN', terminal_reason = 'score_in_band' "
            "WHERE candidate_id = 'cand-ret'",
            (json.dumps({"opportunity": 0.4}),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 5"):
        assert_six_conservation_invariants(inv5_db, "s1", run_id="r1")

    # 9. Inv 6: source_revision=3 + canonical_revision=1 rejects rev 3; checks events.evidence_refs
    inv6_canon_db = tmp_path / "inv6_canon.db"
    _create_m1_style_db(inv6_canon_db)
    with sqlite3.connect(str(inv6_canon_db)) as conn:
        conn.execute(
            "UPDATE suggestions SET evidence_refs_json = ? WHERE suggestion_id = 'sug-1'",
            (json.dumps([{"segment_id": "seg-1", "revision": 3}]),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 6"):
        assert_six_conservation_invariants(inv6_canon_db, "s1", run_id="r1")

    inv6_ledger_db = tmp_path / "inv6_ledger.db"
    _create_m1_style_db(inv6_ledger_db)
    with sqlite3.connect(str(inv6_ledger_db)) as conn:
        bad_ledger = build_ledger_event(
            session_id="s1",
            event_id="l2",
            suggestion_id="sug-1",
            transition="used",
            cause="user_action",
            at_ms=3000,
            evidence_refs=[{"segment_id": "seg-bogus", "revision": 42}],
            source_sequence=9,
        )
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "r1",
                "l2",
                "ledger",
                None,
                None,
                None,
                None,
                3000,
                9,
                None,
                None,
                json.dumps(bad_ledger),
            ),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 6"):
        assert_six_conservation_invariants(inv6_ledger_db, "s1", run_id="r1")

    # 10. §14.2: leaked text in model_cache (no session_id) and flat grounded_slots
    cache_leak_db = tmp_path / "cache_leak.db"
    _create_m1_style_db(cache_leak_db)
    with sqlite3.connect(str(cache_leak_db)) as conn:
        conn.execute(
            "INSERT INTO model_cache VALUES (?, ?, ?)",
            ("sha-1", "gen-v1", json.dumps({"question": "Leaked secret phrase?"})),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="model_cache"):
        assert_session_text_deleted(
            cache_leak_db, "s1", forbidden_substrings=("Leaked secret phrase",)
        )

    slots_leak_db = tmp_path / "slots_leak.db"
    _create_m1_style_db(slots_leak_db)
    with sqlite3.connect(str(slots_leak_db)) as conn:
        conn.execute(
            "UPDATE suggestions SET grounded_slots = ? WHERE suggestion_id = 'sug-1'",
            (
                json.dumps(
                    {"action": {"text": "rollback plan", "segment_id": "seg-1", "revision": 1}}
                ),
            ),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="grounded_slots"):
        assert_session_text_deleted(slots_leak_db, "s1")


def test_passive_ws_rejection_and_mock_edge_cases(
    mock_nemo_ws: MockNemoServer,
    mock_openai: MockOpenAIServer,
    tmp_path: Path,
) -> None:
    import threading

    # 1. Passive unauthenticated WS (accepts handshake & waits without sending) MUST fail auth check
    token = "tok-passive-check"
    ready = threading.Event()
    port_holder = [0]
    loop_holder: list[asyncio.AbstractEventLoop] = []
    stop_holder: list[asyncio.Event] = []

    def _process_req(_conn: ServerConnection, req: Request) -> Response | None:
        path = urlsplit(req.path).path
        if path == "/":
            auth = req.headers.get("Authorization", "")
            qs = parse_qs(urlsplit(req.path).query)
            if auth == f"Bearer {token}" or qs.get("token", [None])[0] == token:
                return Response(
                    200, "OK", Headers([("Content-Type", "text/html")]), b"<html></html>"
                )
            return Response(401, "Unauthorized", Headers(), b"Unauthorized")
        if path == "/ws":
            return None  # Buggy: forgets auth check on /ws!
        return Response(404, "Not Found", Headers(), b"Not Found")

    async def _passive_ws(ws: ServerConnection) -> None:
        try:
            async for _ in ws:
                pass
        except ConnectionClosed:
            return

    async def _serve_passive() -> None:
        stop_ev = asyncio.Event()
        stop_holder.append(stop_ev)
        async with serve(_passive_ws, "127.0.0.1", 0, process_request=_process_req) as srv:
            port_holder[0] = int(next(iter(srv.sockets)).getsockname()[1])
            ready.set()
            await stop_ev.wait()

    def _run_passive() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder.append(loop)
        loop.run_until_complete(_serve_passive())
        loop.close()

    t = threading.Thread(target=_run_passive, daemon=True)
    t.start()
    assert ready.wait(timeout=5.0)
    try:
        client = CardUIClient(f"http://127.0.0.1:{port_holder[0]}", token)
        with pytest.raises(AssertionError, match="Expected WS rejection"):
            client.assert_auth_enforced()
    finally:
        if loop_holder and stop_holder:
            loop_holder[0].call_soon_threadsafe(stop_holder[0].set)
        t.join(timeout=5.0)

    # 2. MockNemoServer emits deltas before drop_connection closes
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=1,
                    deltas=["partial-before-drop"],
                    drop_connection=True,
                )
            ]
        )
    )
    frame = generate_pcm16_silence(duration_ms=160)

    async def _check_deltas_before_drop() -> None:
        async with connect(mock_nemo_ws.url) as ws:
            _ = json.loads(await ws.recv())  # session.created
            await ws.send(frame)
            delta = json.loads(await ws.recv())
            assert delta["delta"] == "partial-before-drop"
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=2.0)

    asyncio.run(_check_deltas_before_drop())

    # 3. MockOpenAIServer handles empty models, empty top_logprobs dict, and multi-pos chat logprobs
    mock_openai.set_models([])
    mock_openai.enqueue(
        MockCompletionSpec(content="ab", top_logprobs=[{}]),
        MockCompletionSpec(
            content="ab",
            top_logprobs=[{"A": -0.1, "B": -2.0}, {"C": -0.2, "D": -1.5}],
        ),
    )
    with httpx.Client(timeout=3.0) as hc:
        r_empty_lp = hc.post(f"{mock_openai.base_url}/chat/completions", json={"messages": []})
        assert r_empty_lp.status_code == 200
        r_multi_lp = hc.post(f"{mock_openai.base_url}/chat/completions", json={"messages": []})
        assert r_multi_lp.status_code == 200
        assert len(r_multi_lp.json()["choices"][0]["logprobs"]["content"]) == 2

    # 4. resolve_coach_cmd on a secondary worktree does not resolve to main repo .venv/bin/coach
    fake_wt = tmp_path / "worktree"
    fake_wt.mkdir()
    cmd = resolve_coach_cmd(fake_wt)
    assert cmd != ["/home/halbritt/git/earshot/.venv/bin/coach"]
    assert cmd[1:] == ["-m", "coach.cli"]

    # 5. Reviewer 2 note: ui_client rejects spaced `< img>` bypass while accepting plain English
    CardUIClient.assert_card_plain_text({"question": "Who owns the rollback plan?"})
    with pytest.raises(AssertionError, match="markup/script"):
        CardUIClient.assert_card_plain_text({"question": "Check < img src=x>?"})


def test_db_assertions_iteration3_regression_checks(tmp_path: Path) -> None:
    # 1. Fix 1: Inv 5 context-segment overlap in egress_log must NOT mask another candidate
    ctx_overlap_db = tmp_path / "inv5_ctx_overlap.db"
    _create_m1_style_db(ctx_overlap_db)
    with sqlite3.connect(str(ctx_overlap_db)) as conn:
        conn.execute(
            "UPDATE egress_log SET segment_ids_json = ? WHERE egress_id = 'eg-1'",
            (json.dumps(["seg-1", "seg-3"]),),
        )
        conn.execute(
            "UPDATE decisions SET outcome = 'ABSTAIN', route = 'generator', "
            "raw_scores_json = ?, terminal_state = 'ABSTAIN', terminal_reason = 'score_in_band' "
            "WHERE candidate_id = 'cand-ret'",
            (json.dumps({"opportunity": 0.4}),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 5"):
        assert_six_conservation_invariants(ctx_overlap_db, "s1", run_id="r1")

    # Also verify suggestion_id mismatch prevents segment-overlap fallback
    sid_mismatch_db = tmp_path / "inv5_sid_mismatch.db"
    _create_m1_style_db(sid_mismatch_db)
    with sqlite3.connect(str(sid_mismatch_db)) as conn:
        conn.execute("ALTER TABLE egress_log ADD COLUMN suggestion_id TEXT")
        conn.execute(
            "UPDATE egress_log SET candidate_id = NULL, suggestion_id = 'sug-1', "
            "segment_ids_json = ? WHERE egress_id = 'eg-1'",
            (json.dumps(["seg-1", "seg-3"]),),
        )
        conn.execute(
            "UPDATE decisions SET outcome = 'ABSTAIN', route = 'generator', "
            "raw_scores_json = ?, terminal_state = 'ABSTAIN', terminal_reason = 'score_in_band' "
            "WHERE candidate_id = 'cand-ret'",
            (json.dumps({"opportunity": 0.4}),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 5"):
        assert_six_conservation_invariants(sid_mismatch_db, "s1", run_id="r1")

    # 2. Fix 2: Inv 5 rejects non-discarded egress_log with empty usage and complete_ms=None,
    # while allowing discarded=1 (timed-out/cancelled) or non-null complete_ms
    empty_usage_db = tmp_path / "inv5_empty_usage.db"
    _create_m1_style_db(empty_usage_db)
    with sqlite3.connect(str(empty_usage_db)) as conn:
        conn.execute("ALTER TABLE egress_log ADD COLUMN complete_ms INTEGER")
        conn.execute("ALTER TABLE egress_log ADD COLUMN discarded INTEGER DEFAULT 0")
        conn.execute(
            "UPDATE egress_log SET usage_json = '{}', complete_ms = NULL, discarded = 0 "
            "WHERE egress_id = 'eg-1'"
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 5"):
        assert_six_conservation_invariants(empty_usage_db, "s1", run_id="r1")

    with sqlite3.connect(str(empty_usage_db)) as conn:
        conn.execute("UPDATE egress_log SET discarded = 1 WHERE egress_id = 'eg-1'")
        conn.commit()
    assert_six_conservation_invariants(empty_usage_db, "s1", run_id="r1")

    with sqlite3.connect(str(empty_usage_db)) as conn:
        conn.execute(
            "UPDATE egress_log SET discarded = 0, complete_ms = 1650 WHERE egress_id = 'eg-1'"
        )
        conn.commit()
    assert_six_conservation_invariants(empty_usage_db, "s1", run_id="r1")

    # 3. Fix 3: Inv 3 rejects a second suggestion row lacking a terminal ReasonCode
    inv3_multi_sug_db = tmp_path / "inv3_multi_sug.db"
    _create_m1_style_db(inv3_multi_sug_db)
    with sqlite3.connect(str(inv3_multi_sug_db)) as conn:
        conn.execute(
            "INSERT INTO suggestions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "s1",
                "r1",
                "sug-2",
                "cand-1",
                "generator",
                "used",
                None,
                "[]",
                json.dumps([{"segment_id": "seg-1", "revision": 1}]),
                None,
                None,
            ),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="Inv 3"):
        assert_six_conservation_invariants(inv3_multi_sug_db, "s1", run_id="r1")

    # 4. Fix 4: §14.2 assert_session_text_deleted rejects flat string values in grounded_slots
    slots_str_db = tmp_path / "slots_str_leak.db"
    _create_m1_style_db(slots_str_db)
    with sqlite3.connect(str(slots_str_db)) as conn:
        conn.execute(
            "UPDATE suggestions SET grounded_slots = ? WHERE suggestion_id = 'sug-1'",
            (json.dumps({"action": "leaked plain string"}),),
        )
        conn.commit()
    with pytest.raises(AssertionError, match="grounded_slots"):
        assert_session_text_deleted(slots_str_db, "s1")


def test_e2_helper_extensions(tmp_path: Path) -> None:
    # 1. deep_merge_dict preserves sibling keys in nested policy dicts
    base = build_policy_dict()
    merged = deep_merge_dict(
        base,
        {
            "display": {"settle_ms": 250},
            "generator": {"breaker": {"open_ms": 5000}},
        },
    )
    assert merged["display"]["settle_ms"] == 250
    assert merged["display"]["display_ttl_ms"] == 15000
    assert merged["generator"]["breaker"]["open_ms"] == 5000
    assert merged["generator"]["breaker"]["consecutive_failures"] == 3
    assert base["display"]["settle_ms"] == 1500

    # 2. write_policy_dict + write_jsonl + build_decision_context_row
    pol_path = write_policy_dict(tmp_path / "custom_policy.yaml", merged)
    loaded_pol = yaml.safe_load(pol_path.read_text(encoding="utf-8"))
    assert loaded_pol["display"]["settle_ms"] == 250

    ctx_row = build_decision_context_row(
        candidate_id="cand-ctx-1",
        context_turns=("S1: Can we deploy Friday?",),
        worthwhile=True,
        categories=("CONFIRM_OWNER",),
        sample_source="random",
    )
    assert ctx_row["schema_version"] == 1
    assert "[ANCHOR]" in ctx_row["serialized_text"]
    assert ctx_row["token_counts"]["limit"] == 512
    jsonl_path = write_jsonl(tmp_path / "contexts.jsonl", [ctx_row])
    assert read_jsonl(jsonl_path) == [ctx_row]

    # 3. write_appendix_b_workspace materializes B04 (flat settle_ms) and B13 (generator_enabled)
    b04_ws = write_appendix_b_workspace(build_appendix_b_fixture("B04"), tmp_path / "b04")
    assert b04_ws["brief_path"].is_file()
    assert b04_ws["events_path"].is_file()
    assert (b04_ws["templates_dir"] / "confirm_owner_v1.yaml").is_file()
    b04_pol = yaml.safe_load(b04_ws["policy_path"].read_text(encoding="utf-8"))
    assert b04_pol["display"]["settle_ms"] == 200
    assert b04_pol["display"]["cooldown_ms"] == 20000
    b13_ws = write_appendix_b_workspace(build_appendix_b_fixture("B13"), tmp_path / "b13")
    b13_pol = yaml.safe_load(b13_ws["policy_path"].read_text(encoding="utf-8"))
    assert b13_pol["generator"]["enabled"] is True
    assert b13_pol["generator"]["breaker"]["open_ms"] == 60000

    # 4. ManagedProcess.wait_for_output, extract_ui_url_and_token, build_coach_test_env
    fake_bin = tmp_path / "bin" / "peecee-speech"
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    env = build_coach_test_env(
        openai_base_url="http://127.0.0.1:9999/v1",
        nemo_ws_url="ws://127.0.0.1:18921/v1/audio/transcriptions/realtime",
        peecee_bin=fake_bin,
        extra_env={"CUSTOM_FLAG": "1"},
    )
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert env["PEECEE_WS_URL"] == "ws://127.0.0.1:18921/v1/audio/transcriptions/realtime"
    assert env["PATH"].startswith(str(fake_bin.parent))
    assert env["CUSTOM_FLAG"] == "1"

    script = (
        "import sys, time; "
        "print('UI listening http://127.0.0.1:43210/?token=tok_e2_abc', flush=True); "
        "print('stderr-marker-ready', file=sys.stderr, flush=True); "
        "time.sleep(10)"
    )
    with ManagedProcess([sys.executable, "-c", script], startup_timeout=5.0) as proc:
        url, token = extract_ui_url_and_token(proc, default_port=9999, timeout=5.0)
        assert url == "http://127.0.0.1:43210"
        assert token == "tok_e2_abc"
        err_match = proc.wait_for_output(r"stderr-marker-(\w+)", timeout=5.0)
        assert err_match.group(1) == "ready"
        with pytest.raises(TimeoutError, match="never-emitted-pattern"):
            proc.wait_for_output("never-emitted-pattern", timeout=0.05)

    # default_port fallback & short-lived process exit behavior
    short_script = "print('token=tok_short_1', flush=True)"
    with ManagedProcess([sys.executable, "-c", short_script], startup_timeout=5.0) as short_proc:
        short_url, short_tok = extract_ui_url_and_token(short_proc, default_port=8765, timeout=5.0)
        assert short_url == "http://127.0.0.1:8765"
        assert short_tok == "tok_short_1"
        with pytest.raises(RuntimeError, match="exited with code 0"):
            short_proc.wait_for_output("missing-after-exit", timeout=1.0)
