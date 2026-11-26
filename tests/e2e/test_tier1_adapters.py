"""Tier 1 opaque-box E2E tests for Area B: peecee-speech, earshot, Deepgram, and replay."""

from __future__ import annotations

import asyncio
import json
import os
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.e2e.helpers.cli_runner import (
    build_coach_test_env,
    extract_ui_url_and_token,
    run_coach,
    run_earshot_beam,
    run_earshot_direction,
    start_coach_process,
)
from tests.e2e.helpers.db_assertions import (
    assert_six_conservation_invariants,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
    build_deepgram_results,
    build_deepgram_utterance_end,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    build_speaker_update_event,
    generate_multichannel_pcm,
    generate_pcm16_sine,
    write_brief_json,
    write_events_jsonl,
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
from tests.e2e.helpers.ui_client import CardUIClient


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 10.0, msg: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out after {timeout}s: {msg}")


def _pid_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _count_committed_segments(db_path: Path, session_id: str) -> int:
    with open_db(db_path, readonly=True) as conn:
        rows = fetch_normalized_rows(conn, "events", session_id)
    return sum(1 for r in rows if r.get("kind") == "segment" and r.get("stability") == "committed")


def test_peecee_realtime_ws_handshake_160ms_frames_and_commit(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-peecee-ws"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    mock_nemo_ws.set_config(
        MockNemoConfig(
            sample_rate=16000,
            script=[
                ScriptedTurn(
                    trigger_after_frames=3,
                    deltas=["Someone"],
                    transcript="Someone should own the rollback plan",
                    words=[
                        MockWord("Someone", 0.0, 0.25, speaker=1),
                        MockWord("should", 0.25, 0.45, speaker=1),
                        MockWord("own", 0.45, 0.65, speaker=1),
                        MockWord("the", 0.65, 0.80, speaker=1),
                        MockWord("rollback", 0.80, 1.15, speaker=1),
                        MockWord("plan", 1.15, 1.40, speaker=1),
                    ],
                ),
                ScriptedTurn(
                    trigger_on_commit=True,
                    transcript="Who owns failover testing",
                    words=[
                        MockWord("Who", 1.50, 1.70, speaker=2),
                        MockWord("owns", 1.70, 1.95, speaker=2),
                        MockWord("failover", 1.95, 2.25, speaker=2),
                        MockWord("testing", 2.25, 2.55, speaker=2),
                    ],
                ),
            ],
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "input_960ms.wav",
        generate_pcm16_sine(duration_ms=960),
    )
    db_path = e2e_workspace["db_path"]
    port = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)

    with start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_path),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port),
            "--local-only",
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
    ):
        _wait_until(
            lambda: (
                len(mock_nemo_ws.session_configs) >= 1
                and len(mock_nemo_ws.received_frames) >= 3
                and any(
                    ev.get("type") == "input_audio_buffer.commit"
                    for ev in mock_nemo_ws.received_events
                )
            ),
            msg="coach serve did not complete WS handshake, 160ms frames, and commit",
        )

    cfg = mock_nemo_ws.session_configs[0]
    assert cfg["sample_rate"] == 16000
    assert cfg["language"] == "en-US"
    assert cfg["word_timestamps"] is True
    assert cfg["speaker_diarization"] is True
    assert all(len(frame) == PCM16_FRAME_BYTES_160MS for frame in mock_nemo_ws.received_frames)

    with open_db(db_path, readonly=True) as conn:
        raw_rows = fetch_normalized_rows(conn, "raw_events", sid)
        ev_rows = fetch_normalized_rows(conn, "events", sid)
        assert len(raw_rows) >= 1
        stabilities = {str(r.get("stability")) for r in ev_rows if r.get("kind") == "segment"}
        assert "committed" in stabilities
        assert "interim" in stabilities

    rep = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert rep.returncode == 0, rep.stderr
    rep_json: dict[str, Any] = rep.json()
    assert rep_json["invariants"]["passed"] is True
    assert rep_json["upstream_delay_ms"]["p50"] is not None


def test_peecee_serve_exit_75_tunnel_drop_and_sigterm_child_cleanup(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-peecee-resilience"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    policy_path = e2e_workspace["policy_path"]

    # (a) peecee-speech --serve exit 75 -> immediate no_transcript & zero inference
    db_75 = e2e_workspace["root"] / "exit75.db"
    p_75 = free_port_factory()
    env_75 = build_coach_test_env(
        nemo_ws_url=mock_nemo_ws.url,
        peecee_bin=mock_peecee_bin,
        extra_env={"MOCK_PEECEE_EXIT_CODE": "75"},
    )
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(policy_path),
            "--db",
            str(db_75),
            "--source",
            "peecee",
            "--ui-port",
            str(p_75),
            "--local-only",
        ],
        env=env_75,
        ready_url=f"http://127.0.0.1:{p_75}/",
    ) as proc_75:
        url_75, tok_75 = extract_ui_url_and_token(proc_75, p_75)
        ui_75 = CardUIClient(url_75, tok_75)

        async def _wait_75() -> None:
            async with ui_75.ws_session() as ws:
                msg = await ws.wait_for_health("no_transcript", timeout=5.0)
                assert (msg.get("health") or msg.get("state")) == "no_transcript"

        asyncio.run(_wait_75())

    rep_75 = run_coach("report", ["--db", str(db_75), "--session", sid, "--json"])
    assert rep_75.returncode == 0, rep_75.stderr
    assert any(
        entry.get("state") == "no_transcript" or entry.get("health") == "no_transcript"
        for entry in rep_75.json()["health_timeline"]
    )
    with open_db(db_75, readonly=True) as conn:
        assert fetch_normalized_rows(conn, "decisions", sid) == []

    # (b) SIGTERM terminates peecee-speech's ssh child PID (no orphan)
    pid_file = e2e_workspace["root"] / "peecee_pids.json"
    db_sig = e2e_workspace["root"] / "sigterm.db"
    p_sig = free_port_factory()
    env_sig = build_coach_test_env(
        nemo_ws_url=mock_nemo_ws.url,
        peecee_bin=mock_peecee_bin,
        extra_env={"MOCK_PEECEE_PID_FILE": str(pid_file)},
    )
    proc_sig = start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(policy_path),
            "--db",
            str(db_sig),
            "--source",
            "peecee",
            "--ui-port",
            str(p_sig),
            "--local-only",
        ],
        env=env_sig,
        ready_url=f"http://127.0.0.1:{p_sig}/",
    )
    with proc_sig:
        _wait_until(pid_file.is_file, msg="peecee-speech did not write MOCK_PEECEE_PID_FILE")
        child_pid = int(json.loads(pid_file.read_text(encoding="utf-8"))["child_pid"])
        proc_sig.stop()
        _wait_until(
            lambda: _pid_dead(child_pid),
            timeout=3.0,
            msg=f"ssh child PID {child_pid} remained alive after SIGTERM",
        )

    # (c) Abrupt tunnel drop mid-session -> immediate no_transcript
    db_drop = e2e_workspace["root"] / "drop.db"
    p_drop = free_port_factory()
    wav_drop = write_wav_mono16(
        e2e_workspace["root"] / "drop_1600ms.wav",
        generate_pcm16_sine(duration_ms=1600),
    )
    mock_nemo_ws.set_config(MockNemoConfig(silence_mode=True))
    env_drop = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(policy_path),
            "--db",
            str(db_drop),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_drop),
            "--ui-port",
            str(p_drop),
            "--local-only",
        ],
        env=env_drop,
        ready_url=f"http://127.0.0.1:{p_drop}/",
    ) as proc_drop:
        url_drop, tok_drop = extract_ui_url_and_token(proc_drop, p_drop)
        ui_drop = CardUIClient(url_drop, tok_drop)

        async def _wait_drop() -> None:
            async with ui_drop.ws_session() as ws:
                _ = await ws.wait_for_health("ok", timeout=5.0)
                deadline = asyncio.get_running_loop().time() + 5.0
                while not mock_nemo_ws._active_clients:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("coach did not connect to mock_nemo_ws before drop")
                    await asyncio.sleep(0.05)
                mock_nemo_ws.drop_active_clients(abrupt=True)
                msg = await ws.wait_for_health("no_transcript", timeout=5.0)
                assert (msg.get("health") or msg.get("state")) == "no_transcript"

        asyncio.run(_wait_drop())


def test_peecee_offline_wav_stream_json_ingestion(
    e2e_workspace: dict[str, Path],
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-peecee-offline-wav"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "offline_1600ms.wav",
        generate_pcm16_sine(duration_ms=1600),
    )
    peecee_json = {
        "file": str(wav_path),
        "text": "Someone should own the rollback plan",
        "confidence": 1.0,
        "duration": 1.6,
        "languages": ["en-US"],
        "words": [
            {"word": "Someone", "start": 0.16, "end": 0.48, "confidence": 1.0, "speaker": 1},
            {"word": "should", "start": 0.48, "end": 0.72, "confidence": 1.0, "speaker": 1},
            {"word": "own", "start": 0.72, "end": 0.90, "confidence": 1.0, "speaker": 1},
            {"word": "the", "start": 0.90, "end": 1.02, "confidence": 1.0, "speaker": 1},
            {"word": "rollback", "start": 1.02, "end": 1.32, "confidence": 1.0, "speaker": 1},
            {"word": "plan", "start": 1.32, "end": 1.56, "confidence": 1.0, "speaker": 1},
        ],
    }
    json_path = e2e_workspace["root"] / "peecee_offline.json"
    json_path.write_text(json.dumps(peecee_json), encoding="utf-8")

    db_path = e2e_workspace["db_path"]
    port = free_port_factory()
    env = build_coach_test_env(
        peecee_bin=mock_peecee_bin,
        extra_env={"MOCK_PEECEE_JSON_PATH": str(json_path)},
    )
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_path),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port),
            "--local-only",
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
    ):

        def _has_committed_segment() -> bool:
            if not db_path.is_file():
                return False
            with open_db(db_path, readonly=True) as conn:
                rows = fetch_normalized_rows(conn, "events", sid)
            return any(
                r.get("kind") == "segment" and r.get("stability") == "committed" for r in rows
            )

        _wait_until(_has_committed_segment, msg="offline peecee WAV ingestion did not commit")

    with open_db(db_path, readonly=True) as conn:
        raw_rows = fetch_normalized_rows(conn, "raw_events", sid)
        ev_rows = fetch_normalized_rows(conn, "events", sid)
    assert len(raw_rows) >= 1
    committed = [
        r for r in ev_rows if r.get("kind") == "segment" and r.get("stability") == "committed"
    ]
    assert any("rollback plan" in str(r.get("text") or "") for r in committed)

    rep = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert rep.returncode == 0, rep.stderr
    assert rep.json()["invariants"]["passed"] is True
    assert_six_conservation_invariants(db_path, sid)


def test_nemotron_punctuation_free_segmenter_and_earshot_gates(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
    repo_root: Path,
) -> None:
    # (a) Real Rust `earshot direction` and `earshot beam` on 8ch raw PCM
    array_toml = repo_root / "arrays" / "respeaker-core-v2.toml"
    raw_pcm = e2e_workspace["root"] / "array_8ch.raw"
    raw_pcm.write_bytes(generate_multichannel_pcm(channels=8, duration_ms=1000, width_bits=32))
    dir_res = run_earshot_direction(array_toml, raw_pcm, every=0.5, repo_root=repo_root)
    assert dir_res.returncode == 0, dir_res.stderr
    dir_rows = dir_res.jsonl()
    assert len(dir_rows) >= 1
    assert {"t", "azimuth", "contrast", "level_db"}.issubset(dir_rows[0].keys())

    beamed_wav = e2e_workspace["root"] / "beamed_mono.wav"
    beam_res = run_earshot_beam(
        array_toml, raw_pcm, beamed_wav, steer="0", verbose=True, repo_root=repo_root
    )
    assert beam_res.returncode == 0, beam_res.stderr
    with wave.open(str(beamed_wav), "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)

    # (b) Punctuation-free Nemotron output splits on silence gap and speaker change
    sid = "sess-nemotron-seg"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    deltas=["Someone"],
                    transcript="Someone should own rollback",
                    words=[
                        MockWord("Someone", 0.1, 0.4, speaker=1),
                        MockWord("should", 0.4, 0.7, speaker=1),
                        MockWord("own", 0.7, 0.9, speaker=1),
                        MockWord("rollback", 0.9, 1.3, speaker=1),
                    ],
                ),
                ScriptedTurn(
                    trigger_after_frames=4,
                    deltas=["Let"],
                    transcript="Let us deploy on Friday",
                    words=[
                        MockWord("Let", 3.0, 3.3, speaker=2),
                        MockWord("us", 3.3, 3.5, speaker=2),
                        MockWord("deploy", 3.5, 3.9, speaker=2),
                        MockWord("on", 3.9, 4.1, speaker=2),
                        MockWord("Friday", 4.1, 4.5, speaker=2),
                    ],
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "nemotron_960ms.wav",
        generate_pcm16_sine(duration_ms=960),
    )
    db_path = e2e_workspace["db_path"]
    port = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_path),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port),
            "--local-only",
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
    ):
        _wait_until(
            lambda: db_path.is_file() and _count_committed_segments(db_path, sid) >= 2,
            msg="Nemotron punctuation-free words did not split into >= 2 committed segments",
        )

    with open_db(db_path, readonly=True) as conn:
        ev_rows = fetch_normalized_rows(conn, "events", sid)
    committed = [
        r for r in ev_rows if r.get("kind") == "segment" and r.get("stability") == "committed"
    ]
    interims = [
        r for r in ev_rows if r.get("kind") == "segment" and r.get("stability") == "interim"
    ]
    boundaries = [r for r in ev_rows if r.get("kind") == "boundary"]
    assert len(committed) >= 2
    assert len(interims) >= 1
    assert len(boundaries) >= 1

    # (c) Speech/echo gate low-confidence span triggers asr_low_confidence abstention
    gate_sid = "sess-echo-gate"
    gate_brief = write_brief_json(
        e2e_workspace["root"] / "gate_brief.json",
        build_session_brief(session_id=gate_sid),
    )
    gate_policy = write_policy_dict(
        e2e_workspace["policies_dir"] / "gate.yaml",
        build_policy_dict(asr_confidence_floor=0.70),
    )
    gate_events = [
        build_segment_event(
            session_id=gate_sid,
            event_id="ev-gate-low",
            segment_id="seg-gate-low",
            text="Someone should own the rollback plan.",
            start_ms=500,
            end_ms=2500,
            received_ms=2700,
            source_sequence=1,
            extra={
                "asr_confidence": 0.35,
                "asr_low_confidence": True,
                "low_quality_gate": True,
            },
        )
    ]
    gate_jsonl = write_events_jsonl(e2e_workspace["sessions_dir"] / "gate.jsonl", gate_events)
    gate_db = e2e_workspace["root"] / "gate.db"
    gate_res = run_coach(
        "replay",
        [
            "--session",
            str(gate_jsonl),
            "--brief",
            str(gate_brief),
            "--policy",
            str(gate_policy),
            "--db",
            str(gate_db),
            "--json",
        ],
    )
    assert gate_res.returncode == 0, gate_res.stderr
    with open_db(gate_db, readonly=True) as conn:
        gate_decs = fetch_normalized_rows(conn, "decisions", gate_sid)
    assert len(gate_decs) == 1
    assert gate_decs[0]["outcome"] == "ABSTAIN"
    assert gate_decs[0]["terminal_reason"] == "asr_low_confidence"


def test_sortformer_recipient_binder_azimuth_speaker_update_and_uncertain(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-sortformer-binder"
    brief_path = write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(
            session_id=sid,
            recipient={"participant_id": "me", "role": "reviewer"},
        ),
    )
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="I will verify the cluster health first",
                    words=[
                        MockWord("I", 0.1, 0.3, speaker=1),
                        MockWord("will", 0.3, 0.5, speaker=1),
                        MockWord("verify", 0.5, 0.8, speaker=1),
                        MockWord("the", 0.8, 1.0, speaker=1),
                        MockWord("cluster", 1.0, 1.3, speaker=1),
                    ],
                ),
                ScriptedTurn(
                    trigger_after_frames=4,
                    transcript="Someone should own the rollback plan",
                    words=[
                        MockWord("Someone", 1.5, 1.8, speaker=2),
                        MockWord("should", 1.8, 2.1, speaker=2),
                        MockWord("own", 2.1, 2.3, speaker=2),
                        MockWord("the", 2.3, 2.5, speaker=2),
                        MockWord("rollback", 2.5, 2.8, speaker=2),
                        MockWord("plan", 2.8, 3.1, speaker=2),
                    ],
                ),
                ScriptedTurn(
                    trigger_after_frames=6,
                    transcript="Wait",
                    words=[MockWord("Wait", 3.3, 3.7, speaker=3)],
                ),
                ScriptedTurn(
                    trigger_on_commit=True,
                    transcript="Who confirms the deployment window",
                    words=[
                        MockWord("Who", 4.0, 4.3, speaker=3),
                        MockWord("confirms", 4.3, 4.7, speaker=3),
                        MockWord("deployment", 4.7, 5.2, speaker=3),
                    ],
                ),
            ],
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "sortformer_1120ms.wav",
        generate_pcm16_sine(duration_ms=1120),
    )
    db_path = e2e_workspace["db_path"]
    port = free_port_factory()
    direction_samples = [
        {"t": 0.5, "azimuth": 0.0, "contrast": 8.5, "level_db": -18.0},
        {"t": 1.0, "azimuth": 0.0, "contrast": 8.5, "level_db": -18.0},
        {"t": 1.5, "azimuth": 90.0, "contrast": 8.2, "level_db": -19.0},
        {"t": 2.0, "azimuth": 90.0, "contrast": 8.2, "level_db": -19.0},
        {"t": 2.5, "azimuth": 90.0, "contrast": 8.2, "level_db": -19.0},
        {"t": 3.0, "azimuth": 90.0, "contrast": 8.2, "level_db": -19.0},
        {"t": 3.5, "azimuth": 180.0, "contrast": 3.0, "level_db": -25.0},
        {"t": 4.5, "azimuth": 90.0, "contrast": 8.0, "level_db": -19.0},
        {"t": 5.0, "azimuth": 90.0, "contrast": 8.0, "level_db": -19.0},
    ]
    env = build_coach_test_env(
        nemo_ws_url=mock_nemo_ws.url,
        peecee_bin=mock_peecee_bin,
        extra_env={
            "COACH_SEAT_AZIMUTHS": json.dumps({"me": 0.0, "S1": 90.0}),
            "COACH_EARSHOT_DIRECTION_JSON": json.dumps(direction_samples),
        },
    )
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_path),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port),
            "--local-only",
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
    ):
        _wait_until(
            lambda: db_path.is_file() and _count_committed_segments(db_path, sid) >= 4,
            msg="Sortformer turns did not commit >= 4 segments",
        )

    with open_db(db_path, readonly=True) as conn:
        ev_rows = fetch_normalized_rows(conn, "events", sid)
        dec_rows = fetch_normalized_rows(conn, "decisions", sid)
    speakers = {str(r.get("speaker_id")) for r in ev_rows if r.get("speaker_id")}
    assert any(s in {"ME", "me", "S1", "S2", "participant-b"} for s in speakers)
    assert any(bool(r.get("speaker_uncertain")) for r in ev_rows) or any(
        r.get("terminal_reason") == "speaker_uncertain" for r in dec_rows
    )
    assert any(r.get("kind") == "speaker_update" for r in ev_rows)

    # Also verify speaker_update revision bump and <1s speaker_uncertain abstention via replay
    rep_sid = "sess-sortformer-replay"
    rep_brief = write_brief_json(
        e2e_workspace["root"] / "sortformer_rep_brief.json",
        build_session_brief(session_id=rep_sid),
    )
    rep_events = [
        build_segment_event(
            session_id=rep_sid,
            event_id="ev-sf-1",
            segment_id="seg-sf-1",
            text="Someone should own the rollback plan.",
            speaker_id="S1",
            start_ms=1000,
            end_ms=3200,
            received_ms=3400,
            source_sequence=1,
        ),
        build_speaker_update_event(
            session_id=rep_sid,
            event_id="ev-sf-upd",
            segment_id="seg-sf-1",
            speaker_id="S2",
            source_revision=2,
            received_ms=3600,
            source_sequence=2,
            speaker_confidence=0.91,
        ),
        build_segment_event(
            session_id=rep_sid,
            event_id="ev-sf-short",
            segment_id="seg-sf-short",
            text="Wait should we check the timing right now?",
            speaker_id="S?",
            start_ms=4000,
            end_ms=4600,
            received_ms=4800,
            source_sequence=3,
            extra={"speaker_uncertain": True},
        ),
    ]
    rep_jsonl = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "sortformer_rep.jsonl", rep_events
    )
    rep_db = e2e_workspace["root"] / "sortformer_rep.db"
    res_sf = run_coach(
        "replay",
        [
            "--session",
            str(rep_jsonl),
            "--brief",
            str(rep_brief),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(rep_db),
            "--json",
        ],
    )
    assert res_sf.returncode == 0, res_sf.stderr
    with open_db(rep_db, readonly=True) as conn:
        sf_ev = fetch_normalized_rows(conn, "events", rep_sid)
        sf_dec = fetch_normalized_rows(conn, "decisions", rep_sid)
    assert any(
        r.get("kind") == "speaker_update" and int(r.get("canonical_revision") or 0) >= 2
        for r in sf_ev
    )
    assert any(
        d.get("outcome") == "ABSTAIN" and d.get("terminal_reason") == "speaker_uncertain"
        for d in sf_dec
    )


def test_deepgram_json_normalizer_interim_committed_utterance_end_and_fallbacks(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-deepgram-norm"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    dg_payloads = [
        build_deepgram_results(
            "Someone should",
            is_final=False,
            speech_final=False,
            start_s=1.0,
            duration_s=1.0,
            channel_index=0,
            speaker=1,
        ),
        build_deepgram_results(
            "Someone should own the rollback plan.",
            is_final=True,
            speech_final=False,
            start_s=1.0,
            duration_s=2.2,
            channel_index=0,
            speaker=1,
        ),
        build_deepgram_results(
            "When are we deploying?",
            is_final=True,
            speech_final=True,
            start_s=3.5,
            duration_s=1.5,
            channel_index=0,
            speaker=2,
        ),
        build_deepgram_utterance_end(5.0, channel=[0, 1]),
    ]
    dg_path = write_jsonl(e2e_workspace["sessions_dir"] / "deepgram.jsonl", dg_payloads)
    db_1 = e2e_workspace["root"] / "dg1.db"
    db_2 = e2e_workspace["root"] / "dg2.db"

    for db in (db_1, db_2):
        res = run_coach(
            "replay",
            [
                "--session",
                str(dg_path),
                "--source",
                "deepgram",
                "--brief",
                str(brief_path),
                "--policy",
                str(e2e_workspace["policy_path"]),
                "--db",
                str(db),
                "--json",
            ],
        )
        assert res.returncode == 0, res.stderr

    def _inspect(db: Path) -> tuple[list[str], list[str], list[str]]:
        with open_db(db, readonly=True) as conn:
            ev_rows = fetch_normalized_rows(conn, "events", sid)
        interims = [
            str(r["segment_id"])
            for r in ev_rows
            if r.get("kind") == "segment" and r.get("stability") == "interim"
        ]
        committed = [
            str(r["segment_id"])
            for r in ev_rows
            if r.get("kind") == "segment" and r.get("stability") == "committed"
        ]
        boundaries = [str(r["event_id"]) for r in ev_rows if r.get("kind") == "boundary"]
        revs = [
            int(r["source_revision"])
            for r in ev_rows
            if r.get("kind") == "segment" and r.get("source_revision") is not None
        ]
        assert len(interims) >= 1
        assert len(committed) >= 2
        assert len(boundaries) >= 2
        assert all(rev >= 1 for rev in revs)
        return interims, committed, boundaries

    interims_1, committed_1, _ = _inspect(db_1)
    interims_2, committed_2, _ = _inspect(db_2)
    assert interims_1 == interims_2
    assert committed_1 == committed_2

    rep = run_coach("report", ["--db", str(db_1), "--session", sid, "--json"])
    assert rep.returncode == 0, rep.stderr
    assert rep.json()["invariants"]["passed"] is True
    assert_six_conservation_invariants(db_1, sid)


def test_deterministic_replay_player_recorded_scaled_and_stepped_modes(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-replay-modes"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    events_3s = [
        build_segment_event(
            session_id=sid,
            event_id="ev-rp-1",
            segment_id="seg-rp-1",
            text="Someone should own the rollback plan.",
            start_ms=500,
            end_ms=900,
            received_ms=1000,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-rp-2",
            segment_id="seg-rp-2",
            text="When will we verify the failover step?",
            start_ms=2800,
            end_ms=3400,
            received_ms=3500,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-rp-3",
            segment_id="seg-rp-3",
            text="How will we measure cutover success?",
            start_ms=5300,
            end_ms=5900,
            received_ms=6000,
            source_sequence=3,
        ),
    ]
    path_3s = write_events_jsonl(e2e_workspace["sessions_dir"] / "rp_3s.jsonl", events_3s)
    db_stepped = e2e_workspace["root"] / "rp_stepped.db"
    db_scaled = e2e_workspace["root"] / "rp_scaled.db"

    res_stepped = run_coach(
        "replay",
        [
            "--session",
            str(path_3s),
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_stepped),
            "--speed",
            "stepped",
            "--json",
        ],
    )
    assert res_stepped.returncode == 0, res_stepped.stderr
    assert res_stepped.duration_s < 4.5

    res_scaled = run_coach(
        "replay",
        [
            "--session",
            str(path_3s),
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_scaled),
            "--speed",
            "10.0",
            "--json",
        ],
    )
    assert res_scaled.returncode == 0, res_scaled.stderr
    assert 0.45 <= res_scaled.duration_s < 4.5

    def _snapshot(db: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        with open_db(db, readonly=True) as conn:
            evs = conn.execute(
                "SELECT segment_id, text FROM events WHERE session_id = ? "
                "ORDER BY start_ms ASC, channel ASC, source_sequence ASC",
                (sid,),
            ).fetchall()
            decs = conn.execute(
                "SELECT candidate_id, serialized_sha256, outcome FROM decisions "
                "WHERE session_id = ? ORDER BY candidate_id ASC",
                (sid,),
            ).fetchall()
        return (
            [(str(r[0]), str(r[1])) for r in evs],
            [(str(r[0]), str(r[1]), str(r[2])) for r in decs],
        )

    assert _snapshot(db_stepped) == _snapshot(db_scaled)

    # Recorded 1.0x speed on short 600ms fixture preserves real inter-event elapsed time
    sid_real = "sess-replay-real"
    brief_real = write_brief_json(
        e2e_workspace["root"] / "brief_real.json", build_session_brief(session_id=sid_real)
    )
    events_real = [
        build_segment_event(
            session_id=sid_real,
            event_id="ev-real-1",
            segment_id="seg-real-1",
            text="Someone should own the rollback plan.",
            start_ms=500,
            end_ms=900,
            received_ms=1000,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid_real,
            event_id="ev-real-2",
            segment_id="seg-real-2",
            text="When does the cutover window open?",
            start_ms=1100,
            end_ms=1500,
            received_ms=1600,
            source_sequence=2,
        ),
    ]
    path_real = write_events_jsonl(e2e_workspace["sessions_dir"] / "rp_real.jsonl", events_real)
    db_real = e2e_workspace["root"] / "rp_real.db"
    res_real = run_coach(
        "replay",
        [
            "--session",
            str(path_real),
            "--brief",
            str(brief_real),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_real),
            "--speed",
            "1.0",
            "--json",
        ],
    )
    assert res_real.returncode == 0, res_real.stderr
    assert res_real.duration_s >= 0.45
    with open_db(db_real, readonly=True) as conn:
        real_evs = fetch_normalized_rows(conn, "events", sid_real)
    assert [str(r["event_id"]) for r in real_evs] == ["ev-real-1", "ev-real-2"]
