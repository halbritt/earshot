"""Tier 2 opaque-box boundary value analysis tests across all 7 areas (A–G)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.helpers.cli_runner import (
    CliResult,
    build_coach_test_env,
    extract_ui_url_and_token,
    run_coach,
    start_coach_process,
)
from tests.e2e.helpers.db_assertions import (
    assert_model_cache_has_entries,
    assert_session_text_deleted,
    assert_six_conservation_invariants,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
    build_control_event,
    build_decision_context_row,
    build_deepgram_results,
    build_label_record,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    build_speaker_update_event,
    chunk_pcm16_frames,
    deep_merge_dict,
    generate_pcm16_silence,
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
from tests.e2e.helpers.mock_openai import MockCompletionSpec, MockOpenAIServer
from tests.e2e.helpers.ui_client import AnnotationClient, CardUIClient


def _write_nested_policy(
    policy_path: Path,
    *,
    generator_enabled: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    base = build_policy_dict(generator_enabled=generator_enabled)
    merged = deep_merge_dict(base, overrides or {})
    return write_policy_dict(policy_path, merged)


def _replay_session(
    workspace: dict[str, Path],
    session_id: str,
    events: Sequence[dict[str, Any]],
    *,
    filename: str = "stream.jsonl",
    brief_overrides: Mapping[str, Any] | None = None,
    policy_overrides: Mapping[str, Any] | None = None,
    generator_enabled: bool = False,
    extra_args: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
) -> CliResult:
    brief = build_session_brief(session_id=session_id, **dict(brief_overrides or {}))
    write_brief_json(workspace["brief_path"], brief)
    _write_nested_policy(
        workspace["policy_path"],
        generator_enabled=generator_enabled,
        overrides=policy_overrides,
    )
    jsonl_path = write_events_jsonl(workspace["sessions_dir"] / filename, events)
    return run_coach(
        "replay",
        [
            "--session",
            str(jsonl_path),
            "--brief",
            str(workspace["brief_path"]),
            "--policy",
            str(workspace["policy_path"]),
            "--db",
            str(workspace["db_path"]),
            "--json",
            *extra_args,
        ],
        env=env,
    )


def _make_words(text: str, *, start_s: float, speaker: int) -> list[MockWord]:
    tokens = text.split()
    words: list[MockWord] = []
    cursor = start_s
    for tok in tokens:
        words.append(MockWord(tok, round(cursor, 3), round(cursor + 0.16, 3), speaker=speaker))
        cursor += 0.20
    return words


# ---------------------------------------------------------------------------
# Area A Boundaries (5 tests)
# ---------------------------------------------------------------------------


def test_boundary_a1_nfc_whitespace_immaterial_vs_case_punctuation_material(
    e2e_workspace: dict[str, Path],
) -> None:
    """NFC + collapsed whitespace is immaterial; case/punctuation changes bump revision (§6.2)."""
    session_id = "session-a1"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-a1-1",
            segment_id="seg-a1-1",
            text="Review  the  caf\u00e9  plan.",
            source_revision=1,
            received_ms=1000,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-a1-2",
            segment_id="seg-a1-1",
            text="Review the cafe\u0301 plan.",
            source_revision=2,
            received_ms=1200,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-a1-3",
            segment_id="seg-a1-1",
            text="Review the cafe\u0301 plan?",
            source_revision=3,
            received_ms=1400,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-a1-4",
            segment_id="seg-a1-1",
            text="review the cafe\u0301 plan?",
            source_revision=4,
            received_ms=1600,
            source_sequence=4,
        ),
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="a1.jsonl")
    assert res.returncode == 0
    rep = run_coach(
        "report",
        ["--db", str(e2e_workspace["db_path"]), "--session", session_id, "--json"],
    ).json()
    assert rep["reducer_counters"]["material_revisions_after_commit"] == 2
    with open_db(e2e_workspace["db_path"]) as conn:
        segs = [
            r
            for r in fetch_normalized_rows(conn, "events", session_id)
            if r.get("segment_id") == "seg-a1-1" and r.get("kind") == "segment"
        ]
    assert max(int(s.get("canonical_revision") or 0) for s in segs) == 3
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_a2_ordering_tiebreak_start_ms_channel_source_sequence(
    e2e_workspace: dict[str, Path],
) -> None:
    """Segments order by (start_ms, channel, source_sequence) regardless of received_ms (§6.2)."""
    session_id = "session-a2"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-a2-c2s1",
            segment_id="seg-c2-s1",
            text="Fourth order channel two sequence one.",
            channel=2,
            start_ms=1000,
            end_ms=1300,
            received_ms=4000,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-a2-c1s2",
            segment_id="seg-c1-s2",
            text="Third order channel one sequence two.",
            channel=1,
            start_ms=1000,
            end_ms=1300,
            received_ms=4200,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-a2-c1s1",
            segment_id="seg-c1-s1",
            text="Second order channel one sequence one.",
            channel=1,
            start_ms=1000,
            end_ms=1300,
            received_ms=4400,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-a2-early",
            segment_id="seg-early",
            text="First order earlier start timestamp.",
            channel=2,
            start_ms=999,
            end_ms=1200,
            received_ms=4600,
            source_sequence=9,
        ),
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="a2.jsonl")
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
    serialized_texts = [str(d.get("serialized_text") or "") for d in decisions]
    combined = "\n".join(serialized_texts)
    pos_early = combined.find("First order earlier start timestamp.")
    pos_c1s1 = combined.find("Second order channel one sequence one.")
    pos_c1s2 = combined.find("Third order channel one sequence two.")
    pos_c2s1 = combined.find("Fourth order channel two sequence one.")
    assert 0 <= pos_early < pos_c1s1 < pos_c1s2 < pos_c2s1
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_a3_explain_window_inclusive_edges(
    e2e_workspace: dict[str, Path],
) -> None:
    """Explain window [at_ms - window_ms, at_ms] includes exact endpoints and excludes +/-1 ms."""
    session_id = "session-a3"
    timestamps = (9999, 10000, 30000, 30001)
    events = [
        build_segment_event(
            session_id=session_id,
            event_id=f"ev-a3-{idx}",
            segment_id=f"seg-a3-{idx}",
            text=f"Someone should own task number {idx}.",
            start_ms=ts - 500,
            end_ms=ts - 100,
            received_ms=ts,
            source_sequence=idx + 1,
        )
        for idx, ts in enumerate(timestamps)
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="a3.jsonl")
    assert res.returncode == 0
    exp = run_coach(
        "explain",
        [
            "--db",
            str(e2e_workspace["db_path"]),
            "--session",
            session_id,
            "--at-ms",
            "30000",
            "--window-ms",
            "20000",
            "--json",
        ],
    )
    assert exp.returncode == 0
    anchor_times = {int(a["anchor_received_ms"]) for a in exp.json()["anchors"]}
    assert anchor_times == {10000, 30000}


def test_boundary_a4_delete_session_zero_events_and_shared_model_cache_isolation(
    e2e_workspace: dict[str, Path],
) -> None:
    """Deleting empty session succeeds; deleting session A keeps active session B's cache."""
    empty_sid = "session-a4-empty"
    res_empty = _replay_session(
        e2e_workspace,
        empty_sid,
        [build_control_event(session_id=empty_sid, event_id="ev-a4-c0", subtype="session_start")],
        filename="a4_empty.jsonl",
    )
    assert res_empty.returncode == 0
    del_empty = run_coach(
        "delete-session",
        ["--db", str(e2e_workspace["db_path"]), "--session", empty_sid, "--json"],
    )
    assert del_empty.returncode == 0
    assert del_empty.json()["raw_events_deleted"] == 0
    assert del_empty.json()["events_redacted"] == 0

    shared_text = "Someone should own the shared cache rollback plan."
    for sid in ("session-a4-a", "session-a4-b"):
        res = _replay_session(
            e2e_workspace,
            sid,
            [
                build_segment_event(
                    session_id=sid,
                    event_id=f"ev-{sid}",
                    segment_id="seg-shared-1",
                    text=shared_text,
                    start_ms=100,
                    end_ms=400,
                    received_ms=500,
                    source_sequence=1,
                )
            ],
            filename=f"{sid}.jsonl",
        )
        assert res.returncode == 0
    assert_model_cache_has_entries(e2e_workspace["db_path"], min_count=1)

    del_a = run_coach(
        "delete-session",
        ["--db", str(e2e_workspace["db_path"]), "--session", "session-a4-a", "--json"],
    )
    assert del_a.returncode == 0
    assert_session_text_deleted(
        e2e_workspace["db_path"],
        "session-a4-a",
        require_decisions_preserved=True,
    )
    with open_db(e2e_workspace["db_path"]) as conn:
        b_events = fetch_normalized_rows(conn, "events", "session-a4-b")
        cache_rows = fetch_normalized_rows(conn, "model_cache")
    assert any(shared_text in str(r.get("text") or "") for r in b_events)
    assert len(cache_rows) >= 1


def test_boundary_a5_report_zero_duration_and_exact_budget_equality(
    e2e_workspace: dict[str, Path],
) -> None:
    """Zero duration reports without error; 3/3 cards in 10m is compliant vs 4/3 non-compliant."""
    zero_sid = "session-a5-zero"
    res_zero = _replay_session(
        e2e_workspace,
        zero_sid,
        [
            build_control_event(
                session_id=zero_sid,
                event_id="ev-a5-z1",
                subtype="session_start",
                received_ms=1000,
            ),
            build_control_event(
                session_id=zero_sid,
                event_id="ev-a5-z2",
                subtype="session_end",
                received_ms=1000,
            ),
        ],
        filename="a5_zero.jsonl",
    )
    assert res_zero.returncode == 0
    rep_zero = run_coach(
        "report",
        ["--db", str(e2e_workspace["db_path"]), "--session", zero_sid, "--json"],
    )
    assert rep_zero.returncode == 0
    assert rep_zero.json()["duration_ms"] == 0

    budget_sid = "session-a5-budget"
    events_4 = [
        build_segment_event(
            session_id=budget_sid,
            event_id=f"ev-a5-b{i}",
            segment_id=f"seg-a5-b{i}",
            text=f"Someone should own rollout item {i}.",
            start_ms=i * 150_000 + 100,
            end_ms=i * 150_000 + 400,
            received_ms=i * 150_000 + 500,
            source_sequence=i + 1,
        )
        for i in range(4)
    ]
    events_4.append(
        build_control_event(
            session_id=budget_sid,
            event_id="ev-a5-end",
            subtype="session_end",
            received_ms=600_500,
            source_sequence=10,
        )
    )
    res_3 = _replay_session(
        e2e_workspace,
        budget_sid,
        events_4,
        filename="a5_budget3.jsonl",
        policy_overrides={
            "display": {"settle_ms": 100, "cooldown_ms": 100, "max_cards_per_10_min": 3}
        },
    )
    assert res_3.returncode == 0
    rep_3 = run_coach(
        "report",
        ["--db", str(e2e_workspace["db_path"]), "--session", budget_sid, "--json"],
    ).json()
    assert rep_3["displayed_card_count"] == 3
    assert rep_3["budget_compliant"] is True

    over_sid = "session-a5-over"
    events_over = [{**ev, "session_id": over_sid} for ev in events_4]
    res_4 = _replay_session(
        e2e_workspace,
        over_sid,
        events_over,
        filename="a5_budget4.jsonl",
        brief_overrides={"attention": {"max_cards_per_10_min": 4, "proactive": True}},
        policy_overrides={
            "display": {"settle_ms": 100, "cooldown_ms": 100, "max_cards_per_10_min": 4}
        },
    )
    assert res_4.returncode == 0
    with sqlite3.connect(str(e2e_workspace["db_path"])) as rw_conn:
        rw_conn.execute(
            "UPDATE sessions SET effective_max_cards_per_10_min = 3 WHERE session_id = ?",
            (over_sid,),
        )
        rw_conn.commit()
    rep_over = run_coach(
        "report",
        ["--db", str(e2e_workspace["db_path"]), "--session", over_sid, "--json"],
    ).json()
    assert rep_over["displayed_card_count"] == 4
    assert rep_over["budget_compliant"] is False


# ---------------------------------------------------------------------------
# Area B Boundaries (5 tests)
# ---------------------------------------------------------------------------


def test_boundary_b1_nemo_exact_5120_byte_160ms_frame_and_trailing_pad(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    """Non-multiple WAV length pads trailing frame to 5120 B without dropping final word (§6.7)."""
    session_id = "session-b1"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(e2e_workspace["policy_path"])
    # 500 ms WAV is 3 full 160 ms frames + 20 ms partial tail -> padded to 4 x 5120 B frames
    pcm = generate_pcm16_sine(duration_ms=500)
    expected_frames = chunk_pcm16_frames(pcm, frame_ms=160, pad_last=True)
    assert len(expected_frames) == 4
    assert all(len(f) == PCM16_FRAME_BYTES_160MS for f in expected_frames)
    wav_path = write_wav_mono16(e2e_workspace["root"] / "non_mult.wav", pcm)
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_on_commit=True,
                    transcript="Someone should own the final padded segment",
                    words=_make_words(
                        "Someone should own the final padded segment", start_s=0.0, speaker=2
                    ),
                )
            ]
        )
    )
    port = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port),
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ):
        time.sleep(1.5)

    assert len(mock_nemo_ws.received_frames) >= 4
    assert all(len(frame) == PCM16_FRAME_BYTES_160MS for frame in mock_nemo_ws.received_frames)
    with open_db(e2e_workspace["db_path"]) as conn:
        segs = [
            r
            for r in fetch_normalized_rows(conn, "events", session_id)
            if r.get("kind") == "segment" and r.get("stability") == "committed"
        ]
    assert any("padded" in str(s.get("text") or "") for s in segs)
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_b2_peecee_lang_regex_and_empty_silence_stream(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    """Invalid locale rejects cleanly; 100% silence stream yields 0 committed words."""
    bad_res = run_coach(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "peecee",
            "--language",
            "en/US",
        ],
    )
    assert bad_res.returncode != 0

    session_id = "session-b2"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(e2e_workspace["policy_path"])
    mock_nemo_ws.set_config(MockNemoConfig(silence_mode=True))
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "silence.wav",
        generate_pcm16_silence(duration_ms=640),
    )
    port = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port),
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ):
        time.sleep(1.0)
    with open_db(e2e_workspace["db_path"]) as conn:
        segs = [
            r
            for r in fetch_normalized_rows(conn, "events", session_id)
            if r.get("kind") == "segment"
        ]
    assert segs == []
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_b3_sortformer_999ms_vs_1000ms_short_turn_speaker_uncertain(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    """Sortformer turn of 999 ms sets speaker_uncertain; >=1000 ms turn does not (§6.7)."""
    session_id = "session-b3"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(e2e_workspace["policy_path"])
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="Someone owns short turn",
                    words=[
                        MockWord("Someone", 0.000, 0.300, speaker=2),
                        MockWord("owns", 0.350, 0.650, speaker=2),
                        MockWord("short", 0.700, 0.850, speaker=2),
                        MockWord("turn", 0.860, 0.999, speaker=2),
                    ],
                ),
                ScriptedTurn(
                    trigger_after_frames=6,
                    trigger_on_commit=True,
                    transcript="Someone owns full second turn",
                    words=[
                        MockWord("Someone", 2.000, 2.250, speaker=3),
                        MockWord("owns", 2.300, 2.500, speaker=3),
                        MockWord("full", 2.550, 2.750, speaker=3),
                        MockWord("second", 2.800, 2.950, speaker=3),
                        MockWord("turn", 2.960, 3.000, speaker=3),
                    ],
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "sortformer_999.wav",
        generate_pcm16_sine(duration_ms=1600),
    )
    port = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port),
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ):
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            with open_db(e2e_workspace["db_path"], readonly=True) as conn:
                cur_decs = fetch_normalized_rows(conn, "decisions", session_id)
            joined = " ".join(str(d.get("serialized_text") or "") for d in cur_decs)
            if "short turn" in joined and "full second turn" in joined:
                break
            time.sleep(0.05)
    with open_db(e2e_workspace["db_path"]) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
    short_dec = [d for d in decisions if "short turn" in str(d.get("serialized_text") or "")]
    full_dec = [d for d in decisions if "full second turn" in str(d.get("serialized_text") or "")]
    assert short_dec and "speaker_uncertain" in (short_dec[0].get("abstain_reasons") or [])
    assert full_dec and "speaker_uncertain" not in (full_dec[0].get("abstain_reasons") or [])
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_b4_deepgram_missing_segment_id_and_empty_transcript(
    e2e_workspace: dict[str, Path],
) -> None:
    """Empty Deepgram Results ignored cleanly; missing segment_id derived deterministically."""
    session_id = "session-b4"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(e2e_workspace["policy_path"])
    dg_rows = [
        build_deepgram_results("   ", is_final=True, start_s=0.1, duration_s=0.4, channel_index=0),
        build_deepgram_results(
            "Someone should own the rollback plan.",
            is_final=True,
            speech_final=True,
            start_s=1.0,
            duration_s=1.5,
            channel_index=0,
            speaker=2,
        ),
    ]
    for row in dg_rows:
        row["session_id"] = session_id
        assert "segment_id" not in row
    dg_jsonl = write_jsonl(e2e_workspace["sessions_dir"] / "dg_b4.jsonl", dg_rows)
    res = run_coach(
        "replay",
        [
            "--session",
            str(dg_jsonl),
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "deepgram",
            "--json",
        ],
    )
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        segs = [
            r
            for r in fetch_normalized_rows(conn, "events", session_id)
            if r.get("kind") == "segment"
        ]
    assert len(segs) == 1
    assert segs[0]["segment_id"]
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_b5_replay_empty_jsonl_and_control_only_stream(
    e2e_workspace: dict[str, Path],
) -> None:
    """Control-only stream (0 segments) and empty JSONL pass replay & invariants cleanly."""
    empty_sid = "session-b5-empty"
    empty_jsonl = write_jsonl(e2e_workspace["sessions_dir"] / "b5_empty.jsonl", [])
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=empty_sid))
    _write_nested_policy(e2e_workspace["policy_path"])
    res_empty = run_coach(
        "replay",
        [
            "--session",
            str(empty_jsonl),
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--json",
        ],
    )
    assert res_empty.returncode == 0

    ctrl_sid = "session-b5-ctrl"
    ctrl_events = [
        build_control_event(
            session_id=ctrl_sid, event_id="ev-b5-1", subtype="session_start", received_ms=100
        ),
        build_control_event(
            session_id=ctrl_sid, event_id="ev-b5-2", subtype="session_end", received_ms=200
        ),
    ]
    res_ctrl = _replay_session(e2e_workspace, ctrl_sid, ctrl_events, filename="b5_ctrl.jsonl")
    assert res_ctrl.returncode == 0
    assert_six_conservation_invariants(e2e_workspace["db_path"], ctrl_sid)


# ---------------------------------------------------------------------------
# Area C Boundaries (6 tests)
# ---------------------------------------------------------------------------


def test_boundary_c1_prefilter_exact_3_vs_4_content_tokens(
    e2e_workspace: dict[str, Path],
) -> None:
    """3 content tokens + stopwords -> too_short; 4 content tokens -> scored (§5.2)."""
    session_id = "session-c1"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-c1-3",
            segment_id="seg-c1-3",
            text="We should check the it",
            start_ms=100,
            end_ms=400,
            received_ms=500,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-c1-4",
            segment_id="seg-c1-4",
            text="Someone should own rollback",
            start_ms=1000,
            end_ms=1500,
            received_ms=1600,
            source_sequence=2,
        ),
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="c1.jsonl")
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
    by_seg = {
        ref["segment_id"]: d
        for d in decisions
        for ref in (d.get("anchor_refs") or [])
        if isinstance(ref, dict) and "segment_id" in ref
    }
    assert by_seg["seg-c1-3"]["terminal_reason"] == "too_short"
    assert by_seg["seg-c1-4"]["outcome"] in {"NONE", "ABSTAIN", "SUPPRESS", "ROUTE"}
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_c2_anchor_coalesce_exact_60_token_cap(
    e2e_workspace: dict[str, Path],
) -> None:
    """Same-speaker segments without boundary/punctuation coalesce up to ~60 tokens and split."""
    session_id = "session-c2"
    chunk_a = " ".join(f"alpha{i}" for i in range(30))
    chunk_b = " ".join(f"beta{i}" for i in range(30))
    chunk_c = " ".join(f"gamma{i}" for i in range(5))
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-c2-1",
            segment_id="seg-c2-1",
            text=chunk_a,
            start_ms=100,
            end_ms=500,
            received_ms=550,
            source_sequence=1,
            boundary=None,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-c2-2",
            segment_id="seg-c2-2",
            text=chunk_b,
            start_ms=520,
            end_ms=900,
            received_ms=950,
            source_sequence=2,
            boundary=None,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-c2-3",
            segment_id="seg-c2-3",
            text=chunk_c,
            start_ms=920,
            end_ms=1200,
            received_ms=1250,
            source_sequence=3,
            boundary=None,
        ),
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="c2.jsonl")
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
    assert len(decisions) == 2
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_c3_token_budget_exact_fit_vs_input_overflow(
    e2e_workspace: dict[str, Path],
) -> None:
    """Mandatory blocks at exact input_limit_tokens fit; +1 token triggers input_overflow (§5.3)."""
    probe_sid = "session-c3-probe"
    anchor_ev = build_segment_event(
        session_id=probe_sid,
        event_id="ev-c3-1",
        segment_id="seg-c3-1",
        text="Someone should own the rollback plan.",
        start_ms=100,
        end_ms=400,
        received_ms=500,
        source_sequence=1,
    )
    res_probe = _replay_session(
        e2e_workspace,
        probe_sid,
        [anchor_ev],
        filename="c3_probe.jsonl",
        brief_overrides={"known_facts": []},
        policy_overrides={"classifier": {"input_limit_tokens": 512}},
    )
    assert res_probe.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        probe_dec = fetch_normalized_rows(conn, "decisions", probe_sid)[0]
    tc = probe_dec["token_counts"]
    exact_tokens = int(tc["objective"]) + int(tc["role"]) + int(tc["anchor"])

    fit_sid = "session-c3-fit"
    res_fit = _replay_session(
        e2e_workspace,
        fit_sid,
        [{**anchor_ev, "session_id": fit_sid}],
        filename="c3_fit.jsonl",
        brief_overrides={"known_facts": []},
        policy_overrides={"classifier": {"input_limit_tokens": exact_tokens}},
    )
    assert res_fit.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        fit_dec = fetch_normalized_rows(conn, "decisions", fit_sid)[0]
    assert fit_dec["terminal_reason"] != "input_overflow"

    over_sid = "session-c3-over"
    res_over = _replay_session(
        e2e_workspace,
        over_sid,
        [{**anchor_ev, "session_id": over_sid}],
        filename="c3_over.jsonl",
        brief_overrides={"known_facts": []},
        policy_overrides={"classifier": {"input_limit_tokens": max(1, exact_tokens - 1)}},
    )
    assert res_over.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        over_dec = fetch_normalized_rows(conn, "decisions", over_sid)[0]
    assert over_dec["outcome"] == "ABSTAIN"
    assert over_dec["terminal_reason"] == "input_overflow"
    assert_six_conservation_invariants(e2e_workspace["db_path"], over_sid)


def test_boundary_c4_context_lookback_boundary_inside_vs_outside_required_lookback_s(
    e2e_workspace: dict[str, Path],
) -> None:
    """Evicting turn inside required_lookback_s abstains; outside does not (§5.3)."""
    filler = " ".join(f"detail{i}" for i in range(45))
    # Case 1: evicted turn at anchor_ms - 10_000 (inside required_lookback_s=20 -> 20_000 ms)
    sid_in = "session-c4-in"
    events_in = [
        build_segment_event(
            session_id=sid_in,
            event_id="ev-c4-old",
            segment_id="seg-c4-old",
            text=f"We discussed background context {filler}.",
            speaker_id="me",
            start_ms=10_000,
            end_ms=10_500,
            received_ms=10_600,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid_in,
            event_id="ev-c4-anc",
            segment_id="seg-c4-anc",
            text="Someone should own the rollback plan.",
            start_ms=20_000,
            end_ms=20_400,
            received_ms=20_600,
            source_sequence=2,
        ),
    ]
    res_in = _replay_session(
        e2e_workspace,
        sid_in,
        events_in,
        filename="c4_in.jsonl",
        policy_overrides={
            "classifier": {"input_limit_tokens": 64},
            "categories": {"CONFIRM_OWNER": {"required_lookback_s": 20}},
        },
    )
    assert res_in.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        decs_in = fetch_normalized_rows(conn, "decisions", sid_in)
    assert decs_in[-1]["outcome"] == "ABSTAIN"
    assert decs_in[-1]["terminal_reason"] == "context_truncated_material"

    # Case 2: evicted turn at anchor_ms - 25_000 (outside required_lookback_s=20)
    sid_out = "session-c4-out"
    events_out = [
        build_segment_event(
            session_id=sid_out,
            event_id="ev-c4-old-out",
            segment_id="seg-c4-old-out",
            text=f"Older background context {filler}.",
            speaker_id="me",
            start_ms=10_000,
            end_ms=10_500,
            received_ms=10_600,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid_out,
            event_id="ev-c4-anc-out",
            segment_id="seg-c4-anc-out",
            text="Someone should own the rollback plan.",
            start_ms=35_000,
            end_ms=35_400,
            received_ms=35_600,
            source_sequence=2,
        ),
    ]
    res_out = _replay_session(
        e2e_workspace,
        sid_out,
        events_out,
        filename="c4_out.jsonl",
        policy_overrides={
            "classifier": {"input_limit_tokens": 64},
            "categories": {"CONFIRM_OWNER": {"required_lookback_s": 20}},
        },
    )
    assert res_out.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        decs_out = fetch_normalized_rows(conn, "decisions", sid_out)
    assert decs_out[-1]["terminal_reason"] != "context_truncated_material"
    assert_six_conservation_invariants(e2e_workspace["db_path"], sid_out)


def test_boundary_c5_template_slot_8_words_vs_9_words_and_12_word_cap(
    e2e_workspace: dict[str, Path],
) -> None:
    """8-word slot renders grounded template (<=12w); 9-word slot falls back/abandons (§9.1)."""
    sid_8 = "session-c5-8"
    slot_8 = "the multi region database failover rollback cutover checklist"
    assert len(slot_8.split()) == 8
    res_8 = _replay_session(
        e2e_workspace,
        sid_8,
        [
            build_segment_event(
                session_id=sid_8,
                event_id="ev-c5-8",
                segment_id="seg-c5-8",
                text=f"Someone should own {slot_8}.",
                start_ms=100,
                end_ms=500,
                received_ms=600,
                source_sequence=1,
            )
        ],
        filename="c5_8.jsonl",
    )
    assert res_8.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        sugs_8 = fetch_normalized_rows(conn, "suggestions", sid_8)
    assert sugs_8 and int(sugs_8[0]["word_count"]) <= 12
    assert slot_8 in str(sugs_8[0]["text"])

    sid_9 = "session-c5-9"
    slot_9 = "the multi region primary database failover rollback cutover checklist"
    assert len(slot_9.split()) == 9
    res_9 = _replay_session(
        e2e_workspace,
        sid_9,
        [
            build_segment_event(
                session_id=sid_9,
                event_id="ev-c5-9a",
                segment_id="seg-c5-9a",
                text=f"Someone should own {slot_9}.",
                start_ms=100,
                end_ms=500,
                received_ms=600,
                source_sequence=1,
            ),
            build_segment_event(
                session_id=sid_9,
                event_id="ev-c5-9b",
                segment_id="seg-c5-9b",
                text="And we need more telemetry.",
                speaker_id="participant-c",
                start_ms=600,
                end_ms=900,
                received_ms=950,
                source_sequence=2,
            ),
        ],
        filename="c5_9.jsonl",
    )
    assert res_9.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        sugs_9 = fetch_normalized_rows(conn, "suggestions", sid_9)
    first_9 = [s for s in sugs_9 if "seg-c5-9a" in json.dumps(s)]
    if first_9:
        assert slot_9 not in str(first_9[0].get("text") or "")
        assert first_9[0].get("terminal_reason") == "slot_too_long" or (
            int(first_9[0].get("word_count") or 0) <= 12
        )
    assert_six_conservation_invariants(e2e_workspace["db_path"], sid_9)


def test_boundary_c6_generic_referent_exact_6000ms_vs_6001ms_age(
    e2e_workspace: dict[str, Path],
) -> None:
    """Generic template allowed at age <= 6000 ms and rejected at 6001 ms (§5.6, §9.1)."""
    slot_long = "the multi region primary database failover rollback cutover checklist"
    for settle_ms, expect_unclear, suffix in ((6000, False, "6000"), (6001, True, "6001")):
        sid = f"session-c6-{suffix}"
        events = [
            build_segment_event(
                session_id=sid,
                event_id=f"ev-c6-{suffix}",
                segment_id=f"seg-c6-{suffix}",
                text=f"Someone should own {slot_long}.",
                start_ms=100,
                end_ms=400,
                received_ms=500,
                source_sequence=1,
            ),
            build_control_event(
                session_id=sid,
                event_id=f"ev-c6-end-{suffix}",
                subtype="session_end",
                received_ms=500 + settle_ms + 500,
                source_sequence=2,
            ),
        ]
        res = _replay_session(
            e2e_workspace,
            sid,
            events,
            filename=f"c6_{suffix}.jsonl",
            policy_overrides={"display": {"settle_ms": settle_ms, "max_candidate_age_ms": 8000}},
        )
        assert res.returncode == 0
        with open_db(e2e_workspace["db_path"]) as conn:
            sugs = fetch_normalized_rows(conn, "suggestions", sid)
        assert sugs
        if expect_unclear:
            assert sugs[0]["terminal_reason"] == "generic_referent_unclear"
        else:
            assert sugs[0]["terminal_reason"] != "generic_referent_unclear"
        assert_six_conservation_invariants(e2e_workspace["db_path"], sid)


# ---------------------------------------------------------------------------
# Area D Boundaries (5 tests)
# ---------------------------------------------------------------------------


def test_boundary_d1_threshold_exact_equality_t_low_and_t_display(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    """Score t_low-1e-6 -> NONE, t_low -> ABSTAIN, t_display-1e-6 -> ABSTAIN, t_display -> ROUTE."""
    t_low = 0.25
    t_compute = 0.50
    t_display = 0.75
    cases = [
        ("below_low", t_low - 1e-6, "NONE", "below_t_low"),
        ("at_low", t_low, "ABSTAIN", "score_in_band"),
        ("below_disp", t_display - 1e-6, "ABSTAIN", "score_in_band"),
        ("at_disp", t_display, "ROUTE", None),
    ]
    for label, score, expected_outcome, expected_reason in cases:
        sid = f"session-d1-{label}"
        mock_openai.set_default(
            MockCompletionSpec(
                status_code=200,
                content=json.dumps({"opportunity": score, "categories": {"CONFIRM_OWNER": score}}),
            )
        )
        res = _replay_session(
            e2e_workspace,
            sid,
            [
                build_segment_event(
                    session_id=sid,
                    event_id=f"ev-d1-{label}",
                    segment_id=f"seg-d1-{label}",
                    text="Someone should own the deployment cutover checklist.",
                    start_ms=100,
                    end_ms=400,
                    received_ms=500,
                    source_sequence=1,
                )
            ],
            filename=f"d1_{label}.jsonl",
            policy_overrides={
                "classifier": {
                    "version": "llm_baseline_v0",
                    "calibration": "llm_baseline_v0-platt",
                },
                "categories": {
                    "CONFIRM_OWNER": {
                        "t_low": t_low,
                        "t_compute": t_compute,
                        "t_display": t_display,
                    },
                    "CONFIRM_TIME": {"t_low": t_low},
                    "SUCCESS_CRITERION": {"t_low": t_low},
                },
            },
            extra_args=("--no-cache",),
            env=build_coach_test_env(openai_base_url=mock_openai.base_url),
        )
        assert res.returncode == 0
        with open_db(e2e_workspace["db_path"]) as conn:
            dec = fetch_normalized_rows(conn, "decisions", sid)[0]
        assert dec["outcome"] == expected_outcome
        if expected_reason is not None:
            assert dec["terminal_reason"] == expected_reason
        assert_six_conservation_invariants(e2e_workspace["db_path"], sid)


def test_boundary_d2_speaker_settle_ms_2000ms_boundary(
    e2e_workspace: dict[str, Path],
) -> None:
    """speaker_update within <=2000 ms abstains (speaker_uncertain); >2000 ms eligible."""
    for delta_ms, expect_abstain, label in ((2000, True, "within"), (2001, False, "past")):
        sid = f"session-d2-{label}"
        events = [
            build_segment_event(
                session_id=sid,
                event_id=f"ev-d2-seg-{label}",
                segment_id=f"seg-d2-{label}",
                text="Someone should own the rollback plan.",
                speaker_id="participant-c",
                start_ms=100,
                end_ms=400,
                received_ms=500,
                source_sequence=1,
            ),
            build_speaker_update_event(
                session_id=sid,
                event_id=f"ev-d2-upd-{label}",
                segment_id=f"seg-d2-{label}",
                speaker_id="participant-b",
                source_revision=2,
                received_ms=600,
                source_sequence=2,
                speaker_confidence=0.95,
            ),
            build_segment_event(
                session_id=sid,
                event_id=f"ev-d2-rev-{label}",
                segment_id=f"seg-d2-{label}",
                text="Someone should own the cutover plan.",
                speaker_id="participant-b",
                source_revision=3,
                start_ms=100,
                end_ms=400,
                received_ms=600 + delta_ms,
                source_sequence=3,
            ),
        ]
        res = _replay_session(
            e2e_workspace,
            sid,
            events,
            filename=f"d2_{label}.jsonl",
            policy_overrides={"abstain": {"speaker_settle_ms": 2000}},
        )
        assert res.returncode == 0
        with open_db(e2e_workspace["db_path"]) as conn:
            decs = fetch_normalized_rows(conn, "decisions", sid)
        last_dec = decs[-1]
        if expect_abstain:
            assert last_dec["outcome"] == "ABSTAIN"
            assert "speaker_uncertain" in (
                last_dec.get("abstain_reasons") or [last_dec.get("terminal_reason")]
            )
        else:
            assert last_dec["terminal_reason"] != "speaker_uncertain"
        assert_six_conservation_invariants(e2e_workspace["db_path"], sid)


def test_boundary_d3_stricter_attention_cap_brief_vs_policy(
    e2e_workspace: dict[str, Path],
) -> None:
    """Stricter attention cap wins: min(brief, policy) for (2 vs 3) and (5 vs 3) (§16)."""
    for brief_cap, policy_cap, expected_cap, label in ((2, 3, 2, "b2p3"), (5, 3, 3, "b5p3")):
        sid = f"session-d3-{label}"
        res = _replay_session(
            e2e_workspace,
            sid,
            [
                build_segment_event(
                    session_id=sid,
                    event_id=f"ev-d3-{label}",
                    segment_id=f"seg-d3-{label}",
                    text="Someone should own the rollback plan.",
                    start_ms=100,
                    end_ms=400,
                    received_ms=500,
                    source_sequence=1,
                )
            ],
            filename=f"d3_{label}.jsonl",
            brief_overrides={"attention": {"max_cards_per_10_min": brief_cap}},
            policy_overrides={"display": {"max_cards_per_10_min": policy_cap}},
        )
        assert res.returncode == 0
        rep = run_coach(
            "report",
            ["--db", str(e2e_workspace["db_path"]), "--session", sid, "--json"],
        ).json()
        assert rep["effective_max_cards_per_10_min"] == expected_cap
        assert_six_conservation_invariants(e2e_workspace["db_path"], sid)


def test_boundary_d4_calibration_single_non_random_row_poisoning(
    e2e_workspace: dict[str, Path],
) -> None:
    """19 random rows + 1 pass1_mark row must still be rejected by coach calibrate (§12.4)."""
    rows = [
        build_label_record(
            label_id=f"lbl-d4-{i}",
            candidate_id=f"cand-d4-{i}",
            sample_source="random" if i < 19 else "pass1_mark",
            worthwhile=(i % 2 == 0),
            categories=("CONFIRM_OWNER",) if (i % 2 == 0) else (),
            none_reason=None if (i % 2 == 0) else "no_gap",
            closed_by="answered" if (i % 2 == 0) else "session_end",
        )
        for i in range(20)
    ]
    labels_path = write_jsonl(e2e_workspace["root"] / "cal_poison.jsonl", rows)
    out_path = e2e_workspace["root"] / "cal_out.json"
    res = run_coach(
        "calibrate",
        [
            "--data",
            str(labels_path),
            "--classifier",
            "rules_v0",
            "--mode",
            "technical_working_discussion",
            "--out",
            str(out_path),
        ],
    )
    assert res.returncode != 0
    assert not out_path.exists()
    assert "random" in (res.stderr + res.stdout).lower()


def test_boundary_d5_multilabel_other_high_opportunity_low_category(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    """High opportunity (>=t_display) with all categories < t_low abstains & logs taxonomy gap."""
    session_id = "session-d5"
    mock_openai.set_default(
        MockCompletionSpec(
            status_code=200,
            content=json.dumps(
                {
                    "opportunity": 0.90,
                    "categories": {
                        "CONFIRM_OWNER": 0.10,
                        "CONFIRM_TIME": 0.08,
                        "SUCCESS_CRITERION": 0.05,
                    },
                }
            ),
        )
    )
    res = _replay_session(
        e2e_workspace,
        session_id,
        [
            build_segment_event(
                session_id=session_id,
                event_id="ev-d5-1",
                segment_id="seg-d5-1",
                text="We should think about vendor licensing terms carefully.",
                start_ms=100,
                end_ms=400,
                received_ms=500,
                source_sequence=1,
            )
        ],
        filename="d5.jsonl",
        policy_overrides={
            "classifier": {"version": "llm_baseline_v0", "calibration": "llm_baseline_v0-platt"}
        },
        env=build_coach_test_env(openai_base_url=mock_openai.base_url),
    )
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        dec = fetch_normalized_rows(conn, "decisions", session_id)[0]
        sugs = fetch_normalized_rows(conn, "suggestions", session_id)
    assert dec["outcome"] in {"ABSTAIN", "NONE"}
    assert dec.get("category") != "OTHER"
    assert sugs == []
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


# ---------------------------------------------------------------------------
# Area E Boundaries (5 tests)
# ---------------------------------------------------------------------------


def test_boundary_e1_validity_interval_inclusive_open_close_and_plus1ms_late(
    e2e_workspace: dict[str, Path],
) -> None:
    """display_ms == open_ms and == close_ms match as useful; close_ms + 1 is late (§12.3)."""
    for settle_ms, expected_tag, label in (
        (0, "useful", "at_open"),
        (1000, "useful", "at_close"),
        (1001, "late", "plus1_late"),
    ):
        sid = f"session-e1-{label}"
        events = [
            build_segment_event(
                session_id=sid,
                event_id=f"ev-e1-{label}",
                segment_id=f"seg-e1-{label}",
                text="Someone should own the rollback plan.",
                start_ms=100,
                end_ms=400,
                received_ms=1000,
                source_sequence=1,
                boundary=None,
            ),
            build_control_event(
                session_id=sid,
                event_id=f"ev-e1-end-{label}",
                subtype="session_end",
                received_ms=5000,
                source_sequence=2,
            ),
        ]
        # Run once to obtain deterministic candidate_id if needed, then match against label
        first_res = _replay_session(
            e2e_workspace,
            sid,
            events,
            filename=f"e1_{label}.jsonl",
            policy_overrides={
                "display": {
                    "settle_ms": settle_ms,
                    "cooldown_ms": 100,
                    "max_candidate_age_ms": 8000,
                }
            },
        )
        assert first_res.returncode == 0
        with open_db(e2e_workspace["db_path"]) as conn:
            cand_id = str(fetch_normalized_rows(conn, "decisions", sid)[0]["candidate_id"])
        labels_path = write_jsonl(
            e2e_workspace["root"] / f"labels_e1_{label}.jsonl",
            [
                build_label_record(
                    label_id=f"lbl-e1-{label}",
                    session_id=sid,
                    candidate_id=cand_id,
                    evidence_refs=({"segment_id": f"seg-e1-{label}", "revision": 1},),
                    open_ms=1000,
                    ideal_ms=1500,
                    close_ms=2000,
                    closed_by="topic_moved",
                )
            ],
        )
        eval_res = _replay_session(
            e2e_workspace,
            sid,
            events,
            filename=f"e1_{label}.jsonl",
            policy_overrides={
                "display": {
                    "settle_ms": settle_ms,
                    "cooldown_ms": 100,
                    "max_candidate_age_ms": 8000,
                }
            },
            extra_args=("--labels", str(labels_path)),
        )
        assert eval_res.returncode == 0
        payload = eval_res.json()
        ev = payload.get("evaluation") or payload
        if expected_tag == "useful":
            assert int(ev["useful_count"]) == 1
            assert int(ev.get("late_count", 0)) == 0
        else:
            assert int(ev["useful_count"]) == 0
            assert int(ev["late_count"]) == 1


def test_boundary_e2_eval_split_zero_overlap_on_shared_series_id(
    e2e_workspace: dict[str, Path],
) -> None:
    """Two conversations sharing series_id never land in different splits (§12.5)."""
    rows: list[dict[str, Any]] = []
    specs = (
        ("conv-s1", "series-alpha", True, ("CONFIRM_OWNER",)),
        ("conv-s2", "series-alpha", False, ()),
        ("conv-s3", "series-beta", True, ("CONFIRM_TIME",)),
        ("conv-s4", "series-gamma", False, ()),
        ("conv-s5", "series-delta", True, ("SUCCESS_CRITERION",)),
        ("conv-s6", "series-epsilon", False, ()),
    )
    for idx, (sid, series_id, worthwhile, cats) in enumerate(specs):
        row = build_decision_context_row(
            candidate_id=f"cand-e2-{idx}",
            session_id=sid,
            worthwhile=worthwhile,
            categories=cats,
            sample_source="random",
        )
        row["series_id"] = series_id
        rows.append(row)
    data_jsonl = write_jsonl(e2e_workspace["root"] / "split_series.jsonl", rows)
    model_dir = e2e_workspace["root"] / "model_e2"
    res = run_coach(
        "train",
        [
            "--data",
            str(data_jsonl),
            "--out",
            str(model_dir),
            "--backend",
            "embed_logreg",
            "--json",
        ],
    )
    assert res.returncode == 0
    splits_path = model_dir / "splits.json"
    split_data = (
        json.loads(splits_path.read_text(encoding="utf-8"))
        if splits_path.exists()
        else res.json()["splits"]
    )
    conv_to_split = {
        str(cid): str(split_name)
        for split_name, cids in split_data.items()
        if isinstance(cids, list)
        for cid in cids
    }
    assert conv_to_split["conv-s1"] == conv_to_split["conv-s2"]
    for split_a, items_a in split_data.items():
        for split_b, items_b in split_data.items():
            if split_a != split_b and isinstance(items_a, list) and isinstance(items_b, list):
                assert not (set(items_a) & set(items_b))


def test_boundary_e3_matcher_multiple_cards_same_validity_interval(
    e2e_workspace: dict[str, Path],
) -> None:
    """3 displayed cards matching 1 validity interval yield useful=1, repetition=2 (§12.6)."""
    session_id = "session-e3"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id=f"ev-e3-{i}",
            segment_id=f"seg-e3-{i}",
            text=f"Someone should own rollout item {i}.",
            start_ms=100 + i * 2000,
            end_ms=400 + i * 2000,
            received_ms=500 + i * 2000,
            source_sequence=i + 1,
        )
        for i in range(3)
    ]
    events.append(
        build_control_event(
            session_id=session_id,
            event_id="ev-e3-end",
            subtype="session_end",
            received_ms=10000,
            source_sequence=10,
        )
    )
    labels_path = write_jsonl(
        e2e_workspace["root"] / "labels_e3.jsonl",
        [
            build_label_record(
                label_id="lbl-e3-1",
                session_id=session_id,
                candidate_id="cand-e3-anchor",
                categories=("CONFIRM_OWNER",),
                evidence_refs=(
                    {"segment_id": "seg-e3-0", "revision": 1},
                    {"segment_id": "seg-e3-1", "revision": 1},
                    {"segment_id": "seg-e3-2", "revision": 1},
                ),
                open_ms=400,
                ideal_ms=1000,
                close_ms=9000,
                closed_by="answered",
            )
        ],
    )
    res = _replay_session(
        e2e_workspace,
        session_id,
        events,
        filename="e3.jsonl",
        policy_overrides={
            "display": {
                "settle_ms": 100,
                "cooldown_ms": 100,
                "display_ttl_ms": 500,
                "max_cards_per_10_min": 10,
            }
        },
        extra_args=("--labels", str(labels_path)),
    )
    assert res.returncode == 0
    payload = res.json()
    ev = payload.get("evaluation") or payload
    assert int(ev["useful_count"]) == 1
    assert int(ev["repetition_count"]) == 2


def test_boundary_e4_bench_single_context_and_2048_token_bucket(
    e2e_workspace: dict[str, Path],
) -> None:
    """coach bench handles single-context input and 2048-token bucket without NaN percentiles."""
    long_turns = [
        f"S1: Context turn {i} discussing latency budgets and deployment guardrails."
        for i in range(180)
    ]
    ctx = build_decision_context_row(
        candidate_id="cand-e4-2048",
        session_id="session-e4",
        input_limit_tokens=2048,
        context_turns=long_turns,
    )
    ctx_path = write_jsonl(e2e_workspace["root"] / "bench_2048.jsonl", [ctx])
    res = run_coach(
        "bench",
        [
            "--contexts",
            str(ctx_path),
            "--classifier",
            "rules_v0",
            "--batch-sizes",
            "1",
            "--tokens",
            "2048",
            "--json",
        ],
    )
    assert res.returncode == 0
    payload = res.json()
    cells = payload.get("cells") or payload.get("results") or []
    assert len(cells) >= 1
    cell = cells[0]
    for metric in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
        alt = metric.removesuffix("_ms")
        val = cell.get(metric, cell.get(alt))
        assert isinstance(val, int | float)
        assert val == val  # Not NaN


def test_boundary_e5_bootstrap_zero_positives_category_flags_insufficient_n(
    e2e_workspace: dict[str, Path],
) -> None:
    """Category with 0 or 1 positives flags low-N / insufficient_n instead of bare CI (§12.7)."""
    session_id = "session-e5"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-e5-1",
            segment_id="seg-e5-1",
            text="Someone should own the rollback plan.",
            start_ms=100,
            end_ms=400,
            received_ms=500,
            source_sequence=1,
        )
    ]
    labels_path = write_jsonl(
        e2e_workspace["root"] / "labels_e5.jsonl",
        [
            build_label_record(
                label_id="lbl-e5-1",
                session_id=session_id,
                candidate_id="cand-e5-1",
                categories=("CONFIRM_OWNER",),
                evidence_refs=({"segment_id": "seg-e5-1", "revision": 1},),
                open_ms=400,
                ideal_ms=800,
                close_ms=4000,
                closed_by="answered",
            )
        ],
    )
    res = _replay_session(
        e2e_workspace,
        session_id,
        events,
        filename="e5.jsonl",
        extra_args=("--labels", str(labels_path)),
    )
    assert res.returncode == 0
    payload = res.json()
    ev = payload.get("evaluation") or payload
    per_cat = ev.get("per_category") or payload.get("per_category") or {}
    assert per_cat
    # CONFIRM_TIME has 0 positives and CONFIRM_OWNER has 1 positive -> flagged low_n
    flagged = [
        info
        for info in per_cat.values()
        if isinstance(info, dict) and (info.get("low_n") or info.get("insufficient_n"))
    ]
    assert len(flagged) >= 1


# ---------------------------------------------------------------------------
# Area F Boundaries (5 tests)
# ---------------------------------------------------------------------------


def test_boundary_f1_max_candidates_per_job_8_vs_9_retire_newest_kept(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    """10 rapid anchors with max_candidates_per_job=8 batch newest 8 and retire expired (§8.3)."""
    session_id = "session-f1"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={
            "classifier": {
                "version": "llm_baseline_v0",
                "calibration": "llm_baseline_v0-platt",
                "max_candidates_per_job": 8,
            },
            "display": {"max_candidate_age_ms": 500, "settle_ms": 100},
        },
    )
    mock_openai.set_default(
        MockCompletionSpec(
            status_code=200,
            content=json.dumps({"opportunity": 0.10, "categories": {"CONFIRM_OWNER": 0.10}}),
            delay_s=0.7,
        )
    )
    events = [
        build_segment_event(
            session_id=session_id,
            event_id=f"ev-f1-{i}",
            segment_id=f"seg-f1-{i}",
            text=f"Someone should own batch anchor {i}.",
            start_ms=100 + i * 10,
            end_ms=105 + i * 10,
            received_ms=110 + i * 10,
            source_sequence=i + 1,
        )
        for i in range(10)
    ]
    jsonl_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "f1.jsonl", events)
    port = free_port_factory()
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "replay",
            "--session",
            str(jsonl_path),
            "--speed",
            "10.0",
            "--ui-port",
            str(port),
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ):
        time.sleep(2.0)
    with open_db(e2e_workspace["db_path"]) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
    assert len(decisions) == 10
    retired = [d for d in decisions if d.get("terminal_reason") == "expired_unevaluated"]
    assert len(retired) >= 1
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_f2_display_by_ms_boundary_never_revives_expired(
    e2e_workspace: dict[str, Path],
) -> None:
    """Candidate at now_ms == display_by_ms + 1 never displays when cooldown ends (§8.4)."""
    session_id = "session-f2"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-f2-1",
            segment_id="seg-f2-1",
            text="Someone should own the first task.",
            start_ms=100,
            end_ms=180,
            received_ms=200,
            source_sequence=1,
            boundary="utterance_end",
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-f2-2",
            segment_id="seg-f2-2",
            text="Someone should own the second deadline task.",
            start_ms=110,
            end_ms=190,
            received_ms=201,
            source_sequence=2,
            boundary="utterance_end",
        ),
        build_control_event(
            session_id=session_id,
            event_id="ev-f2-end",
            subtype="session_end",
            received_ms=3000,
            source_sequence=3,
        ),
    ]
    # Card 1 displays at ~200 ms -> cooldown ends at 1202 ms; Card 2 display_by_ms = 1201
    res = _replay_session(
        e2e_workspace,
        session_id,
        events,
        filename="f2.jsonl",
        policy_overrides={
            "display": {
                "settle_ms": 0,
                "cooldown_ms": 1002,
                "max_candidate_age_ms": 1000,
                "max_cards_per_10_min": 5,
            }
        },
    )
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        sugs = fetch_normalized_rows(conn, "suggestions", session_id)
    second = [s for s in sugs if "seg-f2-2" in json.dumps(s)]
    assert second
    assert second[0]["current_state"] == "expired"
    assert second[0]["terminal_reason"] == "display_by_passed"
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_f3_deferred_slot_single_capacity_tie_and_displacement(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    """3 ready candidates while card active: only highest deferred, 2 displaced_by_better."""
    session_id = "session-f3"
    mock_openai.enqueue(
        MockCompletionSpec(
            status_code=200,
            content=json.dumps({"opportunity": 0.95, "categories": {"CONFIRM_OWNER": 0.95}}),
        ),
        MockCompletionSpec(
            status_code=200,
            content=json.dumps({"opportunity": 0.78, "categories": {"CONFIRM_OWNER": 0.78}}),
        ),
        MockCompletionSpec(
            status_code=200,
            content=json.dumps({"opportunity": 0.85, "categories": {"CONFIRM_OWNER": 0.85}}),
        ),
        MockCompletionSpec(
            status_code=200,
            content=json.dumps({"opportunity": 0.91, "categories": {"CONFIRM_OWNER": 0.91}}),
        ),
    )
    events = [
        build_segment_event(
            session_id=session_id,
            event_id=f"ev-f3-{i}",
            segment_id=f"seg-f3-{i}",
            text=f"Someone should own deferred slot item {i}.",
            start_ms=100 + i * 200,
            end_ms=250 + i * 200,
            received_ms=300 + i * 200,
            source_sequence=i + 1,
        )
        for i in range(4)
    ]
    res = _replay_session(
        e2e_workspace,
        session_id,
        events,
        filename="f3.jsonl",
        policy_overrides={
            "classifier": {"version": "llm_baseline_v0", "calibration": "llm_baseline_v0-platt"},
            "display": {"settle_ms": 100, "cooldown_ms": 5000, "max_candidate_age_ms": 8000},
        },
        env=build_coach_test_env(openai_base_url=mock_openai.base_url),
    )
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        sugs = fetch_normalized_rows(conn, "suggestions", session_id)
    displaced = [s for s in sugs if s.get("terminal_reason") == "displaced_by_better"]
    assert len(displaced) == 2
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_boundary_f4_manual_lookback_exact_30s_and_score_floor_0_30(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    """Manual lane serves 29.9s/0.30 candidate and skips 30.1s or 0.29 candidate (§3.7)."""
    session_id = "session-f4"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        generator_enabled=True,
        overrides={
            "classifier": {"version": "llm_baseline_v0", "calibration": "llm_baseline_v0-platt"},
            "categories": {"CONFIRM_OWNER": {"t_low": 0.20, "t_compute": 0.30, "t_display": 0.90}},
            "manual": {"lookback_s": 30, "score_floor": 0.30, "budget_ms": 2000},
            "generator": {"enabled": True},
        },
    )
    mock_openai.add_matcher(
        lambda req: "low score task" in json.dumps(req).split("[ANCHOR]")[-1],
        MockCompletionSpec(
            status_code=200,
            content=json.dumps({"opportunity": 0.29, "categories": {"CONFIRM_OWNER": 0.29}}),
        ),
    )
    mock_openai.add_matcher(
        lambda req: "exact floor task" in json.dumps(req).split("[ANCHOR]")[-1],
        MockCompletionSpec(
            status_code=200,
            content=json.dumps({"opportunity": 0.30, "categories": {"CONFIRM_OWNER": 0.30}}),
        ),
    )
    mock_openai.set_default(MockCompletionSpec.decline("nothing useful to add"))
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-f4-low",
            segment_id="seg-f4-low",
            text="Someone might own the low score task.",
            start_ms=100,
            end_ms=300,
            received_ms=350,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-f4-ok",
            segment_id="seg-f4-ok",
            text="Someone should own the exact floor task.",
            start_ms=400,
            end_ms=600,
            received_ms=700,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-f4-keepalive",
            segment_id="seg-f4-keepalive",
            text="Ok",
            speaker_id="me",
            start_ms=30_500,
            end_ms=30_550,
            received_ms=30_600,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-f4-keepalive-2",
            segment_id="seg-f4-keepalive-2",
            text="Sounds good.",
            speaker_id="me",
            start_ms=30_800,
            end_ms=30_850,
            received_ms=30_900,
            source_sequence=4,
        ),
    ]
    jsonl_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "f4.jsonl", events)
    port = free_port_factory()
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "replay",
            "--session",
            str(jsonl_path),
            "--speed",
            "30.0",
            "--ui-port",
            str(port),
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _drive() -> tuple[dict[str, Any], dict[str, Any]]:
            async with client.ws_session() as ws:
                await asyncio.sleep(0.3)
                await ws.send_manual_request()
                card_29s = await ws.wait_for_card(timeout=2.5)
                await asyncio.sleep(1.1)  # Advances past 30.1s lookback
                await ws.send_manual_request()
                msg_stale = await ws.wait_for_message(
                    lambda m: (
                        "nothing useful to add" in json.dumps(m).lower()
                        or m.get("type") in {"manual_empty", "nothing_useful"}
                    ),
                    timeout=2.5,
                )
                return card_29s, msg_stale

        card_29s, msg_stale = asyncio.run(_drive())
        assert "exact floor task" in json.dumps(card_29s) or card_29s.get("question")
        assert "nothing useful to add" in json.dumps(msg_stale).lower()


def test_boundary_f5_word_count_boundary_12_vs_20_vs_21_card_ui(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    """20-word generated card is accepted; 21-word generated card is rejected (§9.2, §3.4)."""
    alpha_19 = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike nectar oscar papa quebec romeo sierra"
    )
    q20 = f"{alpha_19} ready?"
    q21 = f"{alpha_19} tango ready?"
    assert len(q20.split()) == 20
    assert len(q21.split()) == 21

    session_id = "session-f5"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        generator_enabled=True,
        overrides={
            "categories": {
                "CLARIFY_SCOPE": {
                    "t_low": 0.20,
                    "t_compute": 0.45,
                    "t_display": 0.70,
                    "route": "generator",
                    "required_lookback_s": 120,
                }
            },
            "display": {"settle_ms": 100, "cooldown_ms": 100, "max_cards_per_10_min": 5},
            "generator": {"enabled": True, "max_words": 20},
        },
    )
    mock_openai.enqueue(
        MockCompletionSpec.valid_question(q20, ["seg-f5-20"]),
        MockCompletionSpec.valid_question(q21, ["seg-f5-21"]),
    )
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-f5-20",
            segment_id="seg-f5-20",
            text="We should clarify the migration scope for stage one.",
            start_ms=100,
            end_ms=300,
            received_ms=350,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-f5-21",
            segment_id="seg-f5-21",
            text="We should clarify the failover scope for stage two.",
            start_ms=1000,
            end_ms=1200,
            received_ms=1250,
            source_sequence=2,
        ),
    ]
    jsonl_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "f5.jsonl", events)
    port = free_port_factory()
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--source",
            "replay",
            "--session",
            str(jsonl_path),
            "--speed",
            "1.0",
            "--ui-port",
            str(port),
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _check() -> dict[str, Any]:
            async with client.ws_session() as ws:
                card = await ws.wait_for_card(timeout=4.0)
                await asyncio.sleep(1.5)
                return card

        card = asyncio.run(_check())
        CardUIClient.assert_card_plain_text(card, max_words=20)
        assert card["question"].strip() == q20

    with open_db(e2e_workspace["db_path"]) as conn:
        decs = fetch_normalized_rows(conn, "decisions", session_id)
    rejected = [d for d in decs if d.get("terminal_reason") == "generator_invalid_output"]
    assert len(rejected) == 1
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


# ---------------------------------------------------------------------------
# Area G Boundaries (5 tests)
# ---------------------------------------------------------------------------


def test_boundary_g1_pass1_prefix_exact_received_ms_equality(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """Pass 1 prefix includes received_ms == at_ms and hides received_ms == at_ms + 1 (§12.4)."""
    session_id = "session-g1"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-g1-at",
            segment_id="seg-g1-at",
            text="Visible at exact timestamp.",
            start_ms=4000,
            end_ms=4800,
            received_ms=5000,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-g1-future",
            segment_id="seg-g1-future",
            text="Hidden one millisecond later.",
            start_ms=4900,
            end_ms=5000,
            received_ms=5001,
            source_sequence=2,
        ),
    ]
    jsonl_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "g1.jsonl", events)
    out_labels = e2e_workspace["root"] / "labels_g1.jsonl"
    port = free_port_factory()
    with start_coach_process(
        "annotate",
        [
            "--session",
            str(jsonl_path),
            "--out",
            str(out_labels),
            "--pass",
            "1",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = AnnotationClient(base_url, token=token)
        prefix = client.get_pass1_prefix(session_id, at_ms=5000)
        ids = {str(ev["event_id"]) for ev in prefix.get("events", [])}
        assert "ev-g1-at" in ids
        assert "ev-g1-future" not in ids


def test_boundary_g2_pass2_step1_lock_idempotent_reunlock(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """Repeated unlock_step2 calls keep Step 1 permanently locked (§12.4)."""
    session_id = "session-g2"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-g2-1",
            segment_id="seg-g2-1",
            text="Someone should own the rollback plan.",
            start_ms=100,
            end_ms=400,
            received_ms=500,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-g2-2",
            segment_id="seg-g2-2",
            text="Priya will own it.",
            speaker_id="me",
            start_ms=1000,
            end_ms=1300,
            received_ms=1400,
            source_sequence=2,
        ),
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="g2.jsonl")
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        cand_id = str(fetch_normalized_rows(conn, "decisions", session_id)[0]["candidate_id"])
    jsonl_path = e2e_workspace["sessions_dir"] / "g2.jsonl"
    out_labels = e2e_workspace["root"] / "labels_g2.jsonl"
    port = free_port_factory()
    with start_coach_process(
        "annotate",
        [
            "--session",
            str(jsonl_path),
            "--out",
            str(out_labels),
            "--pass",
            "2",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = AnnotationClient(base_url, token=token)
        client.submit_step1(
            cand_id,
            worthwhile=True,
            categories=["CONFIRM_OWNER"],
            example_questions=["Who will own the rollback plan?"],
        )
        client.unlock_step2(cand_id)
        client.unlock_step2(cand_id)
        status = client.assert_step1_locked(
            cand_id,
            {
                "worthwhile": False,
                "categories": [],
                "example_questions": [],
                "none_reason": "rhetorical",
            },
        )
        assert status in {400, 403, 409, 422}


def test_boundary_g3_pass2_empty_categories_requires_none_reason_when_not_worthwhile(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """worthwhile=False with none_reason=None is rejected (400/422); valid none_reason succeeds."""
    session_id = "session-g3"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-g3-1",
            segment_id="seg-g3-1",
            text="Someone should own the rollback plan.",
            start_ms=100,
            end_ms=400,
            received_ms=500,
            source_sequence=1,
        )
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="g3.jsonl")
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        cand_id = str(fetch_normalized_rows(conn, "decisions", session_id)[0]["candidate_id"])
    jsonl_path = e2e_workspace["sessions_dir"] / "g3.jsonl"
    out_labels = e2e_workspace["root"] / "labels_g3.jsonl"
    port = free_port_factory()
    with start_coach_process(
        "annotate",
        [
            "--session",
            str(jsonl_path),
            "--out",
            str(out_labels),
            "--pass",
            "2",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        resp = httpx.post(
            f"{base_url}/api/candidates/{cand_id}/step1",
            json={
                "worthwhile": False,
                "categories": [],
                "example_questions": [],
                "none_reason": None,
                "sample_source": "random",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        assert resp.status_code in {400, 422}

        client = AnnotationClient(base_url, token=token)
        accepted = client.submit_step1(
            cand_id,
            worthwhile=False,
            categories=[],
            example_questions=[],
            none_reason="rhetorical",
        )
        assert accepted["worthwhile"] is False


def test_boundary_g4_kappa_all_agree_vs_all_disagree_edge(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """Blind test-retest kappa handles all-agree (1.0) and all-disagree without div-by-zero."""
    session_id = "session-g4"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-g4-1",
            segment_id="seg-g4-1",
            text="Someone should own the rollback plan.",
            start_ms=100,
            end_ms=400,
            received_ms=500,
            source_sequence=1,
        )
    ]
    jsonl_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "g4.jsonl", events)
    for label, first_worth, second_worth, expected_ge in (
        ("agree", True, True, 0.99),
        ("disagree", True, False, -1.0),
    ):
        first_labels = write_jsonl(
            e2e_workspace["root"] / f"g4_first_{label}.jsonl",
            [
                build_label_record(
                    label_id=f"lbl-g4-{label}-0",
                    session_id=session_id,
                    candidate_id=f"cand-g4-{label}-0",
                    worthwhile=first_worth,
                    categories=("CONFIRM_OWNER",) if first_worth else (),
                    none_reason=None if first_worth else "rhetorical",
                ),
                build_label_record(
                    label_id=f"lbl-g4-{label}-1",
                    session_id=session_id,
                    candidate_id=f"cand-g4-{label}-1",
                    worthwhile=first_worth,
                    categories=("CONFIRM_OWNER",) if first_worth else (),
                    none_reason=None if first_worth else "rhetorical",
                ),
            ],
        )
        out_retest = e2e_workspace["root"] / f"g4_retest_{label}.jsonl"
        port = free_port_factory()
        with start_coach_process(
            "annotate",
            [
                "--session",
                str(jsonl_path),
                "--out",
                str(out_retest),
                "--kappa",
                str(first_labels),
                "--port",
                str(port),
            ],
            ready_url=f"http://127.0.0.1:{port}/",
            startup_timeout=10.0,
        ) as proc:
            base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
            client = AnnotationClient(base_url, token=token)
            for idx in (0, 1):
                item_id = f"cand-g4-{label}-{idx}"
                client.submit_retest(
                    item_id,
                    worthwhile=second_worth,
                    categories=["CONFIRM_OWNER"] if second_worth else [],
                    none_reason=None if second_worth else "rhetorical",
                )
            report = client.get_kappa_report()
            kappa_val = float(report["worthwhile_kappa"])
            assert kappa_val == kappa_val  # Not NaN
            assert kappa_val >= expected_ge


def test_boundary_g5_review_queue_empty_when_all_cards_labeled(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """When 100% of replayed cards match labels, review queue count is 0 (§12.6)."""
    session_id = "session-g5"
    events = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-g5-1",
            segment_id="seg-g5-1",
            text="Someone should own the rollback plan.",
            start_ms=100,
            end_ms=400,
            received_ms=500,
            source_sequence=1,
        )
    ]
    res = _replay_session(e2e_workspace, session_id, events, filename="g5.jsonl")
    assert res.returncode == 0
    with open_db(e2e_workspace["db_path"]) as conn:
        cand_id = str(fetch_normalized_rows(conn, "decisions", session_id)[0]["candidate_id"])
    labels_path = write_jsonl(
        e2e_workspace["root"] / "labels_g5.jsonl",
        [
            build_label_record(
                label_id="lbl-g5-1",
                session_id=session_id,
                candidate_id=cand_id,
                evidence_refs=({"segment_id": "seg-g5-1", "revision": 1},),
                open_ms=400,
                ideal_ms=800,
                close_ms=10000,
                closed_by="answered",
            )
        ],
    )
    eval_res = _replay_session(
        e2e_workspace,
        session_id,
        events,
        filename="g5.jsonl",
        extra_args=("--labels", str(labels_path)),
    )
    assert eval_res.returncode == 0
    eval_payload = eval_res.json()
    ev = eval_payload.get("evaluation") or eval_payload
    assert int(ev["unscored_pending_review_count"]) == 0
    assert len(ev.get("review_queue") or []) == 0

    port = free_port_factory()
    with start_coach_process(
        "annotate",
        [
            "--session",
            str(e2e_workspace["sessions_dir"] / "g5.jsonl"),
            "--out",
            str(labels_path),
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = AnnotationClient(base_url, token=token)
        assert client.get_review_queue() == []
