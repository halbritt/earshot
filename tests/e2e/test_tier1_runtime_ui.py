"""Tier 1 opaque-box E2E tests for Area F: 5-task runtime, display controller, and Card UI."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.helpers.cli_runner import (
    ManagedProcess,
    build_coach_test_env,
    extract_ui_url_and_token,
    run_coach,
    start_coach_process,
)
from tests.e2e.helpers.db_assertions import (
    assert_six_conservation_invariants,
    assert_wal_mode,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
    FEEDBACK_REASONS,
    build_boundary_event,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    deep_merge_dict,
    generate_pcm16_sine,
    write_brief_json,
    write_events_jsonl,
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
from tests.e2e.helpers.ui_client import (
    CardUIClient,
    assert_loopback_url,
)


def _write_nested_policy(
    policy_path: Path,
    *,
    generator_enabled: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    base = build_policy_dict(generator_enabled=generator_enabled)
    merged = deep_merge_dict(base, overrides or {})
    return write_policy_dict(policy_path, merged)


def _start_serve(
    workspace: dict[str, Path],
    port: int,
    *,
    source: str,
    extra_args: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
) -> ManagedProcess:
    args = [
        "--brief",
        str(workspace["brief_path"]),
        "--policy",
        str(workspace["policy_path"]),
        "--db",
        str(workspace["db_path"]),
        "--source",
        source,
        "--ui-port",
        str(port),
        *extra_args,
    ]
    return start_coach_process(
        "serve",
        args,
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    )


def _make_words(text: str, *, start_s: float, speaker: int) -> list[MockWord]:
    tokens = text.split()
    words: list[MockWord] = []
    cursor = start_s
    for tok in tokens:
        words.append(MockWord(tok, round(cursor, 3), round(cursor + 0.18, 3), speaker=speaker))
        cursor += 0.22
    return words


def test_five_task_runtime_nonblocking_ingest_under_slow_inference(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    """Proactive slow inference must not block transcript ingest or recorder (§4.2)."""
    session_id = "session-runtime-01"
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
            "display": {"settle_ms": 200, "cooldown_ms": 500, "max_cards_per_10_min": 6},
            "generator": {"enabled": True, "timeout_ms": 4000},
        },
    )
    mock_openai.set_default(
        MockCompletionSpec.valid_question(
            "Has failover testing covered multi-region cutover?",
            ["segment-0001"],
            delay_s=1.2,
        )
    )
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="We should clarify the regional failover scope soon",
                    words=_make_words(
                        "We should clarify the regional failover scope soon",
                        start_s=0.1,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=5,
                    transcript="Someone should also verify backup restore timing",
                    words=_make_words(
                        "Someone should also verify backup restore timing",
                        start_s=0.9,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=8,
                    transcript="Database replicas stay warm for two hours",
                    words=_make_words(
                        "Database replicas stay warm for two hours",
                        start_s=1.6,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=11,
                    trigger_on_commit=True,
                    transcript="Canary alerts fire within thirty seconds",
                    words=_make_words(
                        "Canary alerts fire within thirty seconds",
                        start_s=2.2,
                        speaker=2,
                    ),
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "runtime_ingest.wav",
        generate_pcm16_sine(duration_ms=2560),
    )
    port = free_port_factory()
    env = build_coach_test_env(
        openai_base_url=mock_openai.base_url,
        nemo_ws_url=mock_nemo_ws.url,
        peecee_bin=mock_peecee_bin,
    )
    with _start_serve(
        e2e_workspace,
        port,
        source="peecee",
        extra_args=("--audio-wav", str(wav_path)),
        env=env,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _observe() -> dict[str, Any]:
            async with client.ws_session() as ws:
                await ws.wait_for_health("ok", timeout=5.0)
                return await ws.wait_for_card(timeout=8.0)

        card = asyncio.run(_observe())
        assert card.get("question")
        time.sleep(1.5)

    assert len(mock_nemo_ws.received_frames) >= 12
    assert all(len(frame) == PCM16_FRAME_BYTES_160MS for frame in mock_nemo_ws.received_frames[:-1])
    assert_wal_mode(e2e_workspace["db_path"])
    with open_db(e2e_workspace["db_path"]) as conn:
        events = fetch_normalized_rows(conn, "events", session_id)
        committed_segs = [
            e for e in events if e.get("kind") == "segment" and e.get("stability") == "committed"
        ]
        drops = fetch_normalized_rows(conn, "recorder_drops", session_id)
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
    assert len(committed_segs) >= 4
    assert len(decisions) >= 1
    for drop_row in drops:
        if str(drop_row.get("level")) in {"lifecycle", "decision"}:
            assert int(drop_row.get("drop_count") or 0) == 0
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_all_five_operating_modes_proactive_quiet_shadow_paused_off_record(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    """Exercise Proactive, Quiet, Shadow, Paused, and Off-the-record modes (§3.6)."""
    session_id = "session-modes-02"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={"display": {"settle_ms": 150, "cooldown_ms": 100, "max_cards_per_10_min": 10}},
    )
    turns = [
        ScriptedTurn(
            trigger_after_frames=2,
            transcript="Someone should own the rollback plan",
            words=_make_words("Someone should own the rollback plan", start_s=0.1, speaker=2),
        ),
        ScriptedTurn(
            trigger_after_frames=8,
            transcript="Someone should own the cutover checklist",
            words=_make_words("Someone should own the cutover checklist", start_s=1.3, speaker=2),
        ),
        ScriptedTurn(
            trigger_after_frames=16,
            transcript="Someone should own the smoke verification",
            words=_make_words("Someone should own the smoke verification", start_s=2.6, speaker=2),
        ),
        ScriptedTurn(
            trigger_after_frames=24,
            transcript="Someone should own the paused window item",
            words=_make_words("Someone should own the paused window item", start_s=3.9, speaker=2),
        ),
        ScriptedTurn(
            trigger_after_frames=32,
            transcript="Private salary discussion off the record",
            words=_make_words("Private salary discussion off the record", start_s=5.1, speaker=2),
        ),
        ScriptedTurn(
            trigger_after_frames=40,
            trigger_on_commit=True,
            transcript="Someone should own the post recovery action",
            words=_make_words(
                "Someone should own the post recovery action", start_s=6.4, speaker=2
            ),
        ),
    ]
    mock_nemo_ws.set_config(MockNemoConfig(script=turns))
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "modes.wav",
        generate_pcm16_sine(duration_ms=7200),
    )
    port = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with _start_serve(
        e2e_workspace,
        port,
        source="peecee",
        extra_args=("--audio-wav", str(wav_path)),
        env=env,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _drive_modes() -> tuple[dict[str, Any], dict[str, Any]]:
            async with client.ws_session() as ws:
                proactive_card = await ws.wait_for_card(timeout=4.0)
                await ws.send_mode("Quiet")
                await asyncio.sleep(1.2)
                await ws.send_manual_request()
                manual_in_quiet = await ws.wait_for_card(timeout=3.0)
                await ws.send_mode("Shadow")
                await asyncio.sleep(1.3)
                await ws.send_mode("Paused")
                await ws.wait_for_health("paused", timeout=3.0)
                await asyncio.sleep(1.3)
                await ws.send_off_record(True)
                await asyncio.sleep(1.3)
                await ws.send_off_record(False)
                await ws.send_mode("Proactive")
                await asyncio.sleep(1.4)
                return proactive_card, manual_in_quiet

        proactive_card, manual_in_quiet = asyncio.run(_drive_modes())
        assert proactive_card.get("question")
        assert manual_in_quiet.get("question")

    with open_db(e2e_workspace["db_path"]) as conn:
        raw_rows = fetch_normalized_rows(conn, "raw_events", session_id)
        events = fetch_normalized_rows(conn, "events", session_id)
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
        suggestions = fetch_normalized_rows(conn, "suggestions", session_id)

    for table_rows in (raw_rows, events, decisions, suggestions):
        for row in table_rows:
            assert "Private salary discussion" not in json.dumps(row)

    control_subtypes = {str(e.get("subtype") or "") for e in events if e.get("kind") == "control"}
    assert "off_record_start" in control_subtypes
    assert "gap" in control_subtypes
    assert "off_record_end" in control_subtypes

    decision_reasons = {
        str(code)
        for d in decisions
        for code in (d.get("reason_codes") or [d.get("terminal_reason")])
        if code
    }
    assert "proactive_disabled" in decision_reasons
    assert any(s.get("origin") == "manual" for s in suggestions)
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_display_controller_settle_defer_displace_ttl_cooldown_pin_and_used_inferred(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """Verify boundary early settle, ME deferral, displacement, TTL, pin, and used_inferred."""
    session_id = "session-display-03"
    write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(
            session_id=session_id,
            attention={"max_cards_per_10_min": 10, "proactive": True},
        ),
    )
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={
            "display": {
                "settle_ms": 900,
                "cooldown_ms": 1100,
                "display_ttl_ms": 700,
                "max_candidate_age_ms": 5000,
                "max_cards_per_10_min": 10,
            }
        },
    )
    events_stream = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-disp-1",
            segment_id="seg-disp-1",
            text="Someone should own the rollback runbook.",
            speaker_id="participant-b",
            start_ms=100,
            end_ms=300,
            received_ms=320,
            source_sequence=1,
            boundary=None,
        ),
        build_boundary_event(
            session_id=session_id,
            event_id="ev-disp-1b",
            speaker_id="participant-b",
            end_ms=300,
            received_ms=420,
            source_sequence=2,
            boundary="utterance_end",
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-disp-2",
            segment_id="seg-disp-2",
            text="Maybe we can review the canary metrics.",
            speaker_id="participant-b",
            start_ms=1150,
            end_ms=1350,
            received_ms=1380,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-disp-2-me",
            segment_id="seg-disp-2-me",
            text="Hold on let me speak for a moment.",
            speaker_id="me",
            start_ms=1400,
            end_ms=1550,
            received_ms=1560,
            source_sequence=4,
        ),
    ]
    events_stream.extend(
        [
            build_segment_event(
                session_id=session_id,
                event_id="ev-disp-3",
                segment_id="seg-disp-3",
                text="Someone needs to own the cutover checklist by Friday.",
                speaker_id="participant-b",
                start_ms=1600,
                end_ms=1800,
                received_ms=1850,
                source_sequence=5,
            ),
            build_segment_event(
                session_id=session_id,
                event_id="ev-disp-4",
                segment_id="seg-disp-4",
                text="Someone must own the verification gate.",
                speaker_id="participant-b",
                start_ms=3400,
                end_ms=3600,
                received_ms=3650,
                source_sequence=6,
            ),
            build_segment_event(
                session_id=session_id,
                event_id="ev-disp-4-res",
                segment_id="seg-disp-4-res",
                text="Priya will own the verification gate.",
                speaker_id="me",
                start_ms=4900,
                end_ms=5100,
                received_ms=5150,
                source_sequence=7,
            ),
            build_segment_event(
                session_id=session_id,
                event_id="ev-disp-5",
                segment_id="seg-disp-5",
                text="Someone should own the DNS switch.",
                speaker_id="participant-b",
                start_ms=6600,
                end_ms=6800,
                received_ms=6850,
                source_sequence=8,
            ),
            build_segment_event(
                session_id=session_id,
                event_id="ev-disp-5-rev",
                segment_id="seg-disp-5",
                text="Actually Alex already owns the DNS switch.",
                speaker_id="participant-b",
                start_ms=6600,
                end_ms=6800,
                received_ms=8200,
                source_revision=2,
                source_sequence=9,
            ),
            build_segment_event(
                session_id=session_id,
                event_id="ev-disp-6",
                segment_id="seg-disp-6",
                text="Someone should own the incident pager rotation.",
                speaker_id="participant-b",
                start_ms=9600,
                end_ms=9800,
                received_ms=9850,
                source_sequence=10,
            ),
            build_segment_event(
                session_id=session_id,
                event_id="ev-disp-6-me",
                segment_id="seg-disp-6-me",
                text="Who will own the incident pager rotation?",
                speaker_id="me",
                start_ms=10900,
                end_ms=11100,
                received_ms=11150,
                source_sequence=11,
            ),
        ]
    )
    session_jsonl = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "display.jsonl", events_stream
    )
    port = free_port_factory()
    with _start_serve(
        e2e_workspace,
        port,
        source="replay",
        extra_args=("--session", str(session_jsonl), "--speed", "1.0"),
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _interact() -> list[dict[str, Any]]:
            seen_cards: list[dict[str, Any]] = []
            async with client.ws_session() as ws:
                c1 = await ws.wait_for_card(timeout=4.0)
                seen_cards.append(c1)
                c3 = await ws.wait_for_card(timeout=5.0)
                seen_cards.append(c3)
                c4 = await ws.wait_for_card(timeout=5.0)
                seen_cards.append(c4)
                await ws.send_pin(str(c4["suggestion_id"]))
                await asyncio.sleep(0.85)  # Exceeds display_ttl_ms=700 while pinned
                c5 = await ws.wait_for_card(timeout=5.0)
                seen_cards.append(c5)
                await ws.send_pin(str(c5["suggestion_id"]))
                c6 = await ws.wait_for_card(timeout=6.0)
                seen_cards.append(c6)
                await asyncio.sleep(2.0)
            return seen_cards

        seen = asyncio.run(_interact())
        assert len(seen) >= 4

    with open_db(e2e_workspace["db_path"]) as conn:
        suggestions = fetch_normalized_rows(conn, "suggestions", session_id)
        telemetry_rows = fetch_normalized_rows(conn, "telemetry_records", session_id)
        feedback_rows = fetch_normalized_rows(conn, "feedback", session_id)

    sug_reasons = {str(s.get("terminal_reason") or "") for s in suggestions}
    sug_states = {str(s.get("current_state") or "") for s in suggestions}
    all_transitions = {
        str(tr.get("transition") or "")
        for s in suggestions
        for tr in (s.get("transitions") or [])
        if isinstance(tr, dict)
    }
    all_causes = {
        str(tr.get("cause") or "")
        for s in suggestions
        for tr in (s.get("transitions") or [])
        if isinstance(tr, dict)
    }
    assert "boundary_after_settle" in all_causes
    assert "displaced_by_better" in sug_reasons or "recipient_speaking" in all_causes
    assert "display_ttl_passed" in sug_reasons
    assert "pinned" in all_transitions
    assert "resolved" in sug_states
    assert "outdated" in sug_states
    inferred_logged = any(
        "used_inferred" in json.dumps(row)
        for row in [*telemetry_rows, *feedback_rows, *suggestions]
    )
    assert inferred_logged
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_manual_lane_lookback_generator_fallback_concurrency_and_budget_exclusion(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    """Verify manual 30s lookback, generator fallback, simultaneous press, and budget (§3.7)."""
    session_id = "session-manual-04"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        generator_enabled=True,
        overrides={
            "categories": {"CONFIRM_OWNER": {"t_low": 0.20, "t_compute": 0.30, "t_display": 0.90}},
            "display": {
                "settle_ms": 1500,
                "cooldown_ms": 60000,
                "max_cards_per_10_min": 1,
            },
            "manual": {"lookback_s": 30, "score_floor": 0.30, "budget_ms": 2000},
            "generator": {"enabled": True, "timeout_ms": 1500},
        },
    )
    mock_openai.enqueue(
        MockCompletionSpec.valid_question(
            "What rollback trigger threshold applies during deployment?",
            ["seg-man-2"],
            delay_s=0.35,
        ),
        MockCompletionSpec.decline("nothing useful to add"),
    )
    events_stream = [
        build_segment_event(
            session_id=session_id,
            event_id="ev-man-1",
            segment_id="seg-man-1",
            text="Someone should probably handle the deployment rollback plan.",
            speaker_id="participant-b",
            start_ms=200,
            end_ms=700,
            received_ms=800,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=session_id,
            event_id="ev-man-2",
            segment_id="seg-man-2",
            text="The weather outside looks sunny today.",
            speaker_id="participant-b",
            start_ms=2000,
            end_ms=2500,
            received_ms=2600,
            source_sequence=2,
        ),
    ]
    session_jsonl = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "manual.jsonl", events_stream
    )
    port = free_port_factory()
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    with _start_serve(
        e2e_workspace,
        port,
        source="replay",
        extra_args=("--session", str(session_jsonl), "--speed", "1.0"),
        env=env,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _exercise_manual() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            async with client.ws_session() as ws:
                await asyncio.sleep(1.1)
                await ws.send_manual_request()
                lookback_card = await ws.wait_for_card(timeout=2.5)
                await ws.send_dismiss(str(lookback_card["suggestion_id"]), key="d")
                await asyncio.sleep(2.0)
                await ws.send_manual_request()
                await ws.send_manual_request()  # Simultaneous press: ignored and counted
                gen_card = await ws.wait_for_card(timeout=2.5)
                await ws.send_use(str(gen_card["suggestion_id"]))
                await ws.send_manual_request()
                empty_msg = await ws.wait_for_message(
                    lambda m: (
                        "nothing useful to add" in json.dumps(m).lower()
                        or m.get("type") in {"manual_empty", "nothing_useful"}
                    ),
                    timeout=2.5,
                )
                return lookback_card, gen_card, empty_msg

        lookback_card, gen_card, empty_msg = asyncio.run(_exercise_manual())
        assert lookback_card.get("origin", "manual") == "manual"
        assert gen_card.get("origin", "manual") == "manual"
        assert "nothing useful to add" in json.dumps(empty_msg).lower()

    report_res = run_coach(
        "report",
        ["--db", str(e2e_workspace["db_path"]), "--session", session_id, "--json"],
    )
    assert report_res.returncode == 0
    report = report_res.json()
    assert report["budget_compliant"] is True
    with open_db(e2e_workspace["db_path"]) as conn:
        suggestions = fetch_normalized_rows(conn, "suggestions", session_id)
        telemetry_rows = fetch_normalized_rows(conn, "telemetry_records", session_id)
    assert sum(1 for s in suggestions if s.get("origin") == "manual") >= 2
    assert any(
        "manual" in json.dumps(t).lower() and "ignored" in json.dumps(t).lower()
        for t in telemetry_rows
    )
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_five_health_states_and_outage_recovery_retirement(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_openai: MockOpenAIServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    """Verify all 5 health states (ok, template_only, manual_only, no_transcript, paused)."""
    # Part A: peecee-speech exit 75 -> immediate no_transcript with explicit adapter telemetry
    sid_75 = "session-health-exit75"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid_75))
    _write_nested_policy(e2e_workspace["policy_path"])
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "health.wav",
        generate_pcm16_sine(duration_ms=4800),
    )
    db_75 = e2e_workspace["root"] / "health_exit75.db"
    port_75 = free_port_factory()
    env_75 = build_coach_test_env(
        nemo_ws_url=mock_nemo_ws.url,
        peecee_bin=mock_peecee_bin,
        extra_env={"MOCK_PEECEE_EXIT_CODE": "75"},
    )
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_75),
            "--source",
            "peecee",
            "--audio-wav",
            str(wav_path),
            "--ui-port",
            str(port_75),
        ],
        env=env_75,
        ready_url=f"http://127.0.0.1:{port_75}/",
        startup_timeout=10.0,
    ) as proc_75:
        base_75, token_75 = extract_ui_url_and_token(proc_75, port_75, timeout=5.0)
        client_75 = CardUIClient(base_75, token_75)

        async def _check_75() -> dict[str, Any]:
            async with client_75.ws_session() as ws:
                return await ws.wait_for_health("no_transcript", timeout=5.0)

        msg_75 = asyncio.run(_check_75())
        assert (msg_75.get("health") or msg_75.get("state")) == "no_transcript"
    with open_db(db_75) as conn:
        decisions_75 = fetch_normalized_rows(conn, "decisions", sid_75)
        telemetry_75 = fetch_normalized_rows(conn, "telemetry_records", sid_75)
    assert len(decisions_75) == 0
    assert any("75" in json.dumps(t) or "no_transcript" in json.dumps(t) for t in telemetry_75)

    # Part B: ok -> template_only -> manual_only -> paused -> no_transcript + outage recovery
    sid = "session-health-all"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        generator_enabled=True,
        overrides={
            "classifier": {"version": "llm_baseline_v0", "calibration": "llm_baseline_v0-platt"},
            "categories": {
                "CLARIFY_SCOPE": {
                    "t_low": 0.20,
                    "t_compute": 0.45,
                    "t_display": 0.70,
                    "route": "generator",
                    "required_lookback_s": 120,
                }
            },
            "display": {"settle_ms": 200, "max_candidate_age_ms": 700, "cooldown_ms": 100},
            "generator": {
                "enabled": True,
                "timeout_ms": 400,
                "breaker": {"consecutive_failures": 3, "open_ms": 4000},
            },
            "health": {"no_transcript_after_ms": 700},
        },
    )
    clf_clarify = json.dumps({"opportunity": 0.90, "categories": {"CLARIFY_SCOPE": 0.90}})
    mock_openai.enqueue(
        MockCompletionSpec(status_code=200, content=clf_clarify),
        MockCompletionSpec.http_error(500, "err1"),
        MockCompletionSpec(status_code=200, content=clf_clarify),
        MockCompletionSpec.http_error(500, "err2"),
        MockCompletionSpec(status_code=200, content=clf_clarify),
        MockCompletionSpec.http_error(500, "err3"),
        MockCompletionSpec.timeout(delay_s=2.0),
        MockCompletionSpec.timeout(delay_s=2.0),
        MockCompletionSpec.timeout(delay_s=2.0),
    )
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="We should clarify the migration scope now",
                    words=_make_words(
                        "We should clarify the migration scope now", start_s=0.1, speaker=2
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=5,
                    transcript="Someone should own the second task",
                    words=_make_words("Someone should own the second task", start_s=0.8, speaker=2),
                ),
                ScriptedTurn(
                    trigger_after_frames=8,
                    transcript="Someone should own the third task",
                    words=_make_words("Someone should own the third task", start_s=1.4, speaker=2),
                ),
                ScriptedTurn(
                    trigger_after_frames=12,
                    transcript="Someone should own the outage candidate",
                    words=_make_words(
                        "Someone should own the outage candidate", start_s=2.1, speaker=2
                    ),
                ),
            ]
        )
    )
    port = free_port_factory()
    env = build_coach_test_env(
        openai_base_url=mock_openai.base_url,
        nemo_ws_url=mock_nemo_ws.url,
        peecee_bin=mock_peecee_bin,
    )
    with _start_serve(
        e2e_workspace,
        port,
        source="peecee",
        extra_args=("--audio-wav", str(wav_path)),
        env=env,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _observe_health() -> list[str]:
            observed: list[str] = []
            async with client.ws_session() as ws:
                for expected in ("ok", "template_only", "manual_only"):
                    msg = await ws.wait_for_health(expected, timeout=6.0)
                    observed.append(str(msg.get("health") or msg.get("state")))
                await ws.send_mode("Paused")
                p_msg = await ws.wait_for_health("paused", timeout=3.0)
                observed.append(str(p_msg.get("health") or p_msg.get("state")))
                await ws.send_mode("Proactive")
                mock_nemo_ws.drop_active_clients(abrupt=True)
                nt_msg = await ws.wait_for_health("no_transcript", timeout=4.0)
                observed.append(str(nt_msg.get("health") or nt_msg.get("state")))
            return observed

        states = asyncio.run(_observe_health())
        assert states == ["ok", "template_only", "manual_only", "paused", "no_transcript"]
    assert_six_conservation_invariants(e2e_workspace["db_path"], sid)
    with open_db(e2e_workspace["db_path"]) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", sid)
    assert any(
        d.get("terminal_reason") == "expired_unevaluated"
        or "expired_unevaluated" in (d.get("reason_codes") or [])
        for d in decisions
    )


def test_card_ui_loopback_auth_plain_text_xss_and_no_os_notifications(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """Verify loopback bind, token auth, plain-text DOM XSS resistance, and no Notification API."""
    session_id = "session-ui-sec-06"
    adversarial_text = (
        "<script>alert(1)</script><img src=x onerror=alert(1)> "
        "Someone should own the rollback plan."
    )
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=session_id))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={"display": {"settle_ms": 150, "cooldown_ms": 100}},
    )
    session_jsonl = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "xss.jsonl",
        [
            build_segment_event(
                session_id=session_id,
                event_id="ev-xss-1",
                segment_id="seg-xss-1",
                text=adversarial_text,
                speaker_id="participant-b",
                start_ms=100,
                end_ms=500,
                received_ms=600,
                source_sequence=1,
            )
        ],
    )
    port = free_port_factory()
    with _start_serve(
        e2e_workspace,
        port,
        source="replay",
        extra_args=("--session", str(session_jsonl), "--speed", "1.0"),
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        assert_loopback_url(base_url)
        client = CardUIClient(base_url, token)
        auth_codes = client.assert_auth_enforced()
        assert auth_codes["ok_bearer"] == 200
        assert auth_codes["ok_query"] == 200

        html_resp = httpx.get(
            f"{base_url}/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        assert html_resp.status_code == 200
        CardUIClient.inspect_html_xss_safety(
            html_resp.text,
            forbidden_substrings=("<script>alert(1)</script>", "onerror=alert(1)"),
        )

        async def _recv_card() -> dict[str, Any]:
            async with client.ws_session() as ws:
                return await ws.wait_for_card(timeout=4.0)

        card = asyncio.run(_recv_card())
        CardUIClient.assert_card_plain_text(
            card,
            raw_adversarial_text=adversarial_text,
            max_words=20,
        )
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)


def test_card_ui_anatomy_and_single_key_actions_u_d_p_1_4_persisted(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """Verify card anatomy (<=12w/20w, chip, collapsed evidence) & u/d/p/1..4 actions (§3.4)."""
    session_id = "session-ui-actions-07"
    write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(
            session_id=session_id,
            attention={"max_cards_per_10_min": 10, "proactive": True},
        ),
    )
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={"display": {"settle_ms": 150, "cooldown_ms": 150, "max_cards_per_10_min": 10}},
    )
    prompts = (
        "Someone should own the rollback plan.",
        "We need a deadline for the database migration.",
        "We should define a test criterion for canary acceptance.",
        "Someone should own the incident communication channel.",
        "We need a target date for DNS cutover.",
        "We should confirm success criteria for load testing.",
    )
    events_stream = [
        build_segment_event(
            session_id=session_id,
            event_id=f"ev-act-{idx}",
            segment_id=f"seg-act-{idx}",
            text=text,
            speaker_id="participant-b",
            start_ms=100 + idx * 700,
            end_ms=400 + idx * 700,
            received_ms=450 + idx * 700,
            source_sequence=idx + 1,
        )
        for idx, text in enumerate(prompts)
    ]
    session_jsonl = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "actions.jsonl",
        events_stream,
    )
    port = free_port_factory()
    with _start_serve(
        e2e_workspace,
        port,
        source="replay",
        extra_args=("--session", str(session_jsonl), "--speed", "1.0"),
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        client = CardUIClient(base_url, token)

        async def _exercise_actions() -> list[dict[str, Any]]:
            cards: list[dict[str, Any]] = []
            async with client.ws_session() as ws:
                c0 = await ws.wait_for_card(timeout=4.0)
                cards.append(c0)
                await ws.send_use(str(c0["suggestion_id"]))

                c1 = await ws.wait_for_card(timeout=4.0)
                cards.append(c1)
                await ws.send_dismiss(str(c1["suggestion_id"]), key="d")

                c2 = await ws.wait_for_card(timeout=4.0)
                cards.append(c2)
                await ws.send_pin(str(c2["suggestion_id"]))
                await ws.send_dismiss(str(c2["suggestion_id"]), key="1")

                for key in ("2", "3", "4"):
                    ck = await ws.wait_for_card(timeout=4.0)
                    cards.append(ck)
                    await ws.send_dismiss(str(ck["suggestion_id"]), key=key)
            return cards

        cards = asyncio.run(_exercise_actions())
        assert len(cards) == 6
        first = cards[0]
        question = str(first.get("question") or "").strip()
        assert question.endswith("?")
        assert len(question.split()) <= 12
        assert first.get("category")
        evidence = first.get("evidence")
        assert isinstance(evidence, dict | list) and bool(evidence)
    with open_db(e2e_workspace["db_path"]) as conn:
        suggestions = fetch_normalized_rows(conn, "suggestions", session_id)
        feedback_rows = fetch_normalized_rows(conn, "feedback", session_id)

    transitions_seen = {
        str(tr.get("transition") or "")
        for s in suggestions
        for tr in (s.get("transitions") or [])
        if isinstance(tr, dict)
    }
    assert {"used", "dismissed", "pinned"}.issubset(transitions_seen)
    feedback_seen = {str(fb.get("feedback_reason") or "") for fb in feedback_rows}
    for reason in FEEDBACK_REASONS.values():
        assert reason in feedback_seen
    assert_six_conservation_invariants(e2e_workspace["db_path"], session_id)
