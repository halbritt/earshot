"""Tier 3 opaque-box E2E tests: pairwise cross-feature combinations across Areas A-G."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.e2e.helpers.cli_runner import (
    CliResult,
    ManagedProcess,
    build_coach_test_env,
    extract_ui_url_and_token,
    run_coach,
    run_earshot_direction,
    start_coach_process,
)
from tests.e2e.helpers.db_assertions import (
    assert_local_only_no_egress,
    assert_model_cache_has_entries,
    assert_session_text_deleted,
    assert_six_conservation_invariants,
    assert_wal_mode,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
    REASON_CODES,
    build_boundary_event,
    build_brief_update_event,
    build_control_event,
    build_decision_context_row,
    build_label_record,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    build_speaker_update_event,
    deep_merge_dict,
    generate_multichannel_pcm,
    generate_pcm16_silence,
    read_jsonl,
    write_brief_json,
    write_events_jsonl,
    write_jsonl,
    write_policy_dict,
    write_wav_mono16,
)
from tests.e2e.helpers.mock_nemo_ws import (
    MockNemoConfig,
    MockNemoServer,
    MockWord,
    ScriptedTurn,
)
from tests.e2e.helpers.mock_openai import MockCompletionSpec, MockOpenAIServer
from tests.e2e.helpers.ui_client import AnnotationClient, CardUIClient


def _json_field(row: Mapping[str, Any], name: str) -> Any:
    """Read an unpacked JSON field or parse its `<name>_json` fallback defensively."""
    val = row.get(name)
    if val is not None:
        return val
    raw = row.get(f"{name}_json")
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return raw


def _extract_reason_codes(row: Mapping[str, Any]) -> set[str]:
    """Extract all Appendix C ReasonCode strings from a decision/suggestion/event row."""
    codes: set[str] = set()
    for key in ("terminal_reason", "reason_code", "reason", "cause"):
        val = row.get(key)
        if isinstance(val, str) and val:
            codes.add(val)
    for list_key in ("reason_codes", "abstain_reasons"):
        vals = _json_field(row, list_key)
        if isinstance(vals, list):
            codes.update(str(v) for v in vals if v)
    transitions = _json_field(row, "transitions")
    if isinstance(transitions, list):
        for tr in transitions:
            if isinstance(tr, dict):
                for key in ("cause", "reason", "reason_code"):
                    t_val = tr.get(key)
                    if isinstance(t_val, str) and t_val:
                        codes.add(t_val)
    return codes


def _make_words(
    text: str,
    *,
    start_s: float,
    speaker: int,
    confidence: float = 1.0,
    step_s: float = 0.20,
    word_dur_s: float = 0.16,
) -> list[MockWord]:
    tokens = text.split()
    words: list[MockWord] = []
    cursor = start_s
    for tok in tokens:
        words.append(
            MockWord(
                tok,
                round(cursor, 3),
                round(cursor + word_dur_s, 3),
                confidence=confidence,
                speaker=speaker,
            )
        )
        cursor += step_s
    return words


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
    ws: Mapping[str, Path],
    events: Sequence[dict[str, Any]],
    *,
    brief: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
    name: str = "t3_run",
    extra_args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    brief_path = write_brief_json(ws["root"] / f"{name}.brief.json", dict(brief))
    events_path = write_events_jsonl(ws["sessions_dir"] / f"{name}.jsonl", events)
    policy_path = (
        write_policy_dict(ws["policies_dir"] / f"{name}.yaml", policy)
        if policy is not None
        else ws["policy_path"]
    )
    db_path = ws["root"] / f"{name}.db"
    res: CliResult = run_coach(
        "replay",
        [
            "--session",
            str(events_path),
            "--brief",
            str(brief_path),
            "--policy",
            str(policy_path),
            "--db",
            str(db_path),
            "--json",
            *extra_args,
        ],
        env=env,
    )
    assert res.returncode == 0, (
        f"coach replay failed ({res.returncode}): {res.stderr}\n{res.stdout}"
    )
    payload = res.json()
    assert isinstance(payload, dict)
    return db_path, payload


def _start_serve(
    ws: Mapping[str, Path],
    port: int,
    *,
    source: str,
    db_path: Path | None = None,
    extra_args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> ManagedProcess:
    args = [
        "--brief",
        str(ws["brief_path"]),
        "--policy",
        str(ws["policy_path"]),
        "--db",
        str(db_path or ws["db_path"]),
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


def _run_report(db_path: Path, session_id: str) -> dict[str, Any]:
    res = run_coach("report", ["--db", str(db_path), "--session", session_id, "--json"])
    assert res.returncode == 0, f"coach report failed: {res.stderr}\n{res.stdout}"
    payload = res.json()
    assert isinstance(payload, dict)
    assert payload["invariants"]["passed"] is True, payload["invariants"]
    return payload


def test_t3_01_off_record_toggle_gap_context_flag_and_report_invariants(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    """Off-the-record WS toggle + gap markers + context_builder gap flag + 6 invariants."""
    sid = "t3-01-off-record"
    forbidden = "Private compensation numbers"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={"display": {"settle_ms": 100, "cooldown_ms": 100, "max_cards_per_10_min": 6}},
    )
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="Before off record marker.",
                    words=_make_words("Before off record marker.", start_s=0.1, speaker=2),
                ),
                ScriptedTurn(
                    trigger_after_frames=10,
                    transcript=f"{forbidden} should stay hidden.",
                    words=_make_words(
                        f"{forbidden} should stay hidden.",
                        start_s=1.6,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=20,
                    trigger_on_commit=True,
                    transcript="Someone should own the rollback plan.",
                    words=_make_words(
                        "Someone should own the rollback plan.",
                        start_s=3.2,
                        speaker=2,
                    ),
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "t3_01.wav",
        generate_pcm16_silence(duration_ms=4800),
    )
    port = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with _start_serve(
        e2e_workspace,
        port,
        source="peecee",
        extra_args=("--audio-wav", str(wav_path), "--local-only"),
        env=env,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        ui = CardUIClient(base_url, token)

        async def _drive() -> dict[str, Any]:
            async with ui.ws_session() as ws:
                deadline = asyncio.get_running_loop().time() + 8.0
                while len(mock_nemo_ws.received_frames) < 5:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.05)
                await ws.send_off_record(True)
                while len(mock_nemo_ws.received_frames) < 14:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.05)
                await ws.send_off_record(False)
                return await ws.wait_for_card(timeout=6.0)

        card = asyncio.run(_drive())
        assert card.get("question")

    db_path = e2e_workspace["db_path"]
    assert_wal_mode(db_path)
    with open_db(db_path) as conn:
        raw_rows = fetch_normalized_rows(conn, "raw_events", sid)
        events = fetch_normalized_rows(conn, "events", sid)
        decisions = fetch_normalized_rows(conn, "decisions", sid)
        suggestions = fetch_normalized_rows(conn, "suggestions", sid)
        telemetry = fetch_normalized_rows(conn, "telemetry_records", sid)

    for tbl_rows in (raw_rows, events, decisions, suggestions, telemetry):
        for row in tbl_rows:
            assert forbidden not in json.dumps(row)

    controls = [e for e in events if e.get("kind") == "control"]
    subtypes = {str(c.get("subtype") or "") for c in controls}
    assert {"off_record_start", "gap", "off_record_end"}.issubset(subtypes)
    gaps = [c for c in controls if c.get("subtype") == "gap"]
    assert any(g.get("reason") == "off_record" for g in gaps)

    post_gap_flagged = any(
        bool(
            (
                isinstance(flags := _json_field(d, "flags"), dict)
                and (flags.get("gap") or flags.get("has_gap"))
            )
            or "gap" in str(d.get("serialized_text") or "").lower()
        )
        for d in decisions
    )
    assert post_gap_flagged
    assert_six_conservation_invariants(db_path, sid)
    _run_report(db_path, sid)


def test_t3_02_manual_request_during_active_proactive_job_and_cooldown_bypass(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    """Manual request bypasses 60s cooldown while proactive generator job is active."""
    sid = "t3-02-manual-cooldown"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        generator_enabled=True,
        overrides={
            "categories": {
                "CLARIFY_SCOPE": {
                    "t_low": 0.20,
                    "t_compute": 0.45,
                    "t_display": 0.75,
                    "route": "generator",
                    "required_lookback_s": 120,
                }
            },
            "display": {
                "settle_ms": 100,
                "cooldown_ms": 60000,
                "max_cards_per_10_min": 3,
            },
            "manual": {"lookback_s": 30, "score_floor": 0.30, "budget_ms": 2000},
            "generator": {"enabled": True, "timeout_ms": 4000},
        },
    )
    mock_openai.set_default(
        MockCompletionSpec.valid_question(
            "What scope boundary needs verification?",
            ["seg-ch0-00002000"],
            delay_s=1.2,
        )
    )
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="Someone should own the rollback plan.",
                    words=_make_words(
                        "Someone should own the rollback plan.", start_s=0.1, speaker=2
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=6,
                    transcript="When does the canary window close?",
                    words=_make_words("When does the canary window close?", start_s=1.0, speaker=2),
                ),
                ScriptedTurn(
                    trigger_after_frames=10,
                    trigger_on_commit=True,
                    transcript="We should clarify the regional failover scope boundary now.",
                    words=_make_words(
                        "We should clarify the regional failover scope boundary now.",
                        start_s=2.0,
                        speaker=2,
                    ),
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "t3_02.wav",
        generate_pcm16_silence(duration_ms=3200),
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
        ui = CardUIClient(base_url, token)

        async def _drive() -> tuple[dict[str, Any], dict[str, Any], float]:
            async with ui.ws_session() as ws:
                proactive_card = await ws.wait_for_card(timeout=5.0)
                deadline = asyncio.get_running_loop().time() + 6.0
                while not mock_openai.requests:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.05)
                t0 = time.monotonic()
                await ws.send_manual_request()
                manual_card = await ws.wait_for_card(timeout=2.0)
                elapsed = time.monotonic() - t0
                await asyncio.sleep(1.3)
                return proactive_card, manual_card, elapsed

        proactive_card, manual_card, elapsed_s = asyncio.run(_drive())
        assert elapsed_s <= 2.0
        assert proactive_card.get("origin", "proactive") == "proactive"
        assert manual_card.get("origin", "manual") == "manual"
        assert manual_card.get("question")

    db_path = e2e_workspace["db_path"]
    with open_db(db_path) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", sid)
        suggestions = fetch_normalized_rows(conn, "suggestions", sid)
    assert {"proactive", "manual"} <= {str(d.get("origin") or "") for d in decisions}
    assert {"proactive", "manual"} <= {str(s.get("origin") or "") for s in suggestions}
    report = _run_report(db_path, sid)
    assert report["budget_compliant"] is True
    assert_six_conservation_invariants(db_path, sid)


def test_t3_03_circuit_breaker_template_fallback_recovery_and_ui_health(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    """3 generator timeouts -> template_only -> generic template fallback -> half-open ok."""
    assert build_policy_dict()["generator"]["breaker"]["open_ms"] == 60000
    sid = "t3-03-breaker-fallback"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        generator_enabled=True,
        overrides={
            "categories": {
                "CLARIFY_SCOPE": {
                    "t_low": 0.20,
                    "t_compute": 0.45,
                    "t_display": 0.75,
                    "route": "generator",
                    "required_lookback_s": 120,
                }
            },
            "display": {"settle_ms": 100, "cooldown_ms": 50, "max_cards_per_10_min": 10},
            "generator": {
                "enabled": True,
                "timeout_ms": 200,
                "breaker": {"consecutive_failures": 3, "open_ms": 800},
            },
        },
    )
    mock_openai.enqueue(
        MockCompletionSpec.timeout(delay_s=1.0),
        MockCompletionSpec.timeout(delay_s=1.0),
        MockCompletionSpec.timeout(delay_s=1.0),
    )
    mock_openai.set_default(
        MockCompletionSpec.valid_question(
            "What scope boundary needs verification?",
            ["seg-ch0-00004000"],
        )
    )
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="We should clarify the migration scope boundary first.",
                    words=_make_words(
                        "We should clarify the migration scope boundary first.",
                        start_s=0.1,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=5,
                    transcript="We should clarify the failover scope boundary next.",
                    words=_make_words(
                        "We should clarify the failover scope boundary next.",
                        start_s=0.8,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=8,
                    transcript="We should clarify the rollback scope boundary too.",
                    words=_make_words(
                        "We should clarify the rollback scope boundary too.",
                        start_s=1.5,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=12,
                    transcript=(
                        "Someone should own the extremely complex multi region active active "
                        "failover verification checklist."
                    ),
                    words=_make_words(
                        "Someone should own the extremely complex multi region active active "
                        "failover verification checklist.",
                        start_s=2.2,
                        speaker=2,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=24,
                    trigger_on_commit=True,
                    transcript="We should clarify the backup retention scope boundary now.",
                    words=_make_words(
                        "We should clarify the backup retention scope boundary now.",
                        start_s=4.0,
                        speaker=2,
                    ),
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "t3_03.wav",
        generate_pcm16_silence(duration_ms=4800),
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
        ui = CardUIClient(base_url, token)

        async def _drive() -> tuple[list[str], list[dict[str, Any]]]:
            health_seq: list[str] = []
            cards: list[dict[str, Any]] = []
            async with ui.ws_session() as ws:
                h1 = await ws.wait_for_health("ok", timeout=5.0)
                health_seq.append(str(h1.get("health") or h1.get("state")))
                h2 = await ws.wait_for_health("template_only", timeout=6.0)
                health_seq.append(str(h2.get("health") or h2.get("state")))
                c_generic = await ws.wait_for_card(timeout=5.0)
                cards.append(c_generic)
                await ws.send_dismiss(str(c_generic["suggestion_id"]), key="d")
                h3 = await ws.wait_for_health("ok", timeout=6.0)
                health_seq.append(str(h3.get("health") or h3.get("state")))
                c_gen = await ws.wait_for_card(timeout=5.0)
                cards.append(c_gen)
            return health_seq, cards

        health_seq, cards = asyncio.run(_drive())
        assert health_seq == ["ok", "template_only", "ok"]
        assert any(c.get("question") == "Who owns that next step?" for c in cards)
        assert any(c.get("question") == "What scope boundary needs verification?" for c in cards)

    assert len(mock_openai.requests) == 4
    db_path = e2e_workspace["db_path"]
    assert_six_conservation_invariants(db_path, sid)
    _run_report(db_path, sid)


def test_t3_04_replay_diff_model_cache_hit_and_delete_session_preservation(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    """Replay + model_cache hit on identical prompt + coach diff + §14.2 delete-session."""
    sid = "t3-04-cache-diff-del"
    brief = build_session_brief(session_id=sid)
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-t304-1",
            segment_id="seg-t304-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3000,
            received_ms=3200,
            source_sequence=1,
        ),
        build_boundary_event(
            session_id=sid,
            event_id="ev-t304-bnd",
            channel=1,
            speaker_id="participant-b",
            end_ms=3000,
            received_ms=3300,
            source_sequence=2,
        ),
    ]
    mock_openai.enqueue(
        MockCompletionSpec(
            content=json.dumps(
                {
                    "opportunity": 0.80,
                    "categories": {
                        "CONFIRM_OWNER": 0.80,
                        "CONFIRM_TIME": 0.05,
                        "SUCCESS_CRITERION": 0.05,
                    },
                }
            )
        )
    )
    pol_strict = deep_merge_dict(
        build_policy_dict(calibration="llm_baseline_v0-platt-v0-a", settle_ms=100),
        {
            "classifier": {"version": "llm_baseline_v0"},
            "categories": {"CONFIRM_OWNER": {"t_low": 0.20, "t_compute": 0.50, "t_display": 0.90}},
        },
    )
    pol_loose = deep_merge_dict(
        build_policy_dict(calibration="llm_baseline_v0-platt-v0-a", settle_ms=100),
        {
            "classifier": {"version": "llm_baseline_v0"},
            "categories": {"CONFIRM_OWNER": {"t_low": 0.20, "t_compute": 0.50, "t_display": 0.60}},
        },
    )
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)

    db_path, _ = _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol_strict,
        name="t304_shared",
        extra_args=("--run-id", "run-a"),
        env=env,
    )
    assert len(mock_openai.requests) == 1
    assert_model_cache_has_entries(db_path, min_count=1)

    mock_openai.clear_requests()
    _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol_loose,
        name="t304_shared",
        extra_args=("--run-id", "run-b"),
        env=env,
    )
    assert len(mock_openai.requests) == 0, (
        "Second replay must hit (serialized_sha256, model_version)"
    )

    diff_res = run_coach(
        "diff",
        [
            "--db",
            str(db_path),
            "--run-a",
            "run-a",
            "--run-b",
            "run-b",
            "--session",
            sid,
            "--json",
        ],
    )
    assert diff_res.returncode == 0, diff_res.stderr
    diff_json = diff_res.json()
    assert len(diff_json.get("gained") or []) >= 1

    del_res = run_coach("delete-session", ["--db", str(db_path), "--session", sid, "--json"])
    assert del_res.returncode == 0, del_res.stderr
    assert_session_text_deleted(
        db_path,
        sid,
        forbidden_substrings=("rollback plan",),
        require_decisions_preserved=True,
        require_suggestions_preserved=True,
    )

    diff_after = run_coach(
        "diff",
        [
            "--db",
            str(db_path),
            "--run-a",
            "run-a",
            "--run-b",
            "run-b",
            "--session",
            sid,
            "--json",
        ],
    )
    assert diff_after.returncode == 0, diff_after.stderr
    assert len(diff_after.json().get("gained") or []) >= 1
    _run_report(db_path, sid)


def test_t3_05_speaker_update_invalidation_withdrawal_and_coach_explain(
    e2e_workspace: dict[str, Path],
) -> None:
    """speaker_update (S1->ME / S1->S2) + invalidation + withdrawal + coach explain."""
    sid = "t3-05-speaker-inval"
    brief = build_session_brief(session_id=sid)
    pol = deep_merge_dict(
        build_policy_dict(settle_ms=100, cooldown_ms=100, max_cards_per_10_min=6),
        {"abstain": {"speaker_settle_ms": 500}},
    )
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-t305-1",
            segment_id="seg-t305-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=1900,
            received_ms=2000,
            boundary="utterance_end",
            source_sequence=1,
        ),
        build_speaker_update_event(
            session_id=sid,
            event_id="ev-t305-upd1",
            segment_id="seg-t305-1",
            speaker_id="me",
            source_revision=2,
            received_ms=3000,
            source_sequence=2,
            speaker_confidence=0.95,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t305-2",
            segment_id="seg-t305-2",
            text="Someone should own the cutover checklist.",
            speaker_id="participant-b",
            start_ms=5000,
            end_ms=5900,
            received_ms=6000,
            boundary="utterance_end",
            source_sequence=3,
        ),
        build_speaker_update_event(
            session_id=sid,
            event_id="ev-t305-upd2",
            segment_id="seg-t305-2",
            speaker_id="participant-c",
            source_revision=2,
            received_ms=7000,
            source_sequence=4,
            speaker_confidence=0.93,
        ),
    ]
    db_path, _ = _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol,
        name="t305_spk",
        extra_args=("--speed", "stepped"),
    )
    assert_six_conservation_invariants(db_path, sid)
    _run_report(db_path, sid)

    with open_db(db_path) as conn:
        ev_rows = fetch_normalized_rows(conn, "events", sid)
        sugs = fetch_normalized_rows(conn, "suggestions", sid)

    spk_updates = [e for e in ev_rows if e.get("kind") == "speaker_update"]
    assert len(spk_updates) == 2
    assert all(
        int(e.get("canonical_revision") or e.get("source_revision") or 0) >= 2 for e in spk_updates
    )

    seg1_sugs = [s for s in sugs if "seg-t305-1" in json.dumps(s)]
    assert seg1_sugs
    assert all(s.get("current_state") != "displayed" for s in seg1_sugs)
    assert any(
        s.get("terminal_state") == "withdrawn" or s.get("current_state") == "withdrawn"
        for s in seg1_sugs
    )
    assert any("evidence_revision_changed" in _extract_reason_codes(s) for s in seg1_sugs)

    exp_res = run_coach(
        "explain",
        [
            "--db",
            str(db_path),
            "--session",
            sid,
            "--at-ms",
            "8000",
            "--window-ms",
            "8000",
            "--json",
        ],
    )
    assert exp_res.returncode == 0, exp_res.stderr
    anchors = exp_res.json()["anchors"]
    assert any(
        a.get("terminal_state") == "withdrawn"
        and a.get("terminal_reason") == "evidence_revision_changed"
        for a in anchors
    )


def test_t3_06_brief_update_withdrawal_and_rescoring_under_updated_brief(
    e2e_workspace: dict[str, Path],
) -> None:
    """brief_update (objective + do_not_suggest) withdraws dependents and re-scores."""
    sid = "t3-06-brief-update"
    old_objective = "Verify cutover plan and rollback owner."
    new_objective = "Review post-incident latency SLOs only."
    brief = build_session_brief(
        session_id=sid,
        objective=old_objective,
        do_not_suggest=["budget"],
    )
    pol = build_policy_dict(settle_ms=100, cooldown_ms=100, max_cards_per_10_min=6)
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-t306-1",
            segment_id="seg-t306-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=1900,
            received_ms=2000,
            boundary="utterance_end",
            source_sequence=1,
        ),
        build_brief_update_event(
            session_id=sid,
            event_id="ev-t306-brief",
            brief_patch={
                "objective": new_objective,
                "do_not_suggest": ["budget", "cutover"],
            },
            received_ms=3000,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t306-2",
            segment_id="seg-t306-2",
            text="Someone should own the cutover checklist.",
            speaker_id="participant-c",
            start_ms=4000,
            end_ms=4900,
            received_ms=5000,
            boundary="utterance_end",
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t306-3",
            segment_id="seg-t306-3",
            text="Someone should own the latency SLO dashboard.",
            speaker_id="participant-b",
            start_ms=7000,
            end_ms=7900,
            received_ms=8000,
            boundary="utterance_end",
            source_sequence=4,
        ),
    ]
    db_path, _ = _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol,
        name="t306_brief",
        extra_args=("--speed", "stepped"),
    )
    assert_six_conservation_invariants(db_path, sid)
    _run_report(db_path, sid)

    with open_db(db_path) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", sid)
        suggestions = fetch_normalized_rows(conn, "suggestions", sid)

    by_seg = {
        str(ref["segment_id"]): d
        for d in decisions
        for ref in (d.get("anchor_refs") or [])
        if isinstance(ref, dict) and ref.get("segment_id")
    }
    seg1_sugs = [s for s in suggestions if "seg-t306-1" in json.dumps(s)]
    assert seg1_sugs
    assert any(
        (s.get("terminal_state") == "withdrawn" or s.get("current_state") == "withdrawn")
        and "objective_changed" in _extract_reason_codes(s)
        for s in seg1_sugs
    )

    assert by_seg["seg-t306-2"]["outcome"] == "SUPPRESS"
    assert "do_not_suggest" in _extract_reason_codes(by_seg["seg-t306-2"])

    ser3 = str(by_seg["seg-t306-3"].get("serialized_text") or "")
    assert new_objective in ser3
    assert old_objective not in ser3


def test_t3_07_interim_resolution_hint_commit_vs_revision_away_and_expiry(
    e2e_workspace: dict[str, Path],
) -> None:
    """interim_resolution_hint deferral -> possibly_resolved vs deferred_released vs expiry."""
    sid = "t3-07-interim-hint"
    brief = build_session_brief(
        session_id=sid,
        attention={"max_cards_per_10_min": 10, "proactive": True},
    )
    pol = deep_merge_dict(
        build_policy_dict(settle_ms=200, cooldown_ms=100, max_cards_per_10_min=10),
        {"display": {"max_candidate_age_ms": 1200}},
    )
    events = [
        # Scenario 1: interim resolution hint commits -> possibly_resolved
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-1",
            segment_id="seg-t307-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=500,
            end_ms=950,
            received_ms=1000,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-1t-i",
            segment_id="seg-t307-1t",
            text="Priya, can you take",
            speaker_id="participant-b",
            start_ms=1020,
            end_ms=1120,
            received_ms=1150,
            stability="interim",
            source_revision=1,
            source_sequence=2,
            boundary=None,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-1t-c",
            segment_id="seg-t307-1t",
            text="Priya, can you take that?",
            speaker_id="participant-b",
            start_ms=1020,
            end_ms=1350,
            received_ms=1400,
            stability="committed",
            source_revision=2,
            source_sequence=3,
        ),
        # Scenario 2: interim hint revised away before display_by_ms=4200 -> deferred_released
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-2",
            segment_id="seg-t307-2",
            text="Someone should own the canary checklist.",
            speaker_id="participant-b",
            start_ms=2500,
            end_ms=2950,
            received_ms=3000,
            source_sequence=4,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-2t-i",
            segment_id="seg-t307-2t",
            text="Alex, can you take",
            speaker_id="participant-b",
            start_ms=3020,
            end_ms=3120,
            received_ms=3150,
            stability="interim",
            source_revision=1,
            source_sequence=5,
            boundary=None,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-2t-c",
            segment_id="seg-t307-2t",
            text="Wait, never mind about the canary logs.",
            speaker_id="participant-b",
            start_ms=3020,
            end_ms=3350,
            received_ms=3400,
            stability="committed",
            source_revision=2,
            source_sequence=6,
        ),
        # Scenario 3: interim hint revised away after display_by_ms=7200 -> display_by_passed
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-3",
            segment_id="seg-t307-3",
            text="Someone should own the failover drill.",
            speaker_id="participant-b",
            start_ms=5500,
            end_ms=5950,
            received_ms=6000,
            source_sequence=7,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-3t-i",
            segment_id="seg-t307-3t",
            text="Sam, can you take",
            speaker_id="participant-b",
            start_ms=6020,
            end_ms=6120,
            received_ms=6150,
            stability="interim",
            source_revision=1,
            source_sequence=8,
            boundary=None,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t307-3t-c",
            segment_id="seg-t307-3t",
            text="Wait, never mind about the failover logs.",
            speaker_id="participant-b",
            start_ms=6020,
            end_ms=7450,
            received_ms=7500,
            stability="committed",
            source_revision=2,
            source_sequence=9,
        ),
    ]
    db_path, _ = _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol,
        name="t307_interim",
        extra_args=("--speed", "stepped"),
    )
    assert_six_conservation_invariants(db_path, sid)
    _run_report(db_path, sid)

    with open_db(db_path) as conn:
        sugs = fetch_normalized_rows(conn, "suggestions", sid)

    def _sugs_for(seg_id: str) -> list[dict[str, Any]]:
        return [s for s in sugs if seg_id in json.dumps(s)]

    s1 = _sugs_for("seg-t307-1")
    s2 = _sugs_for("seg-t307-2")
    s3 = _sugs_for("seg-t307-3")
    assert s1 and s2 and s3

    s1_codes = set().union(*(_extract_reason_codes(s) for s in s1))
    s2_codes = set().union(*(_extract_reason_codes(s) for s in s2))
    s3_codes = set().union(*(_extract_reason_codes(s) for s in s3))

    def _transitions(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        return {
            str(t.get("transition"))
            for row in rows
            for t in (_json_field(row, "transitions") or [])
            if isinstance(t, dict)
        }

    assert {"interim_resolution_hint", "possibly_resolved"}.issubset(s1_codes)
    assert "displayed" not in _transitions(s1)

    assert {"interim_resolution_hint", "deferred_released"}.issubset(s2_codes)
    assert "displayed" in _transitions(s2)

    assert {"interim_resolution_hint", "display_by_passed"}.issubset(s3_codes)
    assert "displayed" not in _transitions(s3)


def test_t3_08_earshot_direction_sortformer_binder_abstentions_and_report_delay(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
    repo_root: Path,
) -> None:
    """earshot direction + Sortformer binder + speaker/ASR abstentions + upstream_delay_ms."""
    array_toml = repo_root / "arrays" / "respeaker-core-v2.toml"
    raw_pcm = e2e_workspace["root"] / "t3_08_8ch.raw"
    raw_pcm.write_bytes(generate_multichannel_pcm(channels=8, duration_ms=1000, width_bits=32))
    dir_res = run_earshot_direction(array_toml, raw_pcm, every=0.5, repo_root=repo_root)
    assert dir_res.returncode == 0, dir_res.stderr
    dir_rows = dir_res.jsonl()
    assert len(dir_rows) >= 1
    assert {"t", "azimuth", "contrast", "level_db"}.issubset(dir_rows[0].keys())

    sid = "t3-08-direction-binder"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={
            "display": {"settle_ms": 150, "cooldown_ms": 100, "max_cards_per_10_min": 6},
            "abstain": {"speaker_settle_ms": 200, "asr_confidence_floor": 0.70},
        },
    )
    direction_samples = [
        {"t": 0.5, "azimuth": 90.0, "contrast": 8.5, "level_db": -18.0},
        {"t": 1.0, "azimuth": 90.0, "contrast": 8.5, "level_db": -18.0},
        {"t": 1.5, "azimuth": 90.0, "contrast": 8.5, "level_db": -18.0},
        {"t": 2.2, "azimuth": 180.0, "contrast": 2.5, "level_db": -26.0},
        {"t": 3.5, "azimuth": 90.0, "contrast": 8.2, "level_db": -19.0},
        {"t": 4.2, "azimuth": 90.0, "contrast": 8.2, "level_db": -19.0},
    ]
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=2,
                    transcript="Someone should own the rollback plan",
                    words=_make_words(
                        "Someone should own the rollback plan",
                        start_s=0.2,
                        speaker=2,
                        confidence=0.98,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=8,
                    transcript="Confirm rollback owner checklist today",
                    words=_make_words(
                        "Confirm rollback owner checklist today",
                        start_s=2.0,
                        speaker=3,
                        confidence=0.95,
                        step_s=0.12,
                        word_dur_s=0.10,
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=14,
                    trigger_on_commit=True,
                    transcript="Someone should verify the canary alerts",
                    words=_make_words(
                        "Someone should verify the canary alerts",
                        start_s=3.2,
                        speaker=2,
                        confidence=0.25,
                    ),
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "t3_08.wav",
        generate_pcm16_silence(duration_ms=3200),
    )
    port = free_port_factory()
    env = build_coach_test_env(
        nemo_ws_url=mock_nemo_ws.url,
        peecee_bin=mock_peecee_bin,
        extra_env={
            "COACH_SEAT_AZIMUTHS": json.dumps({"me": 0.0, "S1": 90.0}),
            "COACH_EARSHOT_DIRECTION_JSON": json.dumps(direction_samples),
        },
    )
    with _start_serve(
        e2e_workspace,
        port,
        source="peecee",
        extra_args=("--audio-wav", str(wav_path), "--local-only"),
        env=env,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        ui = CardUIClient(base_url, token)

        async def _observe() -> None:
            async with ui.ws_session() as ws:
                await ws.wait_for_card(timeout=5.0)
                deadline = asyncio.get_running_loop().time() + 8.0
                while len(mock_nemo_ws.received_frames) < 20 or not any(
                    e.get("type") == "input_audio_buffer.commit"
                    for e in mock_nemo_ws.received_events
                ):
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.6)

        asyncio.run(_observe())

    db_path = e2e_workspace["db_path"]
    assert_six_conservation_invariants(db_path, sid)
    with open_db(db_path) as conn:
        decisions = fetch_normalized_rows(conn, "decisions", sid)

    abstain_reasons = {
        code
        for d in decisions
        if d.get("outcome") == "ABSTAIN"
        for code in _extract_reason_codes(d)
    }
    assert "speaker_uncertain" in abstain_reasons
    assert "asr_low_confidence" in abstain_reasons

    report = _run_report(db_path, sid)
    delay = report["upstream_delay_ms"]
    assert delay["p50"] is not None
    assert delay["p95"] is not None
    assert delay["p99"] is not None


def test_t3_09_quiet_shadow_logging_review_queue_and_pass2_shadow_review(
    e2e_workspace: dict[str, Path],
    mock_nemo_ws: MockNemoServer,
    mock_peecee_bin: Path,
    free_port_factory: Callable[[], int],
) -> None:
    """Quiet/Shadow logging -> unmatched review queue -> Pass 2 locked shadow_review."""
    sid = "t3-09-shadow-review"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        overrides={
            "display": {"settle_ms": 100, "cooldown_ms": 100, "max_cards_per_10_min": 6},
            "manual": {"lookback_s": 30, "score_floor": 0.30, "budget_ms": 2000},
        },
    )
    mock_nemo_ws.set_config(
        MockNemoConfig(
            script=[
                ScriptedTurn(
                    trigger_after_frames=5,
                    transcript="Someone should own the rollback plan.",
                    words=_make_words(
                        "Someone should own the rollback plan.", start_s=0.4, speaker=2
                    ),
                ),
                ScriptedTurn(
                    trigger_after_frames=14,
                    transcript="Someone should own the cutover checklist.",
                    words=_make_words(
                        "Someone should own the cutover checklist.",
                        start_s=2.0,
                        speaker=2,
                    ),
                ),
            ]
        )
    )
    wav_path = write_wav_mono16(
        e2e_workspace["root"] / "t3_09.wav",
        generate_pcm16_silence(duration_ms=4000),
    )
    port_serve = free_port_factory()
    env = build_coach_test_env(nemo_ws_url=mock_nemo_ws.url, peecee_bin=mock_peecee_bin)
    with _start_serve(
        e2e_workspace,
        port_serve,
        source="peecee",
        extra_args=("--audio-wav", str(wav_path), "--local-only"),
        env=env,
    ) as proc:
        base_url, token = extract_ui_url_and_token(proc, port_serve, timeout=5.0)
        ui = CardUIClient(base_url, token)

        async def _drive_shadow_quiet() -> dict[str, Any]:
            proactive_cards: list[dict[str, Any]] = []
            async with ui.ws_session() as ws:
                await ws.send_mode("Shadow")
                deadline = asyncio.get_running_loop().time() + 6.0
                while len(mock_nemo_ws.received_frames) < 10:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.4)
                await ws.send_mode("Quiet")
                while len(mock_nemo_ws.received_frames) < 18:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.3)
                # Verify zero proactive cards pushed while in Shadow / Quiet
                while True:
                    try:
                        msg = await ws.recv_json(timeout=0.15)
                    except TimeoutError:
                        break
                    if msg.get("type") in {"card", "card_update", "suggestion"} or msg.get(
                        "question"
                    ):
                        proactive_cards.append(msg)
                assert proactive_cards == []
                await ws.send_manual_request()
                return await ws.wait_for_card(timeout=3.0)

        manual_card = asyncio.run(_drive_shadow_quiet())
        assert manual_card.get("origin", "manual") == "manual"
        assert manual_card.get("question")

    live_db = e2e_workspace["db_path"]
    assert_six_conservation_invariants(live_db, sid)
    _run_report(live_db, sid)

    with open_db(live_db) as conn:
        live_sugs = fetch_normalized_rows(conn, "suggestions", sid)
        live_evs = fetch_normalized_rows(conn, "events", sid)
        live_tel = fetch_normalized_rows(conn, "telemetry_records", sid)

    proactive_shadow_sugs = [
        s
        for s in live_sugs
        if str(s.get("origin") or "proactive") == "proactive"
        and any(
            isinstance(tr, dict) and str(tr.get("transition") or "") == "displayed"
            for tr in (_json_field(s, "transitions") or [])
        )
    ]
    assert proactive_shadow_sugs
    proactive_shadow_ids = {str(s.get("suggestion_id") or "") for s in proactive_shadow_sugs}
    assert any(
        str(t.get("kind") or "") == "shadow_displayed"
        and str(t.get("suggestion_id") or "") in proactive_shadow_ids
        for t in live_tel
    )

    committed_segs = [
        {
            "schema_version": 1,
            "session_id": sid,
            "event_id": str(ev.get("event_id") or f"ev-{idx}"),
            "kind": "segment",
            "segment_id": str(ev["segment_id"]),
            "source_revision": int(ev.get("source_revision") or 1),
            "channel": int(ev.get("channel") or 1),
            "speaker_id": str(ev.get("speaker_id") or "participant-b"),
            "start_ms": int(ev.get("start_ms") or 0),
            "end_ms": int(ev.get("end_ms") or 0),
            "received_ms": int(ev.get("received_ms") or 0),
            "text": str(ev.get("text") or ""),
            "stability": "committed",
            "boundary": ev.get("boundary"),
            "source_sequence": int(ev.get("source_sequence") or idx),
        }
        for idx, ev in enumerate(live_evs, start=1)
        if ev.get("kind") == "segment" and ev.get("stability") == "committed"
    ]
    assert committed_segs
    extracted_path = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "t309_extracted.jsonl",
        committed_segs,
    )
    empty_labels = write_jsonl(e2e_workspace["root"] / "t309_empty.jsonl", [])
    replay_db = e2e_workspace["root"] / "t309_replay.db"
    rep_unlabeled = run_coach(
        "replay",
        [
            "--session",
            str(extracted_path),
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(replay_db),
            "--labels",
            str(empty_labels),
            "--json",
        ],
    )
    assert rep_unlabeled.returncode == 0, rep_unlabeled.stderr
    ev_out1 = rep_unlabeled.json().get("evaluation") or rep_unlabeled.json()
    assert int(ev_out1["unscored_pending_review_count"]) >= 1
    assert len(ev_out1.get("review_queue") or []) >= 1

    saved_labels = e2e_workspace["root"] / "t309_shadow_labels.jsonl"
    port_ann = free_port_factory()
    with start_coach_process(
        "annotate",
        [
            "--session",
            str(extracted_path),
            "--db",
            str(replay_db),
            "--out",
            str(saved_labels),
            "--pass",
            "2",
            "--port",
            str(port_ann),
        ],
        ready_url=f"http://127.0.0.1:{port_ann}/",
        startup_timeout=10.0,
    ) as proc_ann:
        ann_url, ann_tok = extract_ui_url_and_token(proc_ann, port_ann, timeout=5.0)
        ann = AnnotationClient(ann_url, token=ann_tok)
        queue = ann.get_review_queue()
        assert len(queue) >= 1
        for item in queue:
            cid = str(item["candidate_id"])
            ann.get_step1_view(cid)
            ann.submit_step1(
                cid,
                worthwhile=True,
                categories=["CONFIRM_OWNER"],
                example_questions=["Who will own the rollback plan?"],
                sample_source="shadow_review",
            )
            ann.unlock_step2(cid)
            ann.assert_step1_locked(
                cid,
                {
                    "worthwhile": False,
                    "categories": ["CONFIRM_TIME"],
                    "example_questions": [],
                    "none_reason": "no_gap",
                },
            )
            ann.submit_step2(cid, close_ms=30000, closed_by="answered")

    labeled = read_jsonl(saved_labels)
    assert len(labeled) == len(queue)
    assert all(rec["sample_source"] == "shadow_review" for rec in labeled)

    rep_labeled = run_coach(
        "replay",
        [
            "--session",
            str(extracted_path),
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["root"] / "t309_replay_after.db"),
            "--labels",
            str(saved_labels),
            "--json",
        ],
    )
    assert rep_labeled.returncode == 0, rep_labeled.stderr
    ev_out2 = rep_labeled.json().get("evaluation") or rep_labeled.json()
    assert int(ev_out2["unscored_pending_review_count"]) == 0
    assert int(ev_out2["useful_count"]) >= 1


def test_t3_10_local_only_template_only_zero_egress_and_manual_local_model(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    """--local-only -> template_only health + 0 hosted egress + manual local routing."""
    sid = "t3-10-local-only"
    write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    _write_nested_policy(
        e2e_workspace["policy_path"],
        generator_enabled=True,
        overrides={
            "generator": {
                "enabled": True,
                "provider": "hosted",
                "model_id": "gpt-6-luna",
                "reasoning": "low",
                "timeout_ms": 1500,
            },
            "categories": {
                "CLARIFY_SCOPE": {
                    "t_low": 0.20,
                    "t_compute": 0.45,
                    "t_display": 0.70,
                    "route": "generator",
                    "template": "clarify_scope_v1",
                    "required_lookback_s": 120,
                },
                "CONFIRM_OWNER": {
                    "t_low": 0.20,
                    "t_compute": 0.50,
                    "t_display": 0.75,
                    "route": "template",
                    "template": "confirm_owner_v1",
                    "required_lookback_s": 300,
                },
                "CONFIRM_TIME": {
                    "t_low": 0.20,
                    "t_compute": 0.50,
                    "t_display": 0.75,
                    "route": "template",
                    "template": "confirm_time_v1",
                    "required_lookback_s": 300,
                },
            },
            "display": {
                "settle_ms": 100,
                "cooldown_ms": 5000,
                "max_cards_per_10_min": 6,
                "max_candidate_age_ms": 8000,
            },
            "manual": {"lookback_s": 30, "score_floor": 0.30, "budget_ms": 2000},
        },
    )
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-t310-1",
            segment_id="seg-t310-1",
            text="We propose rolling out canary routing except for legacy tenants.",
            speaker_id="participant-b",
            start_ms=100,
            end_ms=400,
            received_ms=450,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t310-2",
            segment_id="seg-t310-2",
            text="Someone should own the rollback plan.",
            speaker_id="participant-c",
            start_ms=700,
            end_ms=1100,
            received_ms=1150,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t310-3",
            segment_id="seg-t310-3",
            text="We need a deadline to finish the staging cutover soon.",
            speaker_id="participant-c",
            start_ms=1300,
            end_ms=1600,
            received_ms=1650,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t310-4",
            segment_id="seg-t310-4",
            text="The chassis fans spin quietly at idle.",
            speaker_id="participant-b",
            start_ms=1900,
            end_ms=2300,
            received_ms=2350,
            source_sequence=4,
        ),
    ]
    sess_jsonl = write_events_jsonl(e2e_workspace["sessions_dir"] / "t310.jsonl", events)
    port = free_port_factory()

    with MockOpenAIServer() as hosted_mock, MockOpenAIServer() as local_mock:
        hosted_mock.set_default(
            MockCompletionSpec.valid_question("Hosted question should never fire?", ["seg-t310-1"])
        )
        local_mock.set_default(
            MockCompletionSpec.valid_question(
                "What verification checks confirm migration readiness?",
                ["seg-t310-2"],
            )
        )
        env = build_coach_test_env(
            openai_base_url=hosted_mock.base_url,
            extra_env={
                "COACH_LOCAL_OPENAI_BASE_URL": local_mock.base_url,
                "LOCAL_OPENAI_BASE_URL": local_mock.base_url,
            },
        )
        with _start_serve(
            e2e_workspace,
            port,
            source="replay",
            extra_args=(
                "--session",
                str(sess_jsonl),
                "--speed",
                "1.0",
                "--local-only",
            ),
            env=env,
        ) as proc:
            base_url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
            ui = CardUIClient(base_url, token)

            async def _drive() -> tuple[dict[str, Any], dict[str, Any], float]:
                async with ui.ws_session() as ws:
                    await ws.wait_for_health("template_only", timeout=5.0)
                    tpl_card = await ws.wait_for_card(timeout=5.0)
                    await asyncio.sleep(1.5)
                    t0 = time.monotonic()
                    await ws.send_manual_request()
                    manual_card = await ws.wait_for_card(timeout=2.0)
                    elapsed = time.monotonic() - t0
                    return tpl_card, manual_card, elapsed

            tpl_card, manual_card, elapsed_s = asyncio.run(_drive())
            assert tpl_card.get("question")
            assert manual_card.get("question")
            assert elapsed_s <= 2.0

        assert len(hosted_mock.requests) == 0

    db_path = e2e_workspace["db_path"]
    assert_local_only_no_egress(db_path, sid)
    assert_six_conservation_invariants(db_path, sid)
    report = _run_report(db_path, sid)
    timeline = [
        str(h.get("health") or h.get("state") or "") for h in report.get("health_timeline", [])
    ]
    assert "template_only" in timeline


def test_t3_11_stricter_attention_cap_single_deferred_slot_and_silence_timers(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    """Stricter-wins attention cap (brief=2 < policy=4) + deferred slot + silence timers."""
    sid = "t3-11-cap-defer-ttl"
    brief = build_session_brief(
        session_id=sid,
        attention={"max_cards_per_10_min": 2, "proactive": True},
    )
    pol = deep_merge_dict(
        build_policy_dict(
            settle_ms=50,
            cooldown_ms=500,
            max_cards_per_10_min=4,
        ),
        {
            "classifier": {
                "version": "llm_baseline_v0",
                "calibration": "llm_baseline_v0-platt-v0-a",
            },
            "display": {
                "display_ttl_ms": 400,
                "max_candidate_age_ms": 2500,
            },
        },
    )
    for score in (0.95, 0.78, 0.85, 0.91, 0.92):
        mock_openai.enqueue(
            MockCompletionSpec(
                status_code=200,
                content=json.dumps(
                    {
                        "opportunity": score,
                        "categories": {
                            "CONFIRM_OWNER": score,
                            "CONFIRM_TIME": 0.05,
                            "SUCCESS_CRITERION": 0.05,
                        },
                    }
                ),
            )
        )
    timings_ms = (100, 250, 400, 550, 1800)
    events: list[dict[str, Any]] = [
        build_segment_event(
            session_id=sid,
            event_id=f"ev-t311-{idx}",
            segment_id=f"seg-t311-{idx}",
            text=f"Someone should own deployment milestone {idx}.",
            speaker_id="participant-b" if idx % 2 == 0 else "participant-c",
            start_ms=max(0, rx_ms - 80),
            end_ms=rx_ms - 10,
            received_ms=rx_ms,
            source_sequence=idx + 1,
        )
        for idx, rx_ms in enumerate(timings_ms)
    ]
    events.append(
        build_control_event(
            session_id=sid,
            event_id="ev-t311-end",
            subtype="session_end",
            received_ms=4500,
            source_sequence=len(events) + 1,
        )
    )
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    db_path, _ = _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol,
        name="t311_cap",
        extra_args=("--speed", "1.0"),
        env=env,
    )
    assert_six_conservation_invariants(db_path, sid)
    report = _run_report(db_path, sid)
    assert report["effective_max_cards_per_10_min"] == 2
    assert int(report["displayed_card_count"]) <= 2
    assert report["budget_compliant"] is True

    with open_db(db_path) as conn:
        sugs = fetch_normalized_rows(conn, "suggestions", sid)
    all_codes = set().union(*(_extract_reason_codes(s) for s in sugs))
    assert "displaced_by_better" in all_codes
    assert "display_ttl_passed" in all_codes
    assert "deferred_released" in all_codes


def test_t3_12_train_masked_bce_calibrate_rejection_fit_routing_and_replay_eval(
    e2e_workspace: dict[str, Path],
) -> None:
    """coach train (masked BCE) + calibrate (non-random reject/random fit) + replay --labels."""
    train_rows = (
        [
            build_decision_context_row(
                candidate_id=f"t312-pos-{i}",
                session_id=f"t312-tr-{i % 3}",
                anchor_text="Someone should own the rollback plan before Friday.",
                worthwhile=True,
                categories=("CONFIRM_OWNER",),
                sample_source="random",
            )
            for i in range(6)
        ]
        + [
            build_decision_context_row(
                candidate_id=f"t312-neg-{i}",
                session_id=f"t312-tr-{i % 3}",
                anchor_text="The server rack uses twenty four volt power supplies.",
                worthwhile=False,
                categories=(),
                sample_source="random",
            )
            for i in range(6)
        ]
        + [
            build_decision_context_row(
                candidate_id=f"t312-mask-{i}",
                session_id=f"t312-tr-{i % 2}",
                anchor_text="What happens if both regions fail simultaneously?",
                worthwhile=True,
                categories=(),
                sample_source="pass1_mark",
            )
            for i in range(3)
        ]
    )
    train_path = write_jsonl(e2e_workspace["root"] / "t312_train.jsonl", train_rows)
    model_dir = e2e_workspace["root"] / "t312_model"
    tr_res = run_coach(
        "train",
        ["--data", str(train_path), "--out", str(model_dir), "--backend", "embed_logreg"],
    )
    assert tr_res.returncode == 0, f"coach train failed: {tr_res.stderr}\n{tr_res.stdout}"
    assert model_dir.exists() and any(model_dir.iterdir())

    mixed_cal_rows = [
        build_label_record(
            label_id="t312-mix-1",
            candidate_id="t312-mix-c1",
            sample_source="random",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
        ),
        build_label_record(
            label_id="t312-mix-2",
            candidate_id="t312-mix-c2",
            sample_source="model_positive",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
        ),
    ]
    mixed_cal_path = write_jsonl(e2e_workspace["root"] / "t312_cal_mixed.jsonl", mixed_cal_rows)
    bad_cal_out = e2e_workspace["root"] / "t312_bad_cal.json"
    bad_cal_res = run_coach(
        "calibrate",
        [
            "--data",
            str(mixed_cal_path),
            "--classifier",
            str(model_dir),
            "--mode",
            "technical_working_discussion",
            "--out",
            str(bad_cal_out),
        ],
    )
    assert bad_cal_res.returncode != 0
    assert not bad_cal_out.exists()

    cal_rows = [
        build_decision_context_row(
            candidate_id=f"t312-cal-pos-{i}",
            session_id=f"t312-cal-{i % 4}",
            anchor_text="Someone should own the cutover checklist.",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            sample_source="random",
        )
        for i in range(6)
    ] + [
        build_decision_context_row(
            candidate_id=f"t312-cal-neg-{i}",
            session_id=f"t312-cal-{i % 4}",
            anchor_text="Everything is already documented in the ticket.",
            worthwhile=False,
            categories=(),
            sample_source="random",
        )
        for i in range(6)
    ]
    cal_path = write_jsonl(e2e_workspace["root"] / "t312_cal_random.jsonl", cal_rows)
    cal_out = e2e_workspace["root"] / "t312_cal_platt.json"
    cal_res = run_coach(
        "calibrate",
        [
            "--data",
            str(cal_path),
            "--classifier",
            str(model_dir),
            "--mode",
            "technical_working_discussion",
            "--out",
            str(cal_out),
        ],
    )
    assert cal_res.returncode == 0, f"coach calibrate failed: {cal_res.stderr}\n{cal_res.stdout}"
    cal_data = json.loads(cal_out.read_text(encoding="utf-8"))
    assert cal_data["calibration_version"]
    assert cal_data["classifier_version"]
    assert cal_data["builder_version"] == "ctx-v0"
    assert cal_data["prefilter_version"] == "pre-v0"
    assert "brier_score" in cal_data
    assert "reliability_bins" in cal_data

    sid = "t3-12-train-cal-eval"
    pol = deep_merge_dict(
        build_policy_dict(
            calibration=str(cal_data["calibration_version"]),
            settle_ms=100,
            cooldown_ms=100,
            max_cards_per_10_min=6,
        ),
        {
            "classifier": {
                "version": str(cal_data["classifier_version"]),
                "model_path": str(model_dir),
                "calibration_path": str(cal_out),
            },
            "categories": {
                "CONFIRM_OWNER": {"t_low": 0.20, "t_compute": 0.50, "t_display": 0.75},
            },
        },
    )
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-t312-1",
            segment_id="seg-t312-1",
            text="Someone should own the rollback plan before Friday.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=2500,
            received_ms=2600,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t312-2",
            segment_id="seg-t312-2",
            text="Someone should own the secondary canary dashboard.",
            speaker_id="participant-c",
            start_ms=6000,
            end_ms=7500,
            received_ms=7600,
            source_sequence=2,
        ),
    ]
    eval_labels = [
        build_label_record(
            label_id="lbl-t312-1",
            session_id=sid,
            candidate_id="cand-t312-1",
            sample_source="random",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            evidence_refs=({"segment_id": "seg-t312-1", "revision": 1},),
            open_ms=2500,
            ideal_ms=2800,
            close_ms=15000,
            closed_by="answered",
        )
    ]
    eval_labels_path = write_jsonl(e2e_workspace["root"] / "t312_eval_labels.jsonl", eval_labels)
    db_path, out = _replay_session(
        e2e_workspace,
        events,
        brief=build_session_brief(session_id=sid),
        policy=pol,
        name="t312_eval",
        extra_args=("--labels", str(eval_labels_path)),
    )
    assert_six_conservation_invariants(db_path, sid)
    _run_report(db_path, sid)

    with open_db(db_path) as conn:
        decs = fetch_normalized_rows(conn, "decisions", sid)
    assert len(decs) == 2
    assert all(d.get("classifier_version") == cal_data["classifier_version"] for d in decs)
    assert all(d.get("calibration_version") == cal_data["calibration_version"] for d in decs)
    for d in decs:
        assert _extract_reason_codes(d).issubset(REASON_CODES)

    ev_metrics = out.get("evaluation") or out
    assert "useful_count" in ev_metrics
    assert "unscored_pending_review_count" in ev_metrics
