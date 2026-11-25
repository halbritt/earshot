"""Tier 1 opaque-box E2E tests for Area A: CLI, contracts, reducer, storage, telemetry."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from tests.e2e.helpers.cli_runner import run_coach
from tests.e2e.helpers.db_assertions import (
    assert_model_cache_has_entries,
    assert_session_text_deleted,
    assert_six_conservation_invariants,
    assert_tables_exist,
    assert_wal_mode,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
    REASON_CODES,
    build_boundary_event,
    build_brief_update_event,
    build_control_event,
    build_ledger_event,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    build_speaker_update_event,
    write_brief_json,
    write_events_jsonl,
    write_policy_dict,
)


def _list_session_run_ids(db_path: Path, session_id: str) -> list[str]:
    with open_db(db_path, readonly=True) as conn:
        rows = conn.execute(
            "SELECT run_id FROM sessions WHERE session_id = ? ORDER BY rowid ASC",
            (session_id,),
        ).fetchall()
    return [str(r[0]) for r in rows]


def _replay_args(
    *,
    session_path: Path,
    brief_path: Path,
    policy_path: Path,
    db_path: Path,
) -> list[str]:
    return [
        "--session",
        str(session_path),
        "--brief",
        str(brief_path),
        "--policy",
        str(policy_path),
        "--db",
        str(db_path),
        "--json",
    ]


def test_session_brief_and_canonical_event_schema_validation(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-schema-v1"
    brief_path = write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(
            session_id=sid,
            consent={"basis": "stated", "note": "verbal notice at kickoff"},
        ),
    )
    events = [
        build_control_event(
            session_id=sid,
            event_id="ev-ctrl-start",
            subtype="session_start",
            received_ms=0,
            source_sequence=0,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-seg-1",
            segment_id="seg-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3500,
            received_ms=3700,
            source_sequence=1,
        ),
        build_speaker_update_event(
            session_id=sid,
            event_id="ev-spk-1",
            segment_id="seg-1",
            speaker_id="participant-c",
            source_revision=2,
            received_ms=3900,
            source_sequence=2,
            speaker_confidence=0.95,
        ),
        build_boundary_event(
            session_id=sid,
            event_id="ev-bnd-1",
            channel=1,
            speaker_id="participant-c",
            end_ms=3500,
            received_ms=4100,
            source_sequence=3,
        ),
        build_brief_update_event(
            session_id=sid,
            event_id="ev-brief-1",
            brief_patch={"objective": "Confirm rollback owner and timing."},
            received_ms=4500,
            source_sequence=4,
        ),
        build_ledger_event(
            session_id=sid,
            event_id="ev-ledger-1",
            suggestion_id="sug-schema-1",
            transition="displayed",
            cause="boundary_after_settle",
            at_ms=5000,
            received_ms=5000,
            source_sequence=5,
        ),
    ]
    session_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "valid.jsonl", events)
    db_path = e2e_workspace["db_path"]

    res = run_coach(
        "replay",
        _replay_args(
            session_path=session_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_path,
        ),
    )
    assert res.returncode == 0, res.stderr

    with open_db(db_path, readonly=True) as conn:
        sess_rows = fetch_normalized_rows(conn, "sessions", sid)
        assert len(sess_rows) == 1
        assert sess_rows[0]["consent_basis"] == "stated"
        assert sess_rows[0]["consent_note"] == "verbal notice at kickoff"
        ev_rows = fetch_normalized_rows(conn, "events", sid)
        kinds = {str(r["kind"]) for r in ev_rows}
        assert {"control", "segment", "speaker_update", "boundary", "brief_update"}.issubset(kinds)

    bad_brief_path = write_brief_json(
        e2e_workspace["root"] / "bad_brief.json",
        build_session_brief(session_id=sid, schema_version=99),
    )
    bad_brief_res = run_coach(
        "replay",
        _replay_args(
            session_path=session_path,
            brief_path=bad_brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=e2e_workspace["root"] / "bad_brief.db",
        ),
    )
    assert bad_brief_res.returncode != 0
    assert (bad_brief_res.stderr + bad_brief_res.stdout).strip()

    bad_kind_ev = dict(events[1])
    bad_kind_ev["kind"] = "bogus"
    bad_kind_path = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "bad_kind.jsonl", [bad_kind_ev]
    )
    bad_kind_res = run_coach(
        "replay",
        _replay_args(
            session_path=bad_kind_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=e2e_workspace["root"] / "bad_kind.db",
        ),
    )
    assert bad_kind_res.returncode != 0
    assert (bad_kind_res.stderr + bad_kind_res.stdout).strip()


def test_closed_reason_code_and_invalid_ledger_transition_rejection(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-reason-ledger"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-seg-1",
            segment_id="seg-1",
            text="Someone should own the rollback plan.",
            start_ms=1000,
            end_ms=3200,
            received_ms=3400,
            source_sequence=1,
        ),
        build_ledger_event(
            session_id=sid,
            event_id="ev-bad-dismiss",
            suggestion_id="sug-bogus-1",
            transition="dismissed",
            cause="user_action",
            at_ms=4000,
            received_ms=4000,
            source_sequence=2,
        ),
        build_ledger_event(
            session_id=sid,
            event_id="ev-bad-used",
            suggestion_id="sug-bogus-1",
            transition="used",
            cause="user_action",
            at_ms=4200,
            received_ms=4200,
            source_sequence=3,
        ),
    ]
    sess_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "ledger.jsonl", events)
    db_path = e2e_workspace["db_path"]

    res = run_coach(
        "replay",
        _replay_args(
            session_path=sess_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_path,
        ),
    )
    assert res.returncode == 0, res.stderr

    with open_db(db_path, readonly=True) as conn:
        sess_rows = fetch_normalized_rows(conn, "sessions", sid)
        assert len(sess_rows) == 1
        counters = json.loads(str(sess_rows[0]["reducer_counters_json"]))
        assert int(counters.get("invalid_ledger_transition", 0)) >= 1
        sug_rows = fetch_normalized_rows(conn, "suggestions", sid)
        for sug in sug_rows:
            transitions = sug.get("transitions") or []
            states = [str(t.get("transition")) for t in transitions if isinstance(t, dict)]
            assert not ("dismissed" in states and "used" in states)

    bad_reason_ev = build_ledger_event(
        session_id=sid,
        event_id="ev-unknown-code",
        suggestion_id="sug-bogus-1",
        transition="displayed",
        cause="invented_code",
        at_ms=5000,
        received_ms=5000,
        source_sequence=4,
    )
    bad_reason_path = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "bad_reason.jsonl", [events[0], bad_reason_ev]
    )
    bad_res = run_coach(
        "replay",
        _replay_args(
            session_path=bad_reason_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=e2e_workspace["root"] / "bad_reason.db",
        ),
    )
    assert bad_res.returncode != 0

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE decisions SET terminal_reason = 'invented_code' WHERE session_id = ?",
            (sid,),
        )
        conn.commit()
    rep_res = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert rep_res.returncode == 2
    assert rep_res.json()["invariants"]["passed"] is False


def test_reducer_dedup_revision_ranking_nfc_whitespace_and_ordering(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-reducer-prop"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    ev_stale_3 = build_segment_event(
        session_id=sid,
        event_id="ev-stale-3",
        segment_id="seg-stale",
        text="Latest rollback owner text.",
        start_ms=1000,
        end_ms=2500,
        received_ms=2600,
        source_revision=3,
        source_sequence=1,
    )
    events = [
        ev_stale_3,
        dict(ev_stale_3),
        build_segment_event(
            session_id=sid,
            event_id="ev-stale-1",
            segment_id="seg-stale",
            text="Older stale text.",
            start_ms=1000,
            end_ms=2500,
            received_ms=2700,
            source_revision=1,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-nfc-1",
            segment_id="seg-nfc",
            text="Someone should own the re\u0301sume\u0301  plan.",
            start_ms=3000,
            end_ms=5000,
            received_ms=5100,
            source_revision=1,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-nfc-2",
            segment_id="seg-nfc",
            text="Someone should own the r\u00e9sum\u00e9 plan.",
            start_ms=3000,
            end_ms=5000,
            received_ms=5200,
            source_revision=2,
            source_sequence=4,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-nfc-3",
            segment_id="seg-nfc",
            text="Someone should own the r\u00e9sum\u00e9 plan?",
            start_ms=3000,
            end_ms=5000,
            received_ms=5300,
            source_revision=3,
            source_sequence=5,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-earlier",
            segment_id="seg-earlier",
            text="Let us review the deployment checklist.",
            start_ms=200,
            end_ms=900,
            received_ms=5400,
            source_revision=1,
            source_sequence=6,
        ),
    ]
    sess_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "reducer.jsonl", events)
    db_path = e2e_workspace["db_path"]

    res = run_coach(
        "replay",
        _replay_args(
            session_path=sess_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_path,
        ),
    )
    assert res.returncode == 0, res.stderr

    rep = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert rep.returncode == 0, rep.stderr
    counters = rep.json()["reducer_counters"]
    assert counters["duplicate"] >= 1
    assert counters["stale_revision"] >= 1
    assert counters["material_revisions_after_commit"] >= 1

    with open_db(db_path, readonly=True) as conn:
        ev_rows = fetch_normalized_rows(conn, "events", sid)
        by_eid = {str(r["event_id"]): r for r in ev_rows}
        assert by_eid["ev-nfc-1"]["canonical_revision"] == 1
        assert by_eid["ev-nfc-2"]["canonical_revision"] == 1
        assert by_eid["ev-nfc-3"]["canonical_revision"] == 2
        assert "ev-stale-1" not in by_eid or by_eid["ev-stale-1"]["canonical_revision"] is None

    seg_a = build_segment_event(
        session_id=sid,
        event_id="ev-ord-a",
        segment_id="seg-ord-a",
        text="First segment by start time.",
        start_ms=500,
        end_ms=1500,
        received_ms=1600,
        source_sequence=1,
    )
    seg_b = build_segment_event(
        session_id=sid,
        event_id="ev-ord-b",
        segment_id="seg-ord-b",
        text="Someone should own the rollback plan.",
        start_ms=2500,
        end_ms=4200,
        received_ms=4400,
        source_sequence=2,
    )
    seg_c = build_segment_event(
        session_id=sid,
        event_id="ev-ord-c",
        segment_id="seg-ord-c",
        text="When will we run the staging cutover?",
        start_ms=5000,
        end_ms=6800,
        received_ms=7000,
        source_sequence=3,
    )
    path_1 = write_events_jsonl(e2e_workspace["sessions_dir"] / "ord1.jsonl", [seg_a, seg_b, seg_c])
    path_2 = write_events_jsonl(e2e_workspace["sessions_dir"] / "ord2.jsonl", [seg_c, seg_a, seg_b])
    db_1 = e2e_workspace["root"] / "ord1.db"
    db_2 = e2e_workspace["root"] / "ord2.db"
    for p, db in ((path_1, db_1), (path_2, db_2)):
        r_ord = run_coach(
            "replay",
            _replay_args(
                session_path=p,
                brief_path=brief_path,
                policy_path=e2e_workspace["policy_path"],
                db_path=db,
            ),
        )
        assert r_ord.returncode == 0, r_ord.stderr

    def _canonical_view(db: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        with open_db(db, readonly=True) as conn:
            segs = conn.execute(
                "SELECT segment_id, text FROM events WHERE session_id = ? AND kind = 'segment' "
                "AND stability = 'committed' "
                "ORDER BY start_ms ASC, channel ASC, source_sequence ASC, segment_id ASC",
                (sid,),
            ).fetchall()
            decs = conn.execute(
                "SELECT candidate_id, serialized_sha256 FROM decisions WHERE session_id = ? "
                "ORDER BY candidate_id ASC",
                (sid,),
            ).fetchall()
        return (
            [(str(r[0]), str(r[1])) for r in segs],
            [(str(r[0]), str(r[1])) for r in decs],
        )

    segs_1, decs_1 = _canonical_view(db_1)
    segs_2, decs_2 = _canonical_view(db_2)
    assert [s[0] for s in segs_1] == ["seg-ord-a", "seg-ord-b", "seg-ord-c"]
    assert segs_1 == segs_2
    assert decs_1 == decs_2


def test_sqlite_wal_tables_and_retention_opt_out(
    e2e_workspace: dict[str, Path],
) -> None:
    sid_out = "sess-ret-opt-out"
    sid_in = "sess-ret-opt-in"
    brief_out = write_brief_json(
        e2e_workspace["root"] / "brief_out.json",
        build_session_brief(
            session_id=sid_out,
            retention={"store_transcript": False, "store_raw_provider_events": False},
        ),
    )
    brief_in = write_brief_json(
        e2e_workspace["root"] / "brief_in.json",
        build_session_brief(
            session_id=sid_in,
            retention={"store_transcript": True, "store_raw_provider_events": True},
        ),
    )
    ev_out = [
        build_segment_event(
            session_id=sid_out,
            event_id="ev-out-1",
            segment_id="seg-out-1",
            text="Someone should own the rollback plan.",
            start_ms=1000,
            end_ms=3200,
            received_ms=3400,
            source_sequence=1,
        )
    ]
    ev_in = [
        build_segment_event(
            session_id=sid_in,
            event_id="ev-in-1",
            segment_id="seg-in-1",
            text="Someone should own the rollback plan.",
            start_ms=1000,
            end_ms=3200,
            received_ms=3400,
            source_sequence=1,
        )
    ]
    p_out = write_events_jsonl(e2e_workspace["sessions_dir"] / "out.jsonl", ev_out)
    p_in = write_events_jsonl(e2e_workspace["sessions_dir"] / "in.jsonl", ev_in)
    db_out = e2e_workspace["root"] / "opt_out.db"
    db_in = e2e_workspace["root"] / "opt_in.db"

    r_out = run_coach(
        "replay",
        _replay_args(
            session_path=p_out,
            brief_path=brief_out,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_out,
        ),
    )
    assert r_out.returncode == 0, r_out.stderr
    assert assert_wal_mode(db_out) == "wal"

    with open_db(db_out, readonly=True) as conn:
        assert_tables_exist(conn)
        assert fetch_normalized_rows(conn, "raw_events", sid_out) == []
        ev_rows = fetch_normalized_rows(conn, "events", sid_out)
        assert len(ev_rows) >= 1
        assert all(r.get("text") is None for r in ev_rows)
        dec_rows = fetch_normalized_rows(conn, "decisions", sid_out)
        assert len(dec_rows) >= 1
        assert all(r.get("serialized_text") is None for r in dec_rows)
        assert all(bool(r.get("serialized_sha256")) for r in dec_rows)
        sug_rows = fetch_normalized_rows(conn, "suggestions", sid_out)
        assert all(r.get("text") is None for r in sug_rows)

    r_in = run_coach(
        "replay",
        _replay_args(
            session_path=p_in,
            brief_path=brief_in,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_in,
        ),
    )
    assert r_in.returncode == 0, r_in.stderr
    with open_db(db_in, readonly=True) as conn:
        ev_in_rows = fetch_normalized_rows(conn, "events", sid_in)
        dec_in_rows = fetch_normalized_rows(conn, "decisions", sid_in)
        assert any(r.get("text") is not None for r in ev_in_rows)
        assert any(r.get("serialized_text") is not None for r in dec_in_rows)


def test_coach_delete_session_preserves_nulled_decisions_and_purges_text(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-delete-apollo"
    secret = "Project Apollo secret"
    brief_path = write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(
            session_id=sid,
            consent={"basis": "stated", "note": f"Notice for {secret}"},
            retention={"store_transcript": True, "store_raw_provider_events": True},
        ),
    )
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-del-1",
            segment_id="seg-del-1",
            text=f"Someone should own the {secret} rollback plan.",
            start_ms=1000,
            end_ms=3500,
            received_ms=3700,
            source_sequence=1,
        ),
        build_boundary_event(
            session_id=sid,
            event_id="ev-del-bnd",
            channel=1,
            speaker_id="participant-b",
            end_ms=3500,
            received_ms=3800,
            source_sequence=2,
        ),
    ]
    sess_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "del.jsonl", events)
    db_path = e2e_workspace["db_path"]

    rep_in = run_coach(
        "replay",
        _replay_args(
            session_path=sess_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_path,
        ),
    )
    assert rep_in.returncode == 0, rep_in.stderr

    with open_db(db_path, readonly=True) as conn:
        ev_before = fetch_normalized_rows(conn, "events", sid)
        assert any(secret in str(r.get("text") or "") for r in ev_before)

    del_res = run_coach(
        "delete-session",
        ["--db", str(db_path), "--session", sid, "--json"],
    )
    assert del_res.returncode == 0, del_res.stderr
    summary: dict[str, Any] = del_res.json()
    expected_keys = {
        "session_id",
        "raw_events_deleted",
        "events_redacted",
        "decisions_redacted",
        "suggestions_redacted",
        "feedback_redacted",
        "model_cache_purged",
        "deleted_at_ms",
        "telemetry_redacted",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["session_id"] == sid
    assert int(summary["deleted_at_ms"]) > 0
    assert int(summary["events_redacted"]) >= 1
    assert int(summary["decisions_redacted"]) >= 1

    assert_session_text_deleted(
        db_path,
        sid,
        forbidden_substrings=(secret,),
        require_decisions_preserved=True,
    )
    rep_after = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert rep_after.returncode == 0, rep_after.stderr


def test_coach_explain_window_query_returns_anchors_scores_and_reasons(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-explain-win"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-early",
            segment_id="seg-early",
            text="We reviewed the agenda before the call.",
            start_ms=1000,
            end_ms=1900,
            received_ms=2000,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-backchannel",
            segment_id="seg-backchannel",
            text="Yeah",
            start_ms=11500,
            end_ms=11900,
            received_ms=12000,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-none",
            segment_id="seg-none",
            text="The weather outside looks sunny and calm today.",
            start_ms=19000,
            end_ms=20800,
            received_ms=21000,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-route",
            segment_id="seg-route",
            text="Someone should own the database cutover plan.",
            start_ms=25000,
            end_ms=26800,
            received_ms=27000,
            source_sequence=4,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-late",
            segment_id="seg-late",
            text="When will the staging verification start?",
            start_ms=37000,
            end_ms=37900,
            received_ms=38000,
            source_sequence=5,
        ),
    ]
    sess_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "explain.jsonl", events)
    db_path = e2e_workspace["db_path"]

    rep_res = run_coach(
        "replay",
        _replay_args(
            session_path=sess_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_path,
        ),
    )
    assert rep_res.returncode == 0, rep_res.stderr

    exp_res = run_coach(
        "explain",
        [
            "--db",
            str(db_path),
            "--session",
            sid,
            "--at-ms",
            "30000",
            "--window-ms",
            "20000",
            "--json",
        ],
    )
    assert exp_res.returncode == 0, exp_res.stderr
    data: dict[str, Any] = exp_res.json()
    assert data["session_id"] == sid
    assert data["at_ms"] == 30000
    assert data["window_ms"] == 20000
    anchors: list[dict[str, Any]] = data["anchors"]
    assert len(anchors) >= 3
    for anchor in anchors:
        rx = int(anchor["anchor_received_ms"])
        assert 10000 <= rx <= 30000
        assert anchor["candidate_id"]
        assert anchor["terminal_state"]
        assert anchor["terminal_reason"] in REASON_CODES
        if anchor.get("outcome") in {"NONE", "ABSTAIN", "SUPPRESS", "ROUTE"}:
            assert isinstance(anchor["raw_scores"], dict)
            assert isinstance(anchor["calibrated_scores"], dict)
    reasons = {str(a["terminal_reason"]) for a in anchors}
    assert "backchannel" in reasons
    assert any(a.get("outcome") == "ROUTE" for a in anchors)


def test_coach_diff_compares_runs_and_uses_model_cache(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-diff-cache"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-diff-1",
            segment_id="seg-diff-1",
            text="Someone should own the rollback plan.",
            start_ms=1000,
            end_ms=3400,
            received_ms=3600,
            source_sequence=1,
        ),
        build_boundary_event(
            session_id=sid,
            event_id="ev-diff-bnd",
            channel=1,
            speaker_id="participant-b",
            end_ms=3400,
            received_ms=3700,
            source_sequence=2,
        ),
    ]
    sess_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "diff.jsonl", events)
    db_path = e2e_workspace["db_path"]

    pol_strict = build_policy_dict()
    for cat in pol_strict["categories"].values():
        cat["t_low"] = 0.90
        cat["t_compute"] = 0.95
        cat["t_display"] = 0.99
    pol_strict_path = write_policy_dict(e2e_workspace["policies_dir"] / "strict.yaml", pol_strict)

    pol_loose = build_policy_dict(settle_ms=100)
    for cat in pol_loose["categories"].values():
        cat["t_low"] = 0.10
        cat["t_compute"] = 0.25
        cat["t_display"] = 0.40
    pol_loose_path = write_policy_dict(e2e_workspace["policies_dir"] / "loose.yaml", pol_loose)

    r_a = run_coach(
        "replay",
        _replay_args(
            session_path=sess_path,
            brief_path=brief_path,
            policy_path=pol_strict_path,
            db_path=db_path,
        ),
    )
    assert r_a.returncode == 0, r_a.stderr
    assert_model_cache_has_entries(db_path, min_count=1)

    r_b = run_coach(
        "replay",
        _replay_args(
            session_path=sess_path,
            brief_path=brief_path,
            policy_path=pol_loose_path,
            db_path=db_path,
        ),
    )
    assert r_b.returncode == 0, r_b.stderr

    run_ids = _list_session_run_ids(db_path, sid)
    assert len(run_ids) >= 2
    run_a, run_b = run_ids[-2], run_ids[-1]

    diff_res = run_coach(
        "diff",
        ["--db", str(db_path), "--run-a", run_a, "--run-b", run_b, "--session", sid, "--json"],
    )
    assert diff_res.returncode == 0, diff_res.stderr
    diff_data: dict[str, Any] = diff_res.json()
    assert diff_data["session_id"] == sid
    assert diff_data["run_a"] == run_a
    assert diff_data["run_b"] == run_b
    entries = [*diff_data["gained"], *diff_data["lost"], *diff_data["changed"]]
    assert len(entries) >= 1
    for entry in entries:
        assert entry["candidate_id"]
        assert entry["change_kind"] in {"gained", "lost", "changed"}
        assert entry["reason_a"]
        assert entry["reason_b"]


def test_coach_report_conservation_invariants_and_bounded_recorder_drops(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-report-inv"
    brief_path = write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(
            session_id=sid,
            consent={"basis": "stated", "note": "team agreed"},
        ),
    )
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-rep-1",
            segment_id="seg-rep-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3400,
            received_ms=3600,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-rep-2",
            segment_id="seg-rep-2",
            text="Okay",
            speaker_id="participant-c",
            start_ms=4000,
            end_ms=4300,
            received_ms=4400,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-rep-3",
            segment_id="seg-rep-3",
            text="I will confirm the staging deploy window.",
            speaker_id="me",
            start_ms=5000,
            end_ms=6800,
            received_ms=7000,
            source_sequence=3,
        ),
    ]
    sess_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "rep.jsonl", events)
    db_path = e2e_workspace["db_path"]

    replay_res = run_coach(
        "replay",
        _replay_args(
            session_path=sess_path,
            brief_path=brief_path,
            policy_path=e2e_workspace["policy_path"],
            db_path=db_path,
        ),
        env={"COACH_RECORDER_MAXSIZE": "2", "COACH_RECORDER_BUFFER_SIZE": "2"},
    )
    assert replay_res.returncode == 0, replay_res.stderr

    rep_res = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert rep_res.returncode == 0, rep_res.stderr
    report: dict[str, Any] = rep_res.json()
    for field in (
        "duration_ms",
        "mode",
        "consent_basis",
        "health_timeline",
        "effective_max_cards_per_10_min",
        "outcomes_by_terminal_state",
        "outcomes_by_reason_code",
        "latency_by_route",
        "upstream_delay_ms",
        "generator_usage",
        "recorder_drops",
        "invariants",
        "displayed_cards",
    ):
        assert field in report, f"Missing report field {field!r}"
    assert report["invariants"]["passed"] is True
    assert report["recorder_drops"].get("lifecycle", 0) == 0
    assert report["recorder_drops"].get("decision", 0) == 0

    summary = assert_six_conservation_invariants(db_path, sid)
    assert summary["non_recipient_segments"] == 2
    assert summary["candidates"] >= 2

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE decisions SET anchor_refs_json = ? WHERE session_id = ? "
            "AND candidate_id = (SELECT candidate_id FROM decisions WHERE session_id = ? LIMIT 1)",
            (json.dumps([{"segment_id": "seg-nonexistent", "revision": 99}]), sid, sid),
        )
        conn.commit()
    bad_rep = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert bad_rep.returncode == 2
    assert bad_rep.json()["invariants"]["passed"] is False
