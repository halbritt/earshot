"""Unit tests for coach.telemetry and coach.cli (§4.2, §13.2–§13.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coach.cli import _try_load_dataset_rows, main
from coach.contracts import (
    CanonicalEvent,
    Category,
    ClassifierResult,
    ConsentBasis,
    ConsentConfig,
    DecisionContext,
    EgressRecord,
    EventKind,
    FeedbackReason,
    LedgerEvent,
    LedgerState,
    Mode,
    Origin,
    Outcome,
    ReasonCode,
    RecipientConfig,
    RetentionConfig,
    Route,
    RoutingDecision,
    ScoreSet,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    SuggestionCandidate,
    TelemetryLevel,
    TelemetryRecord,
)
from coach.storage import SQLiteStore
from coach.telemetry import (
    BoundedRecorderBuffer,
    build_session_report,
    check_conservation_invariants,
    diff_runs,
    explain_silence,
)


def _brief(session_id: str = "s-tel") -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id=session_id,
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="me", role="reviewer"),
        objective="Leave with a rollback owner.",
        consent=ConsentConfig(basis=ConsentBasis.STATED, note="Explained at start"),
    )


def _seg_event(
    session_id: str,
    event_id: str,
    seg_id: str,
    *,
    speaker: str = "S1",
    start_ms: int = 1000,
    end_ms: int = 2000,
    received_ms: int = 2100,
    seq: int = 1,
) -> CanonicalEvent:
    return CanonicalEvent(
        schema_version=1,
        session_id=session_id,
        event_id=event_id,
        kind=EventKind.SEGMENT,
        received_ms=received_ms,
        source_sequence=seq,
        segment_id=seg_id,
        source_revision=1,
        channel=0,
        speaker_id=speaker,
        start_ms=start_ms,
        end_ms=end_ms,
        text="Who owns this?",
        stability=Stability.COMMITTED,
    )


def _ctx(session_id: str, cid: str, seg_id: str, rx_ms: int) -> DecisionContext:
    return DecisionContext(
        schema_version=1,
        candidate_id=cid,
        session_id=session_id,
        origin=Origin.PROACTIVE,
        based_on_state_version=1,
        builder_version="ctx-v0",
        prefilter_version="pre-v0",
        anchor_refs=(SegmentRef(seg_id, 1),),
        context_refs=(),
        anchor_received_ms=rx_ms,
        display_by_ms=rx_ms + 8000,
        token_counts={"total": 20, "limit": 512},
        omitted={"turns": 0},
        flags={"speaker_uncertain": False},
        serialized_sha256=f"sha-{cid}",
    )


def test_bounded_recorder_drops_debug_never_lifecycle_or_decision(tmp_path: Path) -> None:
    buf = BoundedRecorderBuffer(maxsize=3)
    d1 = TelemetryRecord(1, "d1", "s1", 100, TelemetryLevel.DEBUG, "debug_trace")
    d2 = TelemetryRecord(1, "d2", "s1", 101, TelemetryLevel.DEBUG, "debug_trace")
    d3 = TelemetryRecord(1, "d3", "s1", 102, TelemetryLevel.DEBUG, "debug_trace")
    d4 = TelemetryRecord(1, "d4", "s1", 103, TelemetryLevel.DEBUG, "debug_trace")
    lc = TelemetryRecord(1, "lc1", "s1", 104, TelemetryLevel.LIFECYCLE, "displayed")
    dec = TelemetryRecord(1, "dec1", "s1", 105, TelemetryLevel.DECISION, "decision")

    assert buf.emit(d1)
    assert buf.emit(d2)
    assert buf.emit(d3)
    assert not buf.emit(d4)  # incoming debug dropped at capacity
    assert buf.emit(lc)  # evicts oldest debug (d1)
    assert buf.emit(dec)  # evicts oldest debug (d2)
    # Even if critical records exceed maxsize, never drop lifecycle or decision
    lc2 = TelemetryRecord(1, "lc2", "s1", 106, TelemetryLevel.LIFECYCLE, "used")
    dec2 = TelemetryRecord(1, "dec2", "s1", 107, TelemetryLevel.DECISION, "decision")
    assert buf.emit(lc2)
    assert buf.emit(dec2)

    drops = buf.drops_by_level()
    assert drops[TelemetryLevel.DEBUG] == 4
    assert drops[TelemetryLevel.LIFECYCLE] == 0
    assert drops[TelemetryLevel.DECISION] == 0

    with SQLiteStore(tmp_path / "rec.db") as store:
        flushed = buf.flush_to_store(store, "s1", "r1")
        assert flushed == 4
        persisted = store.get_telemetry_records("s1", "r1")
        assert {r.record_id for r in persisted} == {"lc1", "dec1", "lc2", "dec2"}
        assert store.get_recorder_drops("s1", "r1") == {"debug": 4}


def _populate_valid_run(store: SQLiteStore, session_id: str, run_id: str) -> None:
    brief = _brief(session_id)
    store.upsert_session(
        brief=brief,
        run_id=run_id,
        runtime_mode=Mode.PROACTIVE,
        effective_max_cards_per_10_min=3,
        health_timeline=[{"at_ms": 0, "state": "ok"}],
        reducer_counters={"duplicate": 0},
        revision_history=[("seg-1", 1), ("seg-2", 1), ("seg-3", 1), ("seg-4", 1), ("seg-me", 1)],
        started_ms=0,
        ended_ms=60_000,
    )
    for idx, (seg_id, spk, rx) in enumerate(
        [
            ("seg-1", "S1", 2100),
            ("seg-2", "S2", 5100),
            ("seg-3", "S1", 8100),
            ("seg-4", "S2", 12100),
            ("seg-me", "ME", 15100),
        ],
        start=1,
    ):
        store.record_canonical_event(
            run_id=run_id,
            event=_seg_event(
                session_id,
                f"ev-{idx}",
                seg_id,
                speaker=spk,
                start_ms=rx - 1100,
                end_ms=rx - 100,
                received_ms=rx,
                seq=idx,
            ),
            retention=brief.retention,
            canonical_revision=1,
        )

    # 1) Dropped by prefilter
    store.record_decision(
        run_id=run_id,
        context=_ctx(session_id, "c-drop", "seg-1", 2100),
        terminal_state="dropped",
        terminal_reason=ReasonCode.BACKCHANNEL,
    )
    # 2) Retired unevaluated
    store.record_decision(
        run_id=run_id,
        context=_ctx(session_id, "c-ret", "seg-2", 5100),
        terminal_state="retired",
        terminal_reason=ReasonCode.EXPIRED_UNEVALUATED,
    )
    # 3) Scored -> ROUTE -> template -> displayed -> used -> resolved (valid §7.5 chain)
    c3 = _ctx(session_id, "c-ok", "seg-3", 8100)
    clf3 = ClassifierResult(
        1,
        "c-ok",
        1,
        "rules-v0",
        ScoreSet(0.9, {"CONFIRM_OWNER": 0.9}),
        "cal-v0",
        ScoreSet(0.86, {"CONFIRM_OWNER": 0.86}),
        12,
        (SegmentRef("seg-3", 1),),
    )
    rt3 = RoutingDecision(
        1,
        "c-ok",
        "policy-v0-a",
        "psha",
        1,
        Outcome.ROUTE,
        Route.TEMPLATE,
        Category.CONFIRM_OWNER,
        (ReasonCode.ABOVE_T_DISPLAY,),
        (),
        {"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
        8120,
    )
    store.record_decision(
        run_id=run_id,
        context=c3,
        terminal_state="resolved",
        terminal_reason=ReasonCode.RESOLUTION_CUE,
        classifier=clf3,
        routing=rt3,
    )
    sug3 = SuggestionCandidate(
        1,
        "sug-ok",
        "c-ok",
        Origin.PROACTIVE,
        Category.CONFIRM_OWNER,
        Route.TEMPLATE,
        "Who owns rollback?",
        3,
        1,
        (SegmentRef("seg-3", 1),),
        "confirm_owner_v1",
        {},
        8130,
        16100,
    )
    store.record_suggestion(session_id=session_id, run_id=run_id, suggestion=sug3)
    for le in [
        LedgerEvent(
            1, "l1", "sug-ok", LedgerState.DISPLAYED, ReasonCode.BOUNDARY_AFTER_SETTLE, 8600
        ),
        LedgerEvent(1, "l2", "sug-ok", LedgerState.USED, ReasonCode.USER_ACTION, 10000),
        LedgerEvent(
            1,
            "l3",
            "sug-ok",
            LedgerState.RESOLVED,
            ReasonCode.RESOLUTION_CUE,
            15200,
            evidence_refs=(SegmentRef("seg-me", 1),),
        ),
    ]:
        store.record_ledger_event(session_id=session_id, run_id=run_id, event=le)

    # 4) Scored -> ROUTE -> generator -> timed out (discarded egress still logged!)
    c4 = _ctx(session_id, "c-gen", "seg-4", 12100)
    clf4 = ClassifierResult(
        1,
        "c-gen",
        1,
        "rules-v0",
        ScoreSet(0.8, {"CLARIFY_SCOPE": 0.8}),
        "cal-v0",
        ScoreSet(0.79, {"CLARIFY_SCOPE": 0.79}),
        15,
        (SegmentRef("seg-4", 1),),
    )
    rt4 = RoutingDecision(
        1,
        "c-gen",
        "policy-v0-a",
        "psha",
        1,
        Outcome.ROUTE,
        Route.GENERATOR,
        Category.CLARIFY_SCOPE,
        (ReasonCode.ABOVE_T_DISPLAY,),
        (),
        {"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
        12120,
    )
    store.record_decision(
        run_id=run_id,
        context=c4,
        terminal_state="unrendered",
        terminal_reason=ReasonCode.GENERATOR_TIMEOUT,
        classifier=clf4,
        routing=rt4,
    )
    store.record_egress(
        run_id=run_id,
        record=EgressRecord(
            1,
            "eg-gen",
            session_id,
            "c-gen",
            Origin.PROACTIVE,
            "local",
            "qwen3",
            ("seg-4",),
            12125,
            usage={"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 0},
            discarded=True,
        ),
    )


def test_conservation_invariants_pass_on_valid_run(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "inv_ok.db") as store:
        _populate_valid_run(store, "s-tel", "run-1")
        rep = check_conservation_invariants(store, "s-tel", "run-1")
        assert rep.passed, rep.violations
        assert all(rep.checks.values())


@pytest.mark.parametrize("broken_inv", [1, 2, 3, 4, 5, 6])
def test_conservation_invariants_detect_every_violation(tmp_path: Path, broken_inv: int) -> None:
    with SQLiteStore(tmp_path / f"inv_fail_{broken_inv}.db") as store:
        _populate_valid_run(store, "s-tel", "run-1")
        if broken_inv == 1:
            # Unanchored committed non-recipient segment
            store.record_canonical_event(
                run_id="run-1",
                event=_seg_event("s-tel", "ev-orphan", "seg-orphan", speaker="S2", seq=99),
                retention=_brief("s-tel").retention,
                canonical_revision=1,
            )
        elif broken_inv == 2:
            # Scored candidate without routing outcome
            with store.connection:
                store.connection.execute(
                    "UPDATE decisions SET outcome = NULL WHERE candidate_id = 'c-ok'"
                )
        elif broken_inv == 3:
            # Candidate with missing/invalid terminal reason
            with store.connection:
                store.connection.execute(
                    "UPDATE decisions SET terminal_reason = 'not_a_real_code' WHERE candidate_id = 'c-drop'"
                )
        elif broken_inv == 4:
            # Displayed suggestion with no closing transition (stuck in displayed)
            with store.connection:
                store.connection.execute(
                    """
                    UPDATE suggestions
                    SET current_state = 'displayed',
                        terminal_reason = NULL,
                        transitions_json = '[{"transition": "displayed", "at_ms": 8600}]'
                    WHERE suggestion_id = 'sug-ok'
                    """
                )
        elif broken_inv == 5:
            # Generator call without egress usage record
            with store.connection:
                store.connection.execute("DELETE FROM egress_log WHERE candidate_id = 'c-gen'")
        elif broken_inv == 6:
            # Decision referencing non-existent segment revision
            with store.connection:
                store.connection.execute(
                    """
                    UPDATE decisions
                    SET context_refs_json = '[{"segment_id": "seg-1", "revision": 99}]'
                    WHERE candidate_id = 'c-ok'
                    """
                )

        rep = check_conservation_invariants(store, "s-tel", "run-1")
        assert not rep.passed
        assert not rep.checks[
            {
                1: "inv_1_segment_anchor_partition",
                2: "inv_2_scored_has_routing",
                3: "inv_3_candidate_terminal_reason",
                4: "inv_4_displayed_has_closing",
                5: "inv_5_generator_has_usage",
                6: "inv_6_valid_segment_revisions",
            }[broken_inv]
        ]


def test_explain_silence_diff_runs_report_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "telemetry.db"
    with SQLiteStore(db_path) as store:
        _populate_valid_run(store, "s-tel", "run-a")
        store.record_feedback(
            session_id="s-tel",
            run_id="run-a",
            suggestion_id="sug-ok",
            candidate_id="c-ok",
            feedback_reason=FeedbackReason.TOO_LATE,
            created_ms=11000,
        )

        # Second run: c-drop stays dropped, c-ret now displayed (gained!), c-ok suppressed (lost!)
        brief = _brief("s-tel")
        store.upsert_session(
            brief=brief,
            run_id="run-b",
            runtime_mode=Mode.PROACTIVE,
            effective_max_cards_per_10_min=3,
            started_ms=0,
            ended_ms=60_000,
        )
        store.record_decision(
            run_id="run-b",
            context=_ctx("s-tel", "c-drop", "seg-1", 2100),
            terminal_state="dropped",
            terminal_reason=ReasonCode.BACKCHANNEL,
        )
        store.record_decision(
            run_id="run-b",
            context=_ctx("s-tel", "c-ret", "seg-2", 5100),
            terminal_state="dismissed",
            terminal_reason=ReasonCode.USER_ACTION,
            classifier=ClassifierResult(
                1,
                "c-ret",
                1,
                "rules-v0",
                ScoreSet(0.88, {"CONFIRM_TIME": 0.88}),
                "cal-v0",
                ScoreSet(0.82, {"CONFIRM_TIME": 0.82}),
                10,
                (SegmentRef("seg-2", 1),),
            ),
            routing=RoutingDecision(
                1,
                "c-ret",
                "policy-v0-b",
                "psha-b",
                1,
                Outcome.ROUTE,
                Route.TEMPLATE,
                Category.CONFIRM_TIME,
                (ReasonCode.ABOVE_T_DISPLAY,),
                (),
                {"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
                5120,
            ),
        )
        store.record_suggestion(
            session_id="s-tel",
            run_id="run-b",
            suggestion=SuggestionCandidate(
                1,
                "sug-ret",
                "c-ret",
                Origin.PROACTIVE,
                Category.CONFIRM_TIME,
                Route.TEMPLATE,
                "By when is rollback tested?",
                5,
                1,
                (SegmentRef("seg-2", 1),),
                "confirm_time_v1",
                {},
                5130,
                13100,
            ),
        )
        store.record_ledger_event(
            session_id="s-tel",
            run_id="run-b",
            event=LedgerEvent(
                1, "lb1", "sug-ret", LedgerState.DISPLAYED, ReasonCode.SETTLE_TIMEOUT, 6600
            ),
        )
        store.record_ledger_event(
            session_id="s-tel",
            run_id="run-b",
            event=LedgerEvent(
                1,
                "lb2",
                "sug-ret",
                LedgerState.DISMISSED,
                ReasonCode.USER_ACTION,
                7000,
                feedback=FeedbackReason.NOT_RELEVANT,
            ),
        )
        store.record_decision(
            run_id="run-b",
            context=_ctx("s-tel", "c-ok", "seg-3", 8100),
            terminal_state="suppressed",
            terminal_reason=ReasonCode.DUPLICATE,
            classifier=ClassifierResult(
                1,
                "c-ok",
                1,
                "rules-v0",
                ScoreSet(0.9, {"CONFIRM_OWNER": 0.9}),
                "cal-v0",
                ScoreSet(0.86, {"CONFIRM_OWNER": 0.86}),
                12,
                (SegmentRef("seg-3", 1),),
            ),
            routing=RoutingDecision(
                1,
                "c-ok",
                "policy-v0-b",
                "psha-b",
                1,
                Outcome.SUPPRESS,
                None,
                Category.CONFIRM_OWNER,
                (ReasonCode.DUPLICATE,),
                (),
                {"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
                8120,
            ),
        )

        # explain_silence
        exp = explain_silence(store, "s-tel", "run-a", at_ms=10_000, window_ms=8_000)
        assert [a.candidate_id for a in exp.anchors] == ["c-drop", "c-ret", "c-ok"]
        assert exp.anchors[0].terminal_reason == "backchannel"

        # diff_runs
        diff = diff_runs(store, "run-a", "run-b", session_id="s-tel")
        assert [g.candidate_id for g in diff.gained] == ["c-ret"]
        assert diff.gained[0].reason_a == "expired_unevaluated"
        assert diff.gained[0].reason_b == "user_action"
        assert [x.candidate_id for x in diff.lost] == ["c-ok"]
        assert diff.lost[0].reason_a == "resolution_cue"
        assert diff.lost[0].reason_b == "duplicate"

        # build_session_report
        report = build_session_report(store, "s-tel", "run-a")
        assert report.consent_basis == "stated"
        assert report.displayed_card_count == 1
        assert report.budget_compliant is True  # 1 proactive card in 10-min window <= cap 3
        assert report.latency_by_route["template"]["time_to_ready_ms"]["p50"] == 30.0
        assert report.latency_by_route["template"]["time_to_display_ms"]["p50"] == 500.0
        assert report.upstream_delay_ms["p50"] == 100.0
        assert report.generator_usage["total_calls"] == 1
        assert report.generator_usage["timeout_rate"] == 1.0
        assert report.generator_usage["cache_read_share"] == pytest.approx(0.6)
        assert report.displayed_cards[0]["feedback"] == ["too_late"]

    # CLI smoke tests
    assert (
        main(["report", "--db", str(db_path), "--session", "s-tel", "--run-id", "run-a", "--json"])
        == 0
    )
    rep_json = json.loads(capsys.readouterr().out)
    assert rep_json["invariants"]["passed"] is True

    assert (
        main(
            [
                "explain",
                "--db",
                str(db_path),
                "--session",
                "s-tel",
                "--run-id",
                "run-a",
                "--at-ms",
                "10000",
                "--window-ms",
                "8000",
                "--json",
            ]
        )
        == 0
    )
    exp_json = json.loads(capsys.readouterr().out)
    assert len(exp_json["anchors"]) == 3

    assert (
        main(["diff", "--db", str(db_path), "--run-a", "run-a", "--run-b", "run-b", "--json"]) == 0
    )
    diff_json = json.loads(capsys.readouterr().out)
    assert len(diff_json["gained"]) == 1 and len(diff_json["lost"]) == 1

    assert main(["delete-session", "--db", str(db_path), "--session", "s-tel", "--json"]) == 0
    del_json = json.loads(capsys.readouterr().out)
    assert del_json["session_id"] == "s-tel"


def test_invariants_custom_recipient_manual_speaker_order_and_bucket_edge_cases(
    tmp_path: Path,
) -> None:
    # 1) store_transcript=False with custom recipient_id + out-of-order speaker_update + Origin.MANUAL
    with SQLiteStore(tmp_path / "edge_ok.db") as store:
        brief = SessionBrief(
            schema_version=1,
            session_id="s-edge",
            mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
            recipient=RecipientConfig(participant_id="host_user"),
            objective="Test edge invariants.",
            retention=RetentionConfig(store_transcript=False, store_raw_provider_events=False),
        )
        store.upsert_session(
            brief=brief,
            run_id="r1",
            revision_history=[("seg-host", 1), ("seg-other", 1)],
        )
        # Out-of-order speaker_update (persisted before segment) relabeling seg-host S1 -> host_user
        store.record_canonical_event(
            run_id="r1",
            event=CanonicalEvent(
                schema_version=1,
                session_id="s-edge",
                event_id="ev-spk-early",
                kind=EventKind.SPEAKER_UPDATE,
                received_ms=1050,
                source_sequence=2,
                segment_id="seg-host",
                source_revision=2,
                speaker_id="host_user",
            ),
            retention=brief.retention,
            canonical_revision=2,
        )
        store.record_canonical_event(
            run_id="r1",
            event=_seg_event("s-edge", "ev-host", "seg-host", speaker="S1", seq=1),
            retention=brief.retention,
            canonical_revision=1,
        )
        store.record_canonical_event(
            run_id="r1",
            event=_seg_event("s-edge", "ev-other", "seg-other", speaker="S2", seq=3),
            retention=brief.retention,
            canonical_revision=1,
        )
        # Proactive anchor on seg-other + manual lookback on seg-other
        store.record_decision(
            run_id="r1",
            context=_ctx("s-edge", "c-pro", "seg-other", 2100),
            terminal_state="dropped",
            terminal_reason=ReasonCode.BACKCHANNEL,
        )
        manual_ctx = DecisionContext(
            schema_version=1,
            candidate_id="c-man",
            session_id="s-edge",
            origin=Origin.MANUAL,
            based_on_state_version=1,
            builder_version="ctx-v0",
            prefilter_version="pre-v0",
            anchor_refs=(SegmentRef("seg-other", 1),),
            context_refs=(),
            anchor_received_ms=2200,
            display_by_ms=10200,
            token_counts={"total": 20, "limit": 512},
            omitted={"turns": 0},
            flags={"speaker_uncertain": False},
            serialized_sha256="sha-man",
        )
        store.record_decision(
            run_id="r1",
            context=manual_ctx,
            terminal_state="retired",
            terminal_reason=ReasonCode.SESSION_ENDED,
        )
        rep = check_conservation_invariants(store, "s-edge", "r1")
        assert rep.passed, rep.violations

    # 2) Inv 3 mismatched bucket (abstained + below_t_low) & Inv 4 empty transitions with displayed
    with SQLiteStore(tmp_path / "edge_fail.db") as store:
        _populate_valid_run(store, "s-tel", "r-fail")
        with store.connection:
            store.connection.execute(
                """
                UPDATE decisions
                SET terminal_state = 'abstained', terminal_reason = 'below_t_low'
                WHERE candidate_id = 'c-ok'
                """
            )
            store.connection.execute(
                """
                UPDATE suggestions
                SET current_state = 'displayed', terminal_reason = NULL, transitions_json = '[]'
                WHERE suggestion_id = 'sug-ok'
                """
            )
        rep_fail = check_conservation_invariants(store, "s-tel", "r-fail")
        assert not rep_fail.passed
        assert not rep_fail.checks["inv_3_candidate_terminal_reason"]
        assert not rep_fail.checks["inv_4_displayed_has_closing"]

        # 3) Inv 4 empty transitions with current_state='pinned' also fails
        with store.connection:
            store.connection.execute(
                """
                UPDATE decisions
                SET terminal_state = 'resolved', terminal_reason = 'resolution_cue'
                WHERE candidate_id = 'c-ok'
                """
            )
            store.connection.execute(
                """
                UPDATE suggestions
                SET current_state = 'pinned', terminal_reason = NULL, transitions_json = '[]'
                WHERE suggestion_id = 'sug-ok'
                """
            )
        rep_pinned = check_conservation_invariants(store, "s-tel", "r-fail")
        assert not rep_pinned.passed
        assert not rep_pinned.checks["inv_4_displayed_has_closing"]


def test_build_session_report_sliding_window_budget_compliance(tmp_path: Path) -> None:
    def _run_case(run_id: str, display_times: list[int]) -> bool:
        brief = _brief("s-budget")
        with SQLiteStore(tmp_path / f"{run_id}.db") as store:
            store.upsert_session(
                brief=brief,
                run_id=run_id,
                runtime_mode=Mode.PROACTIVE,
                effective_max_cards_per_10_min=2,
                health_timeline=[{"at_ms": 0, "state": "ok"}],
                reducer_counters={"duplicate": 0},
                revision_history=[(f"seg-{idx}", 1) for idx in range(len(display_times))],
                started_ms=0,
                ended_ms=display_times[-1] + 10_000,
            )
            for idx, disp_ms in enumerate(display_times):
                seg_id = f"seg-{idx}"
                cid = f"c-{idx}"
                sid = f"sug-{idx}"
                rx_ms = disp_ms - 500
                store.record_canonical_event(
                    run_id=run_id,
                    event=_seg_event(
                        "s-budget",
                        f"ev-{idx}",
                        seg_id,
                        start_ms=rx_ms - 1000,
                        end_ms=rx_ms - 100,
                        received_ms=rx_ms,
                        seq=idx + 1,
                    ),
                    retention=brief.retention,
                    canonical_revision=1,
                )
                store.record_decision(
                    run_id=run_id,
                    context=_ctx("s-budget", cid, seg_id, rx_ms),
                    terminal_state="dismissed",
                    terminal_reason=ReasonCode.USER_ACTION,
                    classifier=ClassifierResult(
                        1,
                        cid,
                        1,
                        "rules-v0",
                        ScoreSet(0.9, {"CONFIRM_OWNER": 0.9}),
                        "cal-v0",
                        ScoreSet(0.86, {"CONFIRM_OWNER": 0.86}),
                        10,
                        (SegmentRef(seg_id, 1),),
                    ),
                    routing=RoutingDecision(
                        1,
                        cid,
                        "policy-v0-a",
                        "psha",
                        1,
                        Outcome.ROUTE,
                        Route.TEMPLATE,
                        Category.CONFIRM_OWNER,
                        (ReasonCode.ABOVE_T_DISPLAY,),
                        (),
                        {"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
                        rx_ms + 20,
                    ),
                )
                store.record_suggestion(
                    session_id="s-budget",
                    run_id=run_id,
                    suggestion=SuggestionCandidate(
                        1,
                        sid,
                        cid,
                        Origin.PROACTIVE,
                        Category.CONFIRM_OWNER,
                        Route.TEMPLATE,
                        "Who owns rollback?",
                        3,
                        1,
                        (SegmentRef(seg_id, 1),),
                        "confirm_owner_v1",
                        {},
                        rx_ms + 30,
                        rx_ms + 8000,
                    ),
                )
                for le in (
                    LedgerEvent(
                        1,
                        f"le-d-{idx}",
                        sid,
                        LedgerState.DISPLAYED,
                        ReasonCode.BOUNDARY_AFTER_SETTLE,
                        disp_ms,
                    ),
                    LedgerEvent(
                        1,
                        f"le-x-{idx}",
                        sid,
                        LedgerState.DISMISSED,
                        ReasonCode.USER_ACTION,
                        disp_ms + 500,
                    ),
                ):
                    store.record_ledger_event(session_id="s-budget", run_id=run_id, event=le)
            report = build_session_report(store, "s-budget", run_id)
            assert report.displayed_card_count == len(display_times)
            return report.budget_compliant

    assert _run_case("over-cap", [10_000, 100_000, 610_000]) is False
    assert _run_case("spaced-out", [10_000, 610_001, 1_210_002]) is True


def test_try_load_dataset_rows_and_cli_replay_peecee_pretty_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pretty_nemo = tmp_path / "peecee_offline.json"
    pretty_nemo.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "someone", "start": 0.1, "end": 0.3, "speaker": "1"},
                    {"word": "should", "start": 0.3, "end": 0.5, "speaker": "1"},
                    {"word": "own", "start": 0.5, "end": 0.7, "speaker": "1"},
                    {"word": "the", "start": 0.7, "end": 0.8, "speaker": "1"},
                    {"word": "rollback", "start": 0.8, "end": 1.1, "speaker": "1"},
                    {"word": "plan", "start": 1.1, "end": 1.4, "speaker": "1"},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert _try_load_dataset_rows(pretty_nemo) is None

    bad_json = tmp_path / "malformed.json"
    bad_json.write_text("{not valid json\n", encoding="utf-8")
    assert _try_load_dataset_rows(bad_json) is None

    pretty_jsonl = tmp_path / "pretty.jsonl"
    pretty_jsonl.write_text(json.dumps({"words": []}, indent=2), encoding="utf-8")
    assert _try_load_dataset_rows(pretty_jsonl) is None

    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(_brief("s-peecee-json").to_dict(), indent=2), encoding="utf-8")
    db_path = tmp_path / "peecee_json.db"
    policy_path = Path(__file__).resolve().parents[2] / "policies" / "policy-v0-a.yaml"

    rc = main(
        [
            "replay",
            "--source",
            "peecee",
            "--session",
            str(pretty_nemo),
            "--brief",
            str(brief_path),
            "--policy",
            str(policy_path),
            "--db",
            str(db_path),
            "--speed",
            "stepped",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decisions_count"] >= 1
    assert out["invariants"]["passed"] is True
