"""Unit tests for SQLite WAL persistence, JSONL I/O, model_cache, and §14.2 deletion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from coach.contracts import (
    AnnotationPass,
    CanonicalEvent,
    Category,
    ClassifierResult,
    ConsentBasis,
    ConsentConfig,
    DecisionContext,
    EventKind,
    FeedbackReason,
    GroundedSlot,
    LabelRecord,
    LedgerEvent,
    LedgerState,
    Origin,
    Outcome,
    Participant,
    ReasonCode,
    RecipientConfig,
    RetentionConfig,
    Route,
    RoutingDecision,
    SampleSource,
    ScoreSet,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    SuggestionCandidate,
    TelemetryLevel,
    TelemetryRecord,
    ValidityInterval,
)
from coach.storage import SQLiteStore, export_events_jsonl, import_events_jsonl


def _brief(
    session_id: str = "s1",
    *,
    store_transcript: bool = True,
    store_raw: bool = True,
) -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id=session_id,
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="me", role="reviewer"),
        objective="Confirm rollback owner.",
        retention=RetentionConfig(
            store_transcript=store_transcript,
            store_raw_provider_events=store_raw,
        ),
    )


def _event(session_id: str = "s1", event_id: str = "ev-1", seq: int = 1) -> CanonicalEvent:
    return CanonicalEvent(
        schema_version=1,
        session_id=session_id,
        event_id=event_id,
        kind=EventKind.SEGMENT,
        received_ms=1050,
        source_sequence=seq,
        segment_id="seg-1",
        source_revision=1,
        channel=0,
        speaker_id="S1",
        start_ms=100,
        end_ms=1000,
        text="Someone should own rollback.",
        stability=Stability.COMMITTED,
    )


def test_wal_mode_retention_and_multi_run_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "coach.db"
    with SQLiteStore(db_path) as store:
        assert store.journal_mode().lower() == "wal"
        b_no_ret = _brief("s-optout", store_transcript=False, store_raw=False)
        store.upsert_session(brief=b_no_ret, run_id="run-1")
        assert not store.record_raw_event(
            session_id="s-optout",
            run_id="run-1",
            source_id="deepgram",
            received_ms=100,
            payload={"raw": "secret"},
            retention=b_no_ret.retention,
        )
        store.record_canonical_event(
            run_id="run-1",
            event=_event("s-optout"),
            retention=b_no_ret.retention,
            canonical_revision=1,
        )
        assert store.get_raw_events("s-optout", "run-1") == []
        ev_rows = store.get_events("s-optout", "run-1")
        assert len(ev_rows) == 1
        assert ev_rows[0]["text"] is None
        assert ev_rows[0]["payload_json"] is None

        b_keep = _brief("s-multi")
        store.upsert_session(brief=b_keep, run_id="run-a")
        store.upsert_session(brief=b_keep, run_id="run-b")
        store.record_canonical_event(
            run_id="run-a",
            event=_event("s-multi", "e1", 1),
            retention=b_keep.retention,
            canonical_revision=1,
        )
        store.record_canonical_event(
            run_id="run-b",
            event=_event("s-multi", "e1", 1),
            retention=b_keep.retention,
            canonical_revision=1,
        )
        assert len(store.list_runs("s-multi")) == 2
        assert len(store.get_events("s-multi", "run-a")) == 1
        assert len(store.get_events("s-multi", "run-b")) == 1


def test_model_cache_and_jsonl_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    with SQLiteStore(db_path) as store:
        assert store.get_cached_model_output("sha-1", "clf-v0") is None
        store.put_cached_model_output(
            serialized_sha256="sha-1",
            model_version="clf-v0",
            output={"opportunity": 0.88},
            contains_text=False,
            created_ms=1000,
        )
        assert store.get_cached_model_output("sha-1", "clf-v0") == {"opportunity": 0.88}

    brief = _brief("s-jsonl")
    events = [_event("s-jsonl", "ev-1", 1), _event("s-jsonl", "ev-2", 2)]
    jsonl_path = tmp_path / "fixture.jsonl"
    written = export_events_jsonl(events, jsonl_path, brief=brief)
    assert written == 2
    loaded_brief, loaded_events = import_events_jsonl(jsonl_path)
    assert loaded_brief == brief
    assert loaded_events == events


def test_delete_session_text_preserves_nulled_decisions_and_suggestions(tmp_path: Path) -> None:
    db_path = tmp_path / "del.db"
    brief = _brief("s-del")
    ref = SegmentRef("seg-1", 1)
    with SQLiteStore(db_path) as store:
        store.upsert_session(brief=brief, run_id="run-1")
        store.record_raw_event(
            session_id="s-del",
            run_id="run-1",
            source_id="peecee",
            received_ms=1050,
            payload={"transcript": "Someone should own rollback."},
            retention=brief.retention,
        )
        store.record_canonical_event(
            run_id="run-1", event=_event("s-del"), retention=brief.retention, canonical_revision=1
        )
        ctx = DecisionContext(
            schema_version=1,
            candidate_id="cand-1",
            session_id="s-del",
            origin=Origin.PROACTIVE,
            based_on_state_version=1,
            builder_version="ctx-v0",
            prefilter_version="pre-v0",
            anchor_refs=(ref,),
            context_refs=(),
            anchor_received_ms=1050,
            display_by_ms=9050,
            token_counts={"total": 20, "limit": 512},
            omitted={"turns": 0},
            flags={"speaker_uncertain": False},
            serialized_sha256="sha-del-1",
        )
        clf = ClassifierResult(
            schema_version=1,
            candidate_id="cand-1",
            based_on_state_version=1,
            classifier_version="rules-v0",
            raw_scores=ScoreSet(opportunity=0.9, categories={"CONFIRM_OWNER": 0.9}),
            calibration_version="cal-v0",
            calibrated_scores=ScoreSet(opportunity=0.85, categories={"CONFIRM_OWNER": 0.85}),
            latency_ms=5,
            input_refs=(ref,),
        )
        rt = RoutingDecision(
            schema_version=1,
            candidate_id="cand-1",
            policy_version="policy-v0-a",
            policy_sha256="p-sha",
            based_on_state_version=1,
            outcome=Outcome.ROUTE,
            route=Route.TEMPLATE,
            category=Category.CONFIRM_OWNER,
            reason_codes=(ReasonCode.ABOVE_T_DISPLAY,),
            abstain_reasons=(),
            thresholds={"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
            decided_ms=1060,
        )
        store.record_decision(
            run_id="run-1",
            context=ctx,
            terminal_state="dismissed",
            terminal_reason=ReasonCode.USER_ACTION,
            classifier=clf,
            routing=rt,
            serialized_text="[ANCHOR] S1: Someone should own rollback.",
            retention=brief.retention,
        )
        sug = SuggestionCandidate(
            schema_version=1,
            suggestion_id="sug-1",
            candidate_id="cand-1",
            origin=Origin.PROACTIVE,
            category=Category.CONFIRM_OWNER,
            route=Route.TEMPLATE,
            text="Who will own rollback?",
            word_count=4,
            based_on_state_version=1,
            evidence_refs=(ref,),
            template_id="confirm_owner_v1",
            grounded_slots={
                "action": GroundedSlot("rollback", "seg-1", 1, 19, 27, "span_rules_v0")
            },
            ready_ms=1070,
            display_by_ms=9050,
        )
        store.record_suggestion(
            session_id="s-del", run_id="run-1", suggestion=sug, retention=brief.retention
        )
        store.record_ledger_event(
            session_id="s-del",
            run_id="run-1",
            event=LedgerEvent(
                schema_version=1,
                event_id="l1",
                suggestion_id="sug-1",
                transition=LedgerState.DISPLAYED,
                cause=ReasonCode.BOUNDARY_AFTER_SETTLE,
                at_ms=1500,
            ),
        )
        store.record_ledger_event(
            session_id="s-del",
            run_id="run-1",
            event=LedgerEvent(
                schema_version=1,
                event_id="l2",
                suggestion_id="sug-1",
                transition=LedgerState.DISMISSED,
                cause=ReasonCode.USER_ACTION,
                at_ms=2000,
                feedback=FeedbackReason.NOT_RELEVANT,
            ),
        )
        store.record_label(
            run_id="run-1",
            label=LabelRecord(
                schema_version=1,
                label_id="lbl-1",
                session_id="s-del",
                annotator="recipient",
                sample_source=SampleSource.RANDOM,
                annotation_pass=AnnotationPass.OFFLINE,
                candidate_id="cand-1",
                state_version_at_decision=1,
                worthwhile=True,
                categories=(Category.CONFIRM_OWNER,),
                example_questions=("Who owns rollback?",),
                evidence_refs=(ref,),
                validity=ValidityInterval(open_ms=1000, close_ms=3000),
                notes="sensitive note",
            ),
            created_ms=5000,
        )
        store.put_cached_model_output(
            serialized_sha256="sha-del-1",
            model_version="gen-v0",
            output={"text": "Who will own rollback?"},
            contains_text=True,
            created_ms=1065,
        )

        summary = store.delete_session_text("s-del", deleted_at_ms=9999)
        assert summary.raw_events_deleted == 1
        assert summary.events_redacted == 1
        assert summary.decisions_redacted == 1
        assert summary.suggestions_redacted == 1
        assert summary.feedback_redacted == 1
        assert summary.model_cache_purged == 1

        assert store.get_raw_events("s-del", "run-1") == []
        assert store.get_events("s-del", "run-1")[0]["text"] is None
        assert store.get_events("s-del", "run-1")[0]["payload_json"] is None

        dec = store.get_decisions("s-del", "run-1")[0]
        assert dec["serialized_text"] is None
        assert dec["outcome"] == "ROUTE"
        assert dec["terminal_reason"] == "user_action"

        s_row = store.get_suggestions("s-del", "run-1")[0]
        assert s_row["text"] is None
        assert s_row["grounded_slots_json"] is None
        assert s_row["current_state"] == "dismissed"

        fb = store.get_feedback("s-del", "run-1")
        assert len(fb) == 1  # operational feedback reason kept, label row deleted
        assert fb[0]["feedback_reason"] == "not_relevant"
        assert store.get_cached_model_output("sha-del-1", "gen-v0") is None
        sess = store.get_session("s-del", "run-1")
        assert sess is not None and sess["brief_json"] is None and sess["deleted_at_ms"] == 9999


def test_delete_session_text_raw_byte_forensics(tmp_path: Path) -> None:
    db_path = tmp_path / "forensics.db"
    wal_path = Path(f"{db_path}-wal")
    secrets = [
        "SECRET_RAW_EVENT_01",
        "SECRET_TRANSCRIPT_02",
        "SECRET_SERIALIZED_03",
        "SECRET_QUESTION_04",
        "SECRET_SLOT_05",
        "SECRET_LABEL_06",
        "SECRET_BRIEF_OBJ_07",
        "SECRET_PARTICIPANT_08",
        "SECRET_MODEL_CACHE_09",
        "SECRET_CONSENT_NOTE_10",
        "SECRET_TELEMETRY_11",
    ]
    brief = SessionBrief(
        schema_version=1,
        session_id="sess-forensics",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="host_user", role="owner"),
        objective=f"Objective {secrets[6]}",
        participants=(Participant(participant_id="p1", label="S1", role=secrets[7]),),
        consent=ConsentConfig(basis=ConsentBasis.STATED, note=secrets[9]),
    )
    ref = SegmentRef("seg-1", 1)
    with SQLiteStore(db_path) as store:
        store.upsert_session(brief=brief, run_id="run-1")
        store.record_raw_event(
            session_id="sess-forensics",
            run_id="run-1",
            source_id="peecee",
            received_ms=100,
            payload={"raw": secrets[0]},
            retention=brief.retention,
        )
        store.record_canonical_event(
            run_id="run-1",
            event=CanonicalEvent(
                schema_version=1,
                session_id="sess-forensics",
                event_id="ev-1",
                kind=EventKind.SEGMENT,
                received_ms=200,
                source_sequence=1,
                segment_id="seg-1",
                source_revision=1,
                channel=0,
                speaker_id="S1",
                start_ms=100,
                end_ms=190,
                text=secrets[1],
                stability=Stability.COMMITTED,
            ),
            retention=brief.retention,
            canonical_revision=1,
        )
        ctx = DecisionContext(
            schema_version=1,
            candidate_id="cand-1",
            session_id="sess-forensics",
            origin=Origin.PROACTIVE,
            based_on_state_version=1,
            builder_version="ctx-v0",
            prefilter_version="pre-v0",
            anchor_refs=(ref,),
            context_refs=(),
            anchor_received_ms=200,
            display_by_ms=8200,
            token_counts={"total": 10, "limit": 512},
            omitted={"turns": 0},
            flags={"speaker_uncertain": False},
            serialized_sha256="sha-forensics-1",
        )
        store.record_decision(
            run_id="run-1",
            context=ctx,
            terminal_state="dismissed",
            terminal_reason=ReasonCode.USER_ACTION,
            serialized_text=secrets[2],
            retention=brief.retention,
        )
        store.record_suggestion(
            session_id="sess-forensics",
            run_id="run-1",
            suggestion=SuggestionCandidate(
                schema_version=1,
                suggestion_id="sug-1",
                candidate_id="cand-1",
                origin=Origin.PROACTIVE,
                category=Category.CONFIRM_OWNER,
                route=Route.TEMPLATE,
                text=secrets[3],
                word_count=2,
                based_on_state_version=1,
                evidence_refs=(ref,),
                template_id="confirm_owner_v1",
                grounded_slots={
                    "owner": GroundedSlot(secrets[4], "seg-1", 1, 0, 10, "span_rules_v0")
                },
                ready_ms=250,
                display_by_ms=8200,
            ),
            retention=brief.retention,
        )
        store.record_label(
            run_id="run-1",
            label=LabelRecord(
                schema_version=1,
                label_id="lbl-1",
                session_id="sess-forensics",
                annotator="recipient",
                sample_source=SampleSource.RANDOM,
                annotation_pass=AnnotationPass.OFFLINE,
                candidate_id="cand-1",
                state_version_at_decision=1,
                worthwhile=True,
                categories=(Category.CONFIRM_OWNER,),
                example_questions=(secrets[5],),
                evidence_refs=(ref,),
                validity=ValidityInterval(open_ms=100, close_ms=300),
            ),
            created_ms=400,
        )
        store.put_cached_model_output(
            serialized_sha256="sha-forensics-1",
            model_version="gen-v0",
            output={"text": secrets[8]},
            contains_text=True,
            created_ms=260,
        )
        store.record_telemetry(
            run_id="run-1",
            records=[
                TelemetryRecord(
                    schema_version=1,
                    record_id="tr-1",
                    session_id="sess-forensics",
                    at_ms=270,
                    level=TelemetryLevel.DECISION,
                    kind="generator_call",
                    candidate_id="cand-1",
                    payload={"egress_id": "eg-1", "latency_ms": 12, "snippet": secrets[10]},
                )
            ],
        )

        summary = store.delete_session_text("sess-forensics", deleted_at_ms=5000)
        assert summary.telemetry_redacted == 1
        sess = store.get_session("sess-forensics", "run-1")
        assert sess is not None
        assert sess["recipient_id"] == "host_user"
        assert sess["consent_note"] is None
        tel = store.get_telemetry_records("sess-forensics", "run-1")
        assert tel[0].payload == {"egress_id": "eg-1", "latency_ms": 12}

    db_bytes = db_path.read_bytes()
    wal_bytes = wal_path.read_bytes() if wal_path.exists() else b""
    assert len(wal_bytes) == 0
    for secret in secrets:
        assert secret.encode("utf-8") not in db_bytes
        assert secret.encode("utf-8") not in wal_bytes


def test_delete_session_text_wal_checkpoint_clean_and_blocked_reader(tmp_path: Path) -> None:
    db_path = tmp_path / "wal_block.db"
    wal_path = Path(f"{db_path}-wal")
    secret = "SECRET_WAL_BLOCK_PROBE_99"
    brief = _brief("s-wal")

    with SQLiteStore(db_path) as store:
        store.upsert_session(brief=brief, run_id="run-1")
        store.record_canonical_event(
            run_id="run-1",
            event=CanonicalEvent(
                schema_version=1,
                session_id="s-wal",
                event_id="ev-wal-1",
                kind=EventKind.SEGMENT,
                received_ms=200,
                source_sequence=1,
                segment_id="seg-1",
                source_revision=1,
                channel=0,
                speaker_id="S1",
                start_ms=100,
                end_ms=190,
                text=secret,
                stability=Stability.COMMITTED,
            ),
            retention=brief.retention,
            canonical_revision=1,
        )

        # Idle RO connection (no active snapshot tx) does not block WAL truncation
        ro_idle = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            summary_ok = store.delete_session_text("s-wal", deleted_at_ms=3000)
            assert summary_ok.events_redacted == 1
        finally:
            ro_idle.close()
        assert not wal_path.exists() or len(wal_path.read_bytes()) == 0
        assert secret.encode("utf-8") not in db_path.read_bytes()

        # Re-populate and hold an active RO snapshot transaction open across delete_session_text
        store.upsert_session(brief=_brief("s-wal-2"), run_id="run-1")
        store.record_canonical_event(
            run_id="run-1",
            event=CanonicalEvent(
                schema_version=1,
                session_id="s-wal-2",
                event_id="ev-wal-2",
                kind=EventKind.SEGMENT,
                received_ms=400,
                source_sequence=2,
                segment_id="seg-2",
                source_revision=1,
                channel=0,
                speaker_id="S1",
                start_ms=300,
                end_ms=390,
                text=secret,
                stability=Stability.COMMITTED,
            ),
            retention=brief.retention,
            canonical_revision=1,
        )

        ro_active = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            ro_active.execute("BEGIN;")
            ro_active.execute("SELECT COUNT(*) FROM events;").fetchone()
            with pytest.raises(RuntimeError, match="blocked by active WAL reader"):
                store.delete_session_text("s-wal-2", deleted_at_ms=4000)
        finally:
            ro_active.close()

        # Once the reader closes, repeating delete_session_text completes cleanly with 0-byte WAL
        store.delete_session_text("s-wal-2", deleted_at_ms=4001)
        wal_bytes = wal_path.read_bytes() if wal_path.exists() else b""
        assert len(wal_bytes) == 0
        assert secret.encode("utf-8") not in db_path.read_bytes()
