"""Unit tests for schema-v1 contracts and closed enums in coach.contracts."""

from __future__ import annotations

import pytest

from coach.contracts import (
    SCHEMA_VERSION,
    AnnotationPass,
    AttentionConfig,
    CanonicalEvent,
    Category,
    ClassifierResult,
    ClosedBy,
    ConsentBasis,
    ConsentConfig,
    ContractValidationError,
    ControlSubtype,
    DecisionContext,
    EgressRecord,
    EventKind,
    FeedbackReason,
    GroundedSlot,
    HealthState,
    LabelRecord,
    LedgerEvent,
    LedgerState,
    Mode,
    NoneReason,
    Origin,
    Outcome,
    ReasonCode,
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


def _sample_brief_dict() -> dict[str, object]:
    return {
        "schema_version": 1,
        "session_id": "session-001",
        "mode": "technical_working_discussion",
        "recipient": {"participant_id": "me", "role": "reviewer"},
        "objective": "Leave with a rollback plan that has an owner and a test date.",
        "participants": [
            {"participant_id": "participant-a", "label": "S1", "role": "service owner"},
            {"participant_id": "participant-b", "label": "S2", "role": None},
        ],
        "agenda": ["cutover sequence", "rollback", "launch date"],
        "known_facts": [{"key": "launch_date", "value": "March 3", "source": "brief"}],
        "do_not_suggest": ["budget"],
        "attention": {"max_cards_per_10_min": 3, "proactive": True},
        "consent": {"basis": "presumed", "note": None},
        "retention": {"store_transcript": True, "store_raw_provider_events": True},
    }


def test_closed_enums_exact_counts() -> None:
    assert len(ReasonCode) == 47
    assert len(FeedbackReason) == 4
    assert len(Category) == 9
    assert len(HealthState) == 5
    assert len(Mode) == 5
    assert len(SessionMode) == 2
    assert len(Stability) == 2
    assert len(EventKind) == 6
    assert len(ControlSubtype) == 6
    assert len(LedgerState) == 10
    assert len(Outcome) == 4
    assert len(Route) == 2
    assert len(Origin) == 2
    assert len(SampleSource) == 7
    assert len(ClosedBy) == 4
    assert len(NoneReason) == 8
    assert len(ConsentBasis) == 2
    assert len(TelemetryLevel) == 3
    assert SCHEMA_VERSION == 1


def test_session_brief_roundtrip_and_defaults() -> None:
    brief = SessionBrief.from_dict(_sample_brief_dict())
    assert SessionBrief.from_dict(brief.to_dict()) == brief

    minimal = SessionBrief.from_dict(
        {
            "schema_version": 1,
            "session_id": "s-min",
            "mode": "personal_logistics",
            "recipient": {"participant_id": "me"},
            "objective": "Confirm pickup time.",
        }
    )
    assert minimal.consent == ConsentConfig(basis=ConsentBasis.PRESUMED, note=None)
    assert minimal.retention == RetentionConfig(
        store_transcript=True, store_raw_provider_events=True
    )
    assert minimal.attention == AttentionConfig(max_cards_per_10_min=3, proactive=True)


def test_canonical_event_all_six_kinds_roundtrip() -> None:
    events = [
        CanonicalEvent.from_dict(
            {
                "schema_version": 1,
                "session_id": "session-001",
                "event_id": "ev-1",
                "kind": "segment",
                "segment_id": "seg-1",
                "source_revision": 1,
                "channel": 0,
                "speaker_id": "S1",
                "speaker_confidence": 0.9,
                "start_ms": 100,
                "end_ms": 500,
                "received_ms": 550,
                "text": "Who owns this?",
                "stability": "committed",
                "boundary": "segment",
                "source_sequence": 1,
            }
        ),
        CanonicalEvent.from_dict(
            {
                "schema_version": 1,
                "session_id": "session-001",
                "event_id": "ev-2",
                "kind": "speaker_update",
                "segment_id": "seg-1",
                "speaker_id": "ME",
                "source_revision": 2,
                "received_ms": 600,
                "source_sequence": 2,
            }
        ),
        CanonicalEvent.from_dict(
            {
                "schema_version": 1,
                "session_id": "session-001",
                "event_id": "ev-3",
                "kind": "boundary",
                "speaker_id": "S1",
                "end_ms": 500,
                "boundary": "utterance_end",
                "received_ms": 800,
                "source_sequence": 3,
            }
        ),
        CanonicalEvent.from_dict(
            {
                "schema_version": 1,
                "session_id": "session-001",
                "event_id": "ev-4",
                "kind": "brief_update",
                "brief_patch": {"objective": "New objective"},
                "received_ms": 900,
                "source_sequence": 4,
            }
        ),
        CanonicalEvent.from_dict(
            {
                "schema_version": 1,
                "session_id": "session-001",
                "event_id": "ev-5",
                "kind": "ledger",
                "suggestion_id": "sug-1",
                "transition": "displayed",
                "cause": "boundary_after_settle",
                "at_ms": 950,
                "received_ms": 950,
                "source_sequence": 5,
            }
        ),
        CanonicalEvent.from_dict(
            {
                "schema_version": 1,
                "session_id": "session-001",
                "event_id": "ev-6",
                "kind": "control",
                "subtype": "gap",
                "seq_start": 6,
                "seq_end": 10,
                "start_ms": 1000,
                "end_ms": 2000,
                "reason": "tunnel_drop",
                "received_ms": 2100,
                "source_sequence": 11,
            }
        ),
    ]
    for ev in events:
        assert CanonicalEvent.from_dict(ev.to_dict()) == ev


def test_stage_and_observability_records_roundtrip() -> None:
    ref = SegmentRef(segment_id="segment-0042", revision=1)
    ctx = DecisionContext(
        schema_version=1,
        candidate_id="candidate-0028",
        session_id="session-001",
        origin=Origin.PROACTIVE,
        based_on_state_version=176,
        builder_version="ctx-v0",
        prefilter_version="pre-v0",
        anchor_refs=(ref,),
        context_refs=(SegmentRef("segment-0041", 1),),
        anchor_received_ms=129450,
        display_by_ms=137450,
        token_counts={
            "header": 41,
            "suggested": 17,
            "context": 38,
            "anchor": 12,
            "total": 108,
            "limit": 512,
        },
        omitted={"turns": 0, "suggestions": 0, "known_facts": 0},
        flags={"speaker_uncertain": False, "asr_low_confidence": False},
        serialized_sha256="abc123",
    )
    assert DecisionContext.from_dict(ctx.to_dict()) == ctx

    clf = ClassifierResult(
        schema_version=1,
        candidate_id="candidate-0028",
        based_on_state_version=176,
        classifier_version="opportunity-baseline-v0",
        raw_scores=ScoreSet(opportunity=0.95, categories={"CONFIRM_OWNER": 0.91}),
        calibration_version="cal-v0-technical",
        calibrated_scores=ScoreSet(opportunity=0.88, categories={"CONFIRM_OWNER": 0.83}),
        latency_ms=62,
        input_refs=(ref,),
    )
    assert ClassifierResult.from_dict(clf.to_dict()) == clf

    rt = RoutingDecision(
        schema_version=1,
        candidate_id="candidate-0028",
        policy_version="policy-v0-a",
        policy_sha256="deadbeef",
        based_on_state_version=176,
        outcome=Outcome.ROUTE,
        route=Route.TEMPLATE,
        category=Category.CONFIRM_OWNER,
        reason_codes=(ReasonCode.ABOVE_T_DISPLAY,),
        abstain_reasons=(),
        thresholds={"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
        decided_ms=129545,
    )
    assert RoutingDecision.from_dict(rt.to_dict()) == rt

    sug = SuggestionCandidate(
        schema_version=1,
        suggestion_id="suggestion-0017",
        candidate_id="candidate-0028",
        origin=Origin.PROACTIVE,
        category=Category.CONFIRM_OWNER,
        route=Route.TEMPLATE,
        text="Who will own the rollback plan?",
        word_count=6,
        based_on_state_version=176,
        evidence_refs=(ref,),
        template_id="confirm_owner_v1",
        grounded_slots={
            "action": GroundedSlot(
                text="the rollback plan",
                segment_id="segment-0042",
                revision=1,
                char_start=19,
                char_end=36,
                extractor="span_rules_v0",
            )
        },
        ready_ms=129620,
        display_by_ms=137450,
        display_ttl_ms=15000,
        invalidate_on=("evidence_revision_changed", "ownership_resolved"),
    )
    assert SuggestionCandidate.from_dict(sug.to_dict()) == sug

    le = LedgerEvent(
        schema_version=1,
        event_id="ev-181",
        suggestion_id="suggestion-0017",
        transition=LedgerState.DISMISSED,
        cause=ReasonCode.USER_ACTION,
        at_ms=130500,
        feedback=FeedbackReason.TOO_LATE,
    )
    assert LedgerEvent.from_dict(le.to_dict()) == le

    lbl = LabelRecord(
        schema_version=1,
        label_id="lbl-1",
        session_id="session-001",
        annotator="recipient",
        sample_source=SampleSource.RANDOM,
        annotation_pass=AnnotationPass.OFFLINE,
        candidate_id="candidate-0028",
        state_version_at_decision=176,
        worthwhile=True,
        categories=(Category.CONFIRM_OWNER,),
        example_questions=("Who owns rollback?",),
        evidence_refs=(ref,),
        validity=ValidityInterval(
            open_ms=129200, ideal_ms=130400, close_ms=139900, closed_by=ClosedBy.ANSWERED
        ),
    )
    assert LabelRecord.from_dict(lbl.to_dict()) == lbl

    tr = TelemetryRecord(
        schema_version=1,
        record_id="tr-1",
        session_id="session-001",
        at_ms=129500,
        level=TelemetryLevel.DECISION,
        kind="decision",
        candidate_id="candidate-0028",
        reason_code=ReasonCode.ABOVE_T_DISPLAY,
        payload={"score": 0.88},
    )
    assert TelemetryRecord.from_dict(tr.to_dict()) == tr

    eg = EgressRecord(
        schema_version=1,
        egress_id="eg-1",
        session_id="session-001",
        candidate_id="candidate-0028",
        lane=Origin.PROACTIVE,
        provider="openai",
        model_id="gpt-4o-mini",
        segment_ids=("segment-0042",),
        sent_at_ms=129500,
        first_byte_ms=129650,
        complete_ms=129780,
        finish_reason="stop",
        usage={"input_tokens": 120, "cached_input_tokens": 80, "output_tokens": 15},
        discarded=False,
    )
    assert EgressRecord.from_dict(eg.to_dict()) == eg


def test_contract_rejects_invalid_versions_bools_enums_and_codes() -> None:
    bad_version = _sample_brief_dict()
    bad_version["schema_version"] = 2
    with pytest.raises(ContractValidationError):
        SessionBrief.from_dict(bad_version)

    bool_version = _sample_brief_dict()
    bool_version["schema_version"] = True
    with pytest.raises(ContractValidationError):
        SessionBrief.from_dict(bool_version)

    with pytest.raises(ContractValidationError):
        CanonicalEvent.from_dict(
            {
                "schema_version": 1,
                "session_id": "s1",
                "event_id": "e1",
                "kind": "segment",
                "segment_id": "seg1",
                "source_revision": True,
                "channel": 0,
                "speaker_id": "S1",
                "start_ms": 0,
                "end_ms": 100,
                "received_ms": 120,
                "text": "hi",
                "stability": "committed",
                "source_sequence": 1,
            }
        )

    with pytest.raises(ContractValidationError):
        RoutingDecision.from_dict(
            {
                "schema_version": 1,
                "candidate_id": "c1",
                "policy_version": "p1",
                "policy_sha256": "abc",
                "based_on_state_version": 1,
                "outcome": "ABSTAIN",
                "route": None,
                "category": None,
                "reason_codes": ["totally_invented_code"],
                "abstain_reasons": [],
                "thresholds": None,
                "decided_ms": 100,
            }
        )

    with pytest.raises(ContractValidationError):
        RoutingDecision(
            schema_version=1,
            candidate_id="c1",
            policy_version="p1",
            policy_sha256="abc",
            based_on_state_version=1,
            outcome=Outcome.ROUTE,
            route=Route.TEMPLATE,
            category=Category.OTHER,
            reason_codes=(ReasonCode.ABOVE_T_DISPLAY,),
            abstain_reasons=(),
            thresholds={"t_low": 0.2, "t_compute": 0.5, "t_display": 0.75},
            decided_ms=100,
        )

    with pytest.raises(ContractValidationError):
        RoutingDecision(
            schema_version=1,
            candidate_id="c1",
            policy_version="p1",
            policy_sha256="abc",
            based_on_state_version=1,
            outcome=Outcome.NONE,
            route=None,
            category=None,
            reason_codes=(ReasonCode.BELOW_T_LOW,),
            abstain_reasons=(),
            thresholds={"t_low": 0.6, "t_compute": 0.5, "t_display": 0.75},
            decided_ms=100,
        )

    with pytest.raises(ContractValidationError):
        ScoreSet(opportunity=float("nan"), categories={})

    brief = SessionBrief.from_dict(_sample_brief_dict())
    with pytest.raises(AttributeError):
        brief.objective = "mutated"  # type: ignore[misc]


def test_direct_constructor_validation_matches_from_dict() -> None:
    for bad_float in (True, False, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ContractValidationError):
            CanonicalEvent(
                schema_version=1,
                session_id="s1",
                event_id="e1",
                kind=EventKind.SEGMENT,
                received_ms=10,
                source_sequence=1,
                segment_id="seg-1",
                source_revision=1,
                channel=0,
                speaker_id="S1",
                speaker_confidence=bad_float,
                start_ms=0,
                end_ms=10,
                text="hi",
                stability=Stability.COMMITTED,
            )
        with pytest.raises(ContractValidationError):
            CanonicalEvent(
                schema_version=1,
                session_id="s1",
                event_id="e2",
                kind=EventKind.SEGMENT,
                received_ms=10,
                source_sequence=1,
                segment_id="seg-1",
                source_revision=1,
                channel=0,
                speaker_id="S1",
                asr_confidence=bad_float,
                start_ms=0,
                end_ms=10,
                text="hi",
                stability=Stability.COMMITTED,
            )

    with pytest.raises(ContractValidationError):
        CanonicalEvent(
            schema_version=1,
            session_id="s1",
            event_id="e3",
            kind=EventKind.CONTROL,
            received_ms=10,
            source_sequence=1,
            subtype=ControlSubtype.GAP,
            seq_start=True,
        )
    with pytest.raises(ContractValidationError):
        CanonicalEvent(
            schema_version=1,
            session_id="s1",
            event_id="e4",
            kind=EventKind.CONTROL,
            received_ms=10,
            source_sequence=1,
            subtype=ControlSubtype.GAP,
            seq_end=-1,
        )

    with pytest.raises(ContractValidationError):
        DecisionContext(
            schema_version=1,
            candidate_id="c1",
            session_id="s1",
            origin=Origin.PROACTIVE,
            based_on_state_version=1,
            builder_version="b1",
            prefilter_version="p1",
            anchor_refs=(SegmentRef("seg-1", 1),),
            context_refs=(),
            anchor_received_ms=100,
            display_by_ms=200,
            token_counts={"total": True},
            omitted={"turns": 0},
            flags={"speaker_uncertain": False},
            serialized_sha256="sha",
        )
    with pytest.raises(ContractValidationError):
        DecisionContext(
            schema_version=1,
            candidate_id="c1",
            session_id="s1",
            origin=Origin.PROACTIVE,
            based_on_state_version=1,
            builder_version="b1",
            prefilter_version="p1",
            anchor_refs=(SegmentRef("seg-1", 1),),
            context_refs=(),
            anchor_received_ms=100,
            display_by_ms=200,
            token_counts={"total": 10},
            omitted={"turns": 0},
            flags={"x": "yes"},  # type: ignore[dict-item]
            serialized_sha256="sha",
        )
    with pytest.raises(ContractValidationError):
        EgressRecord(
            schema_version=1,
            egress_id="eg1",
            session_id="s1",
            candidate_id="c1",
            lane=Origin.PROACTIVE,
            provider="local",
            model_id="m1",
            segment_ids=("seg-1",),
            sent_at_ms=100,
            usage={"input_tokens": True},
        )
    with pytest.raises(ContractValidationError):
        EgressRecord(
            schema_version=1,
            egress_id="eg1",
            session_id="s1",
            candidate_id="c1",
            lane=Origin.PROACTIVE,
            provider="local",
            model_id="m1",
            segment_ids=("",),
            sent_at_ms=100,
        )
    with pytest.raises(ContractValidationError):
        RoutingDecision(
            schema_version=1,
            candidate_id="c1",
            policy_version="p1",
            policy_sha256="abc",
            based_on_state_version=1,
            outcome=Outcome.NONE,
            route=Route.TEMPLATE,
            category=None,
            reason_codes=(ReasonCode.BELOW_T_LOW,),
            abstain_reasons=(),
            thresholds=None,
            decided_ms=100,
        )

    # BOUNDARY requires non-empty speaker_id when provided
    with pytest.raises(ContractValidationError):
        CanonicalEvent(
            schema_version=1,
            session_id="s1",
            event_id="e-bnd-empty",
            kind=EventKind.BOUNDARY,
            received_ms=100,
            source_sequence=1,
            channel=None,
            speaker_id="",
            end_ms=100,
        )
    valid_bnd = CanonicalEvent(
        schema_version=1,
        session_id="s1",
        event_id="e-bnd-ok",
        kind=EventKind.BOUNDARY,
        received_ms=100,
        source_sequence=1,
        channel=None,
        speaker_id="S1",
        end_ms=100,
    )
    assert valid_bnd.speaker_id == "S1"

    # CONTROL GAP rejects inverted seq_end < seq_start and end_ms < start_ms
    with pytest.raises(ContractValidationError):
        CanonicalEvent(
            schema_version=1,
            session_id="s1",
            event_id="e-gap-seq",
            kind=EventKind.CONTROL,
            received_ms=100,
            source_sequence=1,
            subtype=ControlSubtype.GAP,
            seq_start=10,
            seq_end=5,
        )
    with pytest.raises(ContractValidationError):
        CanonicalEvent(
            schema_version=1,
            session_id="s1",
            event_id="e-gap-ms",
            kind=EventKind.CONTROL,
            received_ms=100,
            source_sequence=1,
            subtype=ControlSubtype.GAP,
            start_ms=200,
            end_ms=100,
        )
    valid_gap = CanonicalEvent(
        schema_version=1,
        session_id="s1",
        event_id="e-gap-ok",
        kind=EventKind.CONTROL,
        received_ms=100,
        source_sequence=1,
        subtype=ControlSubtype.GAP,
        seq_start=5,
        seq_end=5,
        start_ms=100,
        end_ms=100,
    )
    assert valid_gap.seq_end == 5 and valid_gap.end_ms == 100
