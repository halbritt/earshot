"""Unit and Hypothesis property tests for coach.canonical_state (§6.2, §7.5, Appendix B)."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from coach.canonical_state import (
    ALLOWED_LEDGER_TRANSITIONS,
    CanonicalState,
    StateSnapshot,
    normalize_for_material_diff,
)
from coach.contracts import (
    CanonicalEvent,
    Category,
    ContractValidationError,
    ControlSubtype,
    EventKind,
    FeedbackReason,
    LedgerState,
    Origin,
    ReasonCode,
    RecipientConfig,
    Route,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    SuggestionCandidate,
)


def _make_brief(session_id: str = "s-test") -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id=session_id,
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="me", role="reviewer"),
        objective="Leave with an owner and date for rollback.",
    )


def _canonical_content(snap: StateSnapshot) -> tuple[Any, ...]:
    committed = tuple(
        (
            s.segment_id,
            s.start_ms,
            s.channel,
            s.source_sequence,
            normalize_for_material_diff(s.text),
            s.speaker_id,
            s.speaker_confidence,
            s.speaker_uncertain,
            s.source_revision,
            s.overlap,
        )
        for s in snap.segments
    )
    interim = tuple(
        sorted(
            (
                s.segment_id,
                s.start_ms,
                s.channel,
                s.source_sequence,
                normalize_for_material_diff(s.text),
                s.speaker_id,
                s.speaker_confidence,
                s.speaker_uncertain,
                s.source_revision,
                s.overlap,
            )
            for s in snap.interim_segments.values()
        )
    )
    return (committed, interim, snap.last_boundary_ms)


@st.composite
def _segment_stream(draw: st.DrawFn) -> list[CanonicalEvent]:
    num_segs = draw(st.integers(min_value=1, max_value=5))
    events: list[CanonicalEvent] = []
    ev_idx = 0
    for s_idx in range(num_segs):
        seg_id = f"seg-{s_idx}"
        start_ms = draw(st.integers(min_value=0, max_value=20_000))
        dur = draw(st.integers(min_value=50, max_value=2_000))
        channel = draw(st.integers(min_value=0, max_value=2))
        seq = s_idx * 10 + 1
        num_revs = draw(st.integers(min_value=1, max_value=4))
        revs = sorted(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=20),
                    min_size=num_revs,
                    max_size=num_revs,
                    unique=True,
                )
            )
        )
        for rev in revs:
            ev_idx += 1
            stability = draw(st.sampled_from([Stability.INTERIM, Stability.COMMITTED]))
            speaker = draw(st.sampled_from(["S1", "S2", "ME"]))
            text = draw(
                st.sampled_from(
                    ["alpha", "Alpha", "alpha.", "alpha  beta", "caf\u00e9", "cafe\u0301"]
                )
            )
            rx = draw(st.integers(min_value=start_ms, max_value=start_ms + 30_000))
            events.append(
                CanonicalEvent(
                    schema_version=1,
                    session_id="s-test",
                    event_id=f"ev-{ev_idx}",
                    kind=EventKind.SEGMENT,
                    received_ms=rx,
                    source_sequence=seq,
                    segment_id=seg_id,
                    source_revision=rev,
                    channel=channel,
                    speaker_id=speaker,
                    speaker_confidence=draw(st.sampled_from([None, 0.6, 0.85])),
                    start_ms=start_ms,
                    end_ms=start_ms + dur,
                    text=text,
                    stability=stability,
                    overlap=draw(st.booleans()),
                )
            )
        num_spk = draw(st.integers(min_value=0, max_value=3))
        for _k in range(num_spk):
            ev_idx += 1
            spk_rev = draw(st.sampled_from([None, 1, 2, 3]))
            spk_seq = draw(st.integers(min_value=1, max_value=50))
            rx = draw(st.integers(min_value=start_ms, max_value=start_ms + 30_000))
            events.append(
                CanonicalEvent(
                    schema_version=1,
                    session_id="s-test",
                    event_id=f"ev-spk-{ev_idx}",
                    kind=EventKind.SPEAKER_UPDATE,
                    received_ms=rx,
                    source_sequence=spk_seq,
                    segment_id=seg_id,
                    source_revision=spk_rev,
                    speaker_id=draw(st.sampled_from(["S1", "S2", "ME"])),
                    speaker_confidence=draw(st.sampled_from([None, 0.7, 0.95])),
                    speaker_uncertain=draw(st.booleans()),
                )
            )
        if draw(st.booleans()):
            ev_idx += 1
            b_end = start_ms + dur + draw(st.integers(min_value=0, max_value=500))
            events.append(
                CanonicalEvent(
                    schema_version=1,
                    session_id="s-test",
                    event_id=f"ev-bnd-{ev_idx}",
                    kind=EventKind.BOUNDARY,
                    received_ms=b_end,
                    source_sequence=seq + k_offset if (k_offset := ev_idx) else ev_idx,
                    channel=channel,
                    end_ms=b_end,
                    boundary="utterance_end",
                )
            )
    return events


@settings(max_examples=80)
@given(stream=_segment_stream())
def test_property_idempotence(stream: list[CanonicalEvent]) -> None:
    s1 = CanonicalState(_make_brief())
    for ev in stream:
        s1.apply(ev)
    snap1 = s1.snapshot()

    s2 = CanonicalState(_make_brief())
    for ev in stream + stream:
        s2.apply(ev)
    snap2 = s2.snapshot()

    assert _canonical_content(snap1) == _canonical_content(snap2)
    assert snap2.counters["duplicate"] == snap1.counters["duplicate"] + len(stream)


@settings(max_examples=80)
@given(data=st.data())
def test_property_order_convergence(data: st.DataObject) -> None:
    stream = data.draw(_segment_stream())
    permuted = data.draw(st.permutations(stream))

    s_orig = CanonicalState(_make_brief())
    for ev in stream:
        s_orig.apply(ev)

    s_perm = CanonicalState(_make_brief())
    for ev in permuted:
        s_perm.apply(ev)

    assert _canonical_content(s_orig.snapshot()) == _canonical_content(s_perm.snapshot())


@settings(max_examples=80)
@given(data=st.data())
def test_property_prefix_determinism(data: st.DataObject) -> None:
    stream = data.draw(_segment_stream())
    cutoff_ms = data.draw(st.integers(min_value=0, max_value=50_000))

    prefix = [ev for ev in stream if ev.received_ms <= cutoff_ms]
    s_prefix = CanonicalState(_make_brief())
    for ev in prefix:
        s_prefix.apply(ev)

    s_time_ordered = CanonicalState(_make_brief())
    for ev in sorted(stream, key=lambda e: e.received_ms):
        if ev.received_ms <= cutoff_ms:
            s_time_ordered.apply(ev)

    assert _canonical_content(s_prefix.snapshot()) == _canonical_content(s_time_ordered.snapshot())


@settings(max_examples=80)
@given(data=st.data())
def test_property_monotone_commit(data: st.DataObject) -> None:
    stream = data.draw(_segment_stream())
    state = CanonicalState(_make_brief())
    committed_seen: dict[str, tuple[str, str, int, int, int, int]] = {}
    for ev in stream:
        prev_ver = state.state_version
        state.apply(ev)
        snap = state.snapshot()
        for seg_id, prev_vals in committed_seen.items():
            seg = snap.segments_by_id[seg_id]
            cur_vals = (
                seg.text,
                seg.speaker_id,
                seg.start_ms,
                seg.end_ms,
                seg.source_revision,
                seg.revision,
            )
            if (
                ev.kind is EventKind.SEGMENT
                and ev.segment_id == seg_id
                and ev.stability is Stability.INTERIM
            ):
                assert cur_vals == prev_vals
                assert state.state_version == prev_ver
            committed_seen[seg_id] = cur_vals
        for seg_id, seg in snap.segments_by_id.items():
            if seg_id not in committed_seen:
                committed_seen[seg_id] = (
                    seg.text,
                    seg.speaker_id,
                    seg.start_ms,
                    seg.end_ms,
                    seg.source_revision,
                    seg.revision,
                )


def _seg_ev(
    event_id: str,
    segment_id: str,
    *,
    rev: int = 1,
    stability: Stability = Stability.COMMITTED,
    text: str = "Someone should own the rollback plan.",
    speaker: str = "S2",
    channel: int = 0,
    start_ms: int = 1000,
    end_ms: int = 2000,
    received_ms: int = 2100,
    seq: int = 1,
) -> CanonicalEvent:
    return CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id=event_id,
        kind=EventKind.SEGMENT,
        received_ms=received_ms,
        source_sequence=seq,
        segment_id=segment_id,
        source_revision=rev,
        channel=channel,
        speaker_id=speaker,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        stability=stability,
    )


def test_b01_duplicate_events() -> None:
    events = [
        _seg_ev("e1", "seg-1", rev=1),
        _seg_ev("e2", "seg-2", rev=1, start_ms=3000, end_ms=4000, seq=2),
    ]
    s1 = CanonicalState(_make_brief())
    s2 = CanonicalState(_make_brief())
    for ev in events:
        s1.apply(ev)
    for ev in events + events:
        s2.apply(ev)
    assert _canonical_content(s1.snapshot()) == _canonical_content(s2.snapshot())
    assert s2.snapshot().counters["duplicate"] == len(events)


def test_b02_out_of_order_revisions_and_b03_late_interim() -> None:
    state = CanonicalState(_make_brief())
    r_high = state.apply(_seg_ev("e-rev3", "seg-1", rev=3, text="Final rollback plan."))
    assert r_high.accepted and r_high.material_change

    r_stale = state.apply(_seg_ev("e-rev2", "seg-1", rev=2, text="Stale text"))
    assert not r_stale.accepted and r_stale.drop_reason == "stale_revision"

    r_late_interim = state.apply(
        _seg_ev("e-int4", "seg-1", rev=4, stability=Stability.INTERIM, text="Interim after commit")
    )
    assert not r_late_interim.accepted and r_late_interim.drop_reason == "late_interim"

    snap = state.snapshot()
    assert snap.segments_by_id["seg-1"].text == "Final rollback plan."
    assert snap.counters["stale_revision"] == 1
    assert snap.counters["late_interim"] == 1


def test_b04_material_correction_and_immaterial_diff() -> None:
    state = CanonicalState(_make_brief())
    state.apply(_seg_ev("e1", "seg-1", rev=1, text="caf\u00e9   plan."))
    state.register_dependent("suggestion", "sug-1", ["seg-1"])
    state.register_dependent("suggestion", "sug-other", ["seg-2"])
    v_before = state.state_version

    # Immaterial: NFC + whitespace collapse only
    r_immat = state.apply(_seg_ev("e2", "seg-1", rev=2, text="cafe\u0301 plan."))
    assert r_immat.accepted and not r_immat.material_change
    assert r_immat.invalidations == ()
    assert state.state_version == v_before
    assert state.snapshot().segments_by_id["seg-1"].revision == 1
    assert state.snapshot().segments_by_id["seg-1"].source_revision == 2

    # Material: punctuation change (. -> ?)
    r_mat = state.apply(_seg_ev("e3", "seg-1", rev=3, text="caf\u00e9 plan?"))
    assert r_mat.accepted and r_mat.material_change
    assert state.state_version == v_before + 1
    assert state.snapshot().segments_by_id["seg-1"].revision == 2
    assert state.snapshot().counters["material_revisions_after_commit"] == 1
    assert len(r_mat.invalidations) == 1
    assert r_mat.invalidations[0].dependent_id == "sug-1"
    assert r_mat.invalidations[0].reason is ReasonCode.EVIDENCE_REVISION_CHANGED


def test_b05_b22_speaker_update_and_pending_order_convergence() -> None:
    state = CanonicalState(_make_brief())
    state.apply(_seg_ev("e1", "seg-1", rev=1, speaker="S1"))
    state.register_dependent("suggestion", "sug-1", ["seg-1"])

    res = state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="e-spk",
            kind=EventKind.SPEAKER_UPDATE,
            received_ms=2200,
            source_sequence=2,
            segment_id="seg-1",
            speaker_id="ME",
            speaker_confidence=0.95,
        )
    )
    assert res.accepted and res.material_change
    assert state.snapshot().segments_by_id["seg-1"].speaker_id == "ME"
    assert state.snapshot().segments_by_id["seg-1"].revision == 2
    assert res.invalidations[0].reason is ReasonCode.EVIDENCE_REVISION_CHANGED

    # Permuted: speaker_update arrives before segment
    state_perm = CanonicalState(_make_brief())
    state_perm.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="e-spk-early",
            kind=EventKind.SPEAKER_UPDATE,
            received_ms=2200,
            source_sequence=2,
            segment_id="seg-1",
            speaker_id="ME",
        )
    )
    state_perm.apply(_seg_ev("e1-late", "seg-1", rev=1, speaker="S1"))
    assert state_perm.snapshot().segments_by_id["seg-1"].speaker_id == "ME"


def test_b06_b18_gap_and_off_record_and_b15_brief_update() -> None:
    state = CanonicalState(_make_brief())
    state.apply(_seg_ev("e1", "seg-1", rev=1))
    state.register_dependent("suggestion", "sug-1", ["seg-1"])

    r_brief = state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="e-brief",
            kind=EventKind.BRIEF_UPDATE,
            received_ms=2500,
            source_sequence=2,
            brief_patch={"objective": "Focus only on launch readiness."},
        )
    )
    assert r_brief.accepted and r_brief.material_change
    assert r_brief.invalidations[0].reason is ReasonCode.OBJECTIVE_CHANGED

    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="e-off-start",
            kind=EventKind.CONTROL,
            received_ms=3000,
            source_sequence=3,
            subtype=ControlSubtype.OFF_RECORD_START,
        )
    )
    assert state.off_record
    r_dropped = state.apply(
        _seg_ev("e-while-off", "seg-off", rev=1, start_ms=3100, end_ms=3500, seq=4)
    )
    assert not r_dropped.accepted and r_dropped.drop_reason == ReasonCode.OFF_RECORD.value
    assert "seg-off" not in state.snapshot().segments_by_id

    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="e-off-end",
            kind=EventKind.CONTROL,
            received_ms=4000,
            source_sequence=5,
            subtype=ControlSubtype.OFF_RECORD_END,
            start_ms=3000,
            end_ms=4000,
        )
    )
    assert not state.snapshot().off_record
    assert len(state.snapshot().gaps) == 1
    assert state.snapshot().counters["gap"] == 1


def test_b19_overlap_and_transcript_ordering_ignores_received_ms() -> None:
    state = CanonicalState(_make_brief())
    # Later start_ms arrives earlier (smaller received_ms)
    state.apply(
        _seg_ev(
            "e2",
            "seg-2",
            start_ms=1500,
            end_ms=2500,
            received_ms=1600,
            speaker="S2",
            channel=1,
            seq=2,
        )
    )
    state.apply(
        _seg_ev(
            "e1",
            "seg-1",
            start_ms=1000,
            end_ms=2000,
            received_ms=5000,
            speaker="S1",
            channel=0,
            seq=1,
        )
    )
    segs = state.snapshot().segments
    assert [s.segment_id for s in segs] == ["seg-1", "seg-2"]
    assert all(s.overlap for s in segs)


def test_ledger_transition_table_enforcement() -> None:
    state = CanonicalState(_make_brief())
    state.apply(_seg_ev("e1", "seg-1", rev=1))
    cand = SuggestionCandidate(
        schema_version=1,
        suggestion_id="sug-1",
        candidate_id="cand-1",
        origin=Origin.PROACTIVE,
        category=Category.CONFIRM_OWNER,
        route=Route.TEMPLATE,
        text="Who owns rollback?",
        word_count=3,
        based_on_state_version=1,
        evidence_refs=(SegmentRef("seg-1", 1),),
        template_id="confirm_owner_v1",
        grounded_slots={},
        ready_ms=2200,
        display_by_ms=10100,
    )
    state.register_suggestion(cand)

    # Illegal: ready -> used
    bad = state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="l-bad",
            kind=EventKind.LEDGER,
            received_ms=2300,
            source_sequence=2,
            suggestion_id="sug-1",
            transition=LedgerState.USED,
            cause=ReasonCode.USER_ACTION,
            at_ms=2300,
        )
    )
    assert not bad.accepted and bad.drop_reason == "invalid_ledger_transition"
    assert state.snapshot().counters["invalid_ledger_transition"] == 1
    assert state.snapshot().ledger["sug-1"].state is LedgerState.READY

    # Legal chain: ready -> displayed -> used -> resolved
    for idx, (tr, cause, fb) in enumerate(
        [
            (LedgerState.DISPLAYED, ReasonCode.BOUNDARY_AFTER_SETTLE, None),
            (LedgerState.USED, ReasonCode.USER_ACTION, None),
            (LedgerState.RESOLVED, ReasonCode.RESOLUTION_CUE, None),
        ],
        start=10,
    ):
        ok = state.apply(
            CanonicalEvent(
                schema_version=1,
                session_id="s-test",
                event_id=f"l-ok-{idx}",
                kind=EventKind.LEDGER,
                received_ms=2300 + idx,
                source_sequence=idx,
                suggestion_id="sug-1",
                transition=tr,
                cause=cause,
                at_ms=2300 + idx,
                feedback=fb,
            )
        )
        assert ok.accepted
    assert state.snapshot().ledger["sug-1"].state is LedgerState.RESOLVED
    assert LedgerState.USED in ALLOWED_LEDGER_TRANSITIONS[LedgerState.DISPLAYED]

    # Illegal from terminal resolved -> dismissed
    bad_term = state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="l-bad-term",
            kind=EventKind.LEDGER,
            received_ms=2400,
            source_sequence=20,
            suggestion_id="sug-1",
            transition=LedgerState.DISMISSED,
            cause=ReasonCode.USER_ACTION,
            at_ms=2400,
            feedback=FeedbackReason.TOO_LATE,
        )
    )
    assert not bad_term.accepted
    assert state.snapshot().counters["invalid_ledger_transition"] == 2


def test_speaker_update_and_boundary_all_permutations_converge() -> None:
    seg1 = _seg_ev("e-s1", "seg-1", rev=1, text="Initial text.", speaker="S1", seq=1)
    seg2 = _seg_ev("e-s2", "seg-1", rev=2, text="Corrected text?", speaker="S1", seq=1)
    spk_none = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="e-spk-none",
        kind=EventKind.SPEAKER_UPDATE,
        received_ms=2200,
        source_sequence=2,
        segment_id="seg-1",
        source_revision=None,
        speaker_id="S2",
        speaker_confidence=0.7,
        speaker_uncertain=True,
    )
    spk_rev2 = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="e-spk-rev2",
        kind=EventKind.SPEAKER_UPDATE,
        received_ms=2300,
        source_sequence=3,
        segment_id="seg-1",
        source_revision=2,
        speaker_id="ME",
        speaker_confidence=0.95,
        speaker_uncertain=False,
    )
    # 1) Reviewer probe: {seg1, seg2, spk_rev2}
    states_1 = set()
    for perm in itertools.permutations([seg1, seg2, spk_rev2]):
        st_obj = CanonicalState(_make_brief())
        for ev in perm:
            st_obj.apply(ev)
        seg = st_obj.snapshot().segments_by_id["seg-1"]
        states_1.add(
            (
                seg.text,
                seg.speaker_id,
                seg.speaker_confidence,
                seg.speaker_uncertain,
                seg.source_revision,
            )
        )
    assert states_1 == {("Corrected text?", "ME", 0.95, False, 2)}

    # 2) Challenger 2A: {seg1, spk_none, spk_rev2}
    states_2a = set()
    for perm in itertools.permutations([seg1, spk_none, spk_rev2]):
        st_obj = CanonicalState(_make_brief())
        for ev in perm:
            st_obj.apply(ev)
        seg = st_obj.snapshot().segments_by_id["seg-1"]
        states_2a.add(
            (
                seg.text,
                seg.speaker_id,
                seg.speaker_confidence,
                seg.speaker_uncertain,
                seg.source_revision,
            )
        )
    assert states_2a == {("Initial text.", "ME", 0.95, False, 1)}

    # 3) Challenger 2B: tied source_revision=1 with distinct source_sequence
    spk_t1 = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="e-t1",
        kind=EventKind.SPEAKER_UPDATE,
        received_ms=2200,
        source_sequence=2,
        segment_id="seg-1",
        source_revision=1,
        speaker_id="S2",
        speaker_confidence=0.7,
    )
    spk_t2 = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="e-t2",
        kind=EventKind.SPEAKER_UPDATE,
        received_ms=2300,
        source_sequence=3,
        segment_id="seg-1",
        source_revision=1,
        speaker_id="S3",
        speaker_confidence=0.9,
    )
    states_2b = set()
    for perm in itertools.permutations([seg1, spk_t1, spk_t2]):
        st_obj = CanonicalState(_make_brief())
        for ev in perm:
            st_obj.apply(ev)
        seg = st_obj.snapshot().segments_by_id["seg-1"]
        states_2b.add((seg.speaker_id, seg.speaker_confidence))
    assert states_2b == {("S3", 0.9)}

    # 4) Out-of-order boundary events never regress last_boundary_ms
    b1 = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="b1",
        kind=EventKind.BOUNDARY,
        received_ms=2000,
        source_sequence=1,
        channel=0,
        end_ms=2000,
    )
    b2 = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="b2",
        kind=EventKind.BOUNDARY,
        received_ms=5000,
        source_sequence=2,
        channel=0,
        end_ms=5000,
    )
    st_b = CanonicalState(_make_brief())
    st_b.apply(b2)
    st_b.apply(b1)
    assert st_b.snapshot().last_boundary_ms == 5000


def test_brief_update_guards_and_terminal_ledger_unregisters_dependents() -> None:
    state = CanonicalState(_make_brief())
    state.apply(_seg_ev("e1", "seg-1", rev=1))
    cand = SuggestionCandidate(
        schema_version=1,
        suggestion_id="sug-1",
        candidate_id="cand-1",
        origin=Origin.PROACTIVE,
        category=Category.CONFIRM_OWNER,
        route=Route.TEMPLATE,
        text="Who owns rollback?",
        word_count=3,
        based_on_state_version=1,
        evidence_refs=(SegmentRef("seg-1", 1),),
        template_id="confirm_owner_v1",
        grounded_slots={},
        ready_ms=2200,
        display_by_ms=10100,
    )
    state.register_suggestion(cand)

    # Reject session_id and schema_version mutation
    with pytest.raises(ContractValidationError):
        state.apply(
            CanonicalEvent(
                schema_version=1,
                session_id="s-test",
                event_id="bu-hijack",
                kind=EventKind.BRIEF_UPDATE,
                received_ms=2210,
                source_sequence=2,
                brief_patch={"session_id": "hijacked"},
            )
        )
    with pytest.raises(ContractValidationError):
        state.apply(
            CanonicalEvent(
                schema_version=1,
                session_id="s-test",
                event_id="bu-ver",
                kind=EventKind.BRIEF_UPDATE,
                received_ms=2215,
                source_sequence=3,
                brief_patch={"schema_version": 2},
            )
        )
    assert state.brief.session_id == "s-test"

    # Unchanged material field + changed agenda -> no OBJECTIVE_CHANGED invalidation
    r_nonmat = state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="bu-agenda",
            kind=EventKind.BRIEF_UPDATE,
            received_ms=2220,
            source_sequence=4,
            brief_patch={
                "objective": state.brief.objective,
                "agenda": ["cutover"],
            },
        )
    )
    assert r_nonmat.accepted and r_nonmat.material_change
    assert r_nonmat.invalidations == ()

    # Transition sug-1 -> displayed -> dismissed => unregisters from invalidation index
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="l-disp",
            kind=EventKind.LEDGER,
            received_ms=2230,
            source_sequence=5,
            suggestion_id="sug-1",
            transition=LedgerState.DISPLAYED,
            cause=ReasonCode.BOUNDARY_AFTER_SETTLE,
            at_ms=2230,
        )
    )
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="l-dism",
            kind=EventKind.LEDGER,
            received_ms=2240,
            source_sequence=6,
            suggestion_id="sug-1",
            transition=LedgerState.DISMISSED,
            cause=ReasonCode.USER_ACTION,
            at_ms=2240,
        )
    )
    r_seg_rev = state.apply(_seg_ev("e2", "seg-1", rev=2, text="Updated rollback text."))
    assert r_seg_rev.invalidations == ()
    r_obj = state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="s-test",
            event_id="bu-obj",
            kind=EventKind.BRIEF_UPDATE,
            received_ms=2250,
            source_sequence=7,
            brief_patch={"objective": "Brand new objective."},
        )
    )
    assert r_obj.invalidations == ()


def test_overlap_order_convergence_on_revision_overwrite() -> None:
    seg_r1 = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="e-ov-1",
        kind=EventKind.SEGMENT,
        received_ms=2000,
        source_sequence=1,
        segment_id="seg-1",
        source_revision=1,
        channel=0,
        speaker_id="S1",
        start_ms=1000,
        end_ms=2000,
        text="Hello world",
        stability=Stability.COMMITTED,
        overlap=True,
    )
    seg_r2 = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="e-ov-2",
        kind=EventKind.SEGMENT,
        received_ms=2100,
        source_sequence=2,
        segment_id="seg-1",
        source_revision=2,
        channel=0,
        speaker_id="S1",
        start_ms=1000,
        end_ms=2000,
        text="Hello world updated",
        stability=Stability.COMMITTED,
        overlap=False,
    )
    overlaps = set()
    for perm in itertools.permutations([seg_r1, seg_r2]):
        st_obj = CanonicalState(_make_brief())
        for ev in perm:
            st_obj.apply(ev)
        overlaps.add(st_obj.snapshot().segments_by_id["seg-1"].overlap)
    assert overlaps == {False}

    # Pairwise [start_ms, end_ms) overlap recomputation converges too
    seg_other = CanonicalEvent(
        schema_version=1,
        session_id="s-test",
        event_id="e-ov-other",
        kind=EventKind.SEGMENT,
        received_ms=2150,
        source_sequence=3,
        segment_id="seg-2",
        source_revision=1,
        channel=0,
        speaker_id="S2",
        start_ms=1500,
        end_ms=2500,
        text="Interrupting",
        stability=Stability.COMMITTED,
        overlap=False,
    )
    pair_overlaps = set()
    for perm in itertools.permutations([seg_r1, seg_r2, seg_other]):
        st_obj = CanonicalState(_make_brief())
        for ev in perm:
            st_obj.apply(ev)
        snap = st_obj.snapshot()
        pair_overlaps.add(
            (snap.segments_by_id["seg-1"].overlap, snap.segments_by_id["seg-2"].overlap)
        )
    assert pair_overlaps == {(True, True)}
