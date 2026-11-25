"""Deterministic canonical state reducer (§6.2, §7.5) with upward-only imports (§15)."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from coach.contracts import (
    CanonicalEvent,
    ContractValidationError,
    ControlSubtype,
    EventKind,
    FeedbackReason,
    LedgerState,
    ReasonCode,
    SegmentRef,
    SessionBrief,
    Stability,
    SuggestionCandidate,
)

STABILITY_RANK: dict[Stability, int] = {
    Stability.INTERIM: 0,
    Stability.COMMITTED: 1,
}

ALLOWED_LEDGER_TRANSITIONS: dict[LedgerState, frozenset[LedgerState]] = {
    LedgerState.READY: frozenset(
        {LedgerState.DISPLAYED, LedgerState.DEFERRED, LedgerState.WITHDRAWN, LedgerState.EXPIRED}
    ),
    LedgerState.DEFERRED: frozenset(
        {LedgerState.DISPLAYED, LedgerState.WITHDRAWN, LedgerState.EXPIRED}
    ),
    LedgerState.DISPLAYED: frozenset(
        {
            LedgerState.USED,
            LedgerState.DISMISSED,
            LedgerState.PINNED,
            LedgerState.RESOLVED,
            LedgerState.WITHDRAWN,
            LedgerState.EXPIRED,
        }
    ),
    LedgerState.PINNED: frozenset(
        {LedgerState.USED, LedgerState.DISMISSED, LedgerState.RESOLVED, LedgerState.OUTDATED}
    ),
    LedgerState.USED: frozenset({LedgerState.RESOLVED}),
    LedgerState.DISMISSED: frozenset(),
    LedgerState.RESOLVED: frozenset(),
    LedgerState.WITHDRAWN: frozenset(),
    LedgerState.EXPIRED: frozenset(),
    LedgerState.OUTDATED: frozenset(),
}

_MATERIAL_BRIEF_FIELDS = frozenset({"objective", "known_facts", "do_not_suggest", "mode"})
_TERMINAL_LEDGER_STATES = frozenset(
    {
        LedgerState.DISMISSED,
        LedgerState.RESOLVED,
        LedgerState.WITHDRAWN,
        LedgerState.EXPIRED,
        LedgerState.OUTDATED,
    }
)


def normalize_for_material_diff(text: str) -> str:
    """Normalize text with Unicode NFC and whitespace collapsing, preserving case & punctuation (§6.2)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """Canonical committed or interim segment representation."""

    segment_id: str
    channel: int
    speaker_id: str
    speaker_confidence: float | None
    start_ms: int
    end_ms: int
    received_ms: int
    text: str
    stability: Stability
    source_revision: int
    revision: int
    source_sequence: int
    boundary: str | None = None
    asr_confidence: float | None = None
    low_quality_gate: bool = False
    asr_low_confidence: bool = False
    speaker_uncertain: bool = False
    overlap: bool = False

    @property
    def committed(self) -> bool:
        return self.stability is Stability.COMMITTED


@dataclass(frozen=True, slots=True)
class GapRecord:
    """Recorded transcript/control gap (§6.1, B06, B18)."""

    event_id: str
    subtype: ControlSubtype
    received_ms: int
    start_ms: int | None = None
    end_ms: int | None = None
    seq_start: int | None = None
    seq_end: int | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LedgerTransition:
    """Recorded single transition in a suggestion's lifecycle."""

    event_id: str
    from_state: LedgerState
    to_state: LedgerState
    cause: ReasonCode
    at_ms: int
    evidence_refs: tuple[SegmentRef, ...] = ()
    feedback: FeedbackReason | None = None


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """Current lifecycle state and history of a suggestion."""

    suggestion_id: str
    candidate_id: str
    state: LedgerState
    evidence_refs: tuple[SegmentRef, ...]
    history: tuple[LedgerTransition, ...] = ()
    feedback: FeedbackReason | None = None


@dataclass(frozen=True, slots=True)
class InvalidationNotice:
    """Invalidation notification emitted when cited evidence or brief changes (§6.2, §8.5)."""

    dependent_kind: str
    dependent_id: str
    segment_id: str | None
    reason: ReasonCode


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Result returned by CanonicalState.apply(event)."""

    accepted: bool
    state_version: int
    material_change: bool
    drop_reason: str | None = None
    invalidations: tuple[InvalidationNotice, ...] = ()
    affected_segment_id: str | None = None


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable point-in-time view of CanonicalState."""

    session_id: str
    state_version: int
    brief: SessionBrief
    segments: tuple[SegmentRecord, ...]
    segments_by_id: dict[str, SegmentRecord]
    interim_segments: dict[str, SegmentRecord]
    ledger: dict[str, LedgerEntry]
    gaps: tuple[GapRecord, ...]
    counters: dict[str, int]
    revision_history: frozenset[tuple[str, int]]
    off_record: bool
    last_boundary_ms: int | None = None


@dataclass(frozen=True, slots=True)
class _PendingSpeakerUpdate:
    source_revision: int | None
    source_sequence: int
    event_id: str
    speaker_id: str
    speaker_confidence: float | None
    speaker_uncertain: bool

    @property
    def order_key(self) -> tuple[int, int, str]:
        return (
            self.source_revision if self.source_revision is not None else 0,
            self.source_sequence,
            self.event_id,
        )


class CanonicalState:
    """Deterministic reducer maintaining transcript, brief, and suggestion ledger (§6.2)."""

    def __init__(self, brief: SessionBrief) -> None:
        self._brief: SessionBrief = brief
        self._state_version: int = 0
        self._applied_events: set[tuple[str, str]] = set()
        self._committed_segments: dict[str, SegmentRecord] = {}
        self._interim_buffer: dict[str, SegmentRecord] = {}
        self._pending_speaker_updates: dict[str, _PendingSpeakerUpdate] = {}
        self._segment_base_speaker_confidence: dict[str, float | None] = {}
        self._invalidation_index: dict[str, set[tuple[str, str]]] = {}
        self._brief_dependents: set[tuple[str, str]] = set()
        self._ledger: dict[str, LedgerEntry] = {}
        self._gaps: list[GapRecord] = []
        self._revision_history: set[tuple[str, int]] = set()
        self._segment_explicit_overlap: dict[str, bool] = {}
        self._off_record: bool = False
        self._last_boundary_ms: int | None = None
        self._counters: dict[str, int] = {
            "duplicate": 0,
            "stale_revision": 0,
            "late_interim": 0,
            "gap": 0,
            "material_revisions_after_commit": 0,
            "invalid_ledger_transition": 0,
        }

    @property
    def session_id(self) -> str:
        return self._brief.session_id

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def brief(self) -> SessionBrief:
        return self._brief

    @property
    def off_record(self) -> bool:
        return self._off_record

    def register_dependent(
        self, dependent_kind: str, dependent_id: str, segment_ids: Iterable[str]
    ) -> None:
        key = (dependent_kind, dependent_id)
        for seg_id in segment_ids:
            self._invalidation_index.setdefault(seg_id, set()).add(key)
        self._brief_dependents.add(key)

    def unregister_dependent(self, dependent_kind: str, dependent_id: str) -> None:
        key = (dependent_kind, dependent_id)
        empty_segs: list[str] = []
        for seg_id, deps in self._invalidation_index.items():
            deps.discard(key)
            if not deps:
                empty_segs.append(seg_id)
        for seg_id in empty_segs:
            del self._invalidation_index[seg_id]
        self._brief_dependents.discard(key)

    def register_suggestion(self, candidate: SuggestionCandidate) -> None:
        """Enter a ready SuggestionCandidate into the ledger and invalidation index."""
        self._ledger[candidate.suggestion_id] = LedgerEntry(
            suggestion_id=candidate.suggestion_id,
            candidate_id=candidate.candidate_id,
            state=LedgerState.READY,
            evidence_refs=candidate.evidence_refs,
        )
        self.register_dependent(
            "suggestion",
            candidate.suggestion_id,
            (ref.segment_id for ref in candidate.evidence_refs),
        )

    def snapshot(self) -> StateSnapshot:
        ordered = tuple(
            sorted(
                self._committed_segments.values(),
                key=lambda s: (s.start_ms, s.channel, s.source_sequence, s.segment_id),
            )
        )
        return StateSnapshot(
            session_id=self._brief.session_id,
            state_version=self._state_version,
            brief=self._brief,
            segments=ordered,
            segments_by_id=dict(self._committed_segments),
            interim_segments=dict(self._interim_buffer),
            ledger=dict(self._ledger),
            gaps=tuple(self._gaps),
            counters=dict(self._counters),
            revision_history=frozenset(self._revision_history),
            off_record=self._off_record,
            last_boundary_ms=self._last_boundary_ms,
        )

    def _recompute_overlaps(self) -> None:
        segs = list(self._committed_segments.values())
        overlapping_ids: set[str] = set()
        for i, a in enumerate(segs):
            for b in segs[i + 1 :]:
                if (
                    a.start_ms < b.end_ms
                    and b.start_ms < a.end_ms
                    and (a.speaker_id != b.speaker_id or a.channel != b.channel)
                ):
                    overlapping_ids.add(a.segment_id)
                    overlapping_ids.add(b.segment_id)
        for seg_id, seg in list(self._committed_segments.items()):
            new_overlap = seg_id in overlapping_ids or self._segment_explicit_overlap.get(
                seg_id, False
            )
            if seg.overlap != new_overlap:
                self._committed_segments[seg_id] = SegmentRecord(
                    segment_id=seg.segment_id,
                    channel=seg.channel,
                    speaker_id=seg.speaker_id,
                    speaker_confidence=seg.speaker_confidence,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    received_ms=seg.received_ms,
                    text=seg.text,
                    stability=seg.stability,
                    source_revision=seg.source_revision,
                    revision=seg.revision,
                    source_sequence=seg.source_sequence,
                    boundary=seg.boundary,
                    asr_confidence=seg.asr_confidence,
                    low_quality_gate=seg.low_quality_gate,
                    asr_low_confidence=seg.asr_low_confidence,
                    speaker_uncertain=seg.speaker_uncertain,
                    overlap=new_overlap,
                )

    def apply(self, event: CanonicalEvent) -> ApplyResult:
        """Apply a CanonicalEvent deterministically (§6.2, §7.5)."""
        if event.session_id != self._brief.session_id:
            raise ValueError(
                f"Event session_id {event.session_id!r} != state session_id {self._brief.session_id!r}"
            )
        dedup_key = (event.source_id, event.event_id)
        if dedup_key in self._applied_events:
            self._counters["duplicate"] += 1
            return ApplyResult(
                accepted=False,
                state_version=self._state_version,
                material_change=False,
                drop_reason="duplicate",
                affected_segment_id=event.segment_id,
            )
        self._applied_events.add(dedup_key)

        if self._off_record and event.kind is not EventKind.CONTROL:
            return ApplyResult(
                accepted=False,
                state_version=self._state_version,
                material_change=False,
                drop_reason=ReasonCode.OFF_RECORD.value,
                affected_segment_id=event.segment_id,
            )

        if event.kind is EventKind.SEGMENT:
            return self._apply_segment(event)
        if event.kind is EventKind.SPEAKER_UPDATE:
            return self._apply_speaker_update(event)
        if event.kind is EventKind.BOUNDARY:
            return self._apply_boundary(event)
        if event.kind is EventKind.BRIEF_UPDATE:
            return self._apply_brief_update(event)
        if event.kind is EventKind.LEDGER:
            return self._apply_ledger(event)
        return self._apply_control(event)

    def _build_segment_record(
        self,
        ev: CanonicalEvent,
        *,
        revision: int,
        speaker_id: str,
        speaker_confidence: float | None,
        speaker_uncertain: bool,
    ) -> SegmentRecord:
        assert ev.segment_id is not None
        assert ev.channel is not None
        assert ev.start_ms is not None
        assert ev.end_ms is not None
        assert ev.text is not None
        assert ev.stability is not None
        assert ev.source_revision is not None
        return SegmentRecord(
            segment_id=ev.segment_id,
            channel=ev.channel,
            speaker_id=speaker_id,
            speaker_confidence=speaker_confidence,
            start_ms=ev.start_ms,
            end_ms=ev.end_ms,
            received_ms=ev.received_ms,
            text=ev.text,
            stability=ev.stability,
            source_revision=ev.source_revision,
            revision=revision,
            source_sequence=ev.source_sequence,
            boundary=ev.boundary,
            asr_confidence=ev.asr_confidence,
            low_quality_gate=ev.low_quality_gate,
            asr_low_confidence=ev.asr_low_confidence,
            speaker_uncertain=speaker_uncertain,
            overlap=ev.overlap,
        )

    def _invalidations_for_segment(
        self, segment_id: str, reason: ReasonCode
    ) -> tuple[InvalidationNotice, ...]:
        deps = sorted(self._invalidation_index.get(segment_id, ()))
        return tuple(
            InvalidationNotice(
                dependent_kind=kind,
                dependent_id=dep_id,
                segment_id=segment_id,
                reason=reason,
            )
            for kind, dep_id in deps
        )

    def _apply_segment(self, ev: CanonicalEvent) -> ApplyResult:
        assert ev.segment_id is not None
        assert ev.stability is not None
        assert ev.source_revision is not None
        assert ev.speaker_id is not None
        assert ev.text is not None

        seg_id = ev.segment_id
        existing = self._committed_segments.get(seg_id) or self._interim_buffer.get(seg_id)
        incoming_rank = (STABILITY_RANK[ev.stability], ev.source_revision)
        if existing is not None:
            existing_rank = (STABILITY_RANK[existing.stability], existing.source_revision)
            if incoming_rank <= existing_rank:
                late = ev.stability is Stability.INTERIM and existing.committed
                reason = "late_interim" if late else "stale_revision"
                self._counters[reason] += 1
                return ApplyResult(
                    accepted=False,
                    state_version=self._state_version,
                    material_change=False,
                    drop_reason=reason,
                    affected_segment_id=seg_id,
                )

        self._segment_base_speaker_confidence[seg_id] = ev.speaker_confidence
        eff_speaker = ev.speaker_id
        eff_conf = ev.speaker_confidence
        eff_uncertain = ev.speaker_uncertain
        pending = self._pending_speaker_updates.get(seg_id)
        if pending is not None:
            eff_speaker = pending.speaker_id
            if pending.speaker_confidence is not None:
                eff_conf = pending.speaker_confidence
            eff_uncertain = pending.speaker_uncertain

        if ev.stability is Stability.INTERIM:
            self._interim_buffer[seg_id] = self._build_segment_record(
                ev,
                revision=0,
                speaker_id=eff_speaker,
                speaker_confidence=eff_conf,
                speaker_uncertain=eff_uncertain,
            )
            return ApplyResult(
                accepted=True,
                state_version=self._state_version,
                material_change=False,
                affected_segment_id=seg_id,
            )

        self._interim_buffer.pop(seg_id, None)
        self._segment_explicit_overlap[seg_id] = ev.overlap
        committed = self._committed_segments.get(seg_id)
        if committed is None:
            self._committed_segments[seg_id] = self._build_segment_record(
                ev,
                revision=1,
                speaker_id=eff_speaker,
                speaker_confidence=eff_conf,
                speaker_uncertain=eff_uncertain,
            )
            self._revision_history.add((seg_id, 1))
            self._state_version += 1
            self._recompute_overlaps()
            return ApplyResult(
                accepted=True,
                state_version=self._state_version,
                material_change=True,
                affected_segment_id=seg_id,
            )

        material = (
            normalize_for_material_diff(committed.text) != normalize_for_material_diff(ev.text)
            or committed.speaker_id != eff_speaker
        )
        if material:
            new_rev = committed.revision + 1
            self._committed_segments[seg_id] = self._build_segment_record(
                ev,
                revision=new_rev,
                speaker_id=eff_speaker,
                speaker_confidence=eff_conf,
                speaker_uncertain=eff_uncertain,
            )
            self._revision_history.add((seg_id, new_rev))
            self._state_version += 1
            self._counters["material_revisions_after_commit"] += 1
            self._recompute_overlaps()
            invalidations = self._invalidations_for_segment(
                seg_id, ReasonCode.EVIDENCE_REVISION_CHANGED
            )
            return ApplyResult(
                accepted=True,
                state_version=self._state_version,
                material_change=True,
                invalidations=invalidations,
                affected_segment_id=seg_id,
            )

        self._committed_segments[seg_id] = self._build_segment_record(
            ev,
            revision=committed.revision,
            speaker_id=eff_speaker,
            speaker_confidence=eff_conf,
            speaker_uncertain=eff_uncertain,
        )
        self._recompute_overlaps()
        return ApplyResult(
            accepted=True,
            state_version=self._state_version,
            material_change=False,
            affected_segment_id=seg_id,
        )

    def _apply_speaker_update(self, ev: CanonicalEvent) -> ApplyResult:
        assert ev.segment_id is not None
        assert ev.speaker_id is not None
        seg_id = ev.segment_id
        pending = self._pending_speaker_updates.get(seg_id)
        incoming_key = (
            ev.source_revision if ev.source_revision is not None else 0,
            ev.source_sequence,
            ev.event_id,
        )
        if pending is not None and incoming_key <= pending.order_key:
            self._counters["stale_revision"] += 1
            return ApplyResult(
                accepted=False,
                state_version=self._state_version,
                material_change=False,
                drop_reason="stale_revision",
                affected_segment_id=seg_id,
            )
        self._pending_speaker_updates[seg_id] = _PendingSpeakerUpdate(
            source_revision=ev.source_revision,
            source_sequence=ev.source_sequence,
            event_id=ev.event_id,
            speaker_id=ev.speaker_id,
            speaker_confidence=ev.speaker_confidence,
            speaker_uncertain=ev.speaker_uncertain,
        )

        committed = self._committed_segments.get(seg_id)
        if committed is not None:
            changed = (
                committed.speaker_id != ev.speaker_id
                or committed.speaker_uncertain != ev.speaker_uncertain
            )
            new_rev = committed.revision + 1 if changed else committed.revision
            base_conf = self._segment_base_speaker_confidence.get(
                seg_id, committed.speaker_confidence
            )
            new_conf = ev.speaker_confidence if ev.speaker_confidence is not None else base_conf
            self._committed_segments[seg_id] = SegmentRecord(
                segment_id=committed.segment_id,
                channel=committed.channel,
                speaker_id=ev.speaker_id,
                speaker_confidence=new_conf,
                start_ms=committed.start_ms,
                end_ms=committed.end_ms,
                received_ms=committed.received_ms,
                text=committed.text,
                stability=committed.stability,
                source_revision=committed.source_revision,
                revision=new_rev,
                source_sequence=committed.source_sequence,
                boundary=committed.boundary,
                asr_confidence=committed.asr_confidence,
                low_quality_gate=committed.low_quality_gate,
                asr_low_confidence=committed.asr_low_confidence,
                speaker_uncertain=ev.speaker_uncertain,
                overlap=committed.overlap,
            )
            if changed:
                self._revision_history.add((seg_id, new_rev))
                self._state_version += 1
                self._counters["material_revisions_after_commit"] += 1
                self._recompute_overlaps()
                invalidations = self._invalidations_for_segment(
                    seg_id, ReasonCode.EVIDENCE_REVISION_CHANGED
                )
                return ApplyResult(
                    accepted=True,
                    state_version=self._state_version,
                    material_change=True,
                    invalidations=invalidations,
                    affected_segment_id=seg_id,
                )
            return ApplyResult(
                accepted=True,
                state_version=self._state_version,
                material_change=False,
                affected_segment_id=seg_id,
            )

        interim = self._interim_buffer.get(seg_id)
        if interim is not None:
            base_conf = self._segment_base_speaker_confidence.get(
                seg_id, interim.speaker_confidence
            )
            new_conf = ev.speaker_confidence if ev.speaker_confidence is not None else base_conf
            self._interim_buffer[seg_id] = SegmentRecord(
                segment_id=interim.segment_id,
                channel=interim.channel,
                speaker_id=ev.speaker_id,
                speaker_confidence=new_conf,
                start_ms=interim.start_ms,
                end_ms=interim.end_ms,
                received_ms=interim.received_ms,
                text=interim.text,
                stability=interim.stability,
                source_revision=interim.source_revision,
                revision=interim.revision,
                source_sequence=interim.source_sequence,
                boundary=interim.boundary,
                asr_confidence=interim.asr_confidence,
                low_quality_gate=interim.low_quality_gate,
                asr_low_confidence=interim.asr_low_confidence,
                speaker_uncertain=ev.speaker_uncertain,
                overlap=interim.overlap,
            )
        return ApplyResult(
            accepted=True,
            state_version=self._state_version,
            material_change=False,
            affected_segment_id=seg_id,
        )

    def _apply_boundary(self, ev: CanonicalEvent) -> ApplyResult:
        assert ev.end_ms is not None
        if self._last_boundary_ms is None or ev.end_ms > self._last_boundary_ms:
            self._last_boundary_ms = ev.end_ms
        return ApplyResult(
            accepted=True,
            state_version=self._state_version,
            material_change=False,
            affected_segment_id=ev.segment_id,
        )

    def _apply_brief_update(self, ev: CanonicalEvent) -> ApplyResult:
        assert ev.brief_patch is not None
        if (
            "session_id" in ev.brief_patch
            and ev.brief_patch["session_id"] != self._brief.session_id
        ):
            raise ContractValidationError(
                f"brief_patch cannot mutate session_id ({ev.brief_patch['session_id']!r} != {self._brief.session_id!r})"
            )
        if "schema_version" in ev.brief_patch and (
            isinstance(ev.brief_patch["schema_version"], bool)
            or ev.brief_patch["schema_version"] != self._brief.schema_version
        ):
            raise ContractValidationError(
                f"brief_patch cannot mutate schema_version ({ev.brief_patch['schema_version']!r})"
            )
        merged = {**self._brief.to_dict(), **ev.brief_patch}
        updated_brief = SessionBrief.from_dict(merged)
        material = updated_brief != self._brief
        material_field_changed = any(
            getattr(updated_brief, f) != getattr(self._brief, f) for f in _MATERIAL_BRIEF_FIELDS
        )
        self._brief = updated_brief
        invalidations: tuple[InvalidationNotice, ...] = ()
        if material:
            self._state_version += 1
            if material_field_changed:
                invalidations = tuple(
                    InvalidationNotice(
                        dependent_kind=kind,
                        dependent_id=dep_id,
                        segment_id=None,
                        reason=ReasonCode.OBJECTIVE_CHANGED,
                    )
                    for kind, dep_id in sorted(self._brief_dependents)
                )
        return ApplyResult(
            accepted=True,
            state_version=self._state_version,
            material_change=material,
            invalidations=invalidations,
        )

    def _apply_ledger(self, ev: CanonicalEvent) -> ApplyResult:
        assert ev.suggestion_id is not None
        assert ev.transition is not None
        assert ev.cause is not None
        assert ev.at_ms is not None
        entry = self._ledger.get(ev.suggestion_id)
        if entry is None or ev.transition not in ALLOWED_LEDGER_TRANSITIONS.get(
            entry.state, frozenset()
        ):
            self._counters["invalid_ledger_transition"] += 1
            return ApplyResult(
                accepted=False,
                state_version=self._state_version,
                material_change=False,
                drop_reason="invalid_ledger_transition",
            )
        transition = LedgerTransition(
            event_id=ev.event_id,
            from_state=entry.state,
            to_state=ev.transition,
            cause=ev.cause,
            at_ms=ev.at_ms,
            evidence_refs=ev.evidence_refs,
            feedback=ev.feedback,
        )
        self._ledger[ev.suggestion_id] = LedgerEntry(
            suggestion_id=entry.suggestion_id,
            candidate_id=entry.candidate_id,
            state=ev.transition,
            evidence_refs=entry.evidence_refs,
            history=entry.history + (transition,),
            feedback=ev.feedback if ev.feedback is not None else entry.feedback,
        )
        if ev.transition in _TERMINAL_LEDGER_STATES:
            self.unregister_dependent("suggestion", ev.suggestion_id)
        self._state_version += 1
        return ApplyResult(
            accepted=True,
            state_version=self._state_version,
            material_change=True,
        )

    def _apply_control(self, ev: CanonicalEvent) -> ApplyResult:
        assert ev.subtype is not None
        if ev.subtype is ControlSubtype.OFF_RECORD_START:
            self._off_record = True
            self._state_version += 1
            return ApplyResult(
                accepted=True, state_version=self._state_version, material_change=True
            )
        if ev.subtype is ControlSubtype.OFF_RECORD_END:
            self._off_record = False
            self._gaps.append(
                GapRecord(
                    event_id=ev.event_id,
                    subtype=ev.subtype,
                    received_ms=ev.received_ms,
                    start_ms=ev.start_ms,
                    end_ms=ev.end_ms,
                    seq_start=ev.seq_start,
                    seq_end=ev.seq_end,
                    reason=ev.reason or "off_record",
                )
            )
            self._counters["gap"] += 1
            self._state_version += 1
            return ApplyResult(
                accepted=True, state_version=self._state_version, material_change=True
            )
        if ev.subtype is ControlSubtype.GAP:
            self._gaps.append(
                GapRecord(
                    event_id=ev.event_id,
                    subtype=ev.subtype,
                    received_ms=ev.received_ms,
                    start_ms=ev.start_ms,
                    end_ms=ev.end_ms,
                    seq_start=ev.seq_start,
                    seq_end=ev.seq_end,
                    reason=ev.reason,
                )
            )
            self._counters["gap"] += 1
            self._state_version += 1
            return ApplyResult(
                accepted=True, state_version=self._state_version, material_change=True
            )
        return ApplyResult(accepted=True, state_version=self._state_version, material_change=False)
