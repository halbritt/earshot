"""Observability: bounded recorder, 6 conservation invariants, explain, diff, and report (§4.2, §13)."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from coach.contracts import (
    ABSTAIN_REASON_CODES,
    CLOSING_CAUSE_CODES,
    EXPIRE_REASON_CODES,
    PREFILTER_REASON_CODES,
    RENDER_FAIL_REASON_CODES,
    SCHEDULING_REASON_CODES,
    SUPPRESS_REASON_CODES,
    TERMINAL_REASON_CODES,
    WITHDRAW_REASON_CODES,
    ReasonCode,
    SessionBrief,
    TelemetryLevel,
    TelemetryRecord,
)
from coach.storage import SQLiteStore

_TERMINAL_STATE_REASON_BUCKETS: dict[str, frozenset[str]] = {
    "dropped": frozenset(c.value for c in PREFILTER_REASON_CODES),
    "retired": frozenset(c.value for c in SCHEDULING_REASON_CODES),
    "no_opportunity": frozenset({ReasonCode.BELOW_T_LOW.value}),
    "abstained": frozenset(c.value for c in ABSTAIN_REASON_CODES),
    "suppressed": frozenset(c.value for c in SUPPRESS_REASON_CODES),
    "unrendered": frozenset(c.value for c in RENDER_FAIL_REASON_CODES),
    "withdrawn": frozenset(c.value for c in WITHDRAW_REASON_CODES),
    "expired": frozenset(c.value for c in EXPIRE_REASON_CODES),
    "used": frozenset(c.value for c in CLOSING_CAUSE_CODES),
    "dismissed": frozenset(c.value for c in CLOSING_CAUSE_CODES),
    "resolved": frozenset(c.value for c in CLOSING_CAUSE_CODES),
    "outdated": frozenset(c.value for c in CLOSING_CAUSE_CODES),
}

_VALID_CANDIDATE_TERMINAL_STATES = frozenset(
    {
        "dropped",
        "retired",
        "no_opportunity",
        "abstained",
        "suppressed",
        "unrendered",
        "used",
        "dismissed",
        "resolved",
        "withdrawn",
        "expired",
        "outdated",
    }
)
_CLOSING_TRANSITION_STATES = frozenset(
    {"used", "dismissed", "resolved", "withdrawn", "expired", "outdated"}
)
_VALID_OUTCOMES = frozenset({"NONE", "ABSTAIN", "SUPPRESS", "ROUTE"})


@dataclass(frozen=True, slots=True)
class InvariantViolation:
    """Single §13.2 conservation invariant violation."""

    invariant_id: int
    name: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"invariant_id": self.invariant_id, "name": self.name, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class InvariantReport:
    """Result of checking all six §13.2 conservation invariants."""

    session_id: str
    run_id: str
    passed: bool
    checks: dict[str, bool]
    violations: tuple[InvariantViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "passed": self.passed,
            "checks": dict(self.checks),
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass(frozen=True, slots=True)
class SilenceAnchorEntry:
    """Single anchor explanation inside a silence query window (§13.3)."""

    candidate_id: str
    anchor_received_ms: int
    anchor_refs: list[dict[str, Any]]
    outcome: str | None
    route: str | None
    category: str | None
    raw_scores: dict[str, Any] | None
    calibrated_scores: dict[str, Any] | None
    reason_codes: list[str]
    abstain_reasons: list[str]
    terminal_state: str
    terminal_reason: str | None
    suggestion_id: str | None = None
    suggestion_transitions: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "anchor_received_ms": self.anchor_received_ms,
            "anchor_refs": self.anchor_refs,
            "outcome": self.outcome,
            "route": self.route,
            "category": self.category,
            "raw_scores": self.raw_scores,
            "calibrated_scores": self.calibrated_scores,
            "reason_codes": self.reason_codes,
            "abstain_reasons": self.abstain_reasons,
            "terminal_state": self.terminal_state,
            "terminal_reason": self.terminal_reason,
            "suggestion_id": self.suggestion_id,
            "suggestion_transitions": self.suggestion_transitions,
        }


@dataclass(frozen=True, slots=True)
class SilenceExplanation:
    """Result of §13.3 explain_silence query."""

    session_id: str
    run_id: str
    at_ms: int
    window_ms: int
    anchors: tuple[SilenceAnchorEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "at_ms": self.at_ms,
            "window_ms": self.window_ms,
            "anchors": [a.to_dict() for a in self.anchors],
        }


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """Single gained, lost, or changed item in a §13.4 replay diff."""

    candidate_id: str
    change_kind: str
    category_a: str | None
    category_b: str | None
    route_a: str | None
    route_b: str | None
    text_a: str | None
    text_b: str | None
    reason_a: str | None
    reason_b: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "change_kind": self.change_kind,
            "category_a": self.category_a,
            "category_b": self.category_b,
            "route_a": self.route_a,
            "route_b": self.route_b,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "reason_a": self.reason_a,
            "reason_b": self.reason_b,
        }


@dataclass(frozen=True, slots=True)
class RunDiff:
    """Comparison of two runs of the same session (§13.4)."""

    session_id: str
    run_a: str
    run_b: str
    gained: tuple[DiffEntry, ...]
    lost: tuple[DiffEntry, ...]
    changed: tuple[DiffEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_a": self.run_a,
            "run_b": self.run_b,
            "gained": [e.to_dict() for e in self.gained],
            "lost": [e.to_dict() for e in self.lost],
            "changed": [e.to_dict() for e in self.changed],
        }


@dataclass(frozen=True, slots=True)
class SessionReport:
    """Automatic per-session report (§13.5)."""

    session_id: str
    run_id: str
    duration_ms: int
    mode: str
    runtime_mode: str
    consent_basis: str
    consent_note: str | None
    health_timeline: list[dict[str, Any]]
    effective_max_cards_per_10_min: int
    displayed_card_count: int
    cards_per_10_min: float
    budget_compliant: bool
    outcomes_by_terminal_state: dict[str, int]
    outcomes_by_reason_code: dict[str, int]
    latency_by_route: dict[str, dict[str, dict[str, float | None]]]
    upstream_delay_ms: dict[str, float | None]
    generator_usage: dict[str, Any]
    reducer_counters: dict[str, int]
    recorder_drops: dict[str, int]
    invariants: InvariantReport
    displayed_cards: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "duration_ms": self.duration_ms,
            "mode": self.mode,
            "runtime_mode": self.runtime_mode,
            "consent_basis": self.consent_basis,
            "consent_note": self.consent_note,
            "health_timeline": self.health_timeline,
            "effective_max_cards_per_10_min": self.effective_max_cards_per_10_min,
            "displayed_card_count": self.displayed_card_count,
            "cards_per_10_min": self.cards_per_10_min,
            "budget_compliant": self.budget_compliant,
            "outcomes_by_terminal_state": self.outcomes_by_terminal_state,
            "outcomes_by_reason_code": self.outcomes_by_reason_code,
            "latency_by_route": self.latency_by_route,
            "upstream_delay_ms": self.upstream_delay_ms,
            "generator_usage": self.generator_usage,
            "reducer_counters": self.reducer_counters,
            "recorder_drops": self.recorder_drops,
            "invariants": self.invariants.to_dict(),
            "displayed_cards": self.displayed_cards,
        }


class BoundedRecorderBuffer:
    """Non-blocking bounded recorder buffer dropping debug before lifecycle/decision (§4.2)."""

    def __init__(self, maxsize: int = 1024) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._seq = 0
        self._critical: deque[tuple[int, TelemetryRecord]] = deque()
        self._debug: deque[tuple[int, TelemetryRecord]] = deque()
        self._drops: dict[TelemetryLevel, int] = {
            TelemetryLevel.DEBUG: 0,
            TelemetryLevel.LIFECYCLE: 0,
            TelemetryLevel.DECISION: 0,
        }
        self._unflushed_drops: dict[TelemetryLevel, int] = {
            TelemetryLevel.DEBUG: 0,
            TelemetryLevel.LIFECYCLE: 0,
            TelemetryLevel.DECISION: 0,
        }

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def __len__(self) -> int:
        with self._lock:
            return len(self._critical) + len(self._debug)

    def drops_by_level(self) -> dict[TelemetryLevel, int]:
        with self._lock:
            return dict(self._drops)

    def emit(self, record: TelemetryRecord) -> bool:
        """Enqueue a telemetry record without blocking; drops debug first if full."""
        with self._lock:
            total = len(self._critical) + len(self._debug)
            if record.level is TelemetryLevel.DEBUG:
                if total >= self._maxsize:
                    self._drops[TelemetryLevel.DEBUG] += 1
                    self._unflushed_drops[TelemetryLevel.DEBUG] += 1
                    return False
                self._seq += 1
                self._debug.append((self._seq, record))
                return True

            if total >= self._maxsize and self._debug:
                self._debug.popleft()
                self._drops[TelemetryLevel.DEBUG] += 1
                self._unflushed_drops[TelemetryLevel.DEBUG] += 1
            self._seq += 1
            self._critical.append((self._seq, record))
            return True

    def drain(self, max_items: int | None = None) -> list[TelemetryRecord]:
        """Drain buffered records, prioritizing lifecycle/decision over debug if bounded."""
        with self._lock:
            taken: list[tuple[int, TelemetryRecord]] = []
            while self._critical and (max_items is None or len(taken) < max_items):
                taken.append(self._critical.popleft())
            while self._debug and (max_items is None or len(taken) < max_items):
                taken.append(self._debug.popleft())
            taken.sort(key=lambda item: item[0])
            return [rec for _, rec in taken]

    def flush_to_store(self, store: SQLiteStore, session_id: str, run_id: str) -> int:
        """Persist buffered telemetry records and drop counters into SQLiteStore."""
        records = self.drain()
        with self._lock:
            drops = dict(self._unflushed_drops)
            for lvl in self._unflushed_drops:
                self._unflushed_drops[lvl] = 0
        if records:
            matching = [r for r in records if r.session_id == session_id]
            if matching:
                store.record_telemetry(run_id=run_id, records=matching)
        non_zero_drops = {k: v for k, v in drops.items() if v > 0}
        if non_zero_drops:
            store.record_drops(session_id=session_id, run_id=run_id, drops_by_level=non_zero_drops)
        return len(records)


def _safe_json_loads(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    return json.loads(raw)


def _check_ref_list(
    refs: Any,
    valid_revisions: set[tuple[str, int]],
    source_desc: str,
    violations: list[InvariantViolation],
) -> None:
    if not isinstance(refs, list):
        return
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        seg_id = ref.get("segment_id")
        rev = ref.get("revision")
        if (
            isinstance(seg_id, str)
            and isinstance(rev, int)
            and (seg_id, rev) not in valid_revisions
        ):
            violations.append(
                InvariantViolation(
                    invariant_id=6,
                    name="inv_6_valid_segment_revisions",
                    detail=f"{source_desc} references non-existent ({seg_id!r}, rev={rev})",
                )
            )


def check_conservation_invariants(
    store: SQLiteStore, session_id: str, run_id: str
) -> InvariantReport:
    """Verify all six §13.2 conservation invariants for a session run."""
    violations: list[InvariantViolation] = []
    session = store.get_session(session_id, run_id)
    events = store.get_events(session_id, run_id)
    decisions = store.get_decisions(session_id, run_id)
    suggestions = store.get_suggestions(session_id, run_id)
    feedback_rows = store.get_feedback(session_id, run_id)
    egress_records = store.get_egress_records(session_id, run_id)
    telemetry_rows = store.get_telemetry_records(session_id, run_id)

    recipient_ids = {"ME"}
    if session is not None:
        if session.get("recipient_id"):
            recipient_ids.add(str(session["recipient_id"]))
        if session.get("brief_json"):
            brief = SessionBrief.from_dict(json.loads(session["brief_json"]))
            recipient_ids.add(brief.recipient.participant_id)

    base_segments: dict[str, tuple[tuple[int, int], str]] = {}
    speaker_updates: dict[str, tuple[tuple[int, int, str], str]] = {}
    valid_revisions: set[tuple[str, int]] = set()
    latest_committed_rev: dict[str, int] = {}
    if session is not None:
        for pair in _safe_json_loads(session.get("revision_history_json"), []):
            if isinstance(pair, list) and len(pair) == 2:
                seg_key_pair, rev_val = str(pair[0]), int(pair[1])
                valid_revisions.add((seg_key_pair, rev_val))
                if rev_val > latest_committed_rev.get(seg_key_pair, 0):
                    latest_committed_rev[seg_key_pair] = rev_val

    for ev in events:
        seg_id = ev.get("segment_id")
        can_rev = ev.get("canonical_revision")
        if isinstance(seg_id, str) and isinstance(can_rev, int):
            valid_revisions.add((seg_id, can_rev))
            if can_rev > latest_committed_rev.get(seg_id, 0):
                latest_committed_rev[seg_id] = can_rev
        if ev["kind"] == "segment" and ev["stability"] == "committed" and isinstance(seg_id, str):
            if ev.get("speaker_id"):
                seg_key = (
                    int(ev["source_revision"]) if ev.get("source_revision") is not None else 0,
                    int(ev.get("source_sequence") or 0),
                )
                prev_seg = base_segments.get(seg_id)
                if prev_seg is None or seg_key >= prev_seg[0]:
                    base_segments[seg_id] = (seg_key, str(ev["speaker_id"]))
        elif ev["kind"] == "speaker_update" and isinstance(seg_id, str) and ev.get("speaker_id"):
            spk_key = (
                int(ev["source_revision"]) if ev.get("source_revision") is not None else 0,
                int(ev.get("source_sequence") or 0),
                str(ev.get("event_id") or ""),
            )
            prev_spk = speaker_updates.get(seg_id)
            if prev_spk is None or spk_key > prev_spk[0]:
                speaker_updates[seg_id] = (spk_key, str(ev["speaker_id"]))

    committed_speakers: dict[str, str] = {
        seg_id: speaker_updates[seg_id][1] if seg_id in speaker_updates else spk
        for seg_id, (_, spk) in base_segments.items()
    }

    # Invariant 1: Every committed non-recipient segment (at its latest canonical revision)
    # belongs to exactly one proactive anchor (and <= 1 per historical revision),
    # and every anchor is exactly one of scored, dropped, retired.
    seg_rev_anchor_counts: Counter[tuple[str, int]] = Counter()
    prefilter_strs = {c.value for c in PREFILTER_REASON_CODES}
    sched_strs = {c.value for c in SCHEDULING_REASON_CODES}
    for dec in decisions:
        anchor_refs = _safe_json_loads(dec["anchor_refs_json"], [])
        if dec.get("origin", "proactive") != "manual":
            for ref in anchor_refs:
                if isinstance(ref, dict) and isinstance(ref.get("segment_id"), str):
                    rev = ref.get("revision")
                    rev_int = int(rev) if isinstance(rev, int) else 1
                    seg_rev_anchor_counts[(ref["segment_id"], rev_int)] += 1
        is_dropped = dec["terminal_state"] == "dropped"
        is_retired = dec["terminal_state"] == "retired"
        is_scored = dec["raw_scores_json"] is not None
        bucket_sum = int(is_dropped) + int(is_retired) + int(is_scored)
        if bucket_sum != 1:
            violations.append(
                InvariantViolation(
                    invariant_id=1,
                    name="inv_1_segment_anchor_partition",
                    detail=f"Candidate {dec['candidate_id']!r} has invalid partition state "
                    f"(dropped={is_dropped}, retired={is_retired}, scored={is_scored})",
                )
            )
        elif is_dropped and dec["terminal_reason"] not in prefilter_strs:
            violations.append(
                InvariantViolation(
                    invariant_id=1,
                    name="inv_1_segment_anchor_partition",
                    detail=f"Dropped candidate {dec['candidate_id']!r} has non-prefilter reason {dec['terminal_reason']!r}",
                )
            )
        elif is_retired and dec["terminal_reason"] not in sched_strs:
            violations.append(
                InvariantViolation(
                    invariant_id=1,
                    name="inv_1_segment_anchor_partition",
                    detail=f"Retired candidate {dec['candidate_id']!r} has non-scheduling reason {dec['terminal_reason']!r}",
                )
            )

    for (seg_id, rev), cnt in sorted(seg_rev_anchor_counts.items()):
        if cnt > 1:
            violations.append(
                InvariantViolation(
                    invariant_id=1,
                    name="inv_1_segment_anchor_partition",
                    detail=f"Segment {seg_id!r} revision {rev} belongs to {cnt} proactive anchors (expected <= 1)",
                )
            )

    for seg_id, speaker_id in committed_speakers.items():
        if speaker_id in recipient_ids:
            continue
        target_rev = latest_committed_rev.get(seg_id, 1)
        cnt = seg_rev_anchor_counts[(seg_id, target_rev)]
        if cnt != 1:
            violations.append(
                InvariantViolation(
                    invariant_id=1,
                    name="inv_1_segment_anchor_partition",
                    detail=f"Committed non-recipient segment {seg_id!r} (rev={target_rev}) belongs to {cnt} anchors (expected 1)",
                )
            )

    # Invariant 2: Every scored candidate has exactly one routing outcome.
    # Invariant 3: Every candidate reaches exactly one terminal state with one Appendix C reason code.
    term_reason_strs = {c.value for c in TERMINAL_REASON_CODES}
    for dec in decisions:
        if dec["raw_scores_json"] is not None and dec["outcome"] not in _VALID_OUTCOMES:
            violations.append(
                InvariantViolation(
                    invariant_id=2,
                    name="inv_2_scored_has_routing",
                    detail=f"Scored candidate {dec['candidate_id']!r} missing/invalid routing outcome {dec['outcome']!r}",
                )
            )
        t_state = dec["terminal_state"]
        t_reason = dec["terminal_reason"]
        if t_state not in _VALID_CANDIDATE_TERMINAL_STATES:
            violations.append(
                InvariantViolation(
                    invariant_id=3,
                    name="inv_3_candidate_terminal_reason",
                    detail=f"Candidate {dec['candidate_id']!r} has invalid terminal_state {t_state!r}",
                )
            )
        if t_reason is None or t_reason not in term_reason_strs:
            violations.append(
                InvariantViolation(
                    invariant_id=3,
                    name="inv_3_candidate_terminal_reason",
                    detail=f"Candidate {dec['candidate_id']!r} has invalid/missing terminal_reason {t_reason!r}",
                )
            )
        elif (
            t_state in _TERMINAL_STATE_REASON_BUCKETS
            and t_reason not in _TERMINAL_STATE_REASON_BUCKETS[t_state]
        ):
            violations.append(
                InvariantViolation(
                    invariant_id=3,
                    name="inv_3_candidate_terminal_reason",
                    detail=f"Candidate {dec['candidate_id']!r} has mismatched terminal_state {t_state!r} "
                    f"and terminal_reason {t_reason!r}",
                )
            )

    # Invariant 4: Every displayed card has exactly one closing transition (allowing used -> resolved).
    for sug in suggestions:
        transitions = _safe_json_loads(sug["transitions_json"], [])
        states = [t.get("transition") for t in transitions if isinstance(t, dict)]
        if "displayed" in states:
            disp_idx = states.index("displayed")
            closing = [st for st in states[disp_idx + 1 :] if st in _CLOSING_TRANSITION_STATES]
            valid_closing = len(closing) == 1 or closing == ["used", "resolved"]
            if not valid_closing or sug["current_state"] not in _CLOSING_TRANSITION_STATES:
                violations.append(
                    InvariantViolation(
                        invariant_id=4,
                        name="inv_4_displayed_has_closing",
                        detail=f"Displayed suggestion {sug['suggestion_id']!r} has invalid closing transitions: "
                        f"closing={closing}, current_state={sug['current_state']!r}",
                    )
                )
        elif sug["current_state"] in {"displayed", "pinned"}:
            violations.append(
                InvariantViolation(
                    invariant_id=4,
                    name="inv_4_displayed_has_closing",
                    detail=f"Displayed suggestion {sug['suggestion_id']!r} has current_state={sug['current_state']!r} "
                    "with no closing transition",
                )
            )

    # Invariant 5: Every generator call has a usage record, including discarded/timed-out calls.
    egress_by_cand: dict[str, list[Any]] = {}
    egress_ids: set[str] = set()
    for eg in egress_records:
        egress_ids.add(eg.egress_id)
        if eg.candidate_id is not None:
            egress_by_candidate_list = egress_by_cand.setdefault(eg.candidate_id, [])
            egress_by_candidate_list.append(eg)
        if not eg.usage:
            violations.append(
                InvariantViolation(
                    invariant_id=5,
                    name="inv_5_generator_has_usage",
                    detail=f"Egress record {eg.egress_id!r} missing usage metrics",
                )
            )
    for dec in decisions:
        if (
            dec["outcome"] == "ROUTE"
            and dec["route"] == "generator"
            and dec["terminal_reason"] != ReasonCode.INSUFFICIENT_BUDGET.value
            and dec["candidate_id"] not in egress_by_cand
        ):
            violations.append(
                InvariantViolation(
                    invariant_id=5,
                    name="inv_5_generator_has_usage",
                    detail=f"Generator candidate {dec['candidate_id']!r} lacks egress/usage record",
                )
            )
    for tr in telemetry_rows:
        if tr.kind in {"generator_call", "generator_invocation"}:
            eg_id = tr.payload.get("egress_id")
            if isinstance(eg_id, str):
                if eg_id not in egress_ids:
                    violations.append(
                        InvariantViolation(
                            invariant_id=5,
                            name="inv_5_generator_has_usage",
                            detail=f"Generator telemetry {tr.record_id!r} references missing egress_id {eg_id!r}",
                        )
                    )
            elif tr.candidate_id is None or tr.candidate_id not in egress_by_cand:
                violations.append(
                    InvariantViolation(
                        invariant_id=5,
                        name="inv_5_generator_has_usage",
                        detail=f"Generator telemetry {tr.record_id!r} has no matching egress/usage record",
                    )
                )

    # Invariant 6: No record references a segment revision that never existed.
    for dec in decisions:
        cid = dec["candidate_id"]
        _check_ref_list(
            _safe_json_loads(dec["anchor_refs_json"], []),
            valid_revisions,
            f"decision[{cid}].anchor_refs",
            violations,
        )
        _check_ref_list(
            _safe_json_loads(dec["context_refs_json"], []),
            valid_revisions,
            f"decision[{cid}].context_refs",
            violations,
        )
        _check_ref_list(
            _safe_json_loads(dec["input_refs_json"], []),
            valid_revisions,
            f"decision[{cid}].input_refs",
            violations,
        )
    for sug in suggestions:
        sid = sug["suggestion_id"]
        _check_ref_list(
            _safe_json_loads(sug["evidence_refs_json"], []),
            valid_revisions,
            f"suggestion[{sid}].evidence_refs",
            violations,
        )
        slots = _safe_json_loads(sug["grounded_slots_json"], {})
        if isinstance(slots, dict):
            for slot_name, slot in slots.items():
                if isinstance(slot, dict):
                    _check_ref_list(
                        [slot], valid_revisions, f"suggestion[{sid}].slot[{slot_name}]", violations
                    )
        for tr_ev in _safe_json_loads(sug["transitions_json"], []):
            if isinstance(tr_ev, dict):
                _check_ref_list(
                    tr_ev.get("evidence_refs", []),
                    valid_revisions,
                    f"suggestion[{sid}].transition",
                    violations,
                )
    for fb in feedback_rows:
        lbl = _safe_json_loads(fb["label_json"], None)
        if isinstance(lbl, dict):
            _check_ref_list(
                lbl.get("evidence_refs", []),
                valid_revisions,
                f"feedback[{fb.get('label_id')}].evidence_refs",
                violations,
            )

    violated_ids = {v.invariant_id for v in violations}
    checks = {
        "inv_1_segment_anchor_partition": 1 not in violated_ids,
        "inv_2_scored_has_routing": 2 not in violated_ids,
        "inv_3_candidate_terminal_reason": 3 not in violated_ids,
        "inv_4_displayed_has_closing": 4 not in violated_ids,
        "inv_5_generator_has_usage": 5 not in violated_ids,
        "inv_6_valid_segment_revisions": 6 not in violated_ids,
    }
    return InvariantReport(
        session_id=session_id,
        run_id=run_id,
        passed=len(violations) == 0,
        checks=checks,
        violations=tuple(violations),
    )


def explain_silence(
    store: SQLiteStore,
    session_id: str,
    run_id: str,
    at_ms: int,
    window_ms: int = 60_000,
) -> SilenceExplanation:
    """Query anchors formed in [at_ms - window_ms, at_ms] with scores, routing, and terminal reason (§13.3)."""
    start_ms = at_ms - window_ms
    decisions = store.get_decisions(session_id, run_id)
    suggestions = store.get_suggestions(session_id, run_id)
    sug_by_cand = {s["candidate_id"]: s for s in suggestions}

    entries: list[SilenceAnchorEntry] = []
    for dec in decisions:
        rec_ms = int(dec["anchor_received_ms"])
        if start_ms <= rec_ms <= at_ms:
            sug = sug_by_cand.get(dec["candidate_id"])
            entries.append(
                SilenceAnchorEntry(
                    candidate_id=dec["candidate_id"],
                    anchor_received_ms=rec_ms,
                    anchor_refs=_safe_json_loads(dec["anchor_refs_json"], []),
                    outcome=dec["outcome"],
                    route=dec["route"],
                    category=dec["category"],
                    raw_scores=_safe_json_loads(dec["raw_scores_json"], None),
                    calibrated_scores=_safe_json_loads(dec["calibrated_scores_json"], None),
                    reason_codes=_safe_json_loads(dec["reason_codes_json"], []),
                    abstain_reasons=_safe_json_loads(dec["abstain_reasons_json"], []),
                    terminal_state=dec["terminal_state"],
                    terminal_reason=dec["terminal_reason"],
                    suggestion_id=sug["suggestion_id"] if sug is not None else None,
                    suggestion_transitions=_safe_json_loads(sug["transitions_json"], None)
                    if sug is not None
                    else None,
                )
            )
    return SilenceExplanation(
        session_id=session_id,
        run_id=run_id,
        at_ms=at_ms,
        window_ms=window_ms,
        anchors=tuple(entries),
    )


def _was_displayed(sug: dict[str, Any] | None) -> bool:
    if sug is None:
        return False
    transitions = _safe_json_loads(sug["transitions_json"], [])
    for tr in transitions:
        if isinstance(tr, dict) and tr.get("transition") in {"displayed", "shadow_displayed"}:
            return True
    return sug["current_state"] in {
        "displayed",
        "used",
        "dismissed",
        "pinned",
        "resolved",
        "outdated",
    }


def _resolve_session_for_runs(
    store: SQLiteStore, run_a: str, run_b: str, session_id: str | None
) -> str:
    if session_id is not None:
        return session_id
    runs = store.list_runs()
    sess_a = {r["session_id"] for r in runs if r["run_id"] == run_a}
    sess_b = {r["session_id"] for r in runs if r["run_id"] == run_b}
    common = sess_a & sess_b
    if not common:
        raise ValueError(
            f"Could not find common session_id for run_a={run_a!r} and run_b={run_b!r}"
        )
    if len(common) > 1:
        raise ValueError(
            f"Multiple sessions share run_a={run_a!r} and run_b={run_b!r}; pass session_id explicitly"
        )
    return str(next(iter(common)))


def diff_runs(store: SQLiteStore, run_a: str, run_b: str, session_id: str | None = None) -> RunDiff:
    """Compare two runs of the same session: gained, lost, and changed cards (§13.4)."""
    sid = _resolve_session_for_runs(store, run_a, run_b, session_id)
    dec_a = {d["candidate_id"]: d for d in store.get_decisions(sid, run_a)}
    dec_b = {d["candidate_id"]: d for d in store.get_decisions(sid, run_b)}
    sug_a = {s["candidate_id"]: s for s in store.get_suggestions(sid, run_a)}
    sug_b = {s["candidate_id"]: s for s in store.get_suggestions(sid, run_b)}

    all_cands = sorted(set(dec_a) | set(dec_b))
    gained: list[DiffEntry] = []
    lost: list[DiffEntry] = []
    changed: list[DiffEntry] = []

    for cid in all_cands:
        da = dec_a.get(cid)
        db = dec_b.get(cid)
        sa = sug_a.get(cid)
        sb = sug_b.get(cid)
        disp_a = _was_displayed(sa)
        disp_b = _was_displayed(sb)
        cat_a = sa["category"] if sa else (da["category"] if da else None)
        cat_b = sb["category"] if sb else (db["category"] if db else None)
        route_a = sa["route"] if sa else (da["route"] if da else None)
        route_b = sb["route"] if sb else (db["route"] if db else None)
        text_a = sa["text"] if sa else None
        text_b = sb["text"] if sb else None
        reason_a = (sa["terminal_reason"] if sa and sa["terminal_reason"] else None) or (
            da["terminal_reason"] if da else "not_present"
        )
        reason_b = (sb["terminal_reason"] if sb and sb["terminal_reason"] else None) or (
            db["terminal_reason"] if db else "not_present"
        )

        if disp_b and not disp_a:
            gained.append(
                DiffEntry(
                    candidate_id=cid,
                    change_kind="gained",
                    category_a=cat_a,
                    category_b=cat_b,
                    route_a=route_a,
                    route_b=route_b,
                    text_a=text_a,
                    text_b=text_b,
                    reason_a=reason_a,
                    reason_b=reason_b,
                )
            )
        elif disp_a and not disp_b:
            lost.append(
                DiffEntry(
                    candidate_id=cid,
                    change_kind="lost",
                    category_a=cat_a,
                    category_b=cat_b,
                    route_a=route_a,
                    route_b=route_b,
                    text_a=text_a,
                    text_b=text_b,
                    reason_a=reason_a,
                    reason_b=reason_b,
                )
            )
        elif (
            cat_a != cat_b
            or route_a != route_b
            or text_a != text_b
            or reason_a != reason_b
            or (da["outcome"] if da else None) != (db["outcome"] if db else None)
        ):
            changed.append(
                DiffEntry(
                    candidate_id=cid,
                    change_kind="changed",
                    category_a=cat_a,
                    category_b=cat_b,
                    route_a=route_a,
                    route_b=route_b,
                    text_a=text_a,
                    text_b=text_b,
                    reason_a=reason_a,
                    reason_b=reason_b,
                )
            )
    return RunDiff(
        session_id=sid,
        run_a=run_a,
        run_b=run_b,
        gained=tuple(gained),
        lost=tuple(lost),
        changed=tuple(changed),
    )


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _pct_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
    }


def build_session_report(store: SQLiteStore, session_id: str, run_id: str) -> SessionReport:
    """Generate the per-session report (§13.5)."""
    session = store.get_session(session_id, run_id)
    if session is None:
        raise KeyError(f"Session ({session_id!r}, {run_id!r}) not found")
    events = store.get_events(session_id, run_id)
    decisions = store.get_decisions(session_id, run_id)
    suggestions = store.get_suggestions(session_id, run_id)
    feedback_rows = store.get_feedback(session_id, run_id)
    egress_records = store.get_egress_records(session_id, run_id)
    drops = store.get_recorder_drops(session_id, run_id)
    invariants = check_conservation_invariants(store, session_id, run_id)

    started_ms = session.get("started_ms")
    ended_ms = session.get("ended_ms")
    if (started_ms is None or ended_ms is None) and events:
        times = [int(ev["received_ms"]) for ev in events]
        started_ms = min(times) if started_ms is None else started_ms
        ended_ms = max(times) if ended_ms is None else ended_ms
    duration_ms = (
        max(0, int(ended_ms) - int(started_ms))
        if started_ms is not None and ended_ms is not None
        else 0
    )

    state_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    dec_by_cand: dict[str, dict[str, Any]] = {}
    for dec in decisions:
        dec_by_cand[dec["candidate_id"]] = dec
        state_counts[str(dec["terminal_state"])] += 1
        if dec["terminal_reason"] is not None:
            reason_counts[str(dec["terminal_reason"])] += 1

    fb_by_sug: dict[str, list[str]] = {}
    for fb in feedback_rows:
        if fb.get("suggestion_id") and fb.get("feedback_reason"):
            fb_by_sug.setdefault(str(fb["suggestion_id"]), []).append(str(fb["feedback_reason"]))

    ready_lat: dict[str, list[float]] = {"template": [], "generator": []}
    display_lat: dict[str, list[float]] = {"template": [], "generator": []}
    displayed_cards: list[dict[str, Any]] = []
    for sug in suggestions:
        route = str(sug["route"])
        matched_dec = dec_by_cand.get(sug["candidate_id"])
        anchor_rx = (
            int(matched_dec["anchor_received_ms"])
            if matched_dec is not None
            else int(sug["ready_ms"])
        )
        if route in ready_lat:
            ready_lat[route].append(float(int(sug["ready_ms"]) - anchor_rx))
        transitions = _safe_json_loads(sug["transitions_json"], [])
        disp_ts: int | None = None
        for tr in transitions:
            if isinstance(tr, dict) and tr.get("transition") in {"displayed", "shadow_displayed"}:
                disp_ts = int(tr["at_ms"])
                break
        if disp_ts is not None:
            if route in display_lat:
                display_lat[route].append(float(disp_ts - anchor_rx))
            displayed_cards.append(
                {
                    "suggestion_id": sug["suggestion_id"],
                    "candidate_id": sug["candidate_id"],
                    "category": sug["category"],
                    "route": route,
                    "text": sug["text"],
                    "displayed_at_ms": disp_ts,
                    "current_state": sug["current_state"],
                    "terminal_reason": sug["terminal_reason"],
                    "feedback": fb_by_sug.get(sug["suggestion_id"], []),
                }
            )

    upstream_delays = [
        float(int(ev["received_ms"]) - int(ev["end_ms"]))
        for ev in events
        if ev["kind"] == "segment"
        and ev["stability"] == "committed"
        and ev.get("end_ms") is not None
    ]

    total_calls = len(egress_records)
    mins = duration_ms / 60_000.0 if duration_ms > 0 else 0.0
    gen_decisions = [d for d in decisions if d.get("route") == "generator"]
    timeouts = sum(1 for d in gen_decisions if d.get("terminal_reason") == "generator_timeout")
    declines = sum(1 for d in gen_decisions if d.get("terminal_reason") == "generator_declined")
    rejects = sum(
        1 for d in gen_decisions if d.get("terminal_reason") == "generator_invalid_output"
    )
    tok_totals: Counter[str] = Counter()
    for eg in egress_records:
        for k, v in eg.usage.items():
            tok_totals[k] += int(v)
    in_toks = tok_totals.get("input_tokens", 0)
    cached_toks = tok_totals.get("cached_input_tokens", 0)

    eff_cap = int(session["effective_max_cards_per_10_min"])
    displayed_count = len(displayed_cards)
    cards_per_10_min = (
        (displayed_count / (duration_ms / 600_000.0)) if duration_ms > 0 else float(displayed_count)
    )
    proactive_displayed = sum(
        1
        for card in displayed_cards
        if dec_by_cand.get(card["candidate_id"], {}).get("origin", "proactive") == "proactive"
    )
    proactive_rate = (
        (proactive_displayed / (duration_ms / 600_000.0))
        if duration_ms > 0
        else float(proactive_displayed)
    )
    budget_compliant = proactive_rate <= float(eff_cap) + 1e-9

    return SessionReport(
        session_id=session_id,
        run_id=run_id,
        duration_ms=duration_ms,
        mode=str(session["mode"]),
        runtime_mode=str(session["runtime_mode"]),
        consent_basis=str(session["consent_basis"]),
        consent_note=session.get("consent_note"),
        health_timeline=_safe_json_loads(session.get("health_timeline_json"), []),
        effective_max_cards_per_10_min=eff_cap,
        displayed_card_count=displayed_count,
        cards_per_10_min=cards_per_10_min,
        budget_compliant=budget_compliant,
        outcomes_by_terminal_state=dict(state_counts),
        outcomes_by_reason_code=dict(reason_counts),
        latency_by_route={
            route: {
                "time_to_ready_ms": _pct_summary(ready_lat[route]),
                "time_to_display_ms": _pct_summary(display_lat[route]),
            }
            for route in ("template", "generator")
        },
        upstream_delay_ms=_pct_summary(upstream_delays),
        generator_usage={
            "total_calls": total_calls,
            "calls_per_min": (total_calls / mins) if mins > 0 else 0.0,
            "timeout_rate": (timeouts / total_calls) if total_calls > 0 else 0.0,
            "decline_rate": (declines / total_calls) if total_calls > 0 else 0.0,
            "validation_reject_rate": (rejects / total_calls) if total_calls > 0 else 0.0,
            "tokens": dict(tok_totals),
            "cache_read_share": (cached_toks / in_toks) if in_toks > 0 else 0.0,
        },
        reducer_counters=_safe_json_loads(session.get("reducer_counters_json"), {}),
        recorder_drops=drops,
        invariants=invariants,
        displayed_cards=displayed_cards,
    )
