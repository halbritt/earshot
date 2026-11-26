"""2-pass locked annotation tool, review queue, and blind Cohen's kappa (§12.2–§12.4, §15 level 13)."""

from __future__ import annotations

import hmac
import json
import random
import secrets
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

from coach.canonical_state import CanonicalState, SegmentRecord, StateSnapshot
from coach.context_builder import PolicyConfig, build_decision_contexts
from coach.contracts import (
    PREDICTABLE_CATEGORIES,
    SCHEMA_VERSION,
    AnnotationPass,
    CanonicalEvent,
    Category,
    ClosedBy,
    ContractValidationError,
    EventKind,
    LabelRecord,
    NoneReason,
    SampleSource,
    SegmentRef,
    SessionBrief,
    Stability,
    ValidityInterval,
)
from coach.storage import SQLiteStore

_STEP1_FORBIDDEN_IN_STEP2 = frozenset(
    {
        "worthwhile",
        "categories",
        "example_questions",
        "none_reason",
        "sample_source",
        "evidence_refs",
        "confidence",
        "step1_locked",
    }
)


class AnnotationLockError(PermissionError):
    """Raised on Pass 1 scroll-ahead or Pass 2 Step-1 mutation after lock (§12.4)."""


class AnnotationValidationError(ValueError):
    """Raised on malformed annotation payloads or invalid validity intervals."""


def load_labels_jsonl(path: str | Path) -> list[LabelRecord]:
    """Load and validate LabelRecords from a JSONL file."""
    src = Path(path)
    if not src.exists():
        return []
    labels: list[LabelRecord] = []
    with src.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, Mapping):
                raise AnnotationValidationError(f"Line {line_no} in {src} must be a JSON object")
            labels.append(LabelRecord.from_dict(obj))
    return labels


def save_labels_jsonl(
    labels: Sequence[LabelRecord],
    path: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write LabelRecords to JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as fh:
        for lbl in labels:
            fh.write(json.dumps(lbl.to_dict(), sort_keys=True) + "\n")
    return target


def _event_view(ev: CanonicalEvent) -> dict[str, Any]:
    return {
        "event_id": ev.event_id,
        "kind": str(ev.kind),
        "segment_id": ev.segment_id,
        "speaker_id": ev.speaker_id,
        "stability": str(ev.stability) if ev.stability is not None else None,
        "start_ms": ev.start_ms,
        "end_ms": ev.end_ms,
        "received_ms": ev.received_ms,
        "text": ev.text,
    }


def _seg_view(seg: SegmentRecord) -> dict[str, Any]:
    return {
        "segment_id": seg.segment_id,
        "speaker_id": seg.speaker_id,
        "revision": seg.revision,
        "start_ms": seg.start_ms,
        "end_ms": seg.end_ms,
        "received_ms": seg.received_ms,
        "text": seg.text,
        "stability": str(seg.stability),
    }


class Pass1Session:
    """Online prefix-only timed replay marking session (no scroll ahead, §12.4)."""

    def __init__(
        self,
        *,
        brief: SessionBrief,
        events: Sequence[CanonicalEvent],
        annotator: str = "recipient",
        initial_ms: int = 0,
        speed: float = 1.0,
        out_path: str | Path | None = None,
        store: SQLiteStore | None = None,
        run_id: str = "annotate-pass1",
    ) -> None:
        if speed <= 0.0:
            raise AnnotationValidationError(f"speed must be > 0, got {speed}")
        if initial_ms < 0:
            raise AnnotationValidationError(f"initial_ms must be >= 0, got {initial_ms}")
        self._brief = brief
        self._events = tuple(
            sorted(events, key=lambda e: (e.received_ms, e.source_sequence, e.event_id))
        )
        self._annotator = annotator
        self._now_ms = initial_ms
        self._speed = float(speed)
        self._playing = False
        self._out_path = Path(out_path) if out_path is not None else None
        self._store = store
        self._run_id = run_id
        self._state = CanonicalState(brief)
        self._cursor = 0
        self._marks: list[LabelRecord] = []
        self._lock = threading.Lock()
        self._apply_up_to(self._now_ms)

    def _apply_up_to(self, target_ms: int) -> list[CanonicalEvent]:
        applied: list[CanonicalEvent] = []
        while (
            self._cursor < len(self._events) and self._events[self._cursor].received_ms <= target_ms
        ):
            ev = self._events[self._cursor]
            self._state.apply(ev)
            applied.append(ev)
            self._cursor += 1
        return applied

    @property
    def now_ms(self) -> int:
        return self._now_ms

    @property
    def marks(self) -> tuple[LabelRecord, ...]:
        return tuple(self._marks)

    def visible_events(self, at_ms: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if at_ms is not None and at_ms > self._now_ms:
                raise AnnotationLockError(
                    f"scroll_ahead_forbidden: requested at_ms={at_ms} > now_ms={self._now_ms}"
                )
            cutoff = self._now_ms if at_ms is None else at_ms
            return [
                _event_view(ev) for ev in self._events[: self._cursor] if ev.received_ms <= cutoff
            ]

    def seek(self, target_ms: int) -> None:
        with self._lock:
            if target_ms > self._now_ms:
                raise AnnotationLockError(
                    f"scroll_ahead_forbidden: cannot seek to {target_ms}ms past {self._now_ms}ms"
                )
            if target_ms < 0:
                raise AnnotationValidationError(f"target_ms must be >= 0, got {target_ms}")

    def step_next(self) -> CanonicalEvent | None:
        with self._lock:
            if self._cursor >= len(self._events):
                return None
            ev = self._events[self._cursor]
            if ev.received_ms > self._now_ms:
                self._now_ms = ev.received_ms
            self._state.apply(ev)
            self._cursor += 1
            return ev

    def advance_by(self, delta_ms: int) -> list[CanonicalEvent]:
        with self._lock:
            if delta_ms < 0:
                raise AnnotationValidationError(f"delta_ms must be >= 0, got {delta_ms}")
            self._now_ms += delta_ms
            return self._apply_up_to(self._now_ms)

    def prefix_at(self, at_ms: int) -> list[dict[str, Any]]:
        """Advance playhead monotonically up to at_ms (or return earlier prefix) and list events."""
        with self._lock:
            if at_ms < 0:
                raise AnnotationValidationError(f"at_ms must be >= 0, got {at_ms}")
            if at_ms > self._now_ms:
                self._now_ms = at_ms
                self._apply_up_to(self._now_ms)
            return [
                _event_view(ev) for ev in self._events[: self._cursor] if ev.received_ms <= at_ms
            ]

    def state_view(self) -> dict[str, Any]:
        with self._lock:
            snap = self._state.snapshot()
            return {
                "pass": 1,
                "session_id": self._brief.session_id,
                "now_ms": self._now_ms,
                "playing": self._playing,
                "speed": self._speed,
                "state_version": snap.state_version,
                "applied_count": self._cursor,
                "remaining_count": max(0, len(self._events) - self._cursor),
                "visible_events": [_event_view(ev) for ev in self._events[: self._cursor]],
                "segments": [_seg_view(s) for s in snap.segments],
                "marks_count": len(self._marks),
            }

    def mark_opportunity(
        self,
        *,
        example_questions: Sequence[str] = (),
        categories: Sequence[Category | str] = (),
        notes: str | None = None,
        confidence: str = "sure",
        at_ms: int | None = None,
        state_version_override: int | None = None,
    ) -> LabelRecord:
        with self._lock:
            if at_ms is not None and at_ms > self._now_ms:
                self._now_ms = at_ms
                self._apply_up_to(self._now_ms)
            snap = self._state.snapshot()
            recipient_ids = {"ME", snap.brief.recipient.participant_id}
            other_segs = [s for s in snap.segments if s.speaker_id not in recipient_ids]
            latest_seg = other_segs[-1] if other_segs else None
            cand_id: str | None = None
            ev_refs: tuple[SegmentRef, ...] = ()
            mark_ms = self._now_ms if at_ms is None else at_ms
            open_ms = mark_ms
            if latest_seg is not None:
                ev_refs = (SegmentRef(latest_seg.segment_id, latest_seg.revision),)
                open_ms = min(mark_ms, latest_seg.received_ms)
                built = build_decision_contexts(snap, anchor_segments=[latest_seg])
                if built and built[0].context is not None:
                    cand_id = built[0].context.candidate_id
                elif built and built[0].drop_record is not None:
                    cand_id = built[0].drop_record.candidate_id
            parsed_cats = tuple(Category(str(c)) for c in categories)
            lbl = LabelRecord(
                schema_version=SCHEMA_VERSION,
                label_id=f"lbl-p1-{uuid.uuid4().hex[:12]}",
                session_id=self._brief.session_id,
                annotator=self._annotator,
                sample_source=SampleSource.PASS1_MARK,
                annotation_pass=AnnotationPass.ONLINE,
                candidate_id=cand_id,
                state_version_at_decision=(
                    int(state_version_override)
                    if state_version_override is not None
                    else snap.state_version
                ),
                worthwhile=True,
                categories=parsed_cats,
                example_questions=tuple(str(q) for q in example_questions if str(q).strip()),
                evidence_refs=ev_refs,
                validity=ValidityInterval(
                    open_ms=open_ms,
                    ideal_ms=mark_ms,
                    close_ms=None,
                    closed_by=None,
                ),
                none_reason=None,
                confidence=confidence,
                notes=notes,
            )
            self._marks.append(lbl)
            if self._out_path is not None:
                save_labels_jsonl([lbl], self._out_path, append=True)
            if self._store is not None:
                self._store.record_label(run_id=self._run_id, label=lbl, created_ms=self._now_ms)
            return lbl


@dataclass(slots=True)
class Pass2Item:
    """Single Pass-2 annotation item with strict 2-step lock state (§12.4)."""

    item_id: str
    session_id: str
    candidate_id: str | None
    sample_source: SampleSource
    state_version_at_decision: int
    evidence_refs: tuple[SegmentRef, ...]
    open_ms: int
    serialized_context: str
    prefix_segments: tuple[dict[str, Any], ...]
    _continuation: tuple[dict[str, Any], ...]
    dropped: bool = False
    drop_reason: str | None = None
    suggested_question: str | None = None
    suggested_category: str | None = None
    state: Literal["step1_open", "step2_open", "completed"] = "step1_open"
    step1_locked: bool = False
    worthwhile: bool | None = None
    categories: tuple[Category, ...] = ()
    example_questions: tuple[str, ...] = ()
    none_reason: NoneReason | None = None
    confidence: str = "sure"
    close_ms: int | None = None
    closed_by: ClosedBy | None = None
    ideal_ms: int | None = None
    notes: str | None = None
    label: LabelRecord | None = None


def _build_pass2_item_at_anchor(
    *,
    item_id: str,
    snap: StateSnapshot,
    anchor_segs: Sequence[SegmentRecord],
    ordered_events: Sequence[CanonicalEvent],
    policy: PolicyConfig | None,
    sample_source: SampleSource,
    suggested_question: str | None = None,
    suggested_category: str | None = None,
) -> Pass2Item:
    built = build_decision_contexts(snap, anchor_segments=anchor_segs, policy=policy)
    anchor_rx = max(s.received_ms for s in anchor_segs)
    prefix_segs = tuple(_seg_view(s) for s in snap.segments if s.received_ms <= anchor_rx)
    continuation = tuple(
        _event_view(ev)
        for ev in ordered_events
        if ev.received_ms > anchor_rx and ev.kind is EventKind.SEGMENT and ev.text
    )
    if built and built[0].context is not None and built[0].serialized_text is not None:
        ctx = built[0].context
        return Pass2Item(
            item_id=item_id,
            session_id=snap.session_id,
            candidate_id=ctx.candidate_id,
            sample_source=sample_source,
            state_version_at_decision=ctx.based_on_state_version,
            evidence_refs=ctx.anchor_refs,
            open_ms=ctx.anchor_received_ms,
            serialized_context=built[0].serialized_text,
            prefix_segments=prefix_segs,
            _continuation=continuation,
            dropped=False,
            suggested_question=suggested_question,
            suggested_category=suggested_category,
        )
    if built and built[0].drop_record is not None:
        drop = built[0].drop_record
        return Pass2Item(
            item_id=item_id,
            session_id=snap.session_id,
            candidate_id=drop.candidate_id,
            sample_source=sample_source,
            state_version_at_decision=drop.based_on_state_version,
            evidence_refs=drop.anchor_refs,
            open_ms=drop.anchor_received_ms,
            serialized_context=f"[DROPPED:{drop.reason_code.value}] {built[0].anchor.joined_text}",
            prefix_segments=prefix_segs,
            _continuation=continuation,
            dropped=True,
            drop_reason=drop.reason_code.value,
            suggested_question=suggested_question,
            suggested_category=suggested_category,
        )
    refs = tuple(SegmentRef(s.segment_id, s.revision) for s in anchor_segs)
    return Pass2Item(
        item_id=item_id,
        session_id=snap.session_id,
        candidate_id=None,
        sample_source=sample_source,
        state_version_at_decision=snap.state_version,
        evidence_refs=refs,
        open_ms=anchor_rx,
        serialized_context="\n".join(f"{s.speaker_id}: {s.text}" for s in anchor_segs),
        prefix_segments=prefix_segs,
        _continuation=continuation,
        suggested_question=suggested_question,
        suggested_category=suggested_category,
    )


def build_pass2_items(
    brief: SessionBrief,
    events: Sequence[CanonicalEvent],
    *,
    policy: PolicyConfig | None = None,
    sample_source: SampleSource = SampleSource.RANDOM,
    limit: int | None = None,
    seed: int = 0,
) -> list[Pass2Item]:
    """Build Pass-2 items preserving strict prefix vs continuation separation (§12.4)."""
    ordered = sorted(events, key=lambda e: (e.received_ms, e.source_sequence, e.event_id))
    state = CanonicalState(brief)
    recipient_ids = {"ME", brief.recipient.participant_id}
    seen_cands: set[str] = set()
    items: list[Pass2Item] = []

    for ev in ordered:
        state.apply(ev)
        if ev.kind is not EventKind.SEGMENT or ev.stability is not Stability.COMMITTED:
            continue
        if not ev.segment_id:
            continue
        snap = state.snapshot()
        seg = snap.segments_by_id.get(ev.segment_id)
        if seg is None or seg.speaker_id in recipient_ids:
            continue
        item_id = f"item-{len(items) + 1:04d}"
        item = _build_pass2_item_at_anchor(
            item_id=item_id,
            snap=snap,
            anchor_segs=[seg],
            ordered_events=ordered,
            policy=policy,
            sample_source=sample_source,
        )
        key = item.candidate_id or f"{seg.segment_id}@{seg.revision}"
        if key in seen_cands:
            continue
        seen_cands.add(key)
        items.append(item)

    if limit is not None and limit < len(items):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(items)), limit))
        items = [items[i] for i in indices]
    return items


def load_review_queue_items(
    brief: SessionBrief,
    events: Sequence[CanonicalEvent],
    review_queue: Sequence[Mapping[str, Any]],
    *,
    policy: PolicyConfig | None = None,
    sample_source: SampleSource = SampleSource.SHADOW_REVIEW,
) -> list[Pass2Item]:
    """Convert replay/shadow review_queue entries into Pass-2 items (§12.4, §12.6)."""
    ordered = sorted(events, key=lambda e: (e.received_ms, e.source_sequence, e.event_id))
    items: list[Pass2Item] = []
    for idx, entry in enumerate(review_queue, start=1):
        seg_ids = {str(s) for s in (entry.get("segment_ids") or [])}
        if not seg_ids and entry.get("evidence_refs"):
            seg_ids = {
                str(r["segment_id"])
                for r in entry["evidence_refs"]
                if isinstance(r, Mapping) and "segment_id" in r
            }
        state = CanonicalState(brief)
        anchor_rx: int | None = None
        for ev in ordered:
            state.apply(ev)
            if (
                ev.kind is EventKind.SEGMENT
                and ev.stability is Stability.COMMITTED
                and ev.segment_id in seg_ids
            ):
                snap_now = state.snapshot()
                if seg_ids.issubset(snap_now.segments_by_id):
                    anchor_rx = ev.received_ms
                    break
        snap = state.snapshot()
        anchor_segs = [snap.segments_by_id[s] for s in sorted(seg_ids) if s in snap.segments_by_id]
        if not anchor_segs:
            continue
        if anchor_rx is None:
            anchor_rx = max(s.received_ms for s in anchor_segs)
        item_id = str(entry.get("suggestion_id") or f"rev-{idx:04d}")
        items.append(
            _build_pass2_item_at_anchor(
                item_id=item_id,
                snap=snap,
                anchor_segs=anchor_segs,
                ordered_events=ordered,
                policy=policy,
                sample_source=sample_source,
                suggested_question=(str(entry["text"]) if entry.get("text") is not None else None),
                suggested_category=(
                    str(entry["category"]) if entry.get("category") is not None else None
                ),
            )
        )
    return items


class Pass2Session:
    """Two-step locked offline annotation session (§12.4)."""

    def __init__(
        self,
        *,
        items: Sequence[Pass2Item],
        annotator: str = "recipient",
        out_path: str | Path | None = None,
        store: SQLiteStore | None = None,
        run_id: str = "annotate-pass2",
        review_queue_ids: Sequence[str] = (),
    ) -> None:
        self._items: dict[str, Pass2Item] = {it.item_id: it for it in items}
        self._by_cand_id: dict[str, Pass2Item] = {
            it.candidate_id: it for it in items if it.candidate_id is not None
        }
        self._item_order: tuple[str, ...] = tuple(it.item_id for it in items)
        self._review_queue_ids: tuple[str, ...] = tuple(review_queue_ids)
        self._annotator = annotator
        self._out_path = Path(out_path) if out_path is not None else None
        self._store = store
        self._run_id = run_id
        self._lock = threading.Lock()

    def _get(self, item_id: str) -> Pass2Item:
        if item_id in self._items:
            return self._items[item_id]
        if item_id in self._by_cand_id:
            return self._by_cand_id[item_id]
        raise KeyError(f"Unknown Pass 2 item {item_id!r}")

    def review_queue_list(self) -> list[dict[str, Any]]:
        with self._lock:
            rq_set = set(self._review_queue_ids)
            out: list[dict[str, Any]] = []
            for iid in self._item_order:
                it = self._items[iid]
                if (
                    it.item_id in rq_set
                    or (it.candidate_id and it.candidate_id in rq_set)
                    or it.sample_source is SampleSource.SHADOW_REVIEW
                ):
                    out.append(
                        {
                            "item_id": it.item_id,
                            "candidate_id": it.candidate_id,
                            "session_id": it.session_id,
                            "sample_source": str(it.sample_source),
                            "suggested_question": it.suggested_question,
                            "suggested_category": it.suggested_category,
                            "state": it.state,
                        }
                    )
            return out

    def list_items(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "item_id": self._items[iid].item_id,
                    "candidate_id": self._items[iid].candidate_id,
                    "sample_source": str(self._items[iid].sample_source),
                    "state": self._items[iid].state,
                    "step1_locked": self._items[iid].step1_locked,
                    "open_ms": self._items[iid].open_ms,
                }
                for iid in self._item_order
            ]

    def _view(self, item: Pass2Item) -> dict[str, Any]:
        return {
            "item_id": item.item_id,
            "session_id": item.session_id,
            "candidate_id": item.candidate_id,
            "sample_source": str(item.sample_source),
            "state_version_at_decision": item.state_version_at_decision,
            "evidence_refs": [r.to_dict() for r in item.evidence_refs],
            "open_ms": item.open_ms,
            "serialized_context": item.serialized_context,
            "prefix_segments": list(item.prefix_segments),
            "dropped": item.dropped,
            "drop_reason": item.drop_reason,
            "suggested_question": item.suggested_question,
            "suggested_category": item.suggested_category,
            "state": item.state,
            "worthwhile": item.worthwhile,
            "step1_locked": item.step1_locked,
            "step1": {
                "worthwhile": item.worthwhile,
                "categories": [str(c) for c in item.categories],
                "example_questions": list(item.example_questions),
                "none_reason": str(item.none_reason) if item.none_reason is not None else None,
                "confidence": item.confidence,
            },
            "step2": {
                "ideal_ms": item.ideal_ms,
                "close_ms": item.close_ms,
                "closed_by": str(item.closed_by) if item.closed_by is not None else None,
                "notes": item.notes,
            },
            "continuation_available": item.step1_locked,
            "continuation": list(item._continuation) if item.step1_locked else None,
            "continuation_events": list(item._continuation) if item.step1_locked else None,
        }

    def get_item_view(self, item_id: str) -> dict[str, Any]:
        with self._lock:
            return self._view(self._get(item_id))

    def get_continuation(self, item_id: str) -> list[dict[str, Any]]:
        with self._lock:
            item = self._get(item_id)
            if not item.step1_locked:
                raise AnnotationLockError(
                    "step1_not_locked: continuation is hidden until Step 1 is locked"
                )
            return list(item._continuation)

    def _apply_step1_fields(
        self,
        item: Pass2Item,
        *,
        worthwhile: bool,
        categories: Sequence[Category | str] = (),
        example_questions: Sequence[str] = (),
        none_reason: NoneReason | str | None = None,
        confidence: str = "sure",
        sample_source: SampleSource | str | None = None,
    ) -> None:
        if not isinstance(worthwhile, bool):
            raise AnnotationValidationError("worthwhile must be a boolean")
        try:
            parsed_cats = tuple(Category(str(c)) for c in categories)
        except ValueError as exc:
            raise AnnotationValidationError(f"Invalid category: {exc}") from exc
        parsed_nr: NoneReason | None = None
        if none_reason is not None and str(none_reason).strip():
            try:
                parsed_nr = NoneReason(str(none_reason))
            except ValueError as exc:
                raise AnnotationValidationError(f"Invalid none_reason: {none_reason!r}") from exc
        if not worthwhile and parsed_cats:
            raise AnnotationValidationError("categories must be empty when worthwhile=False")
        if not worthwhile and parsed_nr is None:
            raise AnnotationValidationError("none_reason is required when worthwhile=False")
        if worthwhile and parsed_nr is not None:
            raise AnnotationValidationError("none_reason must be null when worthwhile=True")
        if not isinstance(confidence, str) or not confidence.strip():
            raise AnnotationValidationError("confidence must be a non-empty string")
        if sample_source is not None:
            try:
                item.sample_source = SampleSource(str(sample_source))
            except ValueError as exc:
                raise AnnotationValidationError(
                    f"Invalid sample_source: {sample_source!r}"
                ) from exc
        item.worthwhile = worthwhile
        item.categories = parsed_cats
        item.example_questions = tuple(str(q) for q in example_questions if str(q).strip())
        item.none_reason = parsed_nr
        item.confidence = confidence

    def set_step1(
        self,
        item_id: str,
        *,
        worthwhile: bool,
        categories: Sequence[Category | str] = (),
        example_questions: Sequence[str] = (),
        none_reason: NoneReason | str | None = None,
        confidence: str = "sure",
        sample_source: SampleSource | str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            item = self._get(item_id)
            if item.step1_locked:
                raise AnnotationLockError(
                    "step1_locked: Step 1 fields cannot be modified after continuation reveal"
                )
            self._apply_step1_fields(
                item,
                worthwhile=worthwhile,
                categories=categories,
                example_questions=example_questions,
                none_reason=none_reason,
                confidence=confidence,
                sample_source=sample_source,
            )
            return self._view(item)

    def reveal_continuation(
        self,
        item_id: str,
        step1_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            item = self._get(item_id)
            if item.step1_locked:
                if step1_payload:
                    # Reject any attempt to alter locked Step 1 fields
                    raise AnnotationLockError(
                        "step1_locked: Step 1 fields cannot be updated once continuation is revealed"
                    )
                return self._view(item)
            if step1_payload is not None:
                if "worthwhile" not in step1_payload:
                    raise AnnotationValidationError("step1_payload must include 'worthwhile'")
                self._apply_step1_fields(
                    item,
                    worthwhile=step1_payload["worthwhile"],
                    categories=step1_payload.get("categories") or (),
                    example_questions=step1_payload.get("example_questions") or (),
                    none_reason=step1_payload.get("none_reason"),
                    confidence=str(step1_payload.get("confidence", "sure")),
                )
            if item.worthwhile is None:
                raise AnnotationValidationError(
                    "Step 1 must be filled (worthwhile set) before revealing continuation"
                )
            item.step1_locked = True
            item.state = "step2_open"
            return self._view(item)

    def submit_step2(
        self,
        item_id: str,
        *,
        close_ms: int | None = None,
        closed_by: ClosedBy | str | None = None,
        ideal_ms: int | None = None,
        notes: str | None = None,
        **extra: Any,
    ) -> LabelRecord:
        with self._lock:
            item = self._get(item_id)
            if not item.step1_locked or item.worthwhile is None:
                raise AnnotationLockError(
                    "step1_not_locked: must reveal continuation before submitting Step 2"
                )
            bad_step1 = _STEP1_FORBIDDEN_IN_STEP2 & set(extra)
            if bad_step1:
                raise AnnotationLockError(
                    f"step1_locked: cannot include or mutate Step 1 fields {sorted(bad_step1)} in Step 2"
                )
            if extra:
                raise AnnotationValidationError(f"Unknown Step 2 fields: {sorted(extra)}")
            parsed_cb: ClosedBy | None = None
            if closed_by is not None and str(closed_by).strip():
                try:
                    parsed_cb = ClosedBy(str(closed_by))
                except ValueError as exc:
                    raise AnnotationValidationError(f"Invalid closed_by: {closed_by!r}") from exc
            try:
                validity = ValidityInterval(
                    open_ms=item.open_ms,
                    ideal_ms=ideal_ms,
                    close_ms=close_ms,
                    closed_by=parsed_cb,
                )
            except ContractValidationError as exc:
                raise AnnotationValidationError(str(exc)) from exc

            lbl = LabelRecord(
                schema_version=SCHEMA_VERSION,
                label_id=f"lbl-{uuid.uuid4().hex[:12]}",
                session_id=item.session_id,
                annotator=self._annotator,
                sample_source=item.sample_source,
                annotation_pass=AnnotationPass.OFFLINE,
                candidate_id=item.candidate_id,
                state_version_at_decision=item.state_version_at_decision,
                worthwhile=item.worthwhile,
                categories=item.categories,
                example_questions=item.example_questions,
                evidence_refs=item.evidence_refs,
                validity=validity,
                none_reason=item.none_reason,
                confidence=item.confidence,
                notes=notes,
            )
            item.close_ms = close_ms
            item.closed_by = parsed_cb
            item.ideal_ms = ideal_ms
            item.notes = notes
            item.label = lbl
            item.state = "completed"
            if self._out_path is not None:
                save_labels_jsonl(self._all_labels_locked(), self._out_path, append=False)
            if self._store is not None:
                self._store.record_label(
                    run_id=self._run_id, label=lbl, created_ms=close_ms or item.open_ms
                )
            return lbl

    def _all_labels_locked(self) -> list[LabelRecord]:
        out: list[LabelRecord] = []
        for iid in self._item_order:
            lbl = self._items[iid].label
            if lbl is not None:
                out.append(lbl)
        return out

    def completed_labels(self) -> list[LabelRecord]:
        with self._lock:
            return self._all_labels_locked()


def build_blind_retest_queue(
    labels: Sequence[LabelRecord],
    *,
    fraction: float = 0.10,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample a blind ~10% test-retest subset stripped of prior judgment fields (§12.2)."""
    if not labels or fraction <= 0.0:
        return []
    n_take = min(len(labels), max(1, round(len(labels) * fraction)))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(labels)), n_take))
    queue: list[dict[str, Any]] = []
    for idx in indices:
        lbl = labels[idx]
        queue.append(
            {
                "session_id": lbl.session_id,
                "candidate_id": lbl.candidate_id,
                "state_version_at_decision": lbl.state_version_at_decision,
                "evidence_refs": [r.to_dict() for r in lbl.evidence_refs],
                "open_ms": lbl.validity.open_ms,
                "sample_source": str(lbl.sample_source),
            }
        )
    return queue


def _kappa_key(lbl: LabelRecord) -> tuple[str, str]:
    if lbl.candidate_id is not None:
        return (lbl.session_id, lbl.candidate_id)
    seg_key = ",".join(sorted(r.segment_id for r in lbl.evidence_refs))
    return (lbl.session_id, seg_key)


def _binary_kappa(
    y1: Sequence[bool],
    y2: Sequence[bool],
    *,
    min_n: int,
    min_positives: int,
) -> dict[str, Any]:
    n = len(y1)
    n_pos_1 = sum(1 for v in y1 if v)
    n_pos_2 = sum(1 for v in y2 if v)
    if n == 0:
        return {
            "n_paired": 0,
            "n_pos_first": 0,
            "n_pos_second": 0,
            "observed_agreement": None,
            "expected_agreement": None,
            "kappa": None,
            "degenerate_marginals": False,
            "low_n": True,
            "insufficient_n": True,
            "caveat": "No paired labels available.",
        }
    agree = sum(1 for a, b in zip(y1, y2, strict=True) if a == b)
    p_o = agree / float(n)
    p1 = n_pos_1 / float(n)
    p2 = n_pos_2 / float(n)
    p_e = (p1 * p2) + ((1.0 - p1) * (1.0 - p2))
    denom = 1.0 - p_e
    degenerate = denom <= 1e-12
    if degenerate:
        kappa: float | None = 1.0 if p_o >= 1.0 - 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / denom
    low_n = n < min_n or min(n_pos_1, n_pos_2) < min_positives or degenerate
    caveat: str | None = None
    if degenerate:
        caveat = "Degenerate marginals (single-class agreement); kappa defaulted due to zero expected disagreement."
    elif low_n:
        caveat = (
            f"Low sample count (n_paired={n} < {min_n} or "
            f"min_positives={min(n_pos_1, n_pos_2)} < {min_positives})."
        )
    return {
        "n_paired": n,
        "n_pos_first": n_pos_1,
        "n_pos_second": n_pos_2,
        "observed_agreement": p_o,
        "expected_agreement": p_e,
        "kappa": kappa,
        "degenerate_marginals": degenerate,
        "low_n": low_n,
        "insufficient_n": low_n,
        "caveat": caveat,
    }


def compute_cohens_kappa(
    first_labels: Sequence[LabelRecord],
    second_labels: Sequence[LabelRecord],
    *,
    min_n: int = 20,
    min_positives: int = 5,
) -> dict[str, Any]:
    """Compute intra-annotator blind test-retest Cohen's kappa (§12.2)."""
    first_map = {_kappa_key(lbl): lbl for lbl in first_labels}
    second_map = {_kappa_key(lbl): lbl for lbl in second_labels}
    paired_keys = sorted(set(first_map) & set(second_map))
    pairs = [(first_map[k], second_map[k]) for k in paired_keys]

    worth_res = _binary_kappa(
        [a.worthwhile for a, _ in pairs],
        [b.worthwhile for _, b in pairs],
        min_n=min_n,
        min_positives=min_positives,
    )

    seen_cats: set[Category] = set()
    for a, b in pairs:
        seen_cats.update(a.categories)
        seen_cats.update(b.categories)
    all_cats = sorted(seen_cats | set(PREDICTABLE_CATEGORIES), key=lambda c: c.value)
    per_category: dict[str, dict[str, Any]] = {}
    for cat in all_cats:
        if cat not in seen_cats:
            continue
        per_category[cat.value] = _binary_kappa(
            [cat in a.categories for a, _ in pairs],
            [cat in b.categories for _, b in pairs],
            min_n=min_n,
            min_positives=min_positives,
        )

    return {
        "n_first": len(first_labels),
        "n_second": len(second_labels),
        "n_paired": len(pairs),
        "worthwhile": worth_res,
        "per_category": per_category,
    }


class BlindRetestSession:
    """Blind test-retest annotation session stripping prior labels (§12.2)."""

    def __init__(
        self,
        *,
        first_labels: Sequence[LabelRecord],
        events: Sequence[CanonicalEvent] = (),
        annotator: str = "retest_annotator",
        out_path: str | Path | None = None,
    ) -> None:
        self._first_labels = list(first_labels)
        self._by_id: dict[str, LabelRecord] = {}
        for idx, lbl in enumerate(self._first_labels, start=1):
            iid = lbl.candidate_id or lbl.label_id or f"retest-{idx:04d}"
            self._by_id[iid] = lbl
            if lbl.label_id:
                self._by_id[lbl.label_id] = lbl
        self._events_by_seg: dict[str, CanonicalEvent] = {
            ev.segment_id: ev
            for ev in events
            if ev.segment_id is not None and ev.kind is EventKind.SEGMENT
        }
        self._annotator = annotator
        self._out_path = Path(out_path) if out_path is not None else None
        self._submitted: dict[str, LabelRecord] = {}
        self._lock = threading.Lock()

    def _resolve(self, item_id: str) -> tuple[str, LabelRecord]:
        if item_id not in self._by_id:
            raise KeyError(f"Unknown retest item {item_id!r}")
        lbl = self._by_id[item_id]
        key = lbl.candidate_id or lbl.label_id
        return key, lbl

    def get_item(self, item_id: str) -> dict[str, Any]:
        with self._lock:
            key, lbl = self._resolve(item_id)
            ev_views = [
                _event_view(self._events_by_seg[r.segment_id])
                for r in lbl.evidence_refs
                if r.segment_id in self._events_by_seg
            ]
            return {
                "item_id": key,
                "candidate_id": lbl.candidate_id,
                "session_id": lbl.session_id,
                "state_version_at_decision": lbl.state_version_at_decision,
                "evidence_refs": [r.to_dict() for r in lbl.evidence_refs],
                "open_ms": lbl.validity.open_ms,
                "sample_source": str(lbl.sample_source),
                "events": ev_views,
                "worthwhile": None,
                "categories": None,
                "none_reason": None,
                "annotator": None,
            }

    def submit_item(
        self,
        item_id: str,
        *,
        worthwhile: bool,
        categories: Sequence[Category | str] = (),
        none_reason: NoneReason | str | None = None,
        example_questions: Sequence[str] = (),
        confidence: str = "sure",
        notes: str | None = None,
    ) -> LabelRecord:
        with self._lock:
            key, orig = self._resolve(item_id)
            if not isinstance(worthwhile, bool):
                raise AnnotationValidationError("worthwhile must be a boolean")
            parsed_cats = tuple(Category(str(c)) for c in categories)
            parsed_nr: NoneReason | None = None
            if none_reason is not None and str(none_reason).strip():
                parsed_nr = NoneReason(str(none_reason))
            if not worthwhile and parsed_cats:
                raise AnnotationValidationError("categories must be empty when worthwhile=False")
            if not worthwhile and parsed_nr is None:
                raise AnnotationValidationError("none_reason is required when worthwhile=False")
            if worthwhile and parsed_nr is not None:
                raise AnnotationValidationError("none_reason must be null when worthwhile=True")
            lbl = LabelRecord(
                schema_version=SCHEMA_VERSION,
                label_id=f"lbl-retest-{uuid.uuid4().hex[:12]}",
                session_id=orig.session_id,
                annotator=self._annotator,
                sample_source=orig.sample_source,
                annotation_pass=AnnotationPass.OFFLINE,
                candidate_id=orig.candidate_id,
                state_version_at_decision=orig.state_version_at_decision,
                worthwhile=worthwhile,
                categories=parsed_cats,
                example_questions=tuple(str(q) for q in example_questions if str(q).strip()),
                evidence_refs=orig.evidence_refs,
                validity=orig.validity,
                none_reason=parsed_nr,
                confidence=confidence,
                notes=notes,
            )
            self._submitted[key] = lbl
            if self._out_path is not None:
                save_labels_jsonl(list(self._submitted.values()), self._out_path, append=False)
            return lbl

    def kappa_report(self) -> dict[str, Any]:
        with self._lock:
            second = list(self._submitted.values())
        res = compute_cohens_kappa(self._first_labels, second)
        res["worthwhile_kappa"] = res["worthwhile"]["kappa"]
        return res


_ANNOTATION_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Earshot Coach Annotation Tool</title>
</head>
<body>
  <main id="app">
    <h1 id="title">Coach Annotation Tool</h1>
    <pre id="context-box"></pre>
    <pre id="continuation-box"></pre>
  </main>
  <script>
    (function () {
      const p = new URLSearchParams(window.location.search);
      const token = p.get("token") || "";
      fetch("/api/health?token=" + encodeURIComponent(token))
        .then((r) => r.json())
        .then((d) => {
          document.getElementById("context-box").textContent = JSON.stringify(d, null, 2);
        })
        .catch(() => {});
    })();
  </script>
</body>
</html>
"""


class _AnnotationHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        owner: AnnotationServer,
    ) -> None:
        self.owner = owner
        super().__init__(server_address, handler_cls)


class AnnotationServer:
    """Loopback-only token-gated HTTP server for Pass 1 & Pass 2 annotation (§12.4)."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        session_token: str | None = None,
        pass1: Pass1Session | None = None,
        pass2: Pass2Session | None = None,
        retest: BlindRetestSession | None = None,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError(f"AnnotationServer must bind strictly to 127.0.0.1, got {host!r}")
        if port < 0 or port > 65535:
            raise ValueError(f"Invalid port {port}")
        self._host = host
        self._requested_port = port
        self._token = session_token or secrets.token_urlsafe(32)
        self._pass1 = pass1
        self._pass2 = pass2
        self._retest = retest
        self._httpd: _AnnotationHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        if self._httpd is not None:
            return int(self._httpd.server_address[1])
        return self._requested_port

    @property
    def token(self) -> str:
        return self._token

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}/?token={self._token}"

    def start(self) -> None:
        if self._httpd is not None:
            return
        owner = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "CoachAnnotate/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, status: int, data: Any) -> None:
                body = json.dumps(data, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, status: int, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; object-src 'none'",
                )
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                auth = self.headers.get("Authorization", "")
                supplied = ""
                if auth.startswith("Bearer "):
                    supplied = auth[7:].strip()
                else:
                    qs = parse_qs(urlsplit(self.path).query)
                    vals = qs.get("token")
                    supplied = vals[0] if vals else ""
                if not supplied or not hmac.compare_digest(supplied, owner._token):
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                    return False
                return True

            def _read_json_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                loaded = json.loads(raw.decode("utf-8"))
                if not isinstance(loaded, dict):
                    raise AnnotationValidationError("JSON body must be an object")
                return loaded

            def do_GET(self) -> None:
                if not self._authorized():
                    return
                parsed = urlsplit(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)
                try:
                    if path in {"/", "/index.html"}:
                        self._send_html(HTTPStatus.OK, _ANNOTATION_HTML)
                    elif path == "/api/health":
                        self._send_json(
                            HTTPStatus.OK,
                            {
                                "status": "ok",
                                "pass1": owner._pass1 is not None,
                                "pass2": owner._pass2 is not None,
                                "retest": owner._retest is not None,
                            },
                        )
                    elif path.startswith("/api/sessions/") and path.endswith("/prefix"):
                        if owner._pass1 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass1_not_configured"})
                            return
                        at_vals = qs.get("at_ms")
                        if not at_vals:
                            raise AnnotationValidationError("at_ms query parameter is required")
                        at_ms = int(at_vals[0])
                        inc_fut = str((qs.get("include_future") or [""])[0]).strip().lower()
                        scroll_vals = qs.get("scroll_to_ms")
                        if inc_fut in {"1", "true", "yes"} or (
                            scroll_vals and int(scroll_vals[0]) > at_ms
                        ):
                            raise AnnotationLockError(
                                f"scroll_ahead_forbidden: cannot view past playhead at_ms={at_ms}"
                            )
                        sid = path.removeprefix("/api/sessions/").removesuffix("/prefix").strip("/")
                        self._send_json(
                            HTTPStatus.OK,
                            {
                                "session_id": sid,
                                "at_ms": at_ms,
                                "events": owner._pass1.prefix_at(at_ms),
                            },
                        )
                    elif path.startswith("/api/candidates/") and path.endswith("/step1"):
                        if owner._pass2 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass2_not_configured"})
                            return
                        cid = (
                            path.removeprefix("/api/candidates/").removesuffix("/step1").strip("/")
                        )
                        self._send_json(HTTPStatus.OK, owner._pass2.get_item_view(cid))
                    elif path == "/api/review-queue":
                        if owner._pass2 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass2_not_configured"})
                            return
                        self._send_json(HTTPStatus.OK, {"items": owner._pass2.review_queue_list()})
                    elif path == "/api/retest/kappa":
                        if owner._retest is None:
                            self._send_json(
                                HTTPStatus.NOT_FOUND, {"error": "retest_not_configured"}
                            )
                            return
                        self._send_json(HTTPStatus.OK, owner._retest.kappa_report())
                    elif path.startswith("/api/retest/"):
                        if owner._retest is None:
                            self._send_json(
                                HTTPStatus.NOT_FOUND, {"error": "retest_not_configured"}
                            )
                            return
                        rid = path.removeprefix("/api/retest/").strip("/")
                        self._send_json(HTTPStatus.OK, owner._retest.get_item(rid))
                    elif path == "/api/pass1/state":
                        if owner._pass1 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass1_not_configured"})
                            return
                        self._send_json(HTTPStatus.OK, owner._pass1.state_view())
                    elif path == "/api/pass1/events":
                        if owner._pass1 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass1_not_configured"})
                            return
                        at_vals = qs.get("at_ms")
                        opt_at_ms = int(at_vals[0]) if at_vals else None
                        self._send_json(
                            HTTPStatus.OK,
                            {"events": owner._pass1.visible_events(at_ms=opt_at_ms)},
                        )
                    elif path == "/api/pass2/items":
                        if owner._pass2 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass2_not_configured"})
                            return
                        self._send_json(HTTPStatus.OK, {"items": owner._pass2.list_items()})
                    elif path.startswith("/api/pass2/items/"):
                        if owner._pass2 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass2_not_configured"})
                            return
                        rest = path.removeprefix("/api/pass2/items/")
                        if rest.endswith("/continuation"):
                            iid = rest.removesuffix("/continuation").strip("/")
                            cont = owner._pass2.get_continuation(iid)
                            self._send_json(HTTPStatus.OK, {"item_id": iid, "continuation": cont})
                        else:
                            iid = rest.strip("/")
                            self._send_json(HTTPStatus.OK, owner._pass2.get_item_view(iid))
                    elif path == "/api/labels":
                        labels: list[dict[str, Any]] = []
                        if owner._pass1 is not None:
                            labels.extend(lbl.to_dict() for lbl in owner._pass1.marks)
                        if owner._pass2 is not None:
                            labels.extend(lbl.to_dict() for lbl in owner._pass2.completed_labels())
                        self._send_json(HTTPStatus.OK, {"labels": labels})
                    else:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                except AnnotationLockError as exc:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                except (AnnotationValidationError, ValueError, json.JSONDecodeError) as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})

            def do_POST(self) -> None:
                if not self._authorized():
                    return
                path = urlsplit(self.path).path
                try:
                    body = self._read_json_body()
                    if path == "/api/pass1/step":
                        if owner._pass1 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass1_not_configured"})
                            return
                        if "delta_ms" in body:
                            owner._pass1.advance_by(int(body["delta_ms"]))
                        else:
                            owner._pass1.step_next()
                        self._send_json(HTTPStatus.OK, owner._pass1.state_view())
                    elif path == "/api/pass1/seek":
                        if owner._pass1 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass1_not_configured"})
                            return
                        owner._pass1.seek(int(body["target_ms"]))
                        self._send_json(HTTPStatus.OK, owner._pass1.state_view())
                    elif path == "/api/pass1/mark":
                        if owner._pass1 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass1_not_configured"})
                            return
                        lbl = owner._pass1.mark_opportunity(
                            example_questions=body.get("example_questions") or (),
                            categories=body.get("categories") or (),
                            notes=body.get("notes"),
                            confidence=str(body.get("confidence", "sure")),
                        )
                        self._send_json(HTTPStatus.OK, {"label": lbl.to_dict()})
                    elif path.startswith("/api/sessions/") and path.endswith("/marks"):
                        if owner._pass1 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass1_not_configured"})
                            return
                        eq = body.get("example_questions") or ()
                        if not eq and body.get("question"):
                            eq = [str(body["question"])]
                        lbl = owner._pass1.mark_opportunity(
                            example_questions=eq,
                            categories=body.get("categories") or (),
                            notes=body.get("notes"),
                            confidence=str(body.get("confidence", "sure")),
                            at_ms=int(body["at_ms"]) if body.get("at_ms") is not None else None,
                            state_version_override=(
                                int(body["state_version"])
                                if body.get("state_version") is not None
                                else None
                            ),
                        )
                        self._send_json(HTTPStatus.OK, {"label": lbl.to_dict()})
                    elif path.startswith("/api/candidates/"):
                        if owner._pass2 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass2_not_configured"})
                            return
                        rest = path.removeprefix("/api/candidates/")
                        if rest.endswith("/step1"):
                            cid = rest.removesuffix("/step1").strip("/")
                            view = owner._pass2.set_step1(
                                cid,
                                worthwhile=body["worthwhile"],
                                categories=body.get("categories") or (),
                                example_questions=body.get("example_questions") or (),
                                none_reason=body.get("none_reason"),
                                confidence=str(body.get("confidence", "sure")),
                            )
                            self._send_json(HTTPStatus.OK, view)
                        elif rest.endswith("/step2/unlock"):
                            cid = rest.removesuffix("/step2/unlock").strip("/")
                            payload = (
                                body.get("step1")
                                if "step1" in body
                                else (body if "worthwhile" in body else None)
                            )
                            view = owner._pass2.reveal_continuation(cid, step1_payload=payload)
                            self._send_json(HTTPStatus.OK, view)
                        elif rest.endswith("/step2"):
                            cid = rest.removesuffix("/step2").strip("/")
                            lbl = owner._pass2.submit_step2(cid, **body)
                            self._send_json(HTTPStatus.OK, {"label": lbl.to_dict()})
                        else:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    elif path.startswith("/api/retest/"):
                        if owner._retest is None:
                            self._send_json(
                                HTTPStatus.NOT_FOUND, {"error": "retest_not_configured"}
                            )
                            return
                        rid = path.removeprefix("/api/retest/").strip("/")
                        lbl = owner._retest.submit_item(
                            rid,
                            worthwhile=body["worthwhile"],
                            categories=body.get("categories") or (),
                            none_reason=body.get("none_reason"),
                            example_questions=body.get("example_questions") or (),
                            confidence=str(body.get("confidence", "sure")),
                            notes=body.get("notes"),
                        )
                        self._send_json(HTTPStatus.OK, {"label": lbl.to_dict()})
                    elif path.startswith("/api/pass2/items/"):
                        if owner._pass2 is None:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "pass2_not_configured"})
                            return
                        rest = path.removeprefix("/api/pass2/items/")
                        if rest.endswith("/step1"):
                            iid = rest.removesuffix("/step1").strip("/")
                            view = owner._pass2.set_step1(
                                iid,
                                worthwhile=body["worthwhile"],
                                categories=body.get("categories") or (),
                                example_questions=body.get("example_questions") or (),
                                none_reason=body.get("none_reason"),
                                confidence=str(body.get("confidence", "sure")),
                            )
                            self._send_json(HTTPStatus.OK, view)
                        elif rest.endswith("/reveal"):
                            iid = rest.removesuffix("/reveal").strip("/")
                            payload = body.get("step1") if "step1" in body else (body or None)
                            view = owner._pass2.reveal_continuation(iid, step1_payload=payload)
                            self._send_json(HTTPStatus.OK, view)
                        elif rest.endswith("/step2"):
                            iid = rest.removesuffix("/step2").strip("/")
                            lbl = owner._pass2.submit_step2(iid, **body)
                            self._send_json(HTTPStatus.OK, {"label": lbl.to_dict()})
                        else:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    else:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                except AnnotationLockError as exc:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                except (
                    AnnotationValidationError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        self._httpd = _AnnotationHTTPServer((self._host, self._requested_port), _Handler, self)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="coach-annotate-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
