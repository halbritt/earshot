"""5-task asyncio runtime, CoverageCursor, Display controller, and Manual lane (§4.2, §8, §15 level 11)."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import uuid
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, runtime_checkable

from coach.calibration import CalibrationModel, CalibrationRegistry, default_calibration_registry
from coach.candidate_validator import check_do_not_suggest, validate_template_candidate
from coach.canonical_state import ApplyResult, CanonicalState, SegmentRecord, StateSnapshot
from coach.context_builder import (
    ContextBuildResult,
    CoverageCursor,
    PolicyConfig,
    assign_speaker_roles,
    bucket_age_ms,
    build_decision_context,
    coalesce_anchors,
    effective_max_cards_per_10_min,
    normalize_for_matching,
    run_prefilter,
)
from coach.contracts import (
    SCHEMA_VERSION,
    SUPPRESS_REASON_CODES,
    CanonicalEvent,
    Category,
    ClassifierResult,
    ControlSubtype,
    DecisionContext,
    EgressRecord,
    EventKind,
    FeedbackReason,
    HealthState,
    LedgerEvent,
    LedgerState,
    Mode,
    Origin,
    Outcome,
    ReasonCode,
    Route,
    RoutingDecision,
    ScoreSet,
    SessionBrief,
    Stability,
    SuggestionCandidate,
    TelemetryLevel,
    TelemetryRecord,
)
from coach.opportunity_classifier import (
    ClassifierErrorResult,
    ClassifierInput,
    OpportunityClassifier,
)
from coach.provider_adapter import BreakerState, ProviderAdapter
from coach.question_generator import GenerationOutcome, generate_question
from coach.routing_policy import is_equivalent_question, matches_resolution_cue, route_candidate
from coach.storage import SQLiteStore
from coach.template_renderer import (
    TemplateSpec,
    load_templates_dir,
    render_generic_fallback,
    render_template,
)
from coach.transcript_adapter import Clock, MonotonicSessionClock
from coach.ui import CATEGORY_CHIPS, CardUIServer, UIStateSnapshot


@runtime_checkable
class TelemetryBufferProtocol(Protocol):
    """Structural protocol for injected BoundedRecorderBuffer (§15 upward-only boundary)."""

    def emit(self, record: TelemetryRecord) -> bool: ...

    def flush_to_store(self, store: SQLiteStore, session_id: str, run_id: str) -> int: ...


class _BoundedFallbackBuffer:
    """Internal fallback bounded recorder used when telemetry_buffer is omitted."""

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = max(1, maxsize)
        self._lock = threading.Lock()
        self._seq = 0
        self._critical: deque[tuple[int, TelemetryRecord]] = deque()
        self._debug: deque[tuple[int, TelemetryRecord]] = deque()
        self._unflushed_drops: dict[TelemetryLevel, int] = {
            TelemetryLevel.DEBUG: 0,
            TelemetryLevel.LIFECYCLE: 0,
            TelemetryLevel.DECISION: 0,
        }

    def emit(self, record: TelemetryRecord) -> bool:
        with self._lock:
            total = len(self._critical) + len(self._debug)
            if record.level is TelemetryLevel.DEBUG:
                if total >= self._maxsize:
                    self._unflushed_drops[TelemetryLevel.DEBUG] += 1
                    return False
                self._seq += 1
                self._debug.append((self._seq, record))
                return True
            if total >= self._maxsize and self._debug:
                self._debug.popleft()
                self._unflushed_drops[TelemetryLevel.DEBUG] += 1
            self._seq += 1
            self._critical.append((self._seq, record))
            return True

    def flush_to_store(self, store: SQLiteStore, session_id: str, run_id: str) -> int:
        with self._lock:
            taken = list(self._critical) + list(self._debug)
            self._critical.clear()
            self._debug.clear()
            taken.sort(key=lambda item: item[0])
            records = [rec for _, rec in taken]
            drops = {k: v for k, v in self._unflushed_drops.items() if v > 0}
            for lvl in self._unflushed_drops:
                self._unflushed_drops[lvl] = 0
        if records:
            matching = [r for r in records if r.session_id == session_id]
            if matching:
                store.record_telemetry(run_id=run_id, records=matching)
        if drops:
            store.record_drops(session_id=session_id, run_id=run_id, drops_by_level=drops)
        return len(records)


@dataclass(frozen=True, slots=True)
class ManualResult:
    """Outcome returned by SuggestionController.trigger_manual() (§3.7, §8.2, B20)."""

    status: Literal["displayed", "empty", "ignored_busy", "unavailable"]
    suggestion: SuggestionCandidate | None = None
    message: str = ""
    reason: ReasonCode | None = None


@dataclass(slots=True)
class _PendingItem:
    cand: SuggestionCandidate
    context: DecisionContext
    classifier: ClassifierResult
    routing: RoutingDecision
    serialized_text: str
    score: float
    brief_snapshot: SessionBrief
    anchor_text: str
    anchor_speaker_id: str
    anchor_last_key: tuple[int, int, int, str]
    anchor_start_ms: int
    anchor_received_ms: int
    settle_end_ms: int
    settled: bool = False
    settle_cause: ReasonCode = ReasonCode.SETTLE_TIMEOUT
    deferred_reason: ReasonCode | None = None
    cooldown_end_ms: int | None = None


@dataclass(slots=True)
class _ActiveItem:
    cand: SuggestionCandidate
    context: DecisionContext
    classifier: ClassifierResult
    routing: RoutingDecision
    serialized_text: str
    brief_snapshot: SessionBrief
    anchor_text: str
    anchor_last_key: tuple[int, int, int, str]
    anchor_start_ms: int
    anchor_received_ms: int
    displayed_at_ms: int
    expires_at_ms: int
    pinned: bool = False
    shadow: bool = False


@dataclass(frozen=True, slots=True)
class _RecorderJob:
    kind: str
    payload: dict[str, Any]


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "it",
        "this",
        "that",
        "we",
        "you",
        "they",
        "i",
        "will",
        "would",
        "should",
        "could",
        "can",
        "do",
        "does",
        "did",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "and",
        "or",
        "need",
        "needs",
    }
)

_SHOWN_OR_ACTED_STATES: frozenset[LedgerState] = frozenset(
    {
        LedgerState.DISPLAYED,
        LedgerState.PINNED,
        LedgerState.DISMISSED,
        LedgerState.USED,
    }
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize_for_matching(text).split() if t and t not in _STOPWORDS}


_matches_resolution_cue = matches_resolution_cue
_is_equivalent_question = is_equivalent_question


def _brief_material_key(brief: SessionBrief) -> tuple[Any, ...]:
    return (
        brief.objective,
        tuple((f.key, f.value) for f in brief.known_facts),
        brief.do_not_suggest,
        brief.mode,
    )


def _downgrade_unexecuted_generator_routing(
    routing: RoutingDecision,
    *,
    egress_id: str | None = None,
    terminal_reason: ReasonCode | None = None,
) -> RoutingDecision:
    if (
        egress_id is None
        and routing.outcome is Outcome.ROUTE
        and routing.route is Route.GENERATOR
        and terminal_reason is not ReasonCode.INSUFFICIENT_BUDGET
    ):
        return replace(routing, route=Route.TEMPLATE)
    return routing


class SuggestionController:
    """5-task asyncio runtime controller (§4.2, §8, §15 level 11)."""

    def __init__(
        self,
        *,
        brief: SessionBrief,
        policy: PolicyConfig,
        classifier: OpportunityClassifier,
        calibration: CalibrationModel | CalibrationRegistry | None = None,
        templates: Mapping[str, TemplateSpec] | None = None,
        provider: ProviderAdapter | None = None,
        store: SQLiteStore | None = None,
        run_id: str | None = None,
        mode: Mode = Mode.PROACTIVE,
        clock: Clock | None = None,
        telemetry_buffer: TelemetryBufferProtocol | None = None,
        ui_server: CardUIServer | None = None,
        adapter: Any | None = None,
        classifier_error_threshold: int = 3,
    ) -> None:
        self._brief = brief
        self._policy = policy
        self._classifier = classifier
        self._calibration: CalibrationModel | CalibrationRegistry | None = (
            calibration
            if calibration is not None
            else (default_calibration_registry() if policy.classifier.calibration else None)
        )
        self._templates = dict(templates) if templates is not None else load_templates_dir()
        self._provider = provider
        self._store = store
        self._run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self._mode: Mode = mode
        self._pre_otr_mode: Mode = mode if mode is not Mode.OFF_THE_RECORD else Mode.PROACTIVE
        self._clock: Clock = clock or MonotonicSessionClock()
        self._telemetry: TelemetryBufferProtocol = telemetry_buffer or _BoundedFallbackBuffer()
        self._ui = ui_server
        self._adapter = adapter
        self._classifier_error_threshold = max(1, classifier_error_threshold)

        self._state = CanonicalState(brief)
        self._cursor = CoverageCursor()
        local_blocked = (
            provider is not None
            and provider.config.local_only
            and provider.config.effective_is_hosted
        )
        self._health: HealthState
        self._health_reason: str | None
        if mode is Mode.PAUSED:
            self._health = HealthState.PAUSED
            self._health_reason = "paused"
        elif local_blocked:
            self._health = HealthState.TEMPLATE_ONLY
            self._health_reason = "local_only"
        else:
            self._health = HealthState.OK
            self._health_reason = None
        self._health_timeline: list[dict[str, Any]] = [
            {
                "at_ms": self._clock.now_ms(),
                "state": self._health.value,
                "reason": self._health_reason,
            }
        ]

        self._started = False
        self._stopping = False
        self._started_ms: int | None = None
        self._last_transcript_event_ms: int = self._clock.now_ms()
        self._last_gap_ms: int | None = None
        self._recipient_speaking = False
        self._recipient_speaking_until_ms: int = 0
        self._otr_start_ms: int | None = None
        self._speaker_update_times: dict[str, int] = {}
        self._speaker_boundary_end_ms: dict[str, int] = {}
        self._speaker_boundary_rx_ms: dict[str, int] = {}
        self._prev_anchor_norm_text: str | None = None
        self._prev_anchor_seg_revs: dict[str, int] = {}
        self._prev_anchor_rx_ms: int | None = None
        self._consecutive_clf_errors = 0
        self._gen_latencies_ms: deque[int] = deque(maxlen=64)
        self._gen_p95_ms: int | None = None

        self._suggestions_by_id: dict[str, SuggestionCandidate] = {}
        self._ledger_texts: dict[str, str] = {}
        self._ledger_categories: dict[str, Category] = {}
        self._prior_anchor_texts: dict[str, str] = {}
        self._used_inferred_sent: set[tuple[str, str]] = set()
        self._last_displayed_card_info: tuple[SuggestionCandidate, int] | None = None

        self._settling: list[_PendingItem] = []
        self._deferred: _PendingItem | None = None
        self._active: _ActiveItem | None = None
        self._last_proactive_display_ms: int | None = None
        self._proactive_display_times: deque[int] = deque(maxlen=64)
        self._manual_status: dict[str, Any] | None = None
        self._manual_in_flight = False
        self._manual_ignored_count = 0
        self._half_open_probe_in_flight = False

        self._ingest_queue: asyncio.Queue[CanonicalEvent | None] = asyncio.Queue()
        self._manual_queue: asyncio.Queue[asyncio.Future[ManualResult] | None] = asyncio.Queue()
        self._recorder_queue: asyncio.Queue[_RecorderJob | None] = asyncio.Queue()
        self._proactive_wake = asyncio.Event()
        self._display_wake = asyncio.Event()
        self._proactive_idle = asyncio.Event()
        self._proactive_idle.set()
        self._state_lock = asyncio.Lock()

        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._adapter_pump_task: asyncio.Task[None] | None = None
        self._tel_seq = 0
        self._evt_seq = 0
        self._ui_seq = 0
        self._ui_broadcast_queue: asyncio.Queue[UIStateSnapshot] = asyncio.Queue()
        self._ui_broadcast_task: asyncio.Task[None] | None = None

        if self._ui is not None:
            self._ui.set_action_handler(self.handle_ui_action)
        if self._provider is not None:
            self._provider.egress_sink = self._on_provider_egress
            self._provider.on_health_change = self._on_provider_health_change

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def canonical_state(self) -> CanonicalState:
        return self._state

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def health(self) -> HealthState:
        return self._health

    @property
    def health_reason(self) -> str | None:
        return self._health_reason

    @property
    def health_timeline(self) -> list[dict[str, Any]]:
        return list(self._health_timeline)

    @property
    def active_card(self) -> SuggestionCandidate | None:
        return self._active.cand if self._active is not None else None

    @property
    def active_pinned(self) -> bool:
        return self._active.pinned if self._active is not None else False

    @property
    def deferred_card(self) -> SuggestionCandidate | None:
        return self._deferred.cand if self._deferred is not None else None

    @property
    def cursor(self) -> CoverageCursor:
        return self._cursor

    @property
    def manual_ignored_count(self) -> int:
        return self._manual_ignored_count

    @property
    def tasks(self) -> dict[str, asyncio.Task[None]]:
        return dict(self._tasks)

    def _next_tel_id(self, prefix: str = "tel") -> str:
        self._tel_seq += 1
        return f"{prefix}-{self._run_id}-{self._tel_seq}"

    def _next_evt_id(self, prefix: str = "ctl") -> str:
        self._evt_seq += 1
        return f"{prefix}-{self._run_id}-{self._evt_seq}"

    def _emit_tel(
        self,
        *,
        level: TelemetryLevel,
        kind: str,
        stage: str,
        candidate_id: str | None = None,
        suggestion_id: str | None = None,
        reason_code: ReasonCode | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._telemetry.emit(
            TelemetryRecord(
                schema_version=SCHEMA_VERSION,
                record_id=self._next_tel_id(),
                session_id=self._brief.session_id,
                at_ms=self._clock.now_ms(),
                level=level,
                kind=kind,
                candidate_id=candidate_id,
                suggestion_id=suggestion_id,
                stage=stage,
                reason_code=reason_code,
                payload=dict(payload) if payload else {},
            )
        )

    def _enqueue_recorder(self, kind: str, **payload: Any) -> None:
        if self._store is None:
            return
        with contextlib.suppress(asyncio.QueueFull):
            self._recorder_queue.put_nowait(_RecorderJob(kind=kind, payload=payload))

    def _on_provider_egress(self, rec: EgressRecord) -> None:
        self._enqueue_recorder("egress", record=rec)
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind="generator_call",
            stage="provider_adapter",
            candidate_id=rec.candidate_id,
            payload={"egress_id": rec.egress_id, "discarded": rec.discarded},
        )

    def _on_provider_health_change(self, new_state: HealthState) -> None:
        if new_state is HealthState.TEMPLATE_ONLY and self._health is HealthState.OK:
            self._set_health(HealthState.TEMPLATE_ONLY, "generator_breaker_open")
        elif new_state is HealthState.OK and self._health is HealthState.TEMPLATE_ONLY:
            self._set_health(HealthState.OK, "generator_breaker_closed")

    def _set_health(self, state: HealthState, reason: str | None = None) -> None:
        if self._health is state and self._health_reason == reason:
            return
        self._health = state
        self._health_reason = reason
        self._health_timeline.append(
            {"at_ms": self._clock.now_ms(), "state": state.value, "reason": reason}
        )
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind="health_change",
            stage="suggestion_controller",
            payload={"health": state.value, "reason": reason},
        )
        self._schedule_ui_broadcast()

    def build_ui_snapshot(self) -> UIStateSnapshot:
        snap = self._state.snapshot()
        active_dict: dict[str, Any] | None = None
        if self._active is not None and not self._active.shadow:
            cand = self._active.cand
            role_map = assign_speaker_roles(snap)
            now_ms = self._clock.now_ms()
            ev_list: list[dict[str, Any]] = []
            for ref in cand.evidence_refs:
                seg = snap.segments_by_id.get(ref.segment_id)
                if seg is not None:
                    ev_list.append(
                        {
                            "segment_id": seg.segment_id,
                            "speaker": role_map.get(seg.speaker_id, seg.speaker_id),
                            "age": bucket_age_ms(now_ms - seg.received_ms),
                            "text": seg.text,
                        }
                    )
            active_dict = {
                "suggestion_id": cand.suggestion_id,
                "candidate_id": cand.candidate_id,
                "category": cand.category.value,
                "category_chip": CATEGORY_CHIPS.get(cand.category.value, cand.category.value),
                "text": cand.text,
                "question": cand.text,
                "word_count": cand.word_count,
                "pinned": self._active.pinned,
                "origin": cand.origin.value,
                "route": cand.route.value,
                "displayed_at_ms": self._active.displayed_at_ms,
                "evidence": ev_list,
            }
        return UIStateSnapshot(
            session_id=self._brief.session_id,
            mode=self._mode,
            health=self._health,
            health_reason=self._health_reason,
            off_record=self._mode is Mode.OFF_THE_RECORD or snap.off_record,
            active_card=active_dict,
            manual_status=dict(self._manual_status) if self._manual_status is not None else None,
        )

    def _schedule_ui_broadcast(self) -> None:
        if self._ui is None:
            return
        self._ui_seq += 1
        ui_snap = replace(self.build_ui_snapshot(), seq=self._ui_seq)
        if self._stopping:
            self._ui.update_state(ui_snap)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._ui.update_state(ui_snap)
            return
        self._ui_broadcast_queue.put_nowait(ui_snap)
        if self._ui_broadcast_task is None or self._ui_broadcast_task.done():
            self._ui_broadcast_task = loop.create_task(
                self._drain_ui_broadcasts(), name="ui_broadcast"
            )

    async def _drain_ui_broadcasts(self) -> None:
        ui_server = self._ui
        if ui_server is None:
            return
        while not self._ui_broadcast_queue.empty():
            snap = self._ui_broadcast_queue.get_nowait()
            try:
                await ui_server.broadcast_state(snap)
            finally:
                self._ui_broadcast_queue.task_done()

    async def _flush_ui_broadcast(self) -> None:
        while self._ui_broadcast_task is not None and not self._ui_broadcast_task.done():
            task = self._ui_broadcast_task
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cur = asyncio.current_task()
                if not task.cancelled() or (cur is not None and cur.cancelling() > 0):
                    raise

    def set_recipient_speaking(self, speaking: bool) -> None:
        self._recipient_speaking = speaking
        if not speaking:
            self._display_wake.set()

    async def report_adapter_health(
        self,
        state: HealthState,
        reason: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind="adapter_health",
            stage="transcript_adapter",
            payload={"health": state.value, "reason": reason, **dict(detail or {})},
        )
        if state is HealthState.NO_TRANSCRIPT:
            self._set_health(HealthState.NO_TRANSCRIPT, reason)
        elif self._health is HealthState.NO_TRANSCRIPT and state is HealthState.OK:
            await self._recover_from_no_transcript(reason)
        self._display_wake.set()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stopping = False
        now_ms = self._clock.now_ms()
        self._started_ms = now_ms
        self._last_transcript_event_ms = now_ms
        if self._store is not None:
            self._store.upsert_session(
                brief=self._state.brief,
                run_id=self._run_id,
                runtime_mode=self._mode,
                effective_max_cards_per_10_min=effective_max_cards_per_10_min(
                    self._state.brief, self._policy
                ),
                health_timeline=self._health_timeline,
                started_ms=now_ms,
            )
        self._schedule_ui_broadcast()
        self._tasks = {
            "ingest": asyncio.create_task(self._ingest_loop(), name="ingest"),
            "proactive": asyncio.create_task(self._proactive_loop(), name="proactive"),
            "manual": asyncio.create_task(self._manual_loop(), name="manual"),
            "display": asyncio.create_task(self._display_loop(), name="display"),
            "recorder": asyncio.create_task(self._recorder_loop(), name="recorder"),
        }
        if self._adapter is not None:
            self._adapter_pump_task = asyncio.create_task(
                self._pump_adapter(self._adapter), name="adapter_pump"
            )

    async def _pump_adapter(self, adapter: Any) -> None:
        try:
            async for ev in adapter.events():
                await self.submit_event(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.report_adapter_health(
                HealthState.NO_TRANSCRIPT, "adapter_stream_error", {"err": str(exc)}
            )

    async def ingest_from(self, stream: AsyncIterator[CanonicalEvent]) -> int:
        count = 0
        async for ev in stream:
            await self.submit_event(ev)
            count += 1
        return count

    async def submit_event(self, ev: CanonicalEvent) -> ApplyResult:
        if self._mode is Mode.OFF_THE_RECORD and ev.kind is not EventKind.CONTROL:
            return ApplyResult(
                accepted=False,
                state_version=self._state.state_version,
                material_change=False,
                drop_reason="off_the_record",
            )
        if self._started and not self._stopping:
            await self._ingest_queue.put(ev)
            return ApplyResult(
                accepted=True,
                state_version=self._state.state_version,
                material_change=False,
            )
        return await self._process_ingested_event(ev)

    async def drain(self, *, timeout_s: float = 5.0) -> None:
        """Wait until queued ingest events, proactive job, and recorder are idle (for tests/EOF)."""
        await asyncio.wait_for(self._ingest_queue.join(), timeout=timeout_s)
        await self.tick()
        if self._proactive_wake.is_set() or not self._proactive_idle.is_set():
            await asyncio.sleep(0)
            await asyncio.wait_for(self._proactive_idle.wait(), timeout=timeout_s)
        await self.tick()
        await self._flush_ui_broadcast()
        if self._store is not None:
            await asyncio.wait_for(self._recorder_queue.join(), timeout=timeout_s)
            await self.flush_recorder()

    async def _ingest_loop(self) -> None:
        while True:
            ev = await self._ingest_queue.get()
            try:
                if ev is None:
                    return
                await self._process_ingested_event(ev)
            finally:
                self._ingest_queue.task_done()

    async def _recover_from_no_transcript(self, reason: str) -> None:
        snap = self._state.snapshot()
        await self._retire_uncovered_segments(snap, ReasonCode.EXPIRED_UNEVALUATED)
        if self._mode is Mode.PAUSED:
            self._set_health(HealthState.PAUSED, "paused")
        elif self._provider is not None and self._provider.breaker_state is BreakerState.OPEN:
            self._set_health(HealthState.TEMPLATE_ONLY, "generator_breaker_open")
        elif self._consecutive_clf_errors >= self._classifier_error_threshold:
            self._set_health(HealthState.MANUAL_ONLY, "classifier_errors")
        else:
            self._set_health(HealthState.OK, reason)
        self._proactive_wake.set()

    async def _process_ingested_event(self, ev: CanonicalEvent) -> ApplyResult:
        async with self._state_lock:
            if self._mode is Mode.OFF_THE_RECORD and ev.kind is not EventKind.CONTROL:
                return ApplyResult(
                    accepted=False,
                    state_version=self._state.state_version,
                    material_change=False,
                    drop_reason="off_the_record",
                )

            if ev.kind is EventKind.CONTROL and ev.subtype is ControlSubtype.OFF_RECORD_START:
                if self._mode is not Mode.OFF_THE_RECORD:
                    self._pre_otr_mode = self._mode
                    await self._retire_uncovered_segments(
                        self._state.snapshot(), ReasonCode.OFF_RECORD
                    )
                    await self._close_open_items_for_halt()
                    self._mode = Mode.OFF_THE_RECORD
                res = self._state.apply(ev)
                snap = self._state.snapshot()
                if res.accepted:
                    self._enqueue_recorder(
                        "canonical_event",
                        event=ev,
                        retention=snap.brief.retention,
                        canonical_revision=None,
                    )
                self._recipient_speaking = False
                self._schedule_ui_broadcast()
                self._display_wake.set()
                return res

            if ev.kind is EventKind.CONTROL and ev.subtype is ControlSubtype.OFF_RECORD_END:
                res = self._state.apply(ev)
                snap = self._state.snapshot()
                if res.accepted:
                    self._enqueue_recorder(
                        "canonical_event",
                        event=ev,
                        retention=snap.brief.retention,
                        canonical_revision=None,
                    )
                self._mode = self._pre_otr_mode
                self._last_transcript_event_ms = max(
                    self._last_transcript_event_ms, ev.received_ms, self._clock.now_ms()
                )
                self._last_gap_ms = ev.received_ms
                self._schedule_ui_broadcast()
                self._proactive_wake.set()
                self._display_wake.set()
                return res

            if ev.kind in {EventKind.SEGMENT, EventKind.SPEAKER_UPDATE, EventKind.BOUNDARY}:
                self._last_transcript_event_ms = max(
                    self._last_transcript_event_ms, ev.received_ms, self._clock.now_ms()
                )
                if self._health is HealthState.NO_TRANSCRIPT:
                    await self._recover_from_no_transcript("transcript_resumed")

            res = self._state.apply(ev)
            snap = self._state.snapshot()
            if res.accepted:
                can_rev = (
                    snap.segments_by_id[ev.segment_id].revision
                    if ev.segment_id and ev.segment_id in snap.segments_by_id
                    else None
                )
                self._enqueue_recorder(
                    "canonical_event",
                    event=ev,
                    retention=snap.brief.retention,
                    canonical_revision=can_rev,
                )
                self._update_ingest_signals(ev, res, snap)
            return res

    def _update_ingest_signals(
        self, ev: CanonicalEvent, res: ApplyResult, snap: StateSnapshot
    ) -> None:
        recipient_ids = {"ME", snap.brief.recipient.participant_id}
        if ev.kind is EventKind.CONTROL and ev.subtype in {
            ControlSubtype.GAP,
            ControlSubtype.RECONNECT,
        }:
            self._last_gap_ms = ev.received_ms
            self._recipient_speaking = False
        elif ev.kind is EventKind.BOUNDARY:
            if ev.speaker_id is not None and ev.end_ms is not None:
                prev_b = self._speaker_boundary_end_ms.get(ev.speaker_id, 0)
                if ev.end_ms > prev_b:
                    self._speaker_boundary_end_ms[ev.speaker_id] = ev.end_ms
            if ev.speaker_id is not None:
                prev_rx = self._speaker_boundary_rx_ms.get(ev.speaker_id, 0)
                if ev.received_ms > prev_rx:
                    self._speaker_boundary_rx_ms[ev.speaker_id] = ev.received_ms
            if ev.speaker_id is None or ev.speaker_id in recipient_ids:
                self._recipient_speaking = False
        elif ev.kind is EventKind.SPEAKER_UPDATE and ev.segment_id:
            self._speaker_update_times[ev.segment_id] = ev.received_ms
        elif ev.kind is EventKind.SEGMENT and ev.speaker_id in recipient_ids:
            if ev.stability is Stability.INTERIM:
                self._recipient_speaking = True
            elif ev.stability is Stability.COMMITTED and ev.segment_id:
                self._recipient_speaking_until_ms = max(
                    self._recipient_speaking_until_ms, ev.received_ms + 1400
                )
                self._display_wake.set()
                seg = snap.segments_by_id.get(ev.segment_id)
                if seg is not None:
                    self._check_used_inferred(seg)
        if ev.kind is EventKind.BRIEF_UPDATE:
            self._brief = snap.brief
        if res.material_change:
            if res.affected_segment_id is not None:
                self._cursor.note_material_segment(res.affected_segment_id)
            if self._mode in {Mode.PROACTIVE, Mode.QUIET, Mode.SHADOW} and self._health not in {
                HealthState.NO_TRANSCRIPT,
                HealthState.PAUSED,
                HealthState.MANUAL_ONLY,
            }:
                self._proactive_wake.set()
        if res.invalidations or ev.kind in {
            EventKind.SEGMENT,
            EventKind.BOUNDARY,
            EventKind.SPEAKER_UPDATE,
            EventKind.BRIEF_UPDATE,
            EventKind.CONTROL,
        }:
            self._display_wake.set()

    def _check_used_inferred(self, seg: SegmentRecord) -> None:
        cand_to_check: SuggestionCandidate | None = None
        if self._active is not None and not self._active.shadow:
            cand_to_check = self._active.cand
        elif self._last_displayed_card_info is not None:
            prev_cand, disp_ms = self._last_displayed_card_info
            if (self._clock.now_ms() - disp_ms) <= 30_000:
                cand_to_check = prev_cand
        if cand_to_check is None:
            return
        key = (cand_to_check.suggestion_id, seg.segment_id)
        if key in self._used_inferred_sent:
            return
        if _is_equivalent_question(seg.text, cand_to_check.text, cand_to_check.category):
            self._used_inferred_sent.add(key)
            self._emit_tel(
                level=TelemetryLevel.LIFECYCLE,
                kind="used_inferred",
                stage="display",
                candidate_id=cand_to_check.candidate_id,
                suggestion_id=cand_to_check.suggestion_id,
                payload={"segment_id": seg.segment_id, "revision": seg.revision},
            )

    def _local_only_blocked(self) -> bool:
        return bool(
            self._provider is not None
            and self._provider.config.local_only
            and self._provider.config.effective_is_hosted
        )

    def _half_open_probe_ready(self) -> bool:
        return bool(
            self._health is HealthState.TEMPLATE_ONLY
            and self._provider is not None
            and not self._local_only_blocked()
            and self._provider.half_open_probe_available
            and not self._half_open_probe_in_flight
        )

    def _translate_classifier_deadline(self, ctrl_deadline_ms: int) -> int:
        remaining_ms = ctrl_deadline_ms - self._clock.now_ms()
        clf_clock = getattr(self._classifier, "clock_ms", None)
        if callable(clf_clock) and clf_clock != self._clock.now_ms:
            return int(clf_clock()) + remaining_ms
        return ctrl_deadline_ms

    def _is_stopping(self) -> bool:
        return self._stopping

    async def _proactive_loop(self) -> None:
        while not self._is_stopping():
            await self._proactive_wake.wait()
            self._proactive_wake.clear()
            if self._is_stopping():
                return
            self._proactive_idle.clear()
            try:
                more_pending = await self._run_proactive_once()
                if more_pending and not self._is_stopping():
                    self._proactive_wake.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit_tel(
                    level=TelemetryLevel.LIFECYCLE,
                    kind="proactive_error",
                    stage="proactive_lane",
                    payload={"error": str(exc)},
                )
            finally:
                self._proactive_idle.set()
                if self._health is HealthState.NO_TRANSCRIPT:
                    self._display_wake.set()

    def _retire_anchor(
        self,
        anchor: Any,
        *,
        state_version: int,
        reason: ReasonCode,
    ) -> None:
        dummy_ctx = DecisionContext(
            schema_version=SCHEMA_VERSION,
            candidate_id=anchor.candidate_id,
            session_id=anchor.session_id,
            origin=Origin.PROACTIVE,
            based_on_state_version=state_version,
            builder_version=self._policy.classifier.version,
            prefilter_version="pre-v0",
            anchor_refs=anchor.anchor_refs,
            context_refs=(),
            anchor_received_ms=anchor.anchor_received_ms,
            display_by_ms=anchor.display_by_ms,
            token_counts={"total": 0},
            omitted={"turns": 0, "suggestions": 0, "known_facts": 0, "gaps": 0},
            flags={},
            serialized_sha256="retired",
        )
        self._enqueue_recorder(
            "decision",
            context=dummy_ctx,
            terminal_state="retired",
            terminal_reason=reason,
            retention=self._brief.retention,
        )
        self._emit_tel(
            level=TelemetryLevel.DECISION,
            kind="candidate_retired",
            stage="proactive_lane",
            candidate_id=anchor.candidate_id,
            reason_code=reason,
        )

    async def _run_proactive_once(self) -> bool:
        if self._mode not in {Mode.PROACTIVE, Mode.QUIET, Mode.SHADOW}:
            return False
        if self._health in {
            HealthState.NO_TRANSCRIPT,
            HealthState.PAUSED,
            HealthState.MANUAL_ONLY,
        }:
            return False

        async with self._state_lock:
            snap = self._state.snapshot()
            unprocessed = self._cursor.unprocessed_segments(snap)
            if not unprocessed:
                return False
            anchors = coalesce_anchors(
                snap,
                max_candidate_age_ms=self._policy.display.max_candidate_age_ms,
                tokenizer=self._classifier.tokenizer,
                speaker_boundary_end_ms=self._speaker_boundary_end_ms,
                segments_subset=unprocessed,
            )
            now_ms = self._clock.now_ms()
            live_anchors = []
            for anchor in anchors:
                if anchor.display_by_ms <= now_ms:
                    self._retire_anchor(
                        anchor,
                        state_version=snap.state_version,
                        reason=ReasonCode.EXPIRED_UNEVALUATED,
                    )
                    self._cursor.mark_covered(anchor.segments)
                else:
                    live_anchors.append(anchor)

            if not live_anchors:
                return False

            max_per_job = self._policy.classifier.max_candidates_per_job
            has_overflow = len(live_anchors) > max_per_job
            selected = live_anchors[-max_per_job:]

            batch_prev_norm = self._prev_anchor_norm_text
            batch_prev_seg_ids = set(self._prev_anchor_seg_revs)
            batch_prev_rx_ms = self._prev_anchor_rx_ms
            norm_by_cand: dict[str, tuple[str, dict[str, int], int]] = {}
            surviving: list[ContextBuildResult] = []
            for anchor in selected:
                seg_ids = {s.segment_id for s in anchor.segments}
                seg_revs = {s.segment_id: s.revision for s in anchor.segments}
                same_seg_rev = bool(seg_ids & batch_prev_seg_ids)
                gap_exceeded = (
                    batch_prev_rx_ms is not None
                    and (anchor.anchor_received_ms - batch_prev_rx_ms) > 30_000
                )
                eff_prev = None if (same_seg_rev or gap_exceeded) else batch_prev_norm
                pf = run_prefilter(
                    anchor,
                    state_version=snap.state_version,
                    previous_anchor_norm_text=eff_prev,
                )
                batch_prev_norm = pf.normalized_text
                batch_prev_seg_ids = seg_ids
                batch_prev_rx_ms = anchor.anchor_received_ms
                if not pf.passed and pf.drop_record is not None:
                    self._prev_anchor_norm_text = pf.normalized_text
                    self._prev_anchor_seg_revs = seg_revs
                    self._prev_anchor_rx_ms = anchor.anchor_received_ms
                    drop = pf.drop_record
                    dummy_ctx = DecisionContext(
                        schema_version=SCHEMA_VERSION,
                        candidate_id=drop.candidate_id,
                        session_id=drop.session_id,
                        origin=Origin.PROACTIVE,
                        based_on_state_version=drop.based_on_state_version,
                        builder_version=self._policy.classifier.version,
                        prefilter_version=drop.prefilter_version,
                        anchor_refs=drop.anchor_refs,
                        context_refs=(),
                        anchor_received_ms=drop.anchor_received_ms,
                        display_by_ms=drop.display_by_ms,
                        token_counts={"total": 0},
                        omitted={"turns": 0, "suggestions": 0, "known_facts": 0, "gaps": 0},
                        flags={},
                        serialized_sha256="dropped",
                    )
                    self._enqueue_recorder(
                        "decision",
                        context=dummy_ctx,
                        terminal_state="dropped",
                        terminal_reason=drop.reason_code,
                        retention=snap.brief.retention,
                    )
                    self._cursor.mark_covered(anchor.segments)
                else:
                    built = build_decision_context(
                        snap,
                        anchor,
                        self._policy,
                        origin=Origin.PROACTIVE,
                        tokenizer=self._classifier.tokenizer,
                        suggestion_catalog=self._suggestions_by_id,
                    )
                    if built.context is not None:
                        norm_by_cand[built.context.candidate_id] = (
                            pf.normalized_text,
                            seg_revs,
                            anchor.anchor_received_ms,
                        )
                    surviving.append(built)

        if not surviving:
            return has_overflow

        c_inputs = [ClassifierInput.from_build_result(b) for b in surviving]
        ctrl_deadline_ms = min(c.context.display_by_ms for c in c_inputs)
        deadline_ms = self._translate_classifier_deadline(ctrl_deadline_ms)
        clf_results = await self._classifier.classify_batch(c_inputs, deadline_ms=deadline_ms)

        all_err = all(isinstance(r, ClassifierErrorResult) for r in clf_results)
        if all_err:
            self._consecutive_clf_errors += 1
            if self._consecutive_clf_errors >= self._classifier_error_threshold:
                self._set_health(HealthState.MANUAL_ONLY, "classifier_errors")
        else:
            self._consecutive_clf_errors = 0

        pending_gen: list[
            tuple[
                DecisionContext,
                ClassifierResult,
                RoutingDecision,
                StateSnapshot,
                str,
                str,
                dict[str, str],
                dict[str, Category],
                bool,
            ]
        ] = []
        retry_pending = False
        async with self._state_lock:
            current_snap = self._state.snapshot()
            recipient_ids = {"ME", current_snap.brief.recipient.participant_id}
            probe_ready = self._half_open_probe_ready()
            probe_reserved = False
            for built, clf_raw in zip(surviving, clf_results, strict=True):
                assert built.context is not None and built.serialized_text is not None
                ctx = built.context
                if self._calibration is not None and not isinstance(clf_raw, ClassifierErrorResult):
                    if isinstance(self._calibration, CalibrationRegistry):
                        cal_ver = (
                            self._policy.classifier.calibration
                            if self._policy.classifier.calibration
                            and clf_raw.classifier_version == self._policy.classifier.version
                            else None
                        )
                        clf_res = self._calibration.calibrate(
                            ctx,
                            clf_raw,
                            mode=current_snap.brief.mode,
                            calibration_version=cal_ver,
                        )
                    else:
                        clf_res = self._calibration.calibrate(
                            ctx, clf_raw, mode=current_snap.brief.mode
                        )
                else:
                    clf_res = clf_raw

                stale = any(
                    (cur_seg := current_snap.segments_by_id.get(ref.segment_id)) is None
                    or cur_seg.revision != ref.revision
                    or cur_seg.speaker_id in recipient_ids
                    for ref in ctx.anchor_refs
                )
                if (
                    not stale
                    and isinstance(clf_raw, ClassifierErrorResult)
                    and clf_raw.error in {"timeout", "deadline_exceeded"}
                ):
                    if (
                        clf_raw.error == "timeout"
                        and self._consecutive_clf_errors < self._classifier_error_threshold
                        and self._clock.now_ms() < ctx.display_by_ms
                        and self.health is not HealthState.MANUAL_ONLY
                    ):
                        retry_pending = True
                        continue
                    if (
                        self._clock.now_ms() >= ctx.display_by_ms
                        or clf_raw.error == "deadline_exceeded"
                        or self.health is HealthState.MANUAL_ONLY
                    ):
                        self._retire_anchor(
                            built.anchor,
                            state_version=current_snap.state_version,
                            reason=ReasonCode.EXPIRED_UNEVALUATED,
                        )
                        self._cursor.mark_covered(built.anchor.segments)
                        if ctx.candidate_id in norm_by_cand:
                            p_norm, p_revs, p_rx = norm_by_cand[ctx.candidate_id]
                            self._prev_anchor_norm_text = p_norm
                            self._prev_anchor_seg_revs = p_revs
                            self._prev_anchor_rx_ms = p_rx
                        continue
                if ctx.candidate_id in norm_by_cand:
                    p_norm, p_revs, p_rx = norm_by_cand[ctx.candidate_id]
                    self._prev_anchor_norm_text = p_norm
                    self._prev_anchor_seg_revs = p_revs
                    self._prev_anchor_rx_ms = p_rx
                eff_health = (
                    HealthState.OK if (probe_ready and not probe_reserved) else self._health
                )
                routing = route_candidate(
                    ctx,
                    clf_res,
                    current_snap,
                    self._policy,
                    mode=self._mode,
                    health=eff_health,
                    now_ms=self._clock.now_ms(),
                    speaker_update_times_ms=self._speaker_update_times,
                    ledger_texts=self._ledger_texts,
                    ledger_categories=self._ledger_categories,
                    prior_anchor_texts=self._prior_anchor_texts,
                    recent_displayed_ms=list(self._proactive_display_times),
                )
                if stale:
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="withdrawn",
                        terminal_reason=ReasonCode.EVIDENCE_REVISION_CHANGED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing, terminal_reason=ReasonCode.EVIDENCE_REVISION_CHANGED
                        ),
                        serialized_text=built.serialized_text,
                        retention=current_snap.brief.retention,
                    )
                    unchanged = [
                        s
                        for s in built.anchor.segments
                        if (cur := current_snap.segments_by_id.get(s.segment_id)) is not None
                        and cur.revision == s.revision
                    ]
                    if unchanged:
                        self._cursor.mark_covered(unchanged)
                    continue

                self._cursor.mark_covered(built.anchor.segments)
                if routing.outcome is not Outcome.ROUTE or routing.category is None:
                    self._record_non_routed_proactive(
                        ctx, clf_res, routing, current_snap, built.serialized_text
                    )
                    continue
                if routing.route is Route.GENERATOR:
                    manual_busy = self._manual_in_flight or (
                        self._provider is not None and self._provider.manual_waiting > 0
                    )
                    if self._provider is None or manual_busy:
                        cand, eff_rt, fail_r = self._render_fallback_sync(
                            ctx, routing, current_snap
                        )
                        self._finalize_proactive_render(
                            cand,
                            ctx,
                            clf_res,
                            eff_rt,
                            fail_r,
                            current_snap,
                            built.serialized_text,
                            built.anchor.joined_text,
                        )
                    else:
                        is_probe = probe_ready and not probe_reserved
                        if is_probe:
                            probe_reserved = True
                            self._half_open_probe_in_flight = True
                        pending_gen.append(
                            (
                                ctx,
                                clf_res,
                                routing,
                                current_snap,
                                built.serialized_text,
                                built.anchor.joined_text,
                                dict(self._ledger_texts),
                                dict(self._ledger_categories),
                                is_probe,
                            )
                        )
                else:
                    cand, eff_rt, fail_r = self._render_template_sync(ctx, routing, current_snap)
                    self._finalize_proactive_render(
                        cand,
                        ctx,
                        clf_res,
                        eff_rt,
                        fail_r,
                        current_snap,
                        built.serialized_text,
                        built.anchor.joined_text,
                    )

        for (
            ctx,
            clf_res,
            routing,
            snap_at_route,
            ser_text,
            anc_text,
            l_texts,
            l_cats,
            is_probe,
        ) in pending_gen:
            await self._run_proactive_generator_unlocked(
                ctx=ctx,
                clf_res=clf_res,
                routing=routing,
                snap_at_route=snap_at_route,
                serialized_text=ser_text,
                anchor_text=anc_text,
                ledger_texts_copy=l_texts,
                ledger_cats_copy=l_cats,
                is_probe=is_probe,
            )
        return has_overflow or retry_pending

    def _record_gen_latency(self, elapsed_ms: int) -> None:
        self._gen_latencies_ms.append(max(0, elapsed_ms))
        if len(self._gen_latencies_ms) >= 5:
            ordered = sorted(self._gen_latencies_ms)
            idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
            self._gen_p95_ms = ordered[idx]

    def _render_fallback_sync(
        self,
        ctx: DecisionContext,
        routing: RoutingDecision,
        snap: StateSnapshot,
    ) -> tuple[SuggestionCandidate | None, RoutingDecision, ReasonCode | None]:
        assert routing.category is not None
        now_ms = self._clock.now_ms()
        fb = render_generic_fallback(
            ctx,
            routing.category,
            snap,
            self._policy,
            now_ms=now_ms,
            templates=self._templates,
            ready_ms=now_ms,
            origin=ctx.origin,
        )
        if fb.candidate is not None:
            return (fb.candidate, replace(routing, route=Route.TEMPLATE), None)
        return (None, routing, ReasonCode.INSUFFICIENT_BUDGET)

    def _render_template_sync(
        self,
        ctx: DecisionContext,
        routing: RoutingDecision,
        snap: StateSnapshot,
    ) -> tuple[SuggestionCandidate | None, RoutingDecision, ReasonCode | None]:
        assert routing.category is not None
        cat = routing.category
        cpol = self._policy.categories[cat]
        tpl = self._templates.get(cpol.template) if cpol.template else None
        if tpl is None:
            return (None, routing, ReasonCode.SLOT_UNAVAILABLE)
        now_ms = self._clock.now_ms()
        rendered = render_template(
            ctx,
            cat,
            snap,
            tpl,
            now_ms=now_ms,
            ready_ms=now_ms,
            display_ttl_ms=self._policy.display.display_ttl_ms,
            origin=ctx.origin,
        )
        if rendered.candidate is None:
            return (None, routing, rendered.unrendered_reason or ReasonCode.SLOT_UNAVAILABLE)
        val = validate_template_candidate(
            rendered.candidate,
            snap,
            max_words=tpl.max_words,
            ledger_texts=self._ledger_texts,
            ledger_categories=self._ledger_categories,
        )
        if not val.valid:
            return (None, routing, val.reason_code or ReasonCode.SLOT_UNAVAILABLE)
        return (rendered.candidate, routing, None)

    def _record_non_routed_proactive(
        self,
        ctx: DecisionContext,
        clf_res: ClassifierResult,
        routing: RoutingDecision,
        snap: StateSnapshot,
        serialized_text: str,
    ) -> None:
        state_map = {
            Outcome.NONE: "no_opportunity",
            Outcome.ABSTAIN: "abstained",
            Outcome.SUPPRESS: "suppressed",
        }
        t_state = state_map.get(routing.outcome, "abstained")
        t_reason = routing.reason_codes[0] if routing.reason_codes else ReasonCode.BELOW_T_LOW
        self._enqueue_recorder(
            "decision",
            context=ctx,
            terminal_state=t_state,
            terminal_reason=t_reason,
            classifier=clf_res,
            routing=routing,
            serialized_text=serialized_text,
            retention=snap.brief.retention,
        )

    def _finalize_proactive_render(
        self,
        cand: SuggestionCandidate | None,
        ctx: DecisionContext,
        clf_res: ClassifierResult,
        eff_routing: RoutingDecision,
        fail_reason: ReasonCode | None,
        snap: StateSnapshot,
        serialized_text: str,
        anchor_text: str,
    ) -> None:
        if cand is None:
            t_reason = fail_reason or ReasonCode.SLOT_UNAVAILABLE
            t_state = "suppressed" if t_reason in SUPPRESS_REASON_CODES else "unrendered"
            if t_state == "suppressed":
                eff_routing = replace(
                    eff_routing,
                    outcome=Outcome.SUPPRESS,
                    route=None,
                    reason_codes=(t_reason,),
                )
            self._enqueue_recorder(
                "decision",
                context=ctx,
                terminal_state=t_state,
                terminal_reason=t_reason,
                classifier=clf_res,
                routing=eff_routing,
                serialized_text=serialized_text,
                retention=snap.brief.retention,
            )
            return

        self._state.register_suggestion(cand)
        self._suggestions_by_id[cand.suggestion_id] = cand
        self._enqueue_recorder(
            "suggestion",
            suggestion=cand,
            current_state="ready",
            retention=snap.brief.retention,
        )
        score = 0.0
        if clf_res.calibrated_scores is not None:
            score = float(clf_res.calibrated_scores.categories.get(cand.category.value, 0.0))

        anchor_segs = [
            snap.segments_by_id[r.segment_id]
            for r in ctx.anchor_refs
            if r.segment_id in snap.segments_by_id
        ]
        last_key = (
            max(CoverageCursor.seg_key(s) for s in anchor_segs) if anchor_segs else (0, 0, 0, "")
        )
        anchor_start = min((s.start_ms for s in anchor_segs), default=ctx.anchor_received_ms)
        anchor_spk = anchor_segs[0].speaker_id if anchor_segs else "S?"
        settle_end = ctx.anchor_received_ms + self._policy.display.settle_ms
        item = _PendingItem(
            cand=cand,
            context=ctx,
            classifier=clf_res,
            routing=eff_routing,
            serialized_text=serialized_text,
            score=score,
            brief_snapshot=snap.brief,
            anchor_text=anchor_text,
            anchor_speaker_id=anchor_spk,
            anchor_last_key=last_key,
            anchor_start_ms=anchor_start,
            anchor_received_ms=ctx.anchor_received_ms,
            settle_end_ms=settle_end,
        )
        self._settling.append(item)
        self._display_wake.set()

    async def _run_proactive_generator_unlocked(
        self,
        *,
        ctx: DecisionContext,
        clf_res: ClassifierResult,
        routing: RoutingDecision,
        snap_at_route: StateSnapshot,
        serialized_text: str,
        anchor_text: str,
        ledger_texts_copy: dict[str, str],
        ledger_cats_copy: dict[str, Category],
        is_probe: bool,
    ) -> None:
        assert routing.category is not None
        assert self._provider is not None
        try:
            async with self._state_lock:
                pre_snap = self._state.snapshot()
                recipient_ids = {"ME", pre_snap.brief.recipient.participant_id}
                pre_stale = any(
                    (cur_seg := pre_snap.segments_by_id.get(ref.segment_id)) is None
                    or cur_seg.revision != ref.revision
                    or cur_seg.speaker_id in recipient_ids
                    for ref in ctx.anchor_refs
                )
                if pre_stale:
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="withdrawn",
                        terminal_reason=ReasonCode.EVIDENCE_REVISION_CHANGED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing, terminal_reason=ReasonCode.EVIDENCE_REVISION_CHANGED
                        ),
                        serialized_text=serialized_text,
                        retention=pre_snap.brief.retention,
                    )
                    return
                if _brief_material_key(pre_snap.brief) != _brief_material_key(snap_at_route.brief):
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="withdrawn",
                        terminal_reason=ReasonCode.OBJECTIVE_CHANGED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing, terminal_reason=ReasonCode.OBJECTIVE_CHANGED
                        ),
                        serialized_text=serialized_text,
                        retention=pre_snap.brief.retention,
                    )
                    return
                if self._mode in {Mode.PAUSED, Mode.OFF_THE_RECORD} or pre_snap.off_record:
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="suppressed",
                        terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing, terminal_reason=ReasonCode.PROACTIVE_DISABLED
                        ),
                        serialized_text=serialized_text,
                        retention=pre_snap.brief.retention,
                    )
                    return
                if (
                    (self._last_gap_ms is not None and ctx.anchor_received_ms < self._last_gap_ms)
                    or len(pre_snap.gaps) > len(snap_at_route.gaps)
                    or self._clock.now_ms() >= ctx.display_by_ms
                ):
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="expired",
                        terminal_reason=ReasonCode.DISPLAY_BY_PASSED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing, terminal_reason=ReasonCode.DISPLAY_BY_PASSED
                        ),
                        serialized_text=serialized_text,
                        retention=pre_snap.brief.retention,
                    )
                    return
                manual_busy = self._manual_in_flight or self._provider.manual_waiting > 0
                if manual_busy or self._is_stopping():
                    cand, eff_rt, fail_r = self._render_fallback_sync(ctx, routing, pre_snap)
                    self._finalize_proactive_render(
                        cand, ctx, clf_res, eff_rt, fail_r, pre_snap, serialized_text, anchor_text
                    )
                    return

            t0 = self._clock.now_ms()
            gen_out = await generate_question(
                ctx,
                routing.category,
                snap_at_route,
                self._policy,
                self._provider,
                clock_ms=self._clock.now_ms,
                store=self._store,
                templates=self._templates,
                stage_p95_ms=self._gen_p95_ms,
                ledger_texts=ledger_texts_copy,
                ledger_categories=ledger_cats_copy,
            )
            elapsed_ms = self._clock.now_ms() - t0
            async with self._state_lock:
                self._record_gen_latency(elapsed_ms)
                if self._provider.breaker_state is BreakerState.OPEN:
                    self._set_health(HealthState.TEMPLATE_ONLY, "generator_breaker_open")
                fresh_snap = self._state.snapshot()
                recipient_ids = {"ME", fresh_snap.brief.recipient.participant_id}
                stale_rev = any(
                    (cur_seg := fresh_snap.segments_by_id.get(ref.segment_id)) is None
                    or cur_seg.revision != ref.revision
                    or cur_seg.speaker_id in recipient_ids
                    for ref in ctx.anchor_refs
                )
                if not stale_rev and gen_out.candidate is not None:
                    stale_rev = any(
                        (cur_seg := fresh_snap.segments_by_id.get(ref.segment_id)) is None
                        or cur_seg.revision != ref.revision
                        for ref in gen_out.candidate.evidence_refs
                    )
                if stale_rev:
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="withdrawn",
                        terminal_reason=ReasonCode.EVIDENCE_REVISION_CHANGED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing,
                            egress_id=gen_out.egress_id,
                            terminal_reason=ReasonCode.EVIDENCE_REVISION_CHANGED,
                        ),
                        serialized_text=serialized_text,
                        retention=fresh_snap.brief.retention,
                    )
                    return
                if _brief_material_key(fresh_snap.brief) != _brief_material_key(
                    snap_at_route.brief
                ):
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="withdrawn",
                        terminal_reason=ReasonCode.OBJECTIVE_CHANGED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing,
                            egress_id=gen_out.egress_id,
                            terminal_reason=ReasonCode.OBJECTIVE_CHANGED,
                        ),
                        serialized_text=serialized_text,
                        retention=fresh_snap.brief.retention,
                    )
                    return
                if self._mode in {Mode.PAUSED, Mode.OFF_THE_RECORD} or fresh_snap.off_record:
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="suppressed",
                        terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing,
                            egress_id=gen_out.egress_id,
                            terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                        ),
                        serialized_text=serialized_text,
                        retention=fresh_snap.brief.retention,
                    )
                    return
                if (
                    (self._last_gap_ms is not None and ctx.anchor_received_ms < self._last_gap_ms)
                    or len(fresh_snap.gaps) > len(snap_at_route.gaps)
                    or self._clock.now_ms() >= ctx.display_by_ms
                ):
                    self._enqueue_recorder(
                        "decision",
                        context=ctx,
                        terminal_state="expired",
                        terminal_reason=ReasonCode.DISPLAY_BY_PASSED,
                        classifier=clf_res,
                        routing=_downgrade_unexecuted_generator_routing(
                            routing,
                            egress_id=gen_out.egress_id,
                            terminal_reason=ReasonCode.DISPLAY_BY_PASSED,
                        ),
                        serialized_text=serialized_text,
                        retention=fresh_snap.brief.retention,
                    )
                    return
                if gen_out.candidate is None:
                    t_reason = gen_out.terminal_reason or ReasonCode.GENERATOR_INVALID_OUTPUT
                    eff_rt = (
                        routing
                        if t_reason is ReasonCode.GENERATOR_TIMEOUT
                        else _downgrade_unexecuted_generator_routing(
                            routing, egress_id=gen_out.egress_id, terminal_reason=t_reason
                        )
                    )
                    self._finalize_proactive_render(
                        None,
                        ctx,
                        clf_res,
                        eff_rt,
                        t_reason,
                        fresh_snap,
                        serialized_text,
                        anchor_text,
                    )
                    return
                eff_routing = (
                    replace(routing, route=gen_out.candidate.route)
                    if gen_out.fallback_used
                    else routing
                )
                self._finalize_proactive_render(
                    gen_out.candidate,
                    ctx,
                    clf_res,
                    eff_routing,
                    None,
                    fresh_snap,
                    serialized_text,
                    anchor_text,
                )
        finally:
            if is_probe:
                self._half_open_probe_in_flight = False

    async def _manual_loop(self) -> None:
        while True:
            fut = await self._manual_queue.get()
            try:
                if fut is None:
                    return
                if fut.cancelled():
                    self._manual_in_flight = False
                    continue
                res = await self._execute_manual_request()
                if not fut.done():
                    fut.set_result(res)
            except asyncio.CancelledError:
                self._manual_in_flight = False
                if fut is not None and not fut.done():
                    fut.cancel()
                raise
            except Exception as exc:
                self._manual_in_flight = False
                if fut is not None and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._manual_queue.task_done()

    def _manual_unavailable_reason(self) -> ManualResult | None:
        if self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}:
            return ManualResult(
                status="unavailable",
                message=f"Manual requests unavailable in {self._mode.value} mode.",
            )
        if self._health is HealthState.NO_TRANSCRIPT:
            return ManualResult(
                status="unavailable",
                message="Manual requests unavailable while no_transcript.",
            )
        return None

    async def trigger_manual(self) -> ManualResult:
        unavail = self._manual_unavailable_reason()
        if unavail is not None:
            self._manual_status = {"status": "unavailable", "message": unavail.message}
            self._schedule_ui_broadcast()
            return unavail
        if self._manual_in_flight:
            self._manual_ignored_count += 1
            self._emit_tel(
                level=TelemetryLevel.LIFECYCLE,
                kind="manual_ignored_busy",
                stage="manual_lane",
            )
            return ManualResult(status="ignored_busy", message="Manual request already in flight.")

        self._manual_in_flight = True
        if self._started and not self._stopping:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[ManualResult] = loop.create_future()
            try:
                await self._manual_queue.put(fut)
            except BaseException:
                self._manual_in_flight = False
                raise
            return await fut
        return await self._execute_manual_request()

    async def _execute_manual_request(self) -> ManualResult:
        self._manual_in_flight = True
        try:
            now_ms = self._clock.now_ms()
            deadline_ms = now_ms + self._policy.manual.budget_ms
            async with asyncio.timeout(max(0.05, self._policy.manual.budget_ms / 1000.0)):
                return await self._run_manual_once(now_ms=now_ms, deadline_ms=deadline_ms)
        except TimeoutError:
            self._manual_status = {"status": "empty", "message": "Nothing useful to add."}
            self._schedule_ui_broadcast()
            return ManualResult(
                status="empty",
                message="Nothing useful to add.",
                reason=ReasonCode.INSUFFICIENT_BUDGET,
            )
        finally:
            self._manual_in_flight = False

    async def _run_manual_generator_unlocked(
        self,
        *,
        ctx: DecisionContext,
        category: Category,
        snap_at_call: StateSnapshot,
        ledger_texts_copy: dict[str, str],
        ledger_cats_copy: dict[str, Category],
        is_probe: bool,
    ) -> GenerationOutcome | None:
        if self._provider is None:
            return None
        if is_probe:
            self._half_open_probe_in_flight = True
        try:
            t0 = self._clock.now_ms()
            gen_out = await generate_question(
                ctx,
                category,
                snap_at_call,
                self._policy,
                self._provider,
                clock_ms=self._clock.now_ms,
                store=self._store,
                templates=self._templates,
                ledger_texts=ledger_texts_copy,
                ledger_categories=ledger_cats_copy,
            )
            elapsed_ms = self._clock.now_ms() - t0
            async with self._state_lock:
                self._record_gen_latency(elapsed_ms)
                if self._provider.breaker_state is BreakerState.OPEN:
                    self._set_health(HealthState.TEMPLATE_ONLY, "generator_breaker_open")
            return gen_out
        finally:
            if is_probe:
                self._half_open_probe_in_flight = False

    def _manual_empty_result(self, reason: ReasonCode | None = None) -> ManualResult:
        self._manual_status = {"status": "empty", "message": "Nothing useful to add."}
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind="manual_nothing_to_add",
            stage="manual_lane",
            reason_code=reason,
        )
        self._schedule_ui_broadcast()
        return ManualResult(
            status="empty",
            message="Nothing useful to add.",
            reason=reason,
        )

    def _check_manual_staleness_locked(
        self,
        *,
        ctx: DecisionContext,
        snap_at_call: StateSnapshot,
        fresh_snap: StateSnapshot,
        deadline_ms: int,
        cand: SuggestionCandidate | None = None,
    ) -> tuple[str, ReasonCode] | None:
        recipient_ids = {"ME", fresh_snap.brief.recipient.participant_id}
        stale_rev = any(
            (cur_seg := fresh_snap.segments_by_id.get(ref.segment_id)) is None
            or cur_seg.revision != ref.revision
            or cur_seg.speaker_id in recipient_ids
            for ref in ctx.anchor_refs
        )
        if not stale_rev and cand is not None:
            for ref in cand.evidence_refs:
                cur_seg = fresh_snap.segments_by_id.get(ref.segment_id)
                prev_seg = snap_at_call.segments_by_id.get(ref.segment_id)
                flipped_to_recipient = cur_seg is not None and (
                    cur_seg.speaker_id in recipient_ids
                    and (prev_seg is None or prev_seg.speaker_id not in recipient_ids)
                )
                if cur_seg is None or cur_seg.revision != ref.revision or flipped_to_recipient:
                    stale_rev = True
                    break
        if stale_rev:
            return ("withdrawn", ReasonCode.EVIDENCE_REVISION_CHANGED)
        if _brief_material_key(fresh_snap.brief) != _brief_material_key(snap_at_call.brief):
            return ("withdrawn", ReasonCode.OBJECTIVE_CHANGED)
        if (
            (self._last_gap_ms is not None and ctx.anchor_received_ms < self._last_gap_ms)
            or len(fresh_snap.gaps) > len(snap_at_call.gaps)
            or fresh_snap.off_record
            or self._clock.now_ms() >= deadline_ms
        ):
            return ("expired", ReasonCode.DISPLAY_BY_PASSED)
        if (
            cand is not None
            and check_do_not_suggest(cand.text, cand.category, fresh_snap.brief) is not None
        ):
            return ("suppressed", ReasonCode.DO_NOT_SUGGEST)
        return None

    async def _run_manual_once(self, *, now_ms: int, deadline_ms: int) -> ManualResult:
        async with self._state_lock:
            if self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD} or self._is_stopping():
                return ManualResult(
                    status="unavailable",
                    message=f"Manual requests unavailable in {self._mode.value} mode.",
                )
            snap = self._state.snapshot()
            suggestions_copy = dict(self._suggestions_by_id)
            boundary_end_copy = dict(self._speaker_boundary_end_ms)
            recipient_ids = {"ME", snap.brief.recipient.participant_id}
            cutoff = now_ms - self._policy.manual.lookback_s * 1000
            lookback_segs = [
                s
                for s in snap.segments
                if s.committed and s.speaker_id not in recipient_ids and s.received_ms >= cutoff
            ]

        if not lookback_segs:
            return self._manual_empty_result()

        snap_at_build = snap
        anchors = coalesce_anchors(
            snap,
            max_candidate_age_ms=self._policy.manual.budget_ms,
            tokenizer=self._classifier.tokenizer,
            speaker_boundary_end_ms=boundary_end_copy,
            segments_subset=lookback_segs,
        )
        built_pairs: list[tuple[ContextBuildResult, str]] = []
        for anchor in reversed(anchors):
            built = build_decision_context(
                snap,
                anchor,
                self._policy,
                origin=Origin.MANUAL,
                tokenizer=self._classifier.tokenizer,
                suggestion_catalog=suggestions_copy,
            )
            if built.context is not None:
                m_ctx = replace(
                    built.context,
                    candidate_id=f"manual-{uuid.uuid4().hex[:16]}",
                    display_by_ms=deadline_ms,
                )
                built = replace(built, context=m_ctx)
                built_pairs.append((built, anchor.joined_text))

        valid_built_pairs = list(built_pairs)
        brief_invalidated = False
        rejected_cand_ids: set[str] = set()
        recorded_manual_decision = False
        non_routed_manual: list[tuple[ContextBuildResult, ClassifierResult, RoutingDecision]] = []

        if built_pairs and self._health is not HealthState.MANUAL_ONLY:
            c_inputs = [ClassifierInput.from_build_result(b) for b, _ in built_pairs]
            clf_deadline_ms = self._translate_classifier_deadline(deadline_ms)
            clf_results = await self._classifier.classify_batch(
                c_inputs, deadline_ms=clf_deadline_ms
            )
            async with self._state_lock:
                if (
                    self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}
                    or self._is_stopping()
                ):
                    return ManualResult(
                        status="unavailable",
                        message="Manual request halted during mode change.",
                    )
                if self._clock.now_ms() >= deadline_ms:
                    return self._manual_empty_result(ReasonCode.INSUFFICIENT_BUDGET)
                cur_snap = self._state.snapshot()
                if _brief_material_key(cur_snap.brief) != _brief_material_key(snap_at_build.brief):
                    return self._manual_empty_result(ReasonCode.OBJECTIVE_CHANGED)
                probe_ready = self._half_open_probe_ready()
                eff_health = HealthState.OK if probe_ready else self._health
                fresh_valid_pairs: list[tuple[ContextBuildResult, str]] = []
                scored_cands: list[
                    tuple[float, ContextBuildResult, ClassifierResult, RoutingDecision, str]
                ] = []
                for (built, a_txt), clf_raw in zip(built_pairs, clf_results, strict=True):
                    assert built.context is not None and built.serialized_text is not None
                    ctx = built.context
                    if (
                        self._check_manual_staleness_locked(
                            ctx=ctx,
                            snap_at_call=snap_at_build,
                            fresh_snap=cur_snap,
                            deadline_ms=deadline_ms,
                        )
                        is not None
                    ):
                        rejected_cand_ids.add(ctx.candidate_id)
                        continue
                    fresh_valid_pairs.append((built, a_txt))
                    if self._calibration is not None and not isinstance(
                        clf_raw, ClassifierErrorResult
                    ):
                        if isinstance(self._calibration, CalibrationRegistry):
                            cal_ver = (
                                self._policy.classifier.calibration
                                if self._policy.classifier.calibration
                                and clf_raw.classifier_version == self._policy.classifier.version
                                else None
                            )
                            clf_res = self._calibration.calibrate(
                                ctx,
                                clf_raw,
                                mode=cur_snap.brief.mode,
                                calibration_version=cal_ver,
                            )
                        else:
                            clf_res = self._calibration.calibrate(
                                ctx, clf_raw, mode=cur_snap.brief.mode
                            )
                    else:
                        clf_res = clf_raw
                    routing = route_candidate(
                        ctx,
                        clf_res,
                        cur_snap,
                        self._policy,
                        mode=self._mode,
                        health=eff_health,
                        now_ms=self._clock.now_ms(),
                        speaker_update_times_ms=self._speaker_update_times,
                        ledger_texts=self._ledger_texts,
                        ledger_categories=self._ledger_categories,
                        prior_anchor_texts=self._prior_anchor_texts,
                    )
                    if routing.outcome is Outcome.ROUTE and routing.category is not None:
                        s_val = (
                            float(
                                clf_res.calibrated_scores.categories.get(
                                    routing.category.value, 0.0
                                )
                            )
                            if clf_res.calibrated_scores is not None
                            else float(
                                clf_res.raw_scores.categories.get(routing.category.value, 0.0)
                            )
                        )
                        scored_cands.append((s_val, built, clf_res, routing, a_txt))
                    else:
                        non_routed_manual.append((built, clf_res, routing))
                valid_built_pairs = fresh_valid_pairs
                scored_cands.sort(key=lambda x: x[0], reverse=True)

            probe_used = False
            for _sc, built, clf_res, routing, a_txt in scored_cands:
                assert built.context is not None and built.serialized_text is not None
                assert routing.category is not None
                if routing.route is Route.GENERATOR:
                    if self._provider is None or not self._policy.generator.enabled:
                        async with self._state_lock:
                            if (
                                self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}
                                or self._is_stopping()
                            ):
                                return ManualResult(
                                    status="unavailable",
                                    message="Manual request halted during mode change.",
                                )
                            fresh_snap = self._state.snapshot()
                            staleness = self._check_manual_staleness_locked(
                                ctx=built.context,
                                snap_at_call=cur_snap,
                                fresh_snap=fresh_snap,
                                deadline_ms=deadline_ms,
                            )
                            if staleness is not None:
                                t_state, t_reason = staleness
                                self._enqueue_recorder(
                                    "decision",
                                    context=built.context,
                                    terminal_state=t_state,
                                    terminal_reason=t_reason,
                                    classifier=clf_res,
                                    routing=_downgrade_unexecuted_generator_routing(
                                        routing, terminal_reason=t_reason
                                    ),
                                    serialized_text=built.serialized_text,
                                    retention=fresh_snap.brief.retention,
                                )
                                rejected_cand_ids.add(built.context.candidate_id)
                                if t_reason is ReasonCode.OBJECTIVE_CHANGED:
                                    brief_invalidated = True
                                if self._clock.now_ms() >= deadline_ms:
                                    break
                                continue
                            cand, eff_rt, fail_r = self._render_fallback_sync(
                                built.context, routing, fresh_snap
                            )
                            if cand is not None:
                                if (
                                    check_do_not_suggest(cand.text, cand.category, fresh_snap.brief)
                                    is not None
                                ):
                                    self._enqueue_recorder(
                                        "decision",
                                        context=built.context,
                                        terminal_state="suppressed",
                                        terminal_reason=ReasonCode.DO_NOT_SUGGEST,
                                        classifier=clf_res,
                                        routing=replace(
                                            eff_rt,
                                            outcome=Outcome.SUPPRESS,
                                            route=None,
                                            reason_codes=(ReasonCode.DO_NOT_SUGGEST,),
                                        ),
                                        serialized_text=built.serialized_text,
                                        retention=fresh_snap.brief.retention,
                                    )
                                    recorded_manual_decision = True
                                    rejected_cand_ids.add(built.context.candidate_id)
                                    continue
                                cand = replace(
                                    cand, origin=Origin.MANUAL, display_by_ms=deadline_ms
                                )
                                return await self._activate_manual_candidate(
                                    cand,
                                    built.context,
                                    clf_res,
                                    eff_rt,
                                    built.serialized_text,
                                    a_txt,
                                    brief_snapshot=fresh_snap.brief,
                                )
                            t_reason = fail_r or ReasonCode.INSUFFICIENT_BUDGET
                            t_state = (
                                "suppressed" if t_reason in SUPPRESS_REASON_CODES else "unrendered"
                            )
                            rec_rt = (
                                replace(
                                    eff_rt,
                                    outcome=Outcome.SUPPRESS,
                                    route=None,
                                    reason_codes=(t_reason,),
                                )
                                if t_state == "suppressed"
                                else _downgrade_unexecuted_generator_routing(
                                    eff_rt, terminal_reason=t_reason
                                )
                            )
                            self._enqueue_recorder(
                                "decision",
                                context=built.context,
                                terminal_state=t_state,
                                terminal_reason=t_reason,
                                classifier=clf_res,
                                routing=rec_rt,
                                serialized_text=built.serialized_text,
                                retention=fresh_snap.brief.retention,
                            )
                            recorded_manual_decision = True
                        continue
                    async with self._state_lock:
                        if (
                            self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}
                            or self._is_stopping()
                        ):
                            return ManualResult(
                                status="unavailable",
                                message="Manual request halted during mode change.",
                            )
                        pre_snap = self._state.snapshot()
                        pre_stale = self._check_manual_staleness_locked(
                            ctx=built.context,
                            snap_at_call=cur_snap,
                            fresh_snap=pre_snap,
                            deadline_ms=deadline_ms,
                        )
                        if pre_stale is not None:
                            t_state, t_reason = pre_stale
                            self._enqueue_recorder(
                                "decision",
                                context=built.context,
                                terminal_state=t_state,
                                terminal_reason=t_reason,
                                classifier=clf_res,
                                routing=_downgrade_unexecuted_generator_routing(
                                    routing, terminal_reason=t_reason
                                ),
                                serialized_text=built.serialized_text,
                                retention=pre_snap.brief.retention,
                            )
                            rejected_cand_ids.add(built.context.candidate_id)
                            if t_reason is ReasonCode.OBJECTIVE_CHANGED:
                                brief_invalidated = True
                            if self._clock.now_ms() >= deadline_ms:
                                break
                            continue
                        is_probe = self._half_open_probe_ready() and not probe_used
                        if is_probe:
                            probe_used = True
                        call_ledger_texts = dict(self._ledger_texts)
                        call_ledger_cats = dict(self._ledger_categories)
                    gen_out = await self._run_manual_generator_unlocked(
                        ctx=built.context,
                        category=routing.category,
                        snap_at_call=cur_snap,
                        ledger_texts_copy=call_ledger_texts,
                        ledger_cats_copy=call_ledger_cats,
                        is_probe=is_probe,
                    )
                    async with self._state_lock:
                        fresh_snap = self._state.snapshot()
                        gen_egress_id = gen_out.egress_id if gen_out is not None else None
                        cand = gen_out.candidate if gen_out is not None else None
                        if (
                            self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}
                            or self._is_stopping()
                        ):
                            self._enqueue_recorder(
                                "decision",
                                context=built.context,
                                terminal_state="suppressed",
                                terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                                classifier=clf_res,
                                routing=_downgrade_unexecuted_generator_routing(
                                    routing,
                                    egress_id=gen_egress_id,
                                    terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                                ),
                                serialized_text=built.serialized_text,
                                retention=fresh_snap.brief.retention,
                            )
                            return ManualResult(
                                status="unavailable",
                                message="Manual request halted during mode change.",
                            )
                        staleness = self._check_manual_staleness_locked(
                            ctx=built.context,
                            snap_at_call=cur_snap,
                            fresh_snap=fresh_snap,
                            deadline_ms=deadline_ms,
                            cand=cand,
                        )
                        if staleness is not None:
                            t_state, t_reason = staleness
                            self._enqueue_recorder(
                                "decision",
                                context=built.context,
                                terminal_state=t_state,
                                terminal_reason=t_reason,
                                classifier=clf_res,
                                routing=_downgrade_unexecuted_generator_routing(
                                    routing, egress_id=gen_egress_id, terminal_reason=t_reason
                                ),
                                serialized_text=built.serialized_text,
                                retention=fresh_snap.brief.retention,
                            )
                            rejected_cand_ids.add(built.context.candidate_id)
                            if t_reason is ReasonCode.OBJECTIVE_CHANGED:
                                brief_invalidated = True
                            if self._clock.now_ms() >= deadline_ms:
                                break
                            continue
                        if cand is None:
                            t_reason = (
                                gen_out.terminal_reason if gen_out is not None else None
                            ) or ReasonCode.GENERATOR_INVALID_OUTPUT
                            eff_rt = (
                                routing
                                if t_reason is ReasonCode.GENERATOR_TIMEOUT
                                else _downgrade_unexecuted_generator_routing(
                                    routing, egress_id=gen_egress_id, terminal_reason=t_reason
                                )
                            )
                            self._enqueue_recorder(
                                "decision",
                                context=built.context,
                                terminal_state="unrendered",
                                terminal_reason=t_reason,
                                classifier=clf_res,
                                routing=eff_rt,
                                serialized_text=built.serialized_text,
                                retention=fresh_snap.brief.retention,
                            )
                            continue
                        eff_rt = replace(routing, route=cand.route)
                        cand = replace(cand, origin=Origin.MANUAL, display_by_ms=deadline_ms)
                        return await self._activate_manual_candidate(
                            cand,
                            built.context,
                            clf_res,
                            eff_rt,
                            built.serialized_text,
                            a_txt,
                            brief_snapshot=fresh_snap.brief,
                        )
                else:
                    async with self._state_lock:
                        if (
                            self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}
                            or self._is_stopping()
                        ):
                            return ManualResult(
                                status="unavailable",
                                message="Manual request halted during mode change.",
                            )
                        fresh_snap = self._state.snapshot()
                        staleness = self._check_manual_staleness_locked(
                            ctx=built.context,
                            snap_at_call=cur_snap,
                            fresh_snap=fresh_snap,
                            deadline_ms=deadline_ms,
                        )
                        if staleness is not None:
                            t_state, t_reason = staleness
                            self._enqueue_recorder(
                                "decision",
                                context=built.context,
                                terminal_state=t_state,
                                terminal_reason=t_reason,
                                classifier=clf_res,
                                routing=routing,
                                serialized_text=built.serialized_text,
                                retention=fresh_snap.brief.retention,
                            )
                            rejected_cand_ids.add(built.context.candidate_id)
                            if t_reason is ReasonCode.OBJECTIVE_CHANGED:
                                brief_invalidated = True
                            if self._clock.now_ms() >= deadline_ms:
                                break
                            continue
                        cand, eff_rt, fail_r = self._render_template_sync(
                            built.context, routing, fresh_snap
                        )
                        if cand is not None:
                            cand = replace(cand, origin=Origin.MANUAL, display_by_ms=deadline_ms)
                            return await self._activate_manual_candidate(
                                cand,
                                built.context,
                                clf_res,
                                eff_rt,
                                built.serialized_text,
                                a_txt,
                                brief_snapshot=fresh_snap.brief,
                            )
                        t_reason = fail_r or ReasonCode.SLOT_UNAVAILABLE
                        t_state = (
                            "suppressed" if t_reason in SUPPRESS_REASON_CODES else "unrendered"
                        )
                        rec_rt = (
                            replace(
                                eff_rt,
                                outcome=Outcome.SUPPRESS,
                                route=None,
                                reason_codes=(t_reason,),
                            )
                            if t_state == "suppressed"
                            else eff_rt
                        )
                        self._enqueue_recorder(
                            "decision",
                            context=built.context,
                            terminal_state=t_state,
                            terminal_reason=t_reason,
                            classifier=clf_res,
                            routing=rec_rt,
                            serialized_text=built.serialized_text,
                            retention=fresh_snap.brief.retention,
                        )
                        recorded_manual_decision = True

        # Fallback to direct generator call on manual lane if enabled (§3.7, §8.6 manual_only)
        fb_candidate: tuple[ContextBuildResult, str] | None = None
        fb_snap: StateSnapshot | None = None
        allow_fb_probe = False
        fb_ledger_texts: dict[str, str] = {}
        fb_ledger_cats: dict[str, Category] = {}
        fb_empty_reason: ReasonCode | None = (
            ReasonCode.OBJECTIVE_CHANGED if brief_invalidated else None
        )
        if (
            not brief_invalidated
            and self._provider is not None
            and self._policy.generator.enabled
            and valid_built_pairs
            and self._policy.categories
        ):
            async with self._state_lock:
                if (
                    self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}
                    or self._is_stopping()
                ):
                    return ManualResult(
                        status="unavailable",
                        message="Manual request halted during mode change.",
                    )
                cur_snap = self._state.snapshot()
                allow_fb_probe = self._half_open_probe_ready()
                if self._health is not HealthState.TEMPLATE_ONLY or allow_fb_probe:
                    for b_cand, txt_cand in valid_built_pairs:
                        assert b_cand.context is not None
                        if b_cand.context.candidate_id in rejected_cand_ids:
                            continue
                        st = self._check_manual_staleness_locked(
                            ctx=b_cand.context,
                            snap_at_call=snap_at_build,
                            fresh_snap=cur_snap,
                            deadline_ms=deadline_ms,
                        )
                        if st is None:
                            fb_candidate = (b_cand, txt_cand)
                            fb_snap = cur_snap
                            fb_ledger_texts = dict(self._ledger_texts)
                            fb_ledger_cats = dict(self._ledger_categories)
                            break
                        fb_empty_reason = st[1]

        if fb_candidate is not None and fb_snap is not None:
            built0, a_txt0 = fb_candidate
            assert built0.context is not None and built0.serialized_text is not None
            m_cat = (
                Category.FOLLOW_UP
                if Category.FOLLOW_UP in self._policy.categories
                else next(iter(self._policy.categories))
            )
            gen_out_fb = await self._run_manual_generator_unlocked(
                ctx=built0.context,
                category=m_cat,
                snap_at_call=fb_snap,
                ledger_texts_copy=fb_ledger_texts,
                ledger_cats_copy=fb_ledger_cats,
                is_probe=allow_fb_probe,
            )
            async with self._state_lock:
                fresh_snap = self._state.snapshot()
                gen_fb_egress_id = gen_out_fb.egress_id if gen_out_fb is not None else None
                cand_fb = gen_out_fb.candidate if gen_out_fb is not None else None
                dummy_clf = ClassifierResult(
                    schema_version=SCHEMA_VERSION,
                    candidate_id=built0.context.candidate_id,
                    based_on_state_version=fb_snap.state_version,
                    classifier_version=self._classifier.classifier_version,
                    raw_scores=self._zero_scores(m_cat),
                    calibration_version=None,
                    calibrated_scores=self._zero_scores(m_cat),
                    latency_ms=0.0,
                    input_refs=built0.context.anchor_refs + built0.context.context_refs,
                )
                base_rt = RoutingDecision(
                    schema_version=SCHEMA_VERSION,
                    candidate_id=built0.context.candidate_id,
                    policy_version=self._policy.policy_version,
                    policy_sha256=self._policy.policy_sha256,
                    based_on_state_version=fb_snap.state_version,
                    outcome=Outcome.ROUTE,
                    route=cand_fb.route if cand_fb is not None else Route.GENERATOR,
                    category=m_cat,
                    reason_codes=(ReasonCode.MANUAL_REQUEST,),
                    abstain_reasons=(),
                    thresholds=None,
                    decided_ms=self._clock.now_ms(),
                )
                if (
                    self._mode in {Mode.SHADOW, Mode.PAUSED, Mode.OFF_THE_RECORD}
                    or self._is_stopping()
                ):
                    self._enqueue_recorder(
                        "decision",
                        context=built0.context,
                        terminal_state="suppressed",
                        terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                        classifier=dummy_clf,
                        routing=_downgrade_unexecuted_generator_routing(
                            base_rt,
                            egress_id=gen_fb_egress_id,
                            terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                        ),
                        serialized_text=built0.serialized_text,
                        retention=fresh_snap.brief.retention,
                    )
                    return ManualResult(
                        status="unavailable",
                        message="Manual request halted during mode change.",
                    )
                staleness = self._check_manual_staleness_locked(
                    ctx=built0.context,
                    snap_at_call=fb_snap,
                    fresh_snap=fresh_snap,
                    deadline_ms=deadline_ms,
                    cand=cand_fb,
                )
                if staleness is not None:
                    t_state, t_reason = staleness
                    self._enqueue_recorder(
                        "decision",
                        context=built0.context,
                        terminal_state=t_state,
                        terminal_reason=t_reason,
                        classifier=dummy_clf,
                        routing=_downgrade_unexecuted_generator_routing(
                            base_rt, egress_id=gen_fb_egress_id, terminal_reason=t_reason
                        ),
                        serialized_text=built0.serialized_text,
                        retention=fresh_snap.brief.retention,
                    )
                    return self._manual_empty_result(t_reason)
                if cand_fb is None:
                    t_reason = (
                        gen_out_fb.terminal_reason if gen_out_fb is not None else None
                    ) or ReasonCode.GENERATOR_INVALID_OUTPUT
                    eff_rt = (
                        base_rt
                        if t_reason is ReasonCode.GENERATOR_TIMEOUT
                        else _downgrade_unexecuted_generator_routing(
                            base_rt, egress_id=gen_fb_egress_id, terminal_reason=t_reason
                        )
                    )
                    self._enqueue_recorder(
                        "decision",
                        context=built0.context,
                        terminal_state="unrendered",
                        terminal_reason=t_reason,
                        classifier=dummy_clf,
                        routing=eff_rt,
                        serialized_text=built0.serialized_text,
                        retention=fresh_snap.brief.retention,
                    )
                    return self._manual_empty_result(t_reason)
                cand = replace(cand_fb, origin=Origin.MANUAL, display_by_ms=deadline_ms)
                return await self._activate_manual_candidate(
                    cand,
                    built0.context,
                    dummy_clf,
                    base_rt,
                    built0.serialized_text,
                    a_txt0,
                    brief_snapshot=fresh_snap.brief,
                )

        if not recorded_manual_decision and non_routed_manual:
            b0, clf0, rt0 = non_routed_manual[0]
            assert b0.context is not None and b0.serialized_text is not None
            state_map = {
                Outcome.NONE: "no_opportunity",
                Outcome.ABSTAIN: "abstained",
                Outcome.SUPPRESS: "suppressed",
            }
            t_state = state_map.get(rt0.outcome, "abstained")
            t_reason = rt0.reason_codes[0] if rt0.reason_codes else ReasonCode.BELOW_T_LOW
            async with self._state_lock:
                self._enqueue_recorder(
                    "decision",
                    context=b0.context,
                    terminal_state=t_state,
                    terminal_reason=t_reason,
                    classifier=clf0,
                    routing=rt0,
                    serialized_text=b0.serialized_text,
                    retention=self._state.snapshot().brief.retention,
                )

        return self._manual_empty_result(fb_empty_reason)

    @staticmethod
    def _zero_scores(cat: Category) -> ScoreSet:
        return ScoreSet(opportunity=0.0, categories={cat.value: 0.0})

    async def _activate_manual_candidate(
        self,
        cand: SuggestionCandidate,
        ctx: DecisionContext,
        clf_res: ClassifierResult,
        routing: RoutingDecision,
        serialized_text: str,
        anchor_text: str,
        *,
        brief_snapshot: SessionBrief | None = None,
    ) -> ManualResult:
        now_ms = self._clock.now_ms()
        if self._active is not None:
            if self._active.pinned:
                await self._transition_active(
                    LedgerState.OUTDATED, ReasonCode.USER_ACTION, at_ms=now_ms
                )
            else:
                await self._transition_active(
                    LedgerState.EXPIRED, ReasonCode.DISPLAY_TTL_PASSED, at_ms=now_ms
                )

        snap = self._state.snapshot()
        self._state.register_suggestion(cand)
        self._suggestions_by_id[cand.suggestion_id] = cand
        self._enqueue_recorder(
            "suggestion",
            suggestion=cand,
            current_state="ready",
            retention=snap.brief.retention,
        )
        disp_ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=self._next_evt_id("disp"),
            suggestion_id=cand.suggestion_id,
            transition=LedgerState.DISPLAYED,
            cause=ReasonCode.MANUAL_REQUEST,
            at_ms=now_ms,
            evidence_refs=cand.evidence_refs,
        )
        self._state.apply(disp_ev.to_canonical_event(snap.session_id, received_ms=now_ms))
        self._enqueue_recorder("ledger_event", event=disp_ev)
        self._ledger_texts[cand.suggestion_id] = cand.text
        self._ledger_categories[cand.suggestion_id] = cand.category
        if anchor_text:
            self._prior_anchor_texts[cand.candidate_id] = anchor_text
        anchor_segs = [
            snap.segments_by_id[r.segment_id]
            for r in ctx.anchor_refs
            if r.segment_id in snap.segments_by_id
        ]
        last_key = (
            max(CoverageCursor.seg_key(s) for s in anchor_segs) if anchor_segs else (0, 0, 0, "")
        )
        anchor_start = min((s.start_ms for s in anchor_segs), default=ctx.anchor_received_ms)
        self._active = _ActiveItem(
            cand=cand,
            context=ctx,
            classifier=clf_res,
            routing=routing,
            serialized_text=serialized_text,
            brief_snapshot=brief_snapshot if brief_snapshot is not None else snap.brief,
            anchor_text=anchor_text,
            anchor_last_key=last_key,
            anchor_start_ms=anchor_start,
            anchor_received_ms=ctx.anchor_received_ms,
            displayed_at_ms=now_ms,
            expires_at_ms=now_ms + cand.display_ttl_ms,
            pinned=False,
            shadow=False,
        )
        self._manual_status = None
        self._last_displayed_card_info = (cand, now_ms)
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind="manual_displayed",
            stage="manual_lane",
            candidate_id=cand.candidate_id,
            suggestion_id=cand.suggestion_id,
            reason_code=ReasonCode.MANUAL_REQUEST,
        )
        self._schedule_ui_broadcast()
        self._display_wake.set()
        return ManualResult(
            status="displayed",
            suggestion=cand,
            reason=ReasonCode.MANUAL_REQUEST,
        )

    async def handle_ui_action(self, action: Mapping[str, Any]) -> dict[str, Any] | None:
        act = str(action.get("action") or "").strip().lower()
        now_ms = self._clock.now_ms()
        if act == "manual":
            unavail = self._manual_unavailable_reason()
            if unavail is not None:
                self._manual_status = {"status": "unavailable", "message": unavail.message}
                self._schedule_ui_broadcast()
                await self._flush_ui_broadcast()
                return None
            if self._started and not self._stopping:
                if self._manual_in_flight:
                    await self.trigger_manual()
                else:
                    self._manual_in_flight = True
                    loop = asyncio.get_running_loop()
                    fut: asyncio.Future[ManualResult] = loop.create_future()
                    try:
                        await self._manual_queue.put(fut)
                    except BaseException:
                        self._manual_in_flight = False
                        raise
            else:
                await self.trigger_manual()
            await self._flush_ui_broadcast()
            return None
        if act == "off_record":
            active_arg = action.get("active")
            if isinstance(active_arg, bool):
                target = (
                    Mode.OFF_THE_RECORD
                    if active_arg
                    else (self._pre_otr_mode if self._mode is Mode.OFF_THE_RECORD else self._mode)
                )
            else:
                target = (
                    self._pre_otr_mode if self._mode is Mode.OFF_THE_RECORD else Mode.OFF_THE_RECORD
                )
            await self.set_mode(target)
            await self._flush_ui_broadcast()
            return None
        if act == "mode":
            raw_mode = str(action.get("mode") or "").strip()
            if raw_mode:
                await self.set_mode(Mode(raw_mode))
            await self._flush_ui_broadcast()
            return None

        async with self._state_lock:
            if self._active is None or self._active.shadow:
                return None
            req_sid = action.get("suggestion_id")
            if req_sid is not None and str(req_sid) != self._active.cand.suggestion_id:
                return None
            if act == "use":
                await self._transition_active(
                    LedgerState.USED, ReasonCode.USER_ACTION, at_ms=now_ms
                )
            elif act == "dismiss":
                fb_raw = action.get("feedback")
                if fb_raw is None:
                    fb_raw = action.get("key")
                fb_key_map = {
                    "1": "too_late",
                    "2": "already_answered",
                    "3": "not_relevant",
                    "4": "badly_phrased",
                }
                if fb_raw is not None:
                    fb_norm = fb_key_map.get(str(fb_raw).strip(), str(fb_raw).strip())
                    fb = (
                        FeedbackReason(fb_norm)
                        if fb_norm in {e.value for e in FeedbackReason}
                        else None
                    )
                else:
                    fb = None
                await self._transition_active(
                    LedgerState.DISMISSED, ReasonCode.USER_ACTION, at_ms=now_ms, feedback=fb
                )
            elif act == "pin" and not self._active.pinned:
                pin_ev = LedgerEvent(
                    schema_version=SCHEMA_VERSION,
                    event_id=self._next_evt_id("pin"),
                    suggestion_id=self._active.cand.suggestion_id,
                    transition=LedgerState.PINNED,
                    cause=ReasonCode.USER_ACTION,
                    at_ms=now_ms,
                    evidence_refs=self._active.cand.evidence_refs,
                )
                self._state.apply(
                    pin_ev.to_canonical_event(self._brief.session_id, received_ms=now_ms)
                )
                self._enqueue_recorder("ledger_event", event=pin_ev)
                self._active.pinned = True
                self._schedule_ui_broadcast()
        await self._flush_ui_broadcast()
        return None

    async def set_mode(self, new_mode: Mode) -> None:
        async with self._state_lock:
            if new_mode is self._mode:
                return
            now_ms = self._clock.now_ms()
            if new_mode is Mode.OFF_THE_RECORD:
                self._pre_otr_mode = self._mode
                self._otr_start_ms = now_ms
                await self._retire_uncovered_segments(self._state.snapshot(), ReasonCode.OFF_RECORD)
                await self._close_open_items_for_halt()
                self._mode = Mode.OFF_THE_RECORD
                ctrl = CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._brief.session_id,
                    event_id=self._next_evt_id("otr-start"),
                    kind=EventKind.CONTROL,
                    received_ms=now_ms,
                    source_sequence=0,
                    source_id="controller",
                    subtype=ControlSubtype.OFF_RECORD_START,
                    reason="off_record",
                )
                res = self._state.apply(ctrl)
                if res.accepted:
                    self._enqueue_recorder(
                        "canonical_event",
                        event=ctrl,
                        retention=self._brief.retention,
                        canonical_revision=None,
                    )
            elif self._mode is Mode.OFF_THE_RECORD:
                self._mode = new_mode
                otr_start = self._otr_start_ms if self._otr_start_ms is not None else now_ms
                self._otr_start_ms = None
                ctrl = CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._brief.session_id,
                    event_id=self._next_evt_id("otr-end"),
                    kind=EventKind.CONTROL,
                    received_ms=now_ms,
                    source_sequence=0,
                    source_id="controller",
                    subtype=ControlSubtype.OFF_RECORD_END,
                    start_ms=otr_start,
                    end_ms=max(otr_start, now_ms),
                    reason="off_record",
                )
                res = self._state.apply(ctrl)
                if res.accepted:
                    self._enqueue_recorder(
                        "canonical_event",
                        event=ctrl,
                        retention=self._brief.retention,
                        canonical_revision=None,
                    )
                gap_ctrl = CanonicalEvent(
                    schema_version=SCHEMA_VERSION,
                    session_id=self._brief.session_id,
                    event_id=self._next_evt_id("otr-gap"),
                    kind=EventKind.CONTROL,
                    received_ms=now_ms,
                    source_sequence=0,
                    source_id="controller",
                    subtype=ControlSubtype.GAP,
                    start_ms=otr_start,
                    end_ms=max(otr_start, now_ms),
                    reason="off_record",
                )
                gap_res = self._state.apply(gap_ctrl)
                if gap_res.accepted:
                    self._enqueue_recorder(
                        "canonical_event",
                        event=gap_ctrl,
                        retention=self._brief.retention,
                        canonical_revision=None,
                    )
                self._last_transcript_event_ms = max(self._last_transcript_event_ms, now_ms)
                self._last_gap_ms = now_ms
            else:
                self._mode = new_mode

            if self._mode is Mode.PAUSED:
                self._set_health(HealthState.PAUSED, "paused")
            elif self._health is HealthState.PAUSED:
                if self._consecutive_clf_errors >= self._classifier_error_threshold:
                    self._set_health(HealthState.MANUAL_ONLY, "classifier_errors")
                elif self._local_only_blocked() or (
                    self._provider is not None and self._provider.breaker_state is BreakerState.OPEN
                ):
                    self._set_health(HealthState.TEMPLATE_ONLY, "generator_breaker_open")
                else:
                    self._set_health(HealthState.OK, "resumed")
            self._schedule_ui_broadcast()
            if self._mode in {Mode.PROACTIVE, Mode.QUIET, Mode.SHADOW}:
                self._proactive_wake.set()
            self._display_wake.set()

    def next_pending_card_wake_ms(self) -> int | None:
        """Return earliest pending settle/deferred timer ms (used by EOF replay stepping)."""
        if not self._settling and self._deferred is None:
            return None
        deadlines: list[int] = []
        for item in self._settling:
            deadlines.append(min(item.settle_end_ms, item.cand.display_by_ms))
        if self._deferred is not None:
            deadlines.append(self._deferred.cand.display_by_ms)
            if self._deferred.cooldown_end_ms is not None:
                deadlines.append(self._deferred.cooldown_end_ms)
        if self._active is not None and not self._active.pinned:
            deadlines.append(self._active.expires_at_ms)
        if (self._settling or self._deferred is not None) and self._recipient_speaking_until_ms > 0:
            deadlines.append(self._recipient_speaking_until_ms)
        return min(deadlines) if deadlines else None

    def _next_display_wake_ms(self) -> int | None:
        deadlines: list[int] = []
        for item in self._settling:
            deadlines.append(min(item.settle_end_ms, item.cand.display_by_ms))
        if self._deferred is not None:
            deadlines.append(self._deferred.cand.display_by_ms)
            if self._deferred.cooldown_end_ms is not None:
                deadlines.append(self._deferred.cooldown_end_ms)
        if self._active is not None and not self._active.pinned:
            deadlines.append(self._active.expires_at_ms)
        if (
            self._settling or self._deferred is not None
        ) and self._recipient_speaking_until_ms > self._clock.now_ms():
            deadlines.append(self._recipient_speaking_until_ms)
        if (
            self._mode not in {Mode.PAUSED, Mode.OFF_THE_RECORD}
            and self._health is not HealthState.NO_TRANSCRIPT
        ):
            deadlines.append(
                self._last_transcript_event_ms + self._policy.health.no_transcript_after_ms
            )
        if (
            self._health is HealthState.NO_TRANSCRIPT
            and self._proactive_idle.is_set()
            and self._mode is not Mode.OFF_THE_RECORD
        ):
            snap = self._state.snapshot()
            unprocessed = self._cursor.unprocessed_segments(snap)
            if unprocessed:
                anchors = coalesce_anchors(
                    snap,
                    max_candidate_age_ms=self._policy.display.max_candidate_age_ms,
                    tokenizer=self._classifier.tokenizer,
                    speaker_boundary_end_ms=self._speaker_boundary_end_ms,
                    segments_subset=unprocessed,
                )
                if anchors:
                    deadlines.append(min(a.display_by_ms for a in anchors))
        if self._provider is not None and self._provider.breaker_state is BreakerState.OPEN:
            opened_at = self._provider.opened_at_ms
            if opened_at is not None:
                prov_wake = opened_at + self._provider.config.breaker_open_ms
                ctrl_wake = self._clock.now_ms() + (prov_wake - self._provider.clock_ms())
                deadlines.append(max(self._clock.now_ms(), ctrl_wake))
            else:
                deadlines.append(self._clock.now_ms() + self._policy.generator.breaker.open_ms)
        return min(deadlines) if deadlines else None

    async def _display_loop(self) -> None:
        while not self._is_stopping():
            await self.tick()
            next_ms = self._next_display_wake_ms()
            self._display_wake.clear()
            if self._is_stopping():
                return
            if next_ms is None:
                await self._display_wake.wait()
            else:
                delay_s = max(0.005, (next_ms - self._clock.now_ms()) / 1000.0)
                wake_task = asyncio.create_task(self._display_wake.wait())
                try:
                    await asyncio.wait_for(wake_task, timeout=delay_s)
                except TimeoutError:
                    pass
                finally:
                    if not wake_task.done():
                        wake_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await wake_task

    async def tick(self, now_ms: int | None = None) -> None:
        """Run one deterministic display/watchdog pass (safe to call from virtual-clock tests)."""
        async with self._state_lock:
            eff_now = self._clock.now_ms() if now_ms is None else now_ms
            await self._process_display_locked(eff_now)

    async def _process_display_locked(self, now_ms: int) -> None:
        snap = self._state.snapshot()

        # Watchdogs: provider breaker half-open recovery & no_transcript timeout
        if self._provider is not None:
            prev_brk = self._provider.breaker_state
            prov_now = (
                now_ms
                if self._provider.clock_ms == self._clock.now_ms
                else self._provider.clock_ms() + (now_ms - self._clock.now_ms())
            )
            new_brk = self._provider.advance_clock(prov_now)
            if prev_brk is BreakerState.OPEN and new_brk is BreakerState.HALF_OPEN:
                self._emit_tel(
                    level=TelemetryLevel.LIFECYCLE,
                    kind="breaker_half_open",
                    stage="provider_adapter",
                )
        if (
            self._mode not in {Mode.PAUSED, Mode.OFF_THE_RECORD}
            and self._health is not HealthState.NO_TRANSCRIPT
            and self._proactive_idle.is_set()
            and (now_ms - self._last_transcript_event_ms)
            >= self._policy.health.no_transcript_after_ms
        ):
            self._set_health(HealthState.NO_TRANSCRIPT, "no_transcript_timeout")

        if (
            self._health is HealthState.NO_TRANSCRIPT
            and self._proactive_idle.is_set()
            and self._mode is not Mode.OFF_THE_RECORD
        ):
            unprocessed = self._cursor.unprocessed_segments(snap)
            if unprocessed:
                anchors = coalesce_anchors(
                    snap,
                    max_candidate_age_ms=self._policy.display.max_candidate_age_ms,
                    tokenizer=self._classifier.tokenizer,
                    speaker_boundary_end_ms=self._speaker_boundary_end_ms,
                    segments_subset=unprocessed,
                )
                for anchor in anchors:
                    if anchor.display_by_ms <= now_ms:
                        self._retire_anchor(
                            anchor,
                            state_version=snap.state_version,
                            reason=ReasonCode.EXPIRED_UNEVALUATED,
                        )
                        self._cursor.mark_covered(anchor.segments)

        # 1. Revalidate active card
        if self._active is not None:
            fail_reason, _ = self._revalidate_item(self._active, snap, now_ms, is_active=True)
            if fail_reason is not None:
                if fail_reason is ReasonCode.DISPLAY_TTL_PASSED:
                    await self._transition_active(
                        LedgerState.EXPIRED, ReasonCode.DISPLAY_TTL_PASSED, at_ms=now_ms
                    )
                elif fail_reason in {
                    ReasonCode.POSSIBLY_RESOLVED,
                    ReasonCode.EQUIVALENT_QUESTION_ASKED,
                }:
                    await self._transition_active(
                        LedgerState.RESOLVED, ReasonCode.RESOLUTION_CUE, at_ms=now_ms
                    )
                elif self._active.pinned:
                    await self._transition_active(
                        LedgerState.OUTDATED, ReasonCode.RESOLUTION_CUE, at_ms=now_ms
                    )
                else:
                    await self._transition_active(LedgerState.WITHDRAWN, fail_reason, at_ms=now_ms)

        # 2. Revalidate deferred slot
        if self._deferred is not None:
            def_item = self._deferred
            fail_reason, defer_reason = self._revalidate_item(
                def_item, snap, now_ms, is_active=False
            )
            if fail_reason is not None:
                self._deferred = None
                await self._close_pending(def_item, fail_reason, at_ms=now_ms)
            elif defer_reason is None:
                gate = self._display_moment_block(now_ms, snap)
                if gate is None:
                    self._deferred = None
                    await self._display_pending(
                        def_item, cause=ReasonCode.DEFERRED_RELEASED, at_ms=now_ms, snap=snap
                    )
                elif gate is ReasonCode.COOLDOWN and self._last_proactive_display_ms is not None:
                    def_item.cooldown_end_ms = (
                        self._last_proactive_display_ms + self._policy.display.cooldown_ms
                    )

        # 3. Advance settling candidates (boundary early release or settle timeout)
        snap = self._state.snapshot()
        still_settling: list[_PendingItem] = []
        for item in self._settling:
            fail_reason, defer_reason = self._revalidate_item(item, snap, now_ms, is_active=False)
            if fail_reason is not None:
                await self._close_pending(item, fail_reason, at_ms=now_ms)
                continue

            bnd_rx = self._speaker_boundary_rx_ms.get(item.anchor_speaker_id)
            early_bnd = (
                bnd_rx is not None
                and bnd_rx > item.anchor_received_ms
                and now_ms < item.settle_end_ms
            )
            if early_bnd:
                item.settled = True
                item.settle_cause = ReasonCode.BOUNDARY_AFTER_SETTLE
            elif now_ms >= item.settle_end_ms:
                item.settled = True
                item.settle_cause = ReasonCode.SETTLE_TIMEOUT

            if not item.settled:
                still_settling.append(item)
                continue

            if defer_reason is not None:
                await self._offer_deferred(item, defer_reason, at_ms=now_ms)
                continue

            gate = self._display_moment_block(now_ms, snap)
            if gate is not None:
                await self._offer_deferred(item, gate, at_ms=now_ms)
                continue

            await self._display_pending(item, cause=item.settle_cause, at_ms=now_ms, snap=snap)
        self._settling = still_settling

    def _display_moment_block(self, now_ms: int, snap: StateSnapshot) -> ReasonCode | None:
        if self._recipient_speaking or now_ms < self._recipient_speaking_until_ms:
            return ReasonCode.RECIPIENT_SPEAKING
        if self._active is not None:
            return ReasonCode.CARD_ACTIVE
        if self._last_proactive_display_ms is not None:
            cooldown_end = self._last_proactive_display_ms + self._policy.display.cooldown_ms
            if now_ms < cooldown_end:
                return ReasonCode.COOLDOWN
        cap = effective_max_cards_per_10_min(snap.brief, self._policy)
        recent = [t for t in self._proactive_display_times if (now_ms - t) <= 600_000]
        if cap <= 0 or len(recent) >= cap:
            return ReasonCode.CARD_ACTIVE
        return None

    def _revalidate_item(
        self,
        item: _PendingItem | _ActiveItem,
        snap: StateSnapshot,
        now_ms: int,
        *,
        is_active: bool,
    ) -> tuple[ReasonCode | None, ReasonCode | None]:
        cand = item.cand
        # 1. Expiry & pre-gap check
        if is_active:
            assert isinstance(item, _ActiveItem)
            if not item.pinned and now_ms >= item.expires_at_ms:
                return (ReasonCode.DISPLAY_TTL_PASSED, None)
        else:
            if now_ms >= cand.display_by_ms:
                return (ReasonCode.DISPLAY_BY_PASSED, None)
            if self._last_gap_ms is not None and item.anchor_received_ms < self._last_gap_ms:
                return (ReasonCode.DISPLAY_BY_PASSED, None)

        # 2. Evidence ref revision & recipient flip check
        recipient_ids = {"ME", snap.brief.recipient.participant_id}
        for ref in cand.evidence_refs:
            seg = snap.segments_by_id.get(ref.segment_id)
            if seg is None or seg.revision != ref.revision or seg.speaker_id in recipient_ids:
                return (ReasonCode.EVIDENCE_REVISION_CHANGED, None)

        # 3. Material brief fields check
        if _brief_material_key(snap.brief) != _brief_material_key(item.brief_snapshot):
            return (ReasonCode.OBJECTIVE_CHANGED, None)

        # 4 & 5. Ledger blocking duplicate / near-duplicate against shown/acted-on cards
        ev_ids = {r.segment_id for r in cand.evidence_refs}
        norm_q = normalize_for_matching(cand.text)
        q_toks = _content_tokens(cand.text)
        for entry in snap.ledger.values():
            if entry.suggestion_id == cand.suggestion_id:
                continue
            if entry.state not in _SHOWN_OR_ACTED_STATES:
                continue
            e_cat = self._ledger_categories.get(entry.suggestion_id)
            if e_cat is None and entry.suggestion_id in self._suggestions_by_id:
                e_cat = self._suggestions_by_id[entry.suggestion_id].category
            e_ev = {r.segment_id for r in entry.evidence_refs}
            if e_cat is cand.category and (ev_ids & e_ev):
                return (ReasonCode.DUPLICATE, None)
            prev_txt = self._ledger_texts.get(entry.suggestion_id)
            if prev_txt is None and entry.suggestion_id in self._suggestions_by_id:
                prev_txt = self._suggestions_by_id[entry.suggestion_id].text
            if prev_txt:
                if normalize_for_matching(prev_txt) == norm_q:
                    return (ReasonCode.DUPLICATE, None)
                p_toks = _content_tokens(prev_txt)
                if q_toks and p_toks:
                    jac = len(q_toks & p_toks) / len(q_toks | p_toks)
                    if jac >= 0.75:
                        return (ReasonCode.NEAR_DUPLICATE, None)

        # 6. Resolution cues (committed -> withdraw/resolve; interim -> defer)
        cpol = self._policy.categories.get(cand.category)
        tpl = self._templates.get(cpol.template) if cpol and cpol.template else None
        for seg in snap.segments:
            if CoverageCursor.seg_key(seg) > item.anchor_last_key and _matches_resolution_cue(
                seg.text, cand.category, tpl
            ):
                return (ReasonCode.POSSIBLY_RESOLVED, None)
        if not is_active:
            for int_seg in snap.interim_segments.values():
                if (
                    int_seg.segment_id not in ev_ids
                    and (
                        int_seg.start_ms >= item.anchor_start_ms
                        or int_seg.received_ms >= item.anchor_received_ms
                    )
                    and _matches_resolution_cue(int_seg.text, cand.category, tpl)
                ):
                    return (None, ReasonCode.INTERIM_RESOLUTION_HINT)

        # 7. Equivalent question asked by recipient
        for seg in snap.segments:
            if (
                seg.speaker_id in recipient_ids
                and CoverageCursor.seg_key(seg) > item.anchor_last_key
                and _is_equivalent_question(seg.text, cand.text, cand.category)
            ):
                return (ReasonCode.EQUIVALENT_QUESTION_ASKED, None)

        # 8. Display-time generic referent age cap check (§9.1, c6)
        if (
            not is_active
            and cand.template_id is not None
            and not cand.grounded_slots
            and tpl is not None
            and cand.text == tpl.generic.text
            and (
                now_ms < item.anchor_received_ms
                or (now_ms - item.anchor_received_ms) > tpl.generic.allowed_when.max_anchor_age_ms
            )
        ):
            return (ReasonCode.GENERIC_REFERENT_UNCLEAR, None)

        return (None, None)

    async def _offer_deferred(self, item: _PendingItem, reason: ReasonCode, *, at_ms: int) -> None:
        if self._deferred is not None:
            if item.score > self._deferred.score:
                old = self._deferred
                self._deferred = None
                await self._close_pending(old, ReasonCode.DISPLACED_BY_BETTER, at_ms=at_ms)
            else:
                await self._close_pending(item, ReasonCode.DISPLACED_BY_BETTER, at_ms=at_ms)
                return

        entry = self._state.snapshot().ledger.get(item.cand.suggestion_id)
        if entry is not None and entry.state is LedgerState.READY:
            def_ev = LedgerEvent(
                schema_version=SCHEMA_VERSION,
                event_id=self._next_evt_id("def"),
                suggestion_id=item.cand.suggestion_id,
                transition=LedgerState.DEFERRED,
                cause=reason,
                at_ms=at_ms,
                evidence_refs=item.cand.evidence_refs,
            )
            self._state.apply(def_ev.to_canonical_event(self._brief.session_id, received_ms=at_ms))
            self._enqueue_recorder("ledger_event", event=def_ev)

        item.deferred_reason = reason
        if reason is ReasonCode.COOLDOWN and self._last_proactive_display_ms is not None:
            item.cooldown_end_ms = (
                self._last_proactive_display_ms + self._policy.display.cooldown_ms
            )
        else:
            item.cooldown_end_ms = None
        self._deferred = item
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind="suggestion_deferred",
            stage="display",
            candidate_id=item.cand.candidate_id,
            suggestion_id=item.cand.suggestion_id,
            reason_code=reason,
        )

    async def _display_pending(
        self,
        item: _PendingItem,
        *,
        cause: ReasonCode,
        at_ms: int,
        snap: StateSnapshot,
    ) -> None:
        is_shadow = self._mode is Mode.SHADOW
        disp_ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=self._next_evt_id("disp"),
            suggestion_id=item.cand.suggestion_id,
            transition=LedgerState.DISPLAYED,
            cause=cause,
            at_ms=at_ms,
            evidence_refs=item.cand.evidence_refs,
        )
        self._state.apply(disp_ev.to_canonical_event(self._brief.session_id, received_ms=at_ms))
        self._enqueue_recorder("ledger_event", event=disp_ev)
        self._ledger_texts[item.cand.suggestion_id] = item.cand.text
        self._ledger_categories[item.cand.suggestion_id] = item.cand.category
        if item.anchor_text:
            self._prior_anchor_texts[item.cand.candidate_id] = item.anchor_text
        self._last_proactive_display_ms = at_ms
        self._proactive_display_times.append(at_ms)
        self._last_displayed_card_info = (item.cand, at_ms)

        self._active = _ActiveItem(
            cand=item.cand,
            context=item.context,
            classifier=item.classifier,
            routing=item.routing,
            serialized_text=item.serialized_text,
            brief_snapshot=item.brief_snapshot,
            anchor_text=item.anchor_text,
            anchor_last_key=item.anchor_last_key,
            anchor_start_ms=item.anchor_start_ms,
            anchor_received_ms=item.anchor_received_ms,
            displayed_at_ms=at_ms,
            expires_at_ms=at_ms + item.cand.display_ttl_ms,
            pinned=False,
            shadow=is_shadow,
        )
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind="shadow_displayed" if is_shadow else "displayed",
            stage="display",
            candidate_id=item.cand.candidate_id,
            suggestion_id=item.cand.suggestion_id,
            reason_code=cause,
        )
        if not is_shadow:
            self._schedule_ui_broadcast()

    async def _close_pending(
        self,
        item: _PendingItem,
        reason: ReasonCode,
        *,
        at_ms: int,
    ) -> None:
        if reason is ReasonCode.GENERIC_REFERENT_UNCLEAR:
            self._state.unregister_suggestion(item.cand.suggestion_id)
            self._suggestions_by_id.pop(item.cand.suggestion_id, None)
            self._enqueue_recorder(
                "suggestion",
                suggestion=item.cand,
                current_state="unrendered",
                terminal_reason=ReasonCode.GENERIC_REFERENT_UNCLEAR,
                transitions=(),
                retention=self._brief.retention,
            )
            self._enqueue_recorder(
                "decision",
                context=item.context,
                terminal_state="unrendered",
                terminal_reason=ReasonCode.GENERIC_REFERENT_UNCLEAR,
                classifier=item.classifier,
                routing=item.routing,
                serialized_text=item.serialized_text,
                retention=self._brief.retention,
            )
            self._emit_tel(
                level=TelemetryLevel.LIFECYCLE,
                kind="suggestion_unrendered",
                stage="display",
                candidate_id=item.cand.candidate_id,
                suggestion_id=item.cand.suggestion_id,
                reason_code=ReasonCode.GENERIC_REFERENT_UNCLEAR,
            )
            return

        if reason in {ReasonCode.DISPLAY_BY_PASSED, ReasonCode.DISPLAY_TTL_PASSED}:
            t_state_enum = LedgerState.EXPIRED
            t_reason = ReasonCode.DISPLAY_BY_PASSED
            dec_state = "expired"
        else:
            t_state_enum = LedgerState.WITHDRAWN
            t_reason = reason
            dec_state = "withdrawn"

        ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=self._next_evt_id("cls"),
            suggestion_id=item.cand.suggestion_id,
            transition=t_state_enum,
            cause=t_reason,
            at_ms=at_ms,
            evidence_refs=item.cand.evidence_refs,
        )
        self._state.apply(ev.to_canonical_event(self._brief.session_id, received_ms=at_ms))
        self._enqueue_recorder("ledger_event", event=ev)
        self._enqueue_recorder(
            "decision",
            context=item.context,
            terminal_state=dec_state,
            terminal_reason=t_reason,
            classifier=item.classifier,
            routing=item.routing,
            serialized_text=item.serialized_text,
            retention=self._brief.retention,
        )
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind=f"suggestion_{dec_state}",
            stage="display",
            candidate_id=item.cand.candidate_id,
            suggestion_id=item.cand.suggestion_id,
            reason_code=t_reason,
        )

    async def _transition_active(
        self,
        target_state: LedgerState,
        cause: ReasonCode,
        *,
        at_ms: int,
        feedback: FeedbackReason | None = None,
    ) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=self._next_evt_id("act"),
            suggestion_id=active.cand.suggestion_id,
            transition=target_state,
            cause=cause,
            at_ms=at_ms,
            evidence_refs=active.cand.evidence_refs,
            feedback=feedback,
        )
        self._state.apply(ev.to_canonical_event(self._brief.session_id, received_ms=at_ms))
        self._enqueue_recorder("ledger_event", event=ev)
        self._enqueue_recorder(
            "decision",
            context=active.context,
            terminal_state=target_state.value,
            terminal_reason=cause,
            classifier=active.classifier,
            routing=active.routing,
            serialized_text=active.serialized_text,
            retention=self._brief.retention,
        )
        self._emit_tel(
            level=TelemetryLevel.LIFECYCLE,
            kind=f"suggestion_{target_state.value}",
            stage="display",
            candidate_id=active.cand.candidate_id,
            suggestion_id=active.cand.suggestion_id,
            reason_code=cause,
        )
        if not active.shadow:
            self._schedule_ui_broadcast()

    async def _retire_uncovered_segments(self, snap: StateSnapshot, reason: ReasonCode) -> None:
        unprocessed = self._cursor.unprocessed_segments(snap)
        if not unprocessed:
            return
        anchors = coalesce_anchors(
            snap,
            max_candidate_age_ms=self._policy.display.max_candidate_age_ms,
            tokenizer=self._classifier.tokenizer,
            speaker_boundary_end_ms=self._speaker_boundary_end_ms,
            segments_subset=unprocessed,
        )
        now_ms = self._clock.now_ms()
        for anchor in anchors:
            eff_reason = (
                ReasonCode.EXPIRED_UNEVALUATED
                if reason is ReasonCode.SESSION_ENDED and anchor.display_by_ms <= now_ms
                else reason
            )
            self._retire_anchor(anchor, state_version=snap.state_version, reason=eff_reason)
            self._cursor.mark_covered(anchor.segments)

    async def _close_open_items_for_halt(self) -> None:
        now_ms = self._clock.now_ms()
        for item in self._settling:
            await self._close_pending(item, ReasonCode.DISPLAY_BY_PASSED, at_ms=now_ms)
        self._settling.clear()
        if self._deferred is not None:
            def_item = self._deferred
            self._deferred = None
            await self._close_pending(def_item, ReasonCode.DISPLAY_BY_PASSED, at_ms=now_ms)
        if self._active is not None:
            if self._active.pinned:
                await self._transition_active(
                    LedgerState.OUTDATED, ReasonCode.RESOLUTION_CUE, at_ms=now_ms
                )
            else:
                await self._transition_active(
                    LedgerState.EXPIRED, ReasonCode.DISPLAY_TTL_PASSED, at_ms=now_ms
                )

    async def _recorder_loop(self) -> None:
        while True:
            try:
                job = await asyncio.wait_for(self._recorder_queue.get(), timeout=0.1)
            except TimeoutError:
                if self._store is not None:
                    self._telemetry.flush_to_store(
                        self._store, self._brief.session_id, self._run_id
                    )
                continue
            try:
                if job is None:
                    while not self._recorder_queue.empty():
                        extra = self._recorder_queue.get_nowait()
                        if extra is not None:
                            self._execute_recorder_job(extra)
                        self._recorder_queue.task_done()
                    if self._store is not None:
                        self._telemetry.flush_to_store(
                            self._store, self._brief.session_id, self._run_id
                        )
                    return
                self._execute_recorder_job(job)
            finally:
                self._recorder_queue.task_done()

    def _execute_recorder_job(self, job: _RecorderJob) -> None:
        if self._store is None:
            return
        p = job.payload
        k = job.kind
        if k == "canonical_event":
            self._store.record_canonical_event(
                run_id=self._run_id,
                event=p["event"],
                retention=p["retention"],
                canonical_revision=p["canonical_revision"],
            )
        elif k == "decision":
            self._store.record_decision(
                run_id=self._run_id,
                context=p["context"],
                terminal_state=p["terminal_state"],
                terminal_reason=p.get("terminal_reason"),
                classifier=p.get("classifier"),
                routing=p.get("routing"),
                serialized_text=p.get("serialized_text"),
                retention=p.get("retention"),
            )
        elif k == "suggestion":
            self._store.record_suggestion(
                session_id=self._brief.session_id,
                run_id=self._run_id,
                suggestion=p["suggestion"],
                current_state=p.get("current_state", "ready"),
                terminal_reason=p.get("terminal_reason"),
                transitions=p.get("transitions", ()),
                retention=p.get("retention"),
            )
        elif k == "ledger_event":
            self._store.record_ledger_event(
                session_id=self._brief.session_id,
                run_id=self._run_id,
                event=p["event"],
            )
        elif k == "egress":
            self._store.record_egress(run_id=self._run_id, record=p["record"])
        elif k == "session_upsert":
            self._store.upsert_session(**p)

    async def flush_recorder(self) -> None:
        while not self._recorder_queue.empty():
            job = self._recorder_queue.get_nowait()
            if job is not None:
                self._execute_recorder_job(job)
            self._recorder_queue.task_done()
        if self._store is not None:
            self._telemetry.flush_to_store(self._store, self._brief.session_id, self._run_id)

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True

        if self._adapter_pump_task is not None and not self._adapter_pump_task.done():
            self._adapter_pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._adapter_pump_task

        if self._ui_broadcast_task is not None and not self._ui_broadcast_task.done():
            self._ui_broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ui_broadcast_task
        while not self._ui_broadcast_queue.empty():
            self._ui_broadcast_queue.get_nowait()
            self._ui_broadcast_queue.task_done()

        if "ingest" in self._tasks:
            await self._ingest_queue.put(None)
            with contextlib.suppress(asyncio.CancelledError):
                await self._tasks["ingest"]

        self._proactive_wake.set()
        if "proactive" in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await self._tasks["proactive"]

        if "manual" in self._tasks:
            await self._manual_queue.put(None)
            with contextlib.suppress(asyncio.CancelledError):
                await self._tasks["manual"]

        self._display_wake.set()
        if "display" in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await self._tasks["display"]

        async with self._state_lock:
            snap = self._state.snapshot()
            await self._retire_uncovered_segments(snap, ReasonCode.SESSION_ENDED)
            await self._close_open_items_for_halt()
            final_snap = self._state.snapshot()
            if self._store is not None:
                self._enqueue_recorder(
                    "session_upsert",
                    brief=final_snap.brief,
                    run_id=self._run_id,
                    runtime_mode=self._mode,
                    effective_max_cards_per_10_min=effective_max_cards_per_10_min(
                        final_snap.brief, self._policy
                    ),
                    health_timeline=self._health_timeline,
                    reducer_counters=final_snap.counters,
                    revision_history=final_snap.revision_history,
                    started_ms=self._started_ms,
                    ended_ms=self._clock.now_ms(),
                )

        if "recorder" in self._tasks:
            await self._recorder_queue.put(None)
            with contextlib.suppress(asyncio.CancelledError):
                await self._tasks["recorder"]
        else:
            await self.flush_recorder()

        if self._provider is not None:
            self._provider._cancel_recovery_task()
        self._tasks.clear()
        self._started = False
