"""CLI entry point for the Live Conversation Coach (`coach`)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import time
import uuid
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from coach.annotation_tool import (
    AnnotationServer,
    Pass1Session,
    Pass2Session,
    build_pass2_items,
    compute_cohens_kappa,
    load_labels_jsonl,
    load_review_queue_items,
)
from coach.calibration import (
    CalibrationDataError,
    CalibrationModel,
    CalibrationRegistry,
    default_calibration_metadata,
    default_calibration_registry,
    evaluate_calibration,
    fit_platt_calibrator,
)
from coach.canonical_state import CanonicalState
from coach.context_builder import PolicyConfig, build_decision_contexts
from coach.contracts import (
    PREDICTABLE_CATEGORIES,
    CanonicalEvent,
    Category,
    ClassifierResult,
    DecisionContext,
    HealthState,
    LabelRecord,
    Mode,
    Origin,
    Route,
    SampleSource,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
)
from coach.opportunity_classifier import (
    ClassifierInput,
    HashedTfidfEmbedder,
    HuggingFaceEmbedder,
    OllamaEmbedder,
    TextEmbedder,
    create_classifier,
    fit_embed_logreg,
    train_modernbert,
)
from coach.provider_adapter import ProviderAdapter, ProviderConfig
from coach.replay_evaluator import (
    DisplayedCardEvent,
    EvalLookCounter,
    match_opportunities,
    paired_conversation_bootstrap,
    run_latency_benchmark,
    simulate_replay,
    split_by_conversation,
    split_by_series,
    sweep_t_display,
)
from coach.routing_policy import load_policy
from coach.storage import SQLiteStore, import_events_jsonl
from coach.suggestion_controller import SuggestionController
from coach.telemetry import (
    BoundedRecorderBuffer,
    build_session_report,
    check_conservation_invariants,
    diff_runs,
    explain_silence,
)
from coach.transcript_adapter import (
    DEFAULT_PEECEE_SCRIPT,
    DEFAULT_WS_URL,
    BinderConfig,
    Clock,
    DeepgramNormalizer,
    ManualClock,
    MonotonicSessionClock,
    NemotronSegmenter,
    PeeceeConfig,
    PeeceeLeaseUnavailableError,
    PeeceeProcessError,
    PeeceeRealtimeAdapter,
    PeeceeSpeechProcess,
    RecipientBinder,
    ReplayPlayer,
    ReplayTimingMode,
    parse_nemo_offline_json,
)
from coach.ui import CardUIServer


def _resolve_run_id(store: SQLiteStore, session_id: str, run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    runs = store.list_runs(session_id)
    if not runs:
        raise SystemExit(f"No runs found for session {session_id!r}")
    return str(runs[-1]["run_id"])


def _emit_output(payload: dict[str, Any], *, as_json: bool, summary_lines: Sequence[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in summary_lines:
            print(line)


def _load_brief_file(path: str | Path) -> SessionBrief:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SystemExit(f"Brief file {path} must contain a JSON object")
    return SessionBrief.from_dict(raw)


def _load_session_source(
    session_path: str | Path,
    *,
    brief_path: str | Path | None = None,
    run_id: str | None = None,
) -> tuple[SessionBrief, list[CanonicalEvent]]:
    src = Path(session_path)
    brief_override = _load_brief_file(brief_path) if brief_path is not None else None
    if src.suffix in {".db", ".sqlite", ".sqlite3"}:
        with SQLiteStore(src) as store:
            runs = store.list_runs()
            if not runs:
                raise SystemExit(f"No sessions found in database {src}")
            target_run = None
            if run_id is not None:
                for r in runs:
                    if r["run_id"] == run_id:
                        target_run = r
                        break
            if target_run is None:
                target_run = runs[-1]
            sid = str(target_run["session_id"])
            rid = str(target_run["run_id"])
            sess_row = store.get_session(sid, rid)
            brief = brief_override
            if brief is None and sess_row is not None and sess_row.get("brief_json"):
                brief = SessionBrief.from_dict(json.loads(str(sess_row["brief_json"])))
            if brief is None:
                raise SystemExit(f"Database session {sid!r} lacks brief_json; pass --brief")
            events: list[CanonicalEvent] = []
            for row in store.get_events(sid, rid):
                if row.get("payload_json"):
                    events.append(CanonicalEvent.from_dict(json.loads(str(row["payload_json"]))))
            return brief, events

    loaded_brief, events = import_events_jsonl(src)
    brief = brief_override or loaded_brief
    if brief is None:
        raise SystemExit(f"Session {src} has no SessionBrief header; pass --brief")
    return brief, events


def _load_policy_and_cal(
    policy_path: str | Path,
    *,
    calibration_path: str | Path | None = None,
) -> tuple[PolicyConfig, CalibrationModel | CalibrationRegistry | None]:
    if calibration_path is not None:
        cal_model = CalibrationModel.load(calibration_path)
        policy = load_policy(
            policy_path,
            calibration_metadata=cal_model.to_metadata(),
            allow_null_calibration=True,
        )
        return policy, cal_model
    cal_reg = default_calibration_registry()
    policy = load_policy(
        policy_path,
        calibration_metadata=default_calibration_metadata(),
        allow_null_calibration=True,
    )
    return policy, (cal_reg if policy.classifier.calibration else None)


def _make_binder_from_env(brief: SessionBrief) -> RecipientBinder:
    recipient_id = brief.recipient.participant_id or "ME"
    rec_az: float | None = None
    seats: dict[str, float] = {}
    raw_seats = os.environ.get("COACH_SEAT_AZIMUTHS", "").strip()
    if raw_seats:
        parsed = json.loads(raw_seats)
        if isinstance(parsed, Mapping):
            for k, v in parsed.items():
                key = str(k)
                deg = float(v)
                if key in {"ME", recipient_id}:
                    rec_az = deg
                else:
                    seats[key] = deg
    cfg = BinderConfig(
        recipient_id="ME",
        recipient_azimuth_deg=rec_az,
        seat_azimuths_deg=seats,
    )
    binder = RecipientBinder(session_id=brief.session_id, config=cfg)
    raw_dir = os.environ.get("COACH_EARSHOT_DIRECTION_JSON", "").strip()
    if raw_dir:
        if raw_dir.startswith("{") or "\n" in raw_dir:
            binder.feed_direction_jsonl(raw_dir)
        else:
            p = Path(raw_dir)
            if p.exists():
                binder.feed_direction_jsonl(p.read_text(encoding="utf-8"))
    return binder


def _create_classifier_from_args(
    clf_name: str,
    model_dir: str | None,
    *,
    clock_ms: Callable[[], int] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if model_dir is not None:
        kwargs["model_dir"] = model_dir
    if clock_ms is not None:
        kwargs["clock_ms"] = clock_ms
    return create_classifier(clf_name, **kwargs)


def _make_provider_from_args(
    args: argparse.Namespace,
    policy: PolicyConfig,
    *,
    clock_ms: Callable[[], int] | None = None,
) -> ProviderAdapter | None:
    if not args.provider_base_url or not args.provider_model:
        return None
    p_cfg = ProviderConfig(
        provider=args.provider_name,
        base_url=args.provider_base_url,
        model_id=args.provider_model,
        api_key_env=args.provider_api_key_env,
        local_only=bool(args.local_only),
        breaker_consecutive_failures=policy.generator.breaker.consecutive_failures,
        breaker_open_ms=policy.generator.breaker.open_ms,
    )
    if clock_ms is not None:
        return ProviderAdapter(config=p_cfg, clock_ms=clock_ms)
    return ProviderAdapter(config=p_cfg)


def _displayed_cards_from_store(
    store: SQLiteStore, session_id: str, run_id: str
) -> list[DisplayedCardEvent]:
    cards: list[DisplayedCardEvent] = []
    for sug in store.get_suggestions(session_id, run_id):
        transitions = json.loads(sug["transitions_json"])
        disp_ms: int | None = None
        for tr in transitions:
            if isinstance(tr, Mapping) and tr.get("transition") in {
                "displayed",
                "shadow_displayed",
            }:
                disp_ms = int(tr["at_ms"])
                break
        if disp_ms is None:
            continue
        refs_raw = json.loads(sug["evidence_refs_json"])
        refs = tuple(SegmentRef.from_dict(r) for r in refs_raw if isinstance(r, Mapping))
        cards.append(
            DisplayedCardEvent(
                suggestion_id=str(sug["suggestion_id"]),
                candidate_id=str(sug["candidate_id"]),
                session_id=str(sug["session_id"]),
                category=Category(str(sug["category"])),
                display_ms=disp_ms,
                evidence_refs=refs,
                text=str(sug.get("text") or ""),
                route=Route(str(sug["route"])),
                origin=Origin(str(sug["origin"])),
            )
        )
    return cards


def _cmd_report(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        run_id = _resolve_run_id(store, args.session, args.run_id)
        report = build_session_report(store, args.session, run_id)
    payload = report.to_dict()
    lines = [
        f"Session: {report.session_id} (run={report.run_id})",
        f"Mode: {report.mode} | Runtime: {report.runtime_mode} | Consent: {report.consent_basis}",
        f"Duration: {report.duration_ms} ms | Cards: {report.displayed_card_count} "
        f"(budget={report.effective_max_cards_per_10_min}/10m, compliant={report.budget_compliant})",
        f"Invariants passed: {report.invariants.passed} "
        f"({len(report.invariants.violations)} violations)",
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0 if report.invariants.passed else 2


def _cmd_explain(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        run_id = _resolve_run_id(store, args.session, args.run_id)
        explanation = explain_silence(
            store, args.session, run_id, at_ms=args.at_ms, window_ms=args.window_ms
        )
    payload = explanation.to_dict()
    lines = [
        f"Silence explanation for {explanation.session_id} (run={explanation.run_id}) "
        f"at {explanation.at_ms} ms (window={explanation.window_ms} ms): "
        f"{len(explanation.anchors)} anchors"
    ]
    for a in explanation.anchors:
        lines.append(
            f"  - {a.candidate_id} @ {a.anchor_received_ms}ms: "
            f"outcome={a.outcome} state={a.terminal_state} reason={a.terminal_reason}"
        )
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        diff = diff_runs(store, args.run_a, args.run_b, session_id=args.session)
        payload = diff.to_dict()
        if args.labels or args.bootstrap:
            labels = load_labels_jsonl(args.labels) if args.labels else []
            cards_a = _displayed_cards_from_store(store, diff.session_id, args.run_a)
            cards_b = _displayed_cards_from_store(store, diff.session_id, args.run_b)
            payload["bootstrap"] = paired_conversation_bootstrap(
                cards_a,
                cards_b,
                labels,
                metric=args.bootstrap_metric,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
    lines = [
        f"Diff for {diff.session_id}: {diff.run_a} -> {diff.run_b}",
        f"Gained: {len(diff.gained)} | Lost: {len(diff.lost)} | Changed: {len(diff.changed)}",
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_delete_session(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        summary = store.delete_session_text(args.session)
    payload = asdict(summary)
    lines = [
        f"Deleted session text for {summary.session_id}: "
        f"raw_events={summary.raw_events_deleted}, events_redacted={summary.events_redacted}, "
        f"decisions_redacted={summary.decisions_redacted}, suggestions_redacted={summary.suggestions_redacted}"
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


async def _stream_wav_to_peecee(
    adapter: PeeceeRealtimeAdapter, wav_path: str | Path, *, frame_ms: int = 160
) -> None:
    with wave.open(str(wav_path), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    if pcm:
        await adapter.send_audio_pcm16(pcm, pace_realtime=True)
    await adapter.commit_audio()


class _StoreRawSink:
    def __init__(
        self, store: SQLiteStore, run_id: str, controller: SuggestionController, brief: SessionBrief
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._controller = controller
        self._brief = brief

    def record_raw(
        self,
        *,
        session_id: str,
        source_id: str,
        received_ms: int,
        payload: Mapping[str, Any] | str,
    ) -> Any:
        if self._controller.mode is Mode.OFF_THE_RECORD:
            return False
        return self._store.record_raw_event(
            session_id=session_id,
            run_id=self._run_id,
            source_id=source_id,
            received_ms=received_ms,
            payload=payload,
            retention=self._brief.retention,
        )


class _AnchoredSessionClock:
    """Monotonic clock that stays synchronized with replayed event timestamps."""

    def __init__(self, initial_ms: int = 0) -> None:
        self._offset_ms = max(0, initial_ms)
        self._origin_ns = time.monotonic_ns()

    def sync_min_ms(self, target_ms: int) -> None:
        cur = self.now_ms()
        if target_ms > cur:
            self._offset_ms += target_ms - cur

    def now_ms(self) -> int:
        elapsed_ms = (time.monotonic_ns() - self._origin_ns) // 1_000_000
        return self._offset_ms + max(0, int(elapsed_ms))


def _sync_serve_clock(clock: Clock, rx_ms: int) -> None:
    if isinstance(clock, ManualClock):
        if rx_ms > clock.now_ms():
            clock.set_ms(rx_ms)
    elif isinstance(clock, _AnchoredSessionClock):
        clock.sync_min_ms(rx_ms)


async def _advance_replay_timers(controller: SuggestionController, clock: ManualClock) -> None:
    for _ in range(256):
        next_ms = controller.next_pending_card_wake_ms()
        if next_ms is None:
            break
        if next_ms > clock.now_ms():
            clock.set_ms(next_ms)
        await controller.tick()
        await controller.drain()


async def _serve_async(args: argparse.Namespace) -> int:
    brief = _load_brief_file(args.brief)
    policy, cal_obj = _load_policy_and_cal(args.policy, calibration_path=args.calibration)
    is_file_source = args.source in {"replay", "deepgram"} or (
        args.source == "peecee" and bool(args.stream_file)
    )
    clock: Clock
    if is_file_source:
        clock = ManualClock(0) if args.exit_on_eof else _AnchoredSessionClock(0)
    else:
        clock = MonotonicSessionClock(0)
    clf_name = args.classifier or policy.classifier.version
    clf = _create_classifier_from_args(clf_name, args.classifier_model_dir, clock_ms=clock.now_ms)
    provider = _make_provider_from_args(args, policy, clock_ms=clock.now_ms)
    mode = Mode(args.mode)
    run_id = args.run_id or f"serve-{uuid.uuid4().hex[:10]}"

    with SQLiteStore(args.db) as store:
        rec_buf = BoundedRecorderBuffer()
        ui_server = CardUIServer(
            session_id=brief.session_id,
            port=args.ui_port,
            session_token=args.ui_token,
        )
        controller = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=clf,
            calibration=cal_obj,
            provider=provider,
            store=store,
            run_id=run_id,
            mode=mode,
            clock=clock,
            telemetry_buffer=rec_buf,
            ui_server=ui_server,
        )
        await ui_server.start()
        await controller.start()

        banner = {
            "status": "ready",
            "session_id": brief.session_id,
            "run_id": run_id,
            "ui_url": ui_server.http_url,
            "ws_url": ui_server.ws_url,
            "port": ui_server.port,
            "token": ui_server.token,
        }
        if args.json:
            print(json.dumps(banner, sort_keys=True), flush=True)
        else:
            print(f"UI: {ui_server.http_url}", flush=True)
            print(f"WS: {ui_server.ws_url}", flush=True)
            print(f"TOKEN: {ui_server.token}", flush=True)
            print(f"SESSION: {brief.session_id} RUN: {run_id}", flush=True)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, stop_event.set)

        source_task = asyncio.create_task(
            _run_serve_source(
                args,
                brief,
                policy,
                store,
                rec_buf,
                controller,
                stop_event,
                clock=clock,
            )
        )
        stop_task = asyncio.create_task(stop_event.wait())
        try:
            done, _ = await asyncio.wait(
                {source_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                if not source_task.done():
                    source_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await source_task
            else:
                if not source_task.cancelled():
                    source_task.result()
                if args.exit_on_eof:
                    await controller.drain()
                    if isinstance(clock, ManualClock):
                        await _advance_replay_timers(controller, clock)
                    await controller.drain()
                else:
                    await stop_task
        finally:
            for task in (source_task, stop_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                    loop.remove_signal_handler(sig)
            await controller.stop()
            await ui_server.stop()
            if provider is not None:
                await provider.aclose()
        inv = check_conservation_invariants(store, brief.session_id, run_id)
        if not inv.passed:
            return 2
        return 0


async def _run_serve_source(
    args: argparse.Namespace,
    brief: SessionBrief,
    policy: PolicyConfig,
    store: SQLiteStore,
    rec_buf: BoundedRecorderBuffer,
    controller: SuggestionController,
    stop_event: asyncio.Event,
    *,
    clock: Clock,
) -> None:
    binder = _make_binder_from_env(brief)
    if args.source == "replay" or (args.source == "peecee" and args.stream_file):
        if not args.stream_file:
            raise SystemExit(f"--source {args.source} requires --stream-file")
        if args.source == "peecee":
            raw_text = Path(args.stream_file).read_text(encoding="utf-8")
            try:
                events = parse_nemo_offline_json(
                    raw_text, session_id=brief.session_id, binder=binder
                )
            except ValueError, KeyError, json.JSONDecodeError:
                _, events = import_events_jsonl(args.stream_file)
            for ev in events:
                if stop_event.is_set():
                    break
                _sync_serve_clock(clock, ev.received_ms)
                await controller.submit_event(ev)
        else:
            player = ReplayPlayer.from_jsonl(args.stream_file)
            t_mode = ReplayTimingMode.SCALED if args.speed != 1.0 else ReplayTimingMode.RECORDED
            play_clock = clock if isinstance(clock, ManualClock) else None
            async for ev in player.play(mode=t_mode, speed=args.speed, clock=play_clock):
                if stop_event.is_set():
                    break
                _sync_serve_clock(clock, ev.received_ms)
                await controller.submit_event(ev)
        return

    if args.source == "deepgram":
        if not args.stream_file:
            raise SystemExit("--source deepgram requires --stream-file")
        norm = DeepgramNormalizer(
            session_id=brief.session_id,
            confidence_floor=policy.abstain.asr_confidence_floor,
        )
        for line_no, raw in enumerate(
            Path(args.stream_file).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if stop_event.is_set():
                break
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            rx_ms = int(obj.get("received_ms", line_no * 200))
            _sync_serve_clock(clock, rx_ms)
            for ev in norm.normalize(obj, rx_ms):
                await controller.submit_event(ev)
        return

    # Realtime peecee WS source
    ws_url = args.ws_url or DEFAULT_WS_URL
    exe_path = Path(args.peecee_cmd) if args.peecee_cmd else DEFAULT_PEECEE_SCRIPT
    should_spawn = bool(args.peecee_cmd) or (args.ws_url is None and exe_path.exists())
    pc_cfg = PeeceeConfig(ws_url=ws_url, executable=exe_path)
    proc = (
        PeeceeSpeechProcess(
            config=pc_cfg,
            session_id=brief.session_id,
            telemetry_sink=rec_buf,
            health_callback=controller.report_adapter_health,
        )
        if should_spawn
        else None
    )
    segmenter = NemotronSegmenter(session_id=brief.session_id, binder=binder)
    raw_sink = _StoreRawSink(store, controller.run_id, controller, brief)
    adapter = PeeceeRealtimeAdapter(
        session_id=brief.session_id,
        config=pc_cfg,
        segmenter=segmenter,
        binder=binder,
        clock=clock,
        telemetry_sink=rec_buf,
        raw_sink=raw_sink,
        health_callback=controller.report_adapter_health,
        process=proc,
    )
    wav_task: asyncio.Task[None] | None = None
    try:
        try:
            await adapter.connect()
        except (PeeceeLeaseUnavailableError, PeeceeProcessError, OSError) as exc:
            await controller.report_adapter_health(
                HealthState.NO_TRANSCRIPT, "peecee_connect_failed", {"err": str(exc)}
            )
            return
        if args.audio_wav:
            wav_task = asyncio.create_task(_stream_wav_to_peecee(adapter, args.audio_wav))
        async for ev in adapter.events():
            if stop_event.is_set():
                break
            await controller.submit_event(ev)
        if wav_task is not None:
            await wav_task
    finally:
        if wav_task is not None and not wav_task.done():
            wav_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wav_task
        await adapter.close()


def _cmd_serve(args: argparse.Namespace) -> int:
    return asyncio.run(_serve_async(args))


def _cmd_replay(args: argparse.Namespace) -> int:
    brief, events = _load_session_source(args.session, brief_path=args.brief, run_id=args.run_id)
    policy, cal_obj = _load_policy_and_cal(args.policy, calibration_path=args.calibration)
    clf_name = args.classifier or policy.classifier.version
    clf = _create_classifier_from_args(clf_name, args.classifier_model_dir)
    provider = _make_provider_from_args(args, policy)
    mode = Mode(args.mode)
    run_id = args.run_id or f"replay-{uuid.uuid4().hex[:10]}"

    with SQLiteStore(args.db) as store:
        res = simulate_replay(
            brief,
            events,
            policy,
            clf,
            calibration=cal_obj,
            store=store,
            run_id=run_id,
            mode=mode,
            provider=provider,
            simulated_latency_ms=args.simulated_latency_ms,
            cache_classifier=not args.no_cache,
        )
        inv = check_conservation_invariants(store, brief.session_id, run_id)
        report = build_session_report(store, brief.session_id, run_id)

        payload: dict[str, Any] = {
            "session_id": res.session_id,
            "run_id": res.run_id,
            "speed": args.speed,
            "duration_ms": res.duration_ms,
            "decisions_count": res.decisions_count,
            "dropped_count": res.dropped_count,
            "routed_count": res.routed_count,
            "withdrawn_count": res.withdrawn_count,
            "expired_count": res.expired_count,
            "displayed_cards": [
                {
                    "suggestion_id": c.suggestion_id,
                    "candidate_id": c.candidate_id,
                    "category": str(c.category),
                    "display_ms": c.display_ms,
                    "text": c.text,
                    "route": str(c.route),
                    "origin": str(c.origin),
                }
                for c in res.displayed_cards
            ],
            "invariants": inv.to_dict(),
            "report": report.to_dict(),
        }

        labels: list[LabelRecord] = load_labels_jsonl(args.labels) if args.labels else []
        if args.labels:
            payload["evaluation"] = match_opportunities(
                res.displayed_cards,
                labels,
                settle_ms=policy.display.settle_ms,
                duration_ms=res.duration_ms,
            )
        if args.split_by:
            split_items: Sequence[Any] = labels if labels else [brief]
            splits = (
                split_by_series(split_items)
                if args.split_by == "series"
                else split_by_conversation(split_items)
            )
            payload["splits"] = splits.to_dict()
            eval_convs = splits.eval_conversations
        else:
            eval_convs = (brief.session_id,)
        if args.eval_log:
            payload["eval_look_count"] = EvalLookCounter(args.eval_log).record_look(
                run_id,
                split_by=args.split_by or "conversation",
                eval_conversations=eval_convs,
            )
        if args.sweep_t_display:
            t_vals = [float(x.strip()) for x in args.sweep_t_display.split(",") if x.strip()]
            payload["sweep"] = sweep_t_display(
                [(brief, events)],
                labels,
                policy,
                clf,
                t_vals,
                calibration=cal_obj,
                store=store,
                simulated_latency_ms=args.simulated_latency_ms,
            )

    lines = [
        f"Replay {res.session_id} (run={res.run_id}): "
        f"decisions={res.decisions_count}, displayed={len(res.displayed_cards)}, "
        f"invariants={inv.passed}"
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0 if inv.passed else 2


def _cmd_bench(args: argparse.Namespace) -> int:
    brief = _load_brief_file(args.brief) if args.brief else None
    policy = load_policy(args.policy, allow_null_calibration=True) if args.policy else None
    clf = _create_classifier_from_args(args.classifier, args.classifier_model_dir)
    batch_sizes = tuple(int(x.strip()) for x in args.batch_sizes.split(",") if x.strip())
    token_buckets = tuple(int(x.strip()) for x in args.tokens.split(",") if x.strip())
    result = run_latency_benchmark(
        args.contexts,
        clf,
        token_buckets=token_buckets,
        batch_sizes=batch_sizes,
        warmup_calls=args.warmup,
        timed_calls=args.timed_calls,
        contention=args.contention,
        placement=args.placement,
        precision=args.precision,
        policy=policy,
        brief=brief,
    )
    lines = [
        f"Benchmark {result['classifier_version']} ({len(result['cells'])} cells):",
        *[
            f"  tokens={c['tokens']} batch={c['batch_size']} "
            f"p50={c['p50_ms']:.2f}ms p95={c['p95_ms']:.2f}ms p99={c['p99_ms']:.2f}ms"
            for c in result["cells"]
        ],
    ]
    _emit_output(result, as_json=args.json, summary_lines=lines)
    return 0


def _build_inputs_by_candidate_from_session(
    session_path: str | Path,
    *,
    brief_path: str | Path | None = None,
    policy_path: str | Path | None = None,
) -> tuple[SessionBrief, dict[str, ClassifierInput], dict[str, ClassifierInput]]:
    brief, events = _load_session_source(session_path, brief_path=brief_path)
    policy = load_policy(policy_path, allow_null_calibration=True) if policy_path else None
    ordered = sorted(events, key=lambda e: (e.received_ms, e.source_sequence, e.event_id))
    state = CanonicalState(brief)
    by_cand: dict[str, ClassifierInput] = {}
    by_seg: dict[str, ClassifierInput] = {}
    recipient_ids = {"ME", brief.recipient.participant_id}
    for ev in ordered:
        state.apply(ev)
        if ev.kind.value == "segment" and ev.stability is Stability.COMMITTED and ev.segment_id:
            snap = state.snapshot()
            seg = snap.segments_by_id.get(ev.segment_id)
            if seg is None or seg.speaker_id in recipient_ids:
                continue
            for built in build_decision_contexts(snap, anchor_segments=[seg], policy=policy):
                if (
                    not built.dropped
                    and built.context is not None
                    and built.serialized_text is not None
                ):
                    cin = ClassifierInput.from_build_result(built)
                    by_cand[cin.context.candidate_id] = cin
                    for ref in cin.context.anchor_refs:
                        by_seg[ref.segment_id] = cin
    return brief, by_cand, by_seg


def _load_training_rows(
    data_path: str | Path,
    *,
    session_path: str | Path | None = None,
    brief_path: str | Path | None = None,
    policy_path: str | Path | None = None,
) -> tuple[
    list[tuple[ClassifierInput, LabelRecord]],
    list[tuple[ClassifierInput, ClassifierResult | None, LabelRecord]],
    SessionBrief | None,
]:
    by_cand: dict[str, ClassifierInput] = {}
    by_seg: dict[str, ClassifierInput] = {}
    brief: SessionBrief | None = _load_brief_file(brief_path) if brief_path else None
    if session_path is not None:
        s_brief, by_cand, by_seg = _build_inputs_by_candidate_from_session(
            session_path, brief_path=brief_path, policy_path=policy_path
        )
        brief = brief or s_brief

    train_examples: list[tuple[ClassifierInput, LabelRecord]] = []
    cal_rows: list[tuple[ClassifierInput, ClassifierResult | None, LabelRecord]] = []
    with Path(data_path).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, Mapping):
                raise SystemExit(f"Line {line_no} in {data_path} must be a JSON object")
            if "context" in obj and "serialized_text" in obj:
                ctx = DecisionContext.from_dict(obj["context"])
                cin = ClassifierInput(context=ctx, serialized_text=str(obj["serialized_text"]))
                lbl = LabelRecord.from_dict(obj["label"])
                clf_res = (
                    ClassifierResult.from_dict(obj["classifier_result"])
                    if obj.get("classifier_result") is not None
                    else None
                )
                train_examples.append((cin, lbl))
                cal_rows.append((cin, clf_res, lbl))
            else:
                lbl = LabelRecord.from_dict(obj)
                cin_match: ClassifierInput | None = None
                if lbl.candidate_id and lbl.candidate_id in by_cand:
                    cin_match = by_cand[lbl.candidate_id]
                else:
                    for ref in lbl.evidence_refs:
                        if ref.segment_id in by_seg:
                            cin_match = by_seg[ref.segment_id]
                            break
                if cin_match is None:
                    raise SystemExit(
                        f"Label {lbl.label_id!r} cannot resolve ClassifierInput; pass --session"
                    )
                train_examples.append((cin_match, lbl))
                cal_rows.append((cin_match, None, lbl))
    return train_examples, cal_rows, brief


def _cmd_train(args: argparse.Namespace) -> int:
    examples, _, _ = _load_training_rows(
        args.data,
        session_path=args.session,
        brief_path=args.brief,
        policy_path=args.policy,
    )
    if not examples:
        raise SystemExit(f"No training examples found in {args.data}")

    if args.categories:
        cats = tuple(Category(c.strip()) for c in args.categories.split(",") if c.strip())
    else:
        cats = tuple(sorted(PREDICTABLE_CATEGORIES, key=lambda c: c.value))

    out_dir = Path(args.out)
    if args.backend == "modernbert":
        saved_path = train_modernbert(
            examples,
            output_dir=out_dir,
            active_categories=cats,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        payload: dict[str, Any] = {
            "backend": "modernbert",
            "out_dir": str(saved_path),
            "n_examples": len(examples),
            "categories": [c.value for c in cats],
        }
    else:
        embedder: TextEmbedder
        if args.embedder == "ollama":
            embedder = OllamaEmbedder(
                base_url=args.ollama_url or "http://127.0.0.1:11434",
                model=args.ollama_model or "nomic-embed-text:latest",
            )
        elif args.embedder == "hf":
            embedder = HuggingFaceEmbedder(
                model_name=args.hf_model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            embedder = HashedTfidfEmbedder()
        model = fit_embed_logreg(
            examples,
            embedder=embedder,
            active_categories=cats,
            classifier_version=args.classifier_version or "embed_logreg_v0",
        )
        saved_path = model.save(out_dir)
        payload = {
            "backend": "embed_logreg",
            "classifier_version": model.classifier_version,
            "out_dir": str(saved_path),
            "n_examples": len(examples),
            "categories": [c.value for c in cats],
        }

    lines = [f"Trained {payload['backend']} on {len(examples)} examples -> {saved_path}"]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    _, cal_rows, brief = _load_training_rows(
        args.data,
        session_path=args.session,
        brief_path=args.brief,
        policy_path=args.policy,
    )
    if not cal_rows:
        raise SystemExit(f"No calibration rows found in {args.data}")

    mode_str = args.mode or (str(brief.mode) if brief is not None else None)
    if mode_str is None:
        raise SystemExit("calibrate requires --mode or --brief")
    session_mode = SessionMode(mode_str)

    clf = _create_classifier_from_args(args.classifier, args.classifier_model_dir)
    need_clf_idx = [i for i, (_, res, _) in enumerate(cal_rows) if res is None]
    resolved_results: list[ClassifierResult | None] = [res for _, res, _ in cal_rows]
    if need_clf_idx:
        batch_inputs = [cal_rows[i][0] for i in need_clf_idx]
        preds = asyncio.run(clf.classify_batch(batch_inputs))
        for idx, pred in zip(need_clf_idx, preds, strict=True):
            resolved_results[idx] = pred

    fit_rows: list[tuple[DecisionContext, ClassifierResult, LabelRecord]] = []
    for (cin, _, lbl), clf_res in zip(cal_rows, resolved_results, strict=True):
        assert clf_res is not None
        fit_rows.append((cin.context, clf_res, lbl))

    in_space = args.input_space
    if in_space is None:
        prob_names = {"rules_v0", "opportunity-baseline-v0", "llm_baseline_v0"}
        in_space = "probability" if clf.classifier_version in prob_names else "logit"

    try:
        cal_model = fit_platt_calibrator(
            fit_rows,
            calibration_version=args.calibration_version,
            classifier_version=clf.classifier_version,
            mode=session_mode,
            input_space=in_space,
        )
    except CalibrationDataError as exc:
        print(f"Calibration rejected: {exc}", file=sys.stderr)
        return 2

    saved_path = cal_model.save(args.out)
    calibrated_pairs = [
        (cal_model.calibrate(ctx, res, mode=session_mode), lbl) for ctx, res, lbl in fit_rows
    ]
    eval_rep = evaluate_calibration(calibrated_pairs)
    payload: dict[str, Any] = {
        "calibration_version": cal_model.calibration_version,
        "classifier_version": cal_model.classifier_version,
        "mode": str(cal_model.mode),
        "out": str(saved_path),
        "n_samples": len(fit_rows),
        "evaluation": eval_rep.to_dict(),
    }
    lines = [
        f"Saved calibration {cal_model.calibration_version} ({cal_model.classifier_version}) -> {saved_path}"
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    if args.kappa:
        first_path = args.labels or args.out
        if not first_path:
            raise SystemExit("--kappa requires --labels (or --out) with first-pass labels")
        first = load_labels_jsonl(first_path)
        second = load_labels_jsonl(args.kappa)
        kappa_res = compute_cohens_kappa(first, second)
        lines = [
            f"Blind test-retest Cohen's kappa (n_paired={kappa_res['n_paired']}): "
            f"worthwhile={kappa_res['worthwhile']['kappa']} (low_n={kappa_res['worthwhile']['low_n']})"
        ]
        _emit_output(kappa_res, as_json=args.json, summary_lines=lines)
        return 0

    if not args.session:
        raise SystemExit("annotate requires --session unless --kappa is passed")
    if not args.out:
        raise SystemExit("annotate requires --out unless --kappa is passed")

    brief, events = _load_session_source(args.session, brief_path=args.brief)
    policy = load_policy(args.policy, allow_null_calibration=True) if args.policy else None

    pass1: Pass1Session | None = None
    pass2: Pass2Session | None = None
    if args.pass_num == 1:
        pass1 = Pass1Session(
            brief=brief,
            events=events,
            annotator=args.annotator,
            speed=args.speed,
            out_path=args.out,
        )
    else:
        src = SampleSource(args.sample_source)
        if args.review_queue:
            rq_raw = json.loads(Path(args.review_queue).read_text(encoding="utf-8"))
            rq_list = rq_raw.get("review_queue", rq_raw) if isinstance(rq_raw, Mapping) else rq_raw
            items = load_review_queue_items(
                brief, events, rq_list, policy=policy, sample_source=src
            )
        else:
            items = build_pass2_items(brief, events, policy=policy, sample_source=src)
        pass2 = Pass2Session(items=items, annotator=args.annotator, out_path=args.out)

    server = AnnotationServer(
        port=args.port,
        session_token=args.ui_token,
        pass1=pass1,
        pass2=pass2,
    )
    server.start()
    try:
        info = {
            "status": "ready",
            "pass": args.pass_num,
            "url": server.url,
            "port": server.port,
            "token": server.token,
            "items_count": len(pass2.list_items()) if pass2 is not None else 0,
        }
        if args.json:
            print(json.dumps(info, sort_keys=True), flush=True)
        else:
            print(f"Annotation UI (pass {args.pass_num}): {server.url}", flush=True)
            print(f"TOKEN: {server.token}", flush=True)

        stop_evt = False

        def _handle_sig(_signum: int, _frame: Any) -> None:
            nonlocal stop_evt
            stop_evt = True

        prev_int = signal.signal(signal.SIGINT, _handle_sig)
        prev_term = signal.signal(signal.SIGTERM, _handle_sig)
        try:
            while not stop_evt:
                signal.pause()
        except AttributeError:
            import time

            while not stop_evt:
                time.sleep(0.2)
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
    finally:
        server.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coach", description="Live Conversation Coach CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run live/shadow coach runtime + private UI")
    p_serve.add_argument("--brief", required=True, help="Path to SessionBrief JSON")
    p_serve.add_argument("--policy", required=True, help="Path to policy YAML")
    p_serve.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_serve.add_argument("--source", choices=["peecee", "deepgram", "replay"], default="peecee")
    p_serve.add_argument("--audio-wav", default=None, help="Optional 16kHz PCM WAV file to stream")
    p_serve.add_argument("--stream-file", default=None, help="Offline stream/JSONL file")
    p_serve.add_argument("--ws-url", default=None, help="Realtime peecee-speech WebSocket URL")
    p_serve.add_argument("--peecee-cmd", default=None, help="Path to peecee-speech launcher")
    p_serve.add_argument("--ui-port", type=int, default=8765, help="Loopback UI port (0=ephemeral)")
    p_serve.add_argument("--ui-token", default=None, help="Optional fixed UI session token")
    p_serve.add_argument("--local-only", action="store_true", help="Block hosted generator calls")
    p_serve.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.PROACTIVE.value,
        help="Initial runtime mode",
    )
    p_serve.add_argument("--classifier", default=None, help="Classifier name override")
    p_serve.add_argument(
        "--classifier-model-dir", default=None, help="Trained classifier directory"
    )
    p_serve.add_argument("--calibration", default=None, help="Path to CalibrationModel JSON")
    p_serve.add_argument("--provider-base-url", default=None, help="OpenAI-compatible base URL")
    p_serve.add_argument("--provider-model", default=None, help="Generator model ID")
    p_serve.add_argument("--provider-api-key-env", default=None, help="Env var holding API key")
    p_serve.add_argument("--provider-name", default="openai-compat", help="Provider label")
    p_serve.add_argument("--run-id", default=None, help="Run ID")
    p_serve.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    p_serve.add_argument("--exit-on-eof", action="store_true", help="Exit after stream EOF")
    p_serve.add_argument("--json", action="store_true", help="Emit JSON output")
    p_serve.set_defaults(handler=_cmd_serve)

    p_rep_play = sub.add_parser("replay", help="Deterministic policy-level replay & eval (§12.6)")
    p_rep_play.add_argument("--session", required=True, help="Session JSONL or SQLite DB path")
    p_rep_play.add_argument("--policy", required=True, help="Path to policy YAML")
    p_rep_play.add_argument("--brief", default=None, help="Path to SessionBrief JSON")
    p_rep_play.add_argument("--db", default=":memory:", help="Output SQLite DB (default :memory:)")
    p_rep_play.add_argument("--speed", default="1.0", help="Replay speed float or 'stepped'")
    p_rep_play.add_argument("--labels", default=None, help="Path to LabelRecord JSONL")
    p_rep_play.add_argument("--split-by", choices=["conversation", "series"], default=None)
    p_rep_play.add_argument("--eval-log", default=None, help="Path to EvalLookCounter JSON")
    p_rep_play.add_argument("--sweep-t-display", default=None, help="CSV of t_display values")
    p_rep_play.add_argument("--run-id", default=None, help="Run ID")
    p_rep_play.add_argument("--classifier", default=None, help="Classifier name override")
    p_rep_play.add_argument("--classifier-model-dir", default=None)
    p_rep_play.add_argument("--calibration", default=None)
    p_rep_play.add_argument("--provider-base-url", default=None)
    p_rep_play.add_argument("--provider-model", default=None)
    p_rep_play.add_argument("--provider-api-key-env", default=None)
    p_rep_play.add_argument("--provider-name", default="openai-compat")
    p_rep_play.add_argument("--local-only", action="store_true")
    p_rep_play.add_argument("--simulated-latency-ms", type=float, default=None)
    p_rep_play.add_argument("--no-cache", action="store_true")
    p_rep_play.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.PROACTIVE.value)
    p_rep_play.add_argument("--json", action="store_true", help="Emit JSON output")
    p_rep_play.set_defaults(handler=_cmd_replay)

    p_report = sub.add_parser("report", help="Generate per-session report (§13.5)")
    p_report.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_report.add_argument("--session", required=True, help="Session ID")
    p_report.add_argument("--run-id", default=None, help="Run ID (defaults to latest run)")
    p_report.add_argument("--json", action="store_true", help="Emit JSON output")
    p_report.set_defaults(handler=_cmd_report)

    p_explain = sub.add_parser("explain", help="Explain silence in a time window (§13.3)")
    p_explain.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_explain.add_argument("--session", required=True, help="Session ID")
    p_explain.add_argument("--at-ms", required=True, type=int, help="Window end timestamp (ms)")
    p_explain.add_argument("--window-ms", type=int, default=60_000, help="Window length (ms)")
    p_explain.add_argument("--run-id", default=None, help="Run ID (defaults to latest run)")
    p_explain.add_argument("--json", action="store_true", help="Emit JSON output")
    p_explain.set_defaults(handler=_cmd_explain)

    p_diff = sub.add_parser("diff", help="Diff two runs of a session (§13.4)")
    p_diff.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_diff.add_argument("--run-a", required=True, help="Baseline run ID")
    p_diff.add_argument("--run-b", required=True, help="Candidate run ID")
    p_diff.add_argument("--session", default=None, help="Session ID (auto-detected if unique)")
    p_diff.add_argument("--labels", default=None, help="LabelRecord JSONL for paired bootstrap")
    p_diff.add_argument("--bootstrap", action="store_true", help="Compute paired conv bootstrap")
    p_diff.add_argument("--bootstrap-metric", default="useful_count")
    p_diff.add_argument("--n-resamples", type=int, default=1000)
    p_diff.add_argument("--seed", type=int, default=0)
    p_diff.add_argument("--json", action="store_true", help="Emit JSON output")
    p_diff.set_defaults(handler=_cmd_diff)

    p_del = sub.add_parser("delete-session", help="Delete session text per §14.2")
    p_del.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_del.add_argument("--session", required=True, help="Session ID")
    p_del.add_argument("--json", action="store_true", help="Emit JSON output")
    p_del.set_defaults(handler=_cmd_delete_session)

    p_bench = sub.add_parser("bench", help="Benchmark classifier latency & memory (§10.2)")
    p_bench.add_argument(
        "--contexts", required=True, help="Path to serialized contexts/events JSONL"
    )
    p_bench.add_argument("--classifier", required=True, help="Classifier name")
    p_bench.add_argument("--batch-sizes", default="1,4,8", help="CSV batch sizes")
    p_bench.add_argument("--tokens", default="128,256,512,1024,2048", help="CSV token buckets")
    p_bench.add_argument("--warmup", type=int, default=2)
    p_bench.add_argument("--timed-calls", type=int, default=20)
    p_bench.add_argument("--contention", default="idle")
    p_bench.add_argument("--placement", default="in_process")
    p_bench.add_argument("--precision", default="fp32")
    p_bench.add_argument("--brief", default=None)
    p_bench.add_argument("--policy", default=None)
    p_bench.add_argument("--classifier-model-dir", default=None)
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(handler=_cmd_bench)

    p_train = sub.add_parser("train", help="Train embed_logreg or modernbert classifier (§10.1)")
    p_train.add_argument("--data", required=True, help="Path to labeled training JSONL")
    p_train.add_argument("--out", required=True, help="Output model directory")
    p_train.add_argument(
        "--backend", choices=["embed_logreg", "modernbert"], default="embed_logreg"
    )
    p_train.add_argument("--session", default=None, help="Events JSONL if --data has LabelRecords")
    p_train.add_argument("--brief", default=None)
    p_train.add_argument("--policy", default=None)
    p_train.add_argument("--categories", default=None, help="CSV categories")
    p_train.add_argument("--classifier-version", default=None)
    p_train.add_argument(
        "--embedder", choices=["hashed_tfidf", "ollama", "hf"], default="hashed_tfidf"
    )
    p_train.add_argument("--ollama-url", default=None)
    p_train.add_argument("--ollama-model", default=None)
    p_train.add_argument("--hf-model", default=None)
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--max-length", type=int, default=2048)
    p_train.add_argument("--json", action="store_true")
    p_train.set_defaults(handler=_cmd_train)

    p_cal = sub.add_parser("calibrate", help="Fit per-head Platt calibration (§5.7, §12.4)")
    p_cal.add_argument("--data", required=True, help="Path to random-sampled LabelRecord JSONL")
    p_cal.add_argument("--classifier", required=True, help="Classifier name")
    p_cal.add_argument("--mode", choices=[m.value for m in SessionMode], default=None)
    p_cal.add_argument("--out", required=True, help="Output CalibrationModel JSON path")
    p_cal.add_argument("--session", default=None)
    p_cal.add_argument("--brief", default=None)
    p_cal.add_argument("--policy", default=None)
    p_cal.add_argument("--classifier-model-dir", default=None)
    p_cal.add_argument("--calibration-version", default="cal-fitted-v0")
    p_cal.add_argument("--input-space", choices=["logit", "probability"], default=None)
    p_cal.add_argument("--json", action="store_true")
    p_cal.set_defaults(handler=_cmd_calibrate)

    p_ann = sub.add_parser(
        "annotate", help="Run 2-pass locked annotation UI or kappa (§12.2–§12.4)"
    )
    p_ann.add_argument("--session", default=None, help="Session JSONL or DB path")
    p_ann.add_argument("--out", default=None, help="Output LabelRecord JSONL path")
    p_ann.add_argument("--port", type=int, default=8766, help="Loopback HTTP port (0=ephemeral)")
    p_ann.add_argument("--pass", dest="pass_num", type=int, choices=[1, 2], default=2)
    p_ann.add_argument("--kappa", default=None, help="Blind retest LabelRecord JSONL path")
    p_ann.add_argument("--labels", default=None, help="First-pass LabelRecord JSONL for --kappa")
    p_ann.add_argument("--brief", default=None)
    p_ann.add_argument("--policy", default=None)
    p_ann.add_argument("--annotator", default="recipient")
    p_ann.add_argument(
        "--sample-source",
        choices=[s.value for s in SampleSource],
        default=SampleSource.RANDOM.value,
    )
    p_ann.add_argument("--review-queue", default=None, help="Review queue JSON path")
    p_ann.add_argument("--speed", type=float, default=1.0)
    p_ann.add_argument("--ui-token", default=None)
    p_ann.add_argument("--json", action="store_true")
    p_ann.set_defaults(handler=_cmd_annotate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
