"""CLI entry point for the Live Conversation Coach (`coach`)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import signal
import sys
import time
import uuid
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from coach.annotation_tool import (
    AnnotationServer,
    BlindRetestSession,
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
    AnnotationPass,
    AttentionConfig,
    CanonicalEvent,
    Category,
    ClassifierResult,
    DecisionContext,
    HealthState,
    LabelRecord,
    Mode,
    NoneReason,
    Origin,
    RecipientConfig,
    RetentionConfig,
    Route,
    SampleSource,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    ValidityInterval,
)
from coach.opportunity_classifier import (
    ClassifierInput,
    HashedTfidfEmbedder,
    HuggingFaceEmbedder,
    OllamaEmbedder,
    OpenAICompatibleConfig,
    TextEmbedder,
    create_classifier,
    fit_embed_logreg,
    train_modernbert,
)
from coach.provider_adapter import ProviderAdapter, ProviderConfig, ProviderError
from coach.replay_evaluator import (
    DisplayedCardEvent,
    EvalLookCounter,
    ReplayResult,
    async_simulate_replay,
    match_opportunities,
    paired_conversation_bootstrap,
    run_latency_benchmark,
    simulate_dataset_replay,
    split_by_conversation,
    split_by_series,
    sweep_dataset_t_display,
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


def _synthesize_default_brief(session_id: str) -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id=session_id,
        objective="Verify session objectives and open action items.",
        recipient=RecipientConfig(participant_id="ME", role="participant"),
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        retention=RetentionConfig(),
        attention=AttentionConfig(max_cards_per_10_min=3, proactive=True),
    )


def _load_session_source(
    session_path: str | Path,
    *,
    brief_path: str | Path | None = None,
    run_id: str | None = None,
    allow_default_brief: bool = False,
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
            if brief is None and allow_default_brief:
                brief = _synthesize_default_brief(sid)
            if brief is None:
                raise SystemExit(f"Database session {sid!r} lacks brief_json; pass --brief")
            events: list[CanonicalEvent] = []
            for row in store.get_events(sid, rid):
                if row.get("payload_json"):
                    events.append(CanonicalEvent.from_dict(json.loads(str(row["payload_json"]))))
            return brief, events

    loaded_brief, events = import_events_jsonl(src)
    brief = brief_override or loaded_brief
    if brief is None and allow_default_brief:
        sid = events[0].session_id if events else "session-001"
        brief = _synthesize_default_brief(sid)
    if brief is None:
        raise SystemExit(f"Session {src} has no SessionBrief header; pass --brief")
    return brief, events


def _load_policy_and_cal(
    policy_path: str | Path,
    *,
    calibration_path: str | Path | None = None,
) -> tuple[PolicyConfig, CalibrationModel | CalibrationRegistry | None]:
    effective_cal_path: str | Path | None = calibration_path
    if effective_cal_path is None:
        raw_yaml = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8"))
        if isinstance(raw_yaml, Mapping) and isinstance(raw_yaml.get("classifier"), Mapping):
            cp = raw_yaml["classifier"].get("calibration_path")
            if isinstance(cp, str) and cp.strip():
                effective_cal_path = cp.strip()
    if effective_cal_path is not None:
        cal_model = CalibrationModel.load(effective_cal_path)
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
        if raw_dir.startswith("{") or raw_dir.startswith("[") or "\n" in raw_dir:
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
    local_only: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {}
    if model_dir is not None:
        kwargs["model_dir"] = model_dir
    if clock_ms is not None:
        kwargs["clock_ms"] = clock_ms
    key = clf_name.strip()
    if (
        key == "readout_v0"
        or key.startswith("readout_v0:")
        or key == "llm_baseline_v0"
        or key.startswith("llm_baseline_v0:")
    ):
        base_url = (
            os.environ.get("COACH_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://127.0.0.1:8080/v1"
        )
        model_id = key.split(":", 1)[1] if ":" in key else "mock-model"
        api_key_env = "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else None
        kwargs["config"] = OpenAICompatibleConfig(
            provider="openai-compat",
            base_url=base_url,
            model_id=model_id,
            api_key_env=api_key_env,
            local_only=bool(local_only),
        )
    return create_classifier(clf_name, **kwargs)


def _make_provider_from_args(
    args: argparse.Namespace,
    policy: PolicyConfig,
    *,
    clock_ms: Callable[[], int] | None = None,
) -> ProviderAdapter | None:
    env_base_url = (
        os.environ.get("COACH_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
    )
    base_url = args.provider_base_url or policy.generator.base_url or env_base_url
    has_policy_gen_cfg = (
        policy.generator.provider is not None or policy.generator.model_id is not None
    )
    if not (args.provider_base_url and args.provider_model) and not (
        policy.generator.enabled and (base_url is not None or has_policy_gen_cfg)
    ):
        return None
    provider_name = policy.generator.provider or args.provider_name
    model_id = (
        args.provider_model
        or policy.generator.model_id
        or ("gpt-6-luna" if provider_name == "hosted" else "qwen3.8-27b")
    )
    api_key_env = (
        args.provider_api_key_env
        or policy.generator.api_key_env
        or ("OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else None)
    )
    is_hosted: bool | None = None
    if provider_name == "hosted":
        is_hosted = True
    elif provider_name in {"local", "llama_cpp", "ollama"}:
        is_hosted = False
    p_cfg = ProviderConfig(
        provider=provider_name,
        base_url=base_url or "https://api.openai.com/v1",
        model_id=model_id,
        api_key_env=api_key_env,
        reasoning_setting=policy.generator.reasoning,
        is_hosted=is_hosted,
        local_only=bool(args.local_only),
        breaker_consecutive_failures=policy.generator.breaker.consecutive_failures,
        breaker_open_ms=policy.generator.breaker.open_ms,
    )
    try:
        if clock_ms is not None:
            return ProviderAdapter(config=p_cfg, clock_ms=clock_ms)
        return ProviderAdapter(config=p_cfg)
    except (ProviderError, ValueError) as exc:
        print(f"Provider startup capability validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


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
        runs = store.list_runs()
        common_sessions = sorted(
            {str(r["session_id"]) for r in runs if r["run_id"] == args.run_a}
            & {str(r["session_id"]) for r in runs if r["run_id"] == args.run_b}
        )
        if args.session is not None or len(common_sessions) <= 1:
            diff = diff_runs(store, args.run_a, args.run_b, session_id=args.session)
            payload = diff.to_dict()
            target_sessions = [diff.session_id]
            summary_label = diff.session_id
            n_gained, n_lost, n_changed = len(diff.gained), len(diff.lost), len(diff.changed)
        else:
            diffs = [
                diff_runs(store, args.run_a, args.run_b, session_id=sid) for sid in common_sessions
            ]
            diff_dicts = [d.to_dict() for d in diffs]
            payload = {
                "run_a": args.run_a,
                "run_b": args.run_b,
                "sessions": diff_dicts,
                "gained": [e for dd in diff_dicts for e in dd["gained"]],
                "lost": [e for dd in diff_dicts for e in dd["lost"]],
                "changed": [e for dd in diff_dicts for e in dd["changed"]],
            }
            target_sessions = common_sessions
            summary_label = f"{len(common_sessions)} sessions"
            n_gained = len(payload["gained"])
            n_lost = len(payload["lost"])
            n_changed = len(payload["changed"])

        if args.labels or args.bootstrap:
            labels = load_labels_jsonl(args.labels) if args.labels else []
            cards_a: list[DisplayedCardEvent] = []
            cards_b: list[DisplayedCardEvent] = []
            for sid in target_sessions:
                cards_a.extend(_displayed_cards_from_store(store, sid, args.run_a))
                cards_b.extend(_displayed_cards_from_store(store, sid, args.run_b))
            boot_res = paired_conversation_bootstrap(
                cards_a,
                cards_b,
                labels,
                metric=args.bootstrap_metric,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
            payload["bootstrap"] = boot_res
            payload["paired_bootstrap"] = boot_res
    lines = [
        f"Diff for {summary_label}: {args.run_a} -> {args.run_b}",
        f"Gained: {n_gained} | Lost: {n_lost} | Changed: {n_changed}",
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
        await adapter.send_audio_pcm16(pcm, pace_realtime=True, pad_last=True)
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


def _recorder_maxsize_from_env() -> int:
    raw = (
        os.environ.get("COACH_RECORDER_MAXSIZE")
        or os.environ.get("COACH_RECORDER_BUFFER_SIZE")
        or "1024"
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return 1024


async def _serve_async(args: argparse.Namespace) -> int:
    brief = _load_brief_file(args.brief)
    policy, cal_obj = _load_policy_and_cal(args.policy, calibration_path=args.calibration)
    stream_file = args.stream_file or getattr(args, "session", None)
    args.stream_file = stream_file
    is_file_source = args.source in {"replay", "deepgram"} or (
        args.source == "peecee" and bool(stream_file)
    )
    clock: Clock
    if is_file_source:
        clock = ManualClock(0) if args.exit_on_eof else _AnchoredSessionClock(0)
    else:
        clock = MonotonicSessionClock(0)
    clf_name = args.classifier or policy.classifier.version
    model_dir = args.classifier_model_dir or policy.classifier.model_path
    clf = _create_classifier_from_args(
        clf_name,
        model_dir,
        clock_ms=clock.now_ms,
        local_only=bool(args.local_only),
    )
    provider = _make_provider_from_args(args, policy, clock_ms=clock.now_ms)
    mode = Mode(args.mode)
    run_id = args.run_id or f"serve-{uuid.uuid4().hex[:10]}"

    with SQLiteStore(args.db) as store:
        rec_buf = BoundedRecorderBuffer(maxsize=_recorder_maxsize_from_env())
        ui_server = CardUIServer(
            session_id=brief.session_id,
            port=args.ui_port,
            session_token=args.ui_token,
            emit_card_events=True,
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

    # Peecee source: offline WAV transcription vs realtime WS
    env_ws_url = (
        os.environ.get("COACH_PEECEE_WS_URL")
        or os.environ.get("PEECEE_WS_URL")
        or os.environ.get("COACH_NEMO_WS_URL")
        or os.environ.get("NEMO_WS_URL")
        or None
    )
    env_peecee_bin = (
        os.environ.get("COACH_PEECEE_BIN") or os.environ.get("PEECEE_SPEECH_BIN") or None
    )
    exe_path = (
        Path(args.peecee_cmd)
        if args.peecee_cmd
        else (Path(env_peecee_bin) if env_peecee_bin else DEFAULT_PEECEE_SCRIPT)
    )
    raw_sink = _StoreRawSink(store, controller.run_id, controller, brief)
    if args.audio_wav and args.ws_url is None and env_ws_url is None:
        proc_off = await asyncio.create_subprocess_exec(
            str(exe_path),
            str(args.audio_wav),
            "en-US",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc_off.communicate()
        if proc_off.returncode != 0:
            err_msg = stderr_b.decode("utf-8", errors="replace").strip()
            reason = (
                "peecee_lease_unavailable" if proc_off.returncode == 75 else "peecee_process_failed"
            )
            await controller.report_adapter_health(
                HealthState.NO_TRANSCRIPT,
                reason,
                {"returncode": proc_off.returncode, "stderr": err_msg},
            )
            return
        raw_text = stdout_b.decode("utf-8")
        raw_sink.record_raw(
            session_id=brief.session_id,
            source_id="peecee-offline",
            received_ms=clock.now_ms(),
            payload=raw_text,
        )
        for ev in parse_nemo_offline_json(raw_text, session_id=brief.session_id, binder=binder):
            if stop_event.is_set():
                break
            _sync_serve_clock(clock, ev.received_ms)
            await controller.submit_event(ev)
        return

    ws_url = args.ws_url or env_ws_url or DEFAULT_WS_URL
    should_spawn = bool(args.peecee_cmd or env_peecee_bin) or (
        args.ws_url is None and env_ws_url is None and exe_path.exists()
    )
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


def _is_dataset_row(obj: Any) -> bool:
    return isinstance(obj, Mapping) and (
        ("candidate_id" in obj and "serialized_text" in obj and "kind" not in obj)
        or ("context" in obj and "serialized_text" in obj)
    )


def _try_load_dataset_rows(session_path: str | Path) -> list[dict[str, Any]] | None:
    src = Path(session_path)
    if src.suffix in {".db", ".sqlite", ".sqlite3"} or not src.is_file():
        return None
    raw_text = src.read_text(encoding="utf-8")
    if src.suffix.lower() == ".json":
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return [dict(parsed)] if _is_dataset_row(parsed) else None
        if isinstance(parsed, list) and parsed and all(_is_dataset_row(item) for item in parsed):
            return [dict(item) for item in parsed if isinstance(item, Mapping)]
        return None
    rows: list[dict[str, Any]] = []
    for raw in raw_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, Mapping):
            return None
        if not rows and not _is_dataset_row(obj):
            return None
        rows.append(dict(obj))
    return rows if rows else None


def _parse_sweep_t_display(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        raise SystemExit(
            f"--sweep-t-display must be a non-empty CSV of values in [0.0, 1.0], got {raw!r}"
        )
    parts = [part.strip() for part in stripped.split(",")]
    if any(not part for part in parts):
        raise SystemExit(f"--sweep-t-display contains an empty threshold token: {raw!r}")
    values: list[float] = []
    for part in parts:
        try:
            val = float(part)
        except ValueError as exc:
            raise SystemExit(
                f"--sweep-t-display requires numeric thresholds in [0.0, 1.0], got {part!r}"
            ) from exc
        if not math.isfinite(val) or not 0.0 <= val <= 1.0:
            raise SystemExit(
                f"--sweep-t-display thresholds must be finite floats in [0.0, 1.0], got {part!r}"
            )
        values.append(val)
    return values


def _pace_replay_events(events: Sequence[CanonicalEvent], speed_arg: str | None) -> None:
    if speed_arg is None or str(speed_arg).strip().lower() == "stepped":
        return
    speed = float(speed_arg)
    if not math.isfinite(speed) or speed <= 0.0:
        raise SystemExit(f"--speed must be finite and > 0 or 'stepped', got {speed_arg!r}")
    ordered = sorted(events, key=lambda ev: (ev.received_ms, ev.source_sequence, ev.event_id))
    prev_ms: int | None = None
    for ev in ordered:
        if prev_ms is not None and ev.received_ms > prev_ms:
            time.sleep((ev.received_ms - prev_ms) / 1000.0 / speed)
        prev_ms = ev.received_ms


def _load_replay_events(
    args: argparse.Namespace,
    policy: PolicyConfig,
) -> tuple[SessionBrief, list[CanonicalEvent]]:
    source = getattr(args, "source", "canonical")
    if source == "deepgram":
        if not args.brief:
            raise SystemExit("--source deepgram requires --brief")
        brief = _load_brief_file(args.brief)
        norm = DeepgramNormalizer(
            session_id=brief.session_id,
            confidence_floor=policy.abstain.asr_confidence_floor,
        )
        events: list[CanonicalEvent] = []
        for line_no, raw in enumerate(
            Path(args.session).read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            rx_ms = int(obj.get("received_ms", line_no * 200))
            events.extend(norm.normalize(obj, rx_ms))
        return brief, events
    if source == "peecee" and Path(args.session).suffix == ".json":
        if not args.brief:
            raise SystemExit("--source peecee JSON requires --brief")
        brief = _load_brief_file(args.brief)
        binder = _make_binder_from_env(brief)
        raw_text = Path(args.session).read_text(encoding="utf-8")
        return brief, parse_nemo_offline_json(raw_text, session_id=brief.session_id, binder=binder)
    return _load_session_source(args.session, brief_path=args.brief, run_id=args.run_id)


def _cmd_replay(args: argparse.Namespace) -> int:
    policy, cal_obj = _load_policy_and_cal(args.policy, calibration_path=args.calibration)
    t_vals = _parse_sweep_t_display(args.sweep_t_display)
    run_id = args.run_id or f"replay-{uuid.uuid4().hex[:10]}"
    dataset_rows = (
        _try_load_dataset_rows(args.session)
        if getattr(args, "source", "canonical") == "canonical"
        else None
    )
    if dataset_rows is not None:
        split_mode = args.split_by or "conversation"
        splits = (
            split_by_series(dataset_rows)
            if split_mode == "series"
            else split_by_conversation(dataset_rows)
        )
        brief_opt = _load_brief_file(args.brief) if args.brief else None
        clf_name = args.classifier or policy.classifier.version
        model_dir = args.classifier_model_dir or policy.classifier.model_path
        clf = _create_classifier_from_args(clf_name, model_dir, local_only=bool(args.local_only))
        provider = _make_provider_from_args(args, policy)
        mode = Mode(args.mode)
        ds_labels: list[LabelRecord] = []
        sweep_res: list[dict[str, Any]] | None = None
        try:
            with SQLiteStore(args.db) as store:
                session_results = simulate_dataset_replay(
                    dataset_rows,
                    policy,
                    clf,
                    brief=brief_opt,
                    calibration=cal_obj,
                    store=store,
                    run_id=run_id,
                    mode=mode,
                    provider=provider,
                    simulated_latency_ms=args.simulated_latency_ms,
                    cache_classifier=not args.no_cache,
                )
            if args.labels:
                ds_labels = load_labels_jsonl(args.labels)
            if t_vals is not None:
                if provider is not None:
                    provider._cancel_recovery_task()
                sweep_provider = provider.clone_isolated() if provider is not None else None
                try:
                    sweep_res = sweep_dataset_t_display(
                        dataset_rows,
                        ds_labels,
                        policy,
                        clf,
                        t_vals,
                        brief=brief_opt,
                        calibration=cal_obj,
                        mode=mode,
                        provider=sweep_provider,
                        simulated_latency_ms=args.simulated_latency_ms,
                        cache_classifier=not args.no_cache,
                    )
                finally:
                    if sweep_provider is not None:
                        sweep_provider._cancel_recovery_task()
        finally:
            if provider is not None:
                asyncio.run(provider.aclose())
        all_cards = [c for r in session_results for c in r.displayed_cards]
        total_dur_ms = sum(r.duration_ms for r in session_results)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "n_rows": len(dataset_rows),
            "splits": splits.to_dict(),
            "duration_ms": total_dur_ms,
            "decisions_count": sum(r.decisions_count for r in session_results),
            "dropped_count": sum(r.dropped_count for r in session_results),
            "routed_count": sum(r.routed_count for r in session_results),
            "withdrawn_count": sum(r.withdrawn_count for r in session_results),
            "expired_count": sum(r.expired_count for r in session_results),
            "displayed_cards": [
                {
                    "suggestion_id": c.suggestion_id,
                    "candidate_id": c.candidate_id,
                    "session_id": c.session_id,
                    "category": str(c.category),
                    "display_ms": c.display_ms,
                    "text": c.text,
                    "route": str(c.route),
                    "origin": str(c.origin),
                }
                for c in all_cards
            ],
        }
        if args.labels:
            payload["evaluation"] = match_opportunities(
                all_cards,
                ds_labels,
                settle_ms=policy.display.settle_ms,
                duration_ms=total_dur_ms,
            )
        if args.eval_log:
            looks = EvalLookCounter(args.eval_log).record_look(
                run_id, split_by=split_mode, eval_conversations=splits.eval_conversations
            )
            payload["look_count"] = looks
            payload["eval_look_count"] = looks
        if sweep_res is not None:
            payload["sweep"] = sweep_res
            payload["sweep_curve"] = sweep_res
        _emit_output(
            payload,
            as_json=args.json,
            summary_lines=[
                f"Dataset replay ({split_mode}): {len(dataset_rows)} rows, "
                f"decisions={payload['decisions_count']}, displayed={len(all_cards)}"
            ],
        )
        return 0

    brief, events = _load_replay_events(args, policy)
    clf_name = args.classifier or policy.classifier.version
    model_dir = args.classifier_model_dir or policy.classifier.model_path
    clf = _create_classifier_from_args(clf_name, model_dir, local_only=bool(args.local_only))
    provider = _make_provider_from_args(args, policy)
    mode = Mode(args.mode)
    _ = _recorder_maxsize_from_env()
    _pace_replay_events(events, args.speed)

    grouped_events: dict[str, list[CanonicalEvent]] = {}
    for ev in events:
        grouped_events.setdefault(ev.session_id, []).append(ev)
    if not grouped_events:
        grouped_events[brief.session_id] = []

    try:
        with SQLiteStore(args.db) as store:

            async def _run_session_replay(
                sess_brief: SessionBrief, sess_evs: list[CanonicalEvent]
            ) -> ReplayResult:
                sess_provider = provider.clone_isolated() if provider is not None else None
                try:
                    return await async_simulate_replay(
                        sess_brief,
                        sess_evs,
                        policy,
                        clf,
                        calibration=cal_obj,
                        store=store,
                        run_id=run_id,
                        mode=mode,
                        provider=sess_provider,
                        simulated_latency_ms=args.simulated_latency_ms,
                        cache_classifier=not args.no_cache,
                    )
                finally:
                    if sess_provider is not None:
                        await sess_provider.cancel_recovery()

            session_results = []
            session_briefs: list[tuple[SessionBrief, list[CanonicalEvent]]] = []
            for sid, sess_evs in grouped_events.items():
                sess_brief = replace(brief, session_id=sid)
                res_s = asyncio.run(_run_session_replay(sess_brief, sess_evs))
                session_results.append(res_s)
                session_briefs.append((sess_brief, sess_evs))

            res = session_results[0]
            all_cards = [c for r in session_results for c in r.displayed_cards]
            total_dur_ms = sum(r.duration_ms for r in session_results)
            invs = [
                check_conservation_invariants(store, sb.session_id, run_id)
                for sb, _ in session_briefs
            ]
            all_inv_passed = all(i.passed for i in invs)
            inv = invs[0]
            reports = [
                build_session_report(store, r.session_id, run_id).to_dict() for r in session_results
            ]
            report = reports[0]

            payload = {
                "session_id": res.session_id,
                "run_id": res.run_id,
                "speed": args.speed if args.speed is not None else "stepped",
                "duration_ms": total_dur_ms,
                "decisions_count": sum(r.decisions_count for r in session_results),
                "dropped_count": sum(r.dropped_count for r in session_results),
                "routed_count": sum(r.routed_count for r in session_results),
                "withdrawn_count": sum(r.withdrawn_count for r in session_results),
                "expired_count": sum(r.expired_count for r in session_results),
                "displayed_cards": [
                    {
                        "suggestion_id": c.suggestion_id,
                        "candidate_id": c.candidate_id,
                        "session_id": c.session_id,
                        "category": str(c.category),
                        "display_ms": c.display_ms,
                        "text": c.text,
                        "route": str(c.route),
                        "origin": str(c.origin),
                    }
                    for c in all_cards
                ],
                "invariants": {**inv.to_dict(), "passed": all_inv_passed},
                "report": report,
            }
            if len(session_results) > 1:
                payload["session_ids"] = [r.session_id for r in session_results]
                payload["reports"] = reports

            labels: list[LabelRecord] = load_labels_jsonl(args.labels) if args.labels else []
            if args.labels:
                payload["evaluation"] = match_opportunities(
                    all_cards,
                    labels,
                    settle_ms=policy.display.settle_ms,
                    duration_ms=total_dur_ms,
                )
            if args.split_by:
                split_items: Sequence[Any] = labels if labels else [sb for sb, _ in session_briefs]
                splits = (
                    split_by_series(split_items)
                    if args.split_by == "series"
                    else split_by_conversation(split_items)
                )
                payload["splits"] = splits.to_dict()
                eval_convs = splits.eval_conversations
            else:
                eval_convs = tuple(sb.session_id for sb, _ in session_briefs)
            if args.eval_log:
                looks = EvalLookCounter(args.eval_log).record_look(
                    run_id,
                    split_by=args.split_by or "conversation",
                    eval_conversations=eval_convs,
                )
                payload["eval_look_count"] = looks
                payload["look_count"] = looks
            if t_vals is not None:
                if provider is not None:
                    provider._cancel_recovery_task()
                sweep_provider = provider.clone_isolated() if provider is not None else None
                try:
                    sweep_res = sweep_t_display(
                        session_briefs,
                        labels,
                        policy,
                        clf,
                        t_vals,
                        calibration=cal_obj,
                        mode=mode,
                        provider=sweep_provider,
                        simulated_latency_ms=args.simulated_latency_ms,
                        cache_classifier=not args.no_cache,
                    )
                finally:
                    if sweep_provider is not None:
                        sweep_provider._cancel_recovery_task()
                payload["sweep"] = sweep_res
                payload["sweep_curve"] = sweep_res
    finally:
        if provider is not None:
            asyncio.run(provider.aclose())

    lines = [
        f"Replay {res.session_id} (run={res.run_id}): "
        f"decisions={payload['decisions_count']}, displayed={len(all_cards)}, "
        f"invariants={all_inv_passed}"
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0 if all_inv_passed else 2


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


_DECISION_CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "session_id",
        "origin",
        "based_on_state_version",
        "builder_version",
        "prefilter_version",
        "anchor_refs",
        "context_refs",
        "anchor_received_ms",
        "display_by_ms",
        "token_counts",
        "omitted",
        "flags",
        "serialized_sha256",
    }
)


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
            elif "candidate_id" in obj and "serialized_text" in obj and "kind" not in obj:
                ctx_payload = {k: v for k, v in obj.items() if k in _DECISION_CONTEXT_FIELDS}
                ctx = DecisionContext.from_dict(ctx_payload)
                cin = ClassifierInput(context=ctx, serialized_text=str(obj["serialized_text"]))
                if "label" in obj and isinstance(obj["label"], Mapping):
                    lbl = LabelRecord.from_dict(obj["label"])
                else:
                    worthwhile = bool(obj.get("worthwhile", False))
                    raw_cats = obj.get("categories") or ()
                    cats = tuple(Category(str(c)) for c in raw_cats) if worthwhile else ()
                    src_str = str(obj.get("sample_source") or SampleSource.RANDOM.value)
                    nr = (
                        None
                        if worthwhile
                        else NoneReason(str(obj.get("none_reason") or NoneReason.NO_GAP.value))
                    )
                    lbl = LabelRecord(
                        schema_version=1,
                        label_id=str(obj.get("label_id") or f"lbl-{ctx.candidate_id}"),
                        session_id=ctx.session_id,
                        annotator=str(obj.get("annotator") or "dataset"),
                        sample_source=SampleSource(src_str),
                        annotation_pass=AnnotationPass.OFFLINE,
                        candidate_id=ctx.candidate_id,
                        state_version_at_decision=ctx.based_on_state_version,
                        worthwhile=worthwhile,
                        categories=cats,
                        example_questions=tuple(
                            str(q) for q in (obj.get("example_questions") or ()) if str(q).strip()
                        ),
                        evidence_refs=ctx.anchor_refs,
                        validity=ValidityInterval(open_ms=ctx.anchor_received_ms),
                        none_reason=nr,
                    )
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


def _extract_row_series_ids(data_path: str | Path) -> list[str | None]:
    out: list[str | None] = []
    with Path(data_path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, Mapping):
                raw_ser = obj.get("series_id")
                if raw_ser is None and isinstance(obj.get("label"), Mapping):
                    raw_ser = obj["label"].get("series_id")
                if raw_ser is None and isinstance(obj.get("context"), Mapping):
                    raw_ser = obj["context"].get("series_id")
                ser_str = str(raw_ser).strip() if raw_ser is not None else ""
                out.append(ser_str or None)
    return out


def _cmd_train(args: argparse.Namespace) -> int:
    examples, _, _ = _load_training_rows(
        args.data,
        session_path=args.session,
        brief_path=args.brief,
        policy_path=args.policy,
    )
    if not examples:
        raise SystemExit(f"No training examples found in {args.data}")

    series_ids = _extract_row_series_ids(args.data)
    row_meta: list[tuple[str, str]] = []
    for idx, (cin, lbl) in enumerate(examples):
        sess_id = lbl.session_id or cin.context.session_id
        raw_ser = series_ids[idx] if idx < len(series_ids) else None
        ser_id = raw_ser if raw_ser is not None else sess_id
        row_meta.append((sess_id, ser_id))
    splits = split_by_series(
        list(range(len(examples))),
        series_id_fn=lambda i: row_meta[i][1],
        session_id_fn=lambda i: row_meta[i][0],
    )
    splits_payload: dict[str, Any] = {
        "train": list(splits.train_conversations),
        "cal": list(splits.cal_conversations),
        "eval": list(splits.eval_conversations),
        "series": {
            "train": list(splits.train_series),
            "cal": list(splits.cal_series),
            "eval": list(splits.eval_series),
        },
        "counts": {
            "train": len(splits.train),
            "cal": len(splits.cal),
            "eval": len(splits.eval),
        },
    }

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

    (out_dir / "splits.json").write_text(
        json.dumps(splits_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    payload["splits"] = splits_payload

    lines = [f"Trained {payload['backend']} on {len(examples)} examples -> {saved_path}"]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    with Path(args.data).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, Mapping):
                raise SystemExit(f"Line {line_no} in {args.data} must be a JSON object")
            raw_src = obj.get("sample_source")
            if raw_src is None and isinstance(obj.get("label"), Mapping):
                raw_src = obj["label"].get("sample_source")
            if raw_src is None or str(raw_src) != SampleSource.RANDOM.value:
                print(
                    f"Calibration rejected: calibration requires sample_source == 'random', "
                    f"found {raw_src!r} at line {line_no}",
                    file=sys.stderr,
                )
                return 2

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

    calibrated_pairs = [
        (cal_model.calibrate(ctx, res, mode=session_mode), lbl) for ctx, res, lbl in fit_rows
    ]
    eval_rep = evaluate_calibration(calibrated_pairs)
    eval_dict = eval_rep.to_dict()
    opp_eval = eval_rep.heads.get("opportunity")
    out_data = cal_model.to_dict()
    if opp_eval is not None:
        out_data["brier_score"] = opp_eval.brier_score
        out_data["reliability_bins"] = [b.to_dict() for b in opp_eval.bins]
    out_data["evaluation"] = eval_dict
    saved_path = Path(args.out)
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_text(json.dumps(out_data, indent=2, sort_keys=True), encoding="utf-8")
    payload: dict[str, Any] = {
        "calibration_version": cal_model.calibration_version,
        "classifier_version": cal_model.classifier_version,
        "mode": str(cal_model.mode),
        "out": str(saved_path),
        "n_samples": len(fit_rows),
        "evaluation": eval_dict,
    }
    if opp_eval is not None:
        payload["brier_score"] = opp_eval.brier_score
        payload["reliability_bins"] = [b.to_dict() for b in opp_eval.bins]
    lines = [
        f"Saved calibration {cal_model.calibration_version} ({cal_model.classifier_version}) -> {saved_path}"
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    if args.kappa and not (args.session and args.out and not args.labels):
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

    brief, events = _load_session_source(
        args.session, brief_path=args.brief, allow_default_brief=True
    )
    if args.db and args.brief is None:
        with SQLiteStore(args.db) as db_store:
            runs = db_store.list_runs(brief.session_id)
            if runs:
                srow = db_store.get_session(brief.session_id, str(runs[-1]["run_id"]))
                if srow is not None and srow.get("brief_json"):
                    brief = SessionBrief.from_dict(json.loads(str(srow["brief_json"])))
    policy = load_policy(args.policy, allow_null_calibration=True) if args.policy else None

    pass1: Pass1Session | None = None
    pass2: Pass2Session | None = None
    retest: BlindRetestSession | None = None
    if args.kappa:
        first_labels = load_labels_jsonl(args.kappa)
        retest = BlindRetestSession(
            first_labels=first_labels,
            events=events,
            annotator=args.annotator,
            out_path=args.out,
        )
    elif args.pass_num == 1:
        pass1 = Pass1Session(
            brief=brief,
            events=events,
            annotator=args.annotator,
            speed=args.speed,
            out_path=args.out,
        )
    else:
        src = SampleSource(args.sample_source)
        review_queue_ids: list[str] = []
        if args.review_queue:
            rq_raw = json.loads(Path(args.review_queue).read_text(encoding="utf-8"))
            rq_list = rq_raw.get("review_queue", rq_raw) if isinstance(rq_raw, Mapping) else rq_raw
            items = load_review_queue_items(
                brief, events, rq_list, policy=policy, sample_source=src
            )
            review_queue_ids = [it.candidate_id or it.item_id for it in items]
        else:
            items = build_pass2_items(brief, events, policy=policy, sample_source=src)
            if args.db:
                with SQLiteStore(args.db) as db_store:
                    runs = db_store.list_runs(brief.session_id)
                    if runs:
                        rid = str(runs[-1]["run_id"])
                        rq_entries: list[dict[str, Any]] = []
                        for sug in db_store.get_suggestions(brief.session_id, rid):
                            trs = json.loads(str(sug["transitions_json"]))
                            if any(
                                isinstance(t, Mapping)
                                and t.get("transition") in {"displayed", "shadow_displayed"}
                                for t in trs
                            ):
                                rq_entries.append(
                                    {
                                        "suggestion_id": sug["suggestion_id"],
                                        "candidate_id": sug["candidate_id"],
                                        "evidence_refs": json.loads(str(sug["evidence_refs_json"])),
                                        "text": sug.get("text"),
                                        "category": sug.get("category"),
                                    }
                                )
                        if rq_entries:
                            rq_items = load_review_queue_items(
                                brief,
                                events,
                                rq_entries,
                                policy=policy,
                                sample_source=SampleSource.SHADOW_REVIEW,
                            )
                            by_cand = {it.candidate_id: it for it in items if it.candidate_id}
                            for rit in rq_items:
                                if rit.candidate_id and rit.candidate_id in by_cand:
                                    existing = by_cand[rit.candidate_id]
                                    existing.suggested_question = rit.suggested_question
                                    existing.suggested_category = rit.suggested_category
                                    existing.sample_source = rit.sample_source
                                    review_queue_ids.append(rit.candidate_id)
                                else:
                                    items.append(rit)
                                    review_queue_ids.append(rit.candidate_id or rit.item_id)
        pass2 = Pass2Session(
            items=items,
            annotator=args.annotator,
            out_path=args.out,
            review_queue_ids=review_queue_ids,
        )

    server = AnnotationServer(
        port=args.port,
        session_token=args.ui_token,
        pass1=pass1,
        pass2=pass2,
        retest=retest,
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
    p_serve.add_argument(
        "--session", default=None, help="Alias for --stream-file (offline stream/JSONL)"
    )
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
    p_rep_play.add_argument(
        "--source",
        choices=["canonical", "replay", "deepgram", "peecee"],
        default="canonical",
        help="Input session format",
    )
    p_rep_play.add_argument("--speed", default=None, help="Replay speed float or 'stepped'")
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
    p_ann.add_argument("--db", default=None, help="Optional SQLite WAL database")
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
