"""Unit tests for the 5-task SuggestionController runtime (§3.5–§3.7, §4.2, §8, §12.9)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from coach.calibration import default_calibration_registry
from coach.context_builder import PolicyConfig, Tokenizer, WhitespaceTokenizer
from coach.contracts import (
    CanonicalEvent,
    Category,
    ClassifierResult,
    EventKind,
    HealthState,
    LedgerState,
    Mode,
    Outcome,
    ReasonCode,
    RecipientConfig,
    Route,
    ScoreSet,
    SessionBrief,
    SessionMode,
    Stability,
    SuggestionCandidate,
)
from coach.opportunity_classifier import (
    ClassifierInput,
    coerce_classifier_input,
    create_classifier,
    make_classifier_error_result,
)
from coach.provider_adapter import ProviderAdapter, ProviderConfig
from coach.routing_policy import load_policy
from coach.storage import SQLiteStore
from coach.suggestion_controller import SuggestionController
from coach.telemetry import BoundedRecorderBuffer, check_conservation_invariants
from coach.transcript_adapter import ManualClock

POLICY_PATH = Path(__file__).resolve().parents[2] / "policies" / "policy-v0-a.yaml"


def _brief(session_id: str = "sess-sc") -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id=session_id,
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="me", role="reviewer"),
        objective="Confirm rollback owner and migration cutover deadline.",
    )


def _policy(*, generator_enabled: bool = False) -> PolicyConfig:
    base = load_policy(POLICY_PATH, allow_null_calibration=True)
    if not generator_enabled:
        return base
    from dataclasses import replace

    return replace(base, generator=replace(base.generator, enabled=True))


def _seg(
    sid: str,
    ev_id: str,
    seg_id: str,
    text: str,
    *,
    speaker: str = "S1",
    rx_ms: int = 1000,
    start_ms: int | None = None,
    end_ms: int | None = None,
    seq: int = 1,
    rev: int = 1,
    stability: Stability = Stability.COMMITTED,
) -> CanonicalEvent:
    s_ms = rx_ms - 500 if start_ms is None else start_ms
    e_ms = rx_ms - 50 if end_ms is None else end_ms
    return CanonicalEvent(
        schema_version=1,
        session_id=sid,
        event_id=ev_id,
        kind=EventKind.SEGMENT,
        received_ms=rx_ms,
        source_sequence=seq,
        segment_id=seg_id,
        source_revision=rev,
        channel=0,
        speaker_id=speaker,
        start_ms=max(0, s_ms),
        end_ms=max(0, e_ms),
        text=text,
        stability=stability,
    )


class _ScriptedClassifier:
    def __init__(
        self,
        *,
        opp: float = 0.90,
        cats: dict[str, float] | None = None,
        fail: bool = False,
        score_fn: Any | None = None,
    ) -> None:
        self.classifier_version = "opportunity-baseline-v0"
        self.tokenizer: Tokenizer = WhitespaceTokenizer()
        self._opp = opp
        self._cats = cats or {"CONFIRM_OWNER": 0.90}
        self._fail = fail
        self._score_fn = score_fn

    async def classify_batch(
        self,
        contexts: Sequence[Any],
        *,
        deadline_ms: int | None = None,
    ) -> list[ClassifierResult]:
        out: list[ClassifierResult] = []
        for item in contexts:
            cin = coerce_classifier_input(item)
            ctx = cin.context
            if self._fail:
                out.append(make_classifier_error_result(ctx, self.classifier_version, error="boom"))
                continue
            opp, cats = (
                self._score_fn(cin) if self._score_fn is not None else (self._opp, dict(self._cats))
            )
            out.append(
                ClassifierResult(
                    schema_version=1,
                    candidate_id=ctx.candidate_id,
                    based_on_state_version=ctx.based_on_state_version,
                    classifier_version=self.classifier_version,
                    raw_scores=ScoreSet(opportunity=opp, categories=cats),
                    calibration_version="cal-v0-technical",
                    calibrated_scores=ScoreSet(opportunity=opp, categories=cats),
                    latency_ms=1.0,
                    input_refs=ctx.anchor_refs + ctx.context_refs,
                )
            )
        return out


def _active(ctrl: SuggestionController) -> SuggestionCandidate | None:
    return ctrl.active_card


def _deferred(ctrl: SuggestionController) -> SuggestionCandidate | None:
    return ctrl.deferred_card


def _health(ctrl: SuggestionController) -> HealthState:
    return ctrl.health


async def test_five_tasks_boundary_early_release_and_pin_lifecycle(tmp_path: Path) -> None:
    brief = _brief("s-pin")
    policy = _policy()
    clock = ManualClock(1000)
    buf = BoundedRecorderBuffer()
    with SQLiteStore(tmp_path / "sc.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_ScriptedClassifier(),
            calibration=default_calibration_registry(),
            store=store,
            run_id="r-pin",
            clock=clock,
            telemetry_buffer=buf,
        )
        await ctrl.start()
        assert set(ctrl.tasks) == {"ingest", "proactive", "manual", "display", "recorder"}
        assert all(not t.done() for t in ctrl.tasks.values())

        await ctrl.submit_event(
            _seg(
                "s-pin",
                "e1",
                "seg-1",
                "Someone needs to handle the database rollback tonight.",
                rx_ms=1000,
                seq=1,
            )
        )
        await ctrl.drain()
        assert _active(ctrl) is None  # Still within 1500 ms settle window

        # Speaker boundary before settle_ms releases early!
        clock.set_ms(1400)
        await ctrl.submit_event(
            CanonicalEvent(
                schema_version=1,
                session_id="s-pin",
                event_id="b1",
                kind=EventKind.BOUNDARY,
                received_ms=1400,
                source_sequence=2,
                speaker_id="S1",
                end_ms=1380,
                boundary="utterance_end",
            )
        )
        await ctrl.drain()
        card = _active(ctrl)
        assert card is not None
        sid = card.suggestion_id

        # Pin survives display_ttl_ms
        await ctrl.handle_ui_action({"action": "pin", "suggestion_id": sid})
        assert ctrl.active_pinned
        clock.set_ms(1400 + policy.display.display_ttl_ms + 5000)
        await ctrl.tick()
        assert ctrl.active_card is not None
        assert ctrl.active_pinned

        # Resolution cue resolves pinned card cleanly
        await ctrl.submit_event(
            _seg(
                "s-pin",
                "e2",
                "seg-2",
                "I will handle the database rollback tonight.",
                speaker="S2",
                rx_ms=clock.now_ms(),
                seq=3,
            )
        )
        await ctrl.drain()
        assert _active(ctrl) is None
        await ctrl.stop()

        inv = check_conservation_invariants(store, "s-pin", "r-pin")
        assert inv.passed, inv.violations
        sugs = {s["suggestion_id"]: s for s in store.get_suggestions("s-pin", "r-pin")}
        assert sugs[sid]["current_state"] == LedgerState.RESOLVED.value


async def test_modes_shadow_quiet_paused_and_off_the_record(tmp_path: Path) -> None:
    brief = _brief("s-modes")
    policy = _policy()
    clock = ManualClock(1000)
    with SQLiteStore(tmp_path / "modes.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_ScriptedClassifier(),
            calibration=default_calibration_registry(),
            store=store,
            run_id="r-modes",
            mode=Mode.QUIET,
            clock=clock,
            telemetry_buffer=BoundedRecorderBuffer(),
        )
        await ctrl.start()
        await ctrl.submit_event(
            _seg(
                "s-modes",
                "e1",
                "seg-1",
                "Someone should own the cutover checklist.",
                rx_ms=1000,
                seq=1,
            )
        )
        await ctrl.drain()
        assert ctrl.active_card is None  # Quiet: no proactive card
        decs = store.get_decisions("s-modes", "r-modes")
        assert len(decs) == 1
        assert decs[0]["terminal_state"] == "suppressed"
        assert decs[0]["terminal_reason"] == ReasonCode.PROACTIVE_DISABLED.value

        # Shadow mode: logs display without showing UI card
        await ctrl.set_mode(Mode.SHADOW)
        clock.set_ms(3000)
        await ctrl.submit_event(
            _seg(
                "s-modes",
                "e2",
                "seg-2",
                "Someone needs to manage the firewall rollout.",
                rx_ms=3000,
                seq=2,
            )
        )
        await ctrl.drain()
        clock.set_ms(5000)
        await ctrl.tick()
        assert ctrl.build_ui_snapshot().active_card is None

        # Paused: health paused, no inference
        await ctrl.set_mode(Mode.PAUSED)
        assert ctrl.health is HealthState.PAUSED
        clock.set_ms(7000)
        await ctrl.submit_event(
            _seg(
                "s-modes",
                "e3",
                "seg-3",
                "Someone needs to handle DNS propagation.",
                rx_ms=7000,
                seq=3,
            )
        )
        await ctrl.drain()
        await ctrl.set_mode(Mode.PROACTIVE)
        await ctrl.drain()

        # Off the record: halts ingest, drops segment, writes gap on exit
        await ctrl.set_mode(Mode.OFF_THE_RECORD)
        clock.set_ms(9000)
        res_otr = await ctrl.submit_event(
            _seg(
                "s-modes",
                "e-otr",
                "seg-otr",
                "Secret compensation discussion here.",
                rx_ms=9000,
                seq=4,
            )
        )
        assert not res_otr.accepted
        await ctrl.drain()
        await ctrl.set_mode(Mode.PROACTIVE)
        await ctrl.stop()

        events = store.get_events("s-modes", "r-modes")
        assert all(ev["segment_id"] != "seg-otr" for ev in events)
        assert any(
            ev["kind"] == "control" and ev["payload_json"] and "off_record" in ev["payload_json"]
            for ev in events
        )
        inv = check_conservation_invariants(store, "s-modes", "r-modes")
        assert inv.passed, inv.violations


async def test_deferred_slot_displacement_and_silence_cooldown_timer(tmp_path: Path) -> None:
    brief = _brief("s-def")
    policy = _policy()
    clock = ManualClock(1000)

    def _score_fn(cin: ClassifierInput) -> tuple[float, dict[str, float]]:
        if "firewall" in cin.serialized_text:
            return (0.95, {"CONFIRM_OWNER": 0.95})
        return (0.82, {"CONFIRM_OWNER": 0.82})

    with SQLiteStore(tmp_path / "def.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_ScriptedClassifier(score_fn=_score_fn),
            calibration=default_calibration_registry(),
            store=store,
            run_id="r-def",
            clock=clock,
            telemetry_buffer=BoundedRecorderBuffer(),
        )
        await ctrl.start()
        # First card displayed at 2600 ms
        await ctrl.submit_event(
            _seg(
                "s-def", "e1", "seg-1", "Someone needs to handle rollback prep.", rx_ms=1000, seq=1
            )
        )
        await ctrl.drain()
        clock.set_ms(2600)
        await ctrl.tick()
        assert ctrl.active_card is not None
        first_sid = ctrl.active_card.suggestion_id
        await ctrl.handle_ui_action(
            {"action": "dismiss", "suggestion_id": first_sid, "feedback": "not_relevant"}
        )
        assert _active(ctrl) is None

        # Candidate A arrives during 20s cooldown -> deferred(cooldown)
        clock.set_ms(4000)
        await ctrl.submit_event(
            _seg("s-def", "e2", "seg-2", "Someone needs to check staging logs.", rx_ms=4000, seq=2)
        )
        await ctrl.drain()
        clock.set_ms(5600)
        await ctrl.tick()
        assert ctrl.deferred_card is not None
        cand_a_sid = ctrl.deferred_card.suggestion_id

        # Candidate B has higher score -> displaces A (DISPLACED_BY_BETTER)
        clock.set_ms(6000)
        await ctrl.submit_event(
            _seg(
                "s-def",
                "e3",
                "seg-3",
                "Someone needs to configure the firewall rules.",
                rx_ms=6000,
                seq=3,
            )
        )
        await ctrl.drain()
        clock.set_ms(7600)
        await ctrl.tick()
        assert ctrl.deferred_card is not None
        assert ctrl.deferred_card.suggestion_id != cand_a_sid
        cand_b_sid = ctrl.deferred_card.suggestion_id

        # Advance to cooldown end + re-check: B displays with zero new transcript events!
        # Keep display_by_ms in range: 6000 + 8000 = 14000; wait cooldown is 20000 so 2600+20000=22600 > 14000!
        # Let's verify timer fires at 14000 -> B expires(display_by_passed) during silence!
        clock.set_ms(14000)
        await ctrl.tick()
        assert _deferred(ctrl) is None
        await ctrl.stop()

        sugs = {s["suggestion_id"]: s for s in store.get_suggestions("s-def", "r-def")}
        assert sugs[cand_a_sid]["current_state"] == "withdrawn"
        assert sugs[cand_a_sid]["terminal_reason"] == ReasonCode.DISPLACED_BY_BETTER.value
        assert sugs[cand_b_sid]["current_state"] == "expired"
        assert sugs[cand_b_sid]["terminal_reason"] == ReasonCode.DISPLAY_BY_PASSED.value
        inv = check_conservation_invariants(store, "s-def", "r-def")
        assert inv.passed, inv.violations


async def test_used_inferred_and_health_states_and_manual_lane(tmp_path: Path) -> None:
    brief = _brief("s-hlth")
    policy = _policy(generator_enabled=True)
    clock = ManualClock(1000)
    buf = BoundedRecorderBuffer()

    def _gen_handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(
            {
                "id": "cmpl-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "question": "Who owns the rollback checklist?",
                                    "evidence_segment_ids": ["seg-1"],
                                    "decline_reason": None,
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        )
        return httpx.Response(200, text=body)

    provider = ProviderAdapter(
        config=ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:9999/v1",
            model_id="qwen3:14b",
            use_streaming=False,
        ),
        transport=httpx.MockTransport(_gen_handler),
        clock_ms=clock.now_ms,
    )

    with SQLiteStore(tmp_path / "hlth.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_ScriptedClassifier(fail=True),
            calibration=default_calibration_registry(),
            provider=provider,
            store=store,
            run_id="r-hlth",
            clock=clock,
            telemetry_buffer=buf,
            classifier_error_threshold=2,
        )
        await ctrl.start()
        # Two failing batches -> HealthState.MANUAL_ONLY
        await ctrl.submit_event(
            _seg(
                "s-hlth",
                "e1",
                "seg-1",
                "Someone needs to handle rollback checklist.",
                rx_ms=1000,
                seq=1,
            )
        )
        await ctrl.drain()
        clock.set_ms(2000)
        await ctrl.submit_event(
            _seg(
                "s-hlth",
                "e2",
                "seg-2",
                "Someone needs to verify cutover metrics.",
                rx_ms=2000,
                seq=2,
            )
        )
        await ctrl.drain()
        assert _health(ctrl) is HealthState.MANUAL_ONLY

        # Manual lane still works via generator fallback in MANUAL_ONLY!
        man_res = await ctrl.trigger_manual()
        assert man_res.status == "displayed"
        assert man_res.suggestion is not None
        assert man_res.suggestion.origin.value == "manual"
        sid = man_res.suggestion.suggestion_id

        # Recipient speaks equivalent question -> used_inferred logged WITHOUT mutating LedgerState to USED
        clock.set_ms(2500)
        await ctrl.submit_event(
            _seg(
                "s-hlth",
                "e-me",
                "seg-me",
                "Who owns the rollback checklist?",
                speaker="ME",
                rx_ms=2500,
                seq=3,
            )
        )
        await ctrl.drain()
        tel_rows = store.get_telemetry_records("s-hlth", "r-hlth")
        assert any(r.kind == "used_inferred" and r.suggestion_id == sid for r in tel_rows)
        snap = ctrl.canonical_state.snapshot()
        assert snap.ledger[sid].state is not LedgerState.USED

        # Watchdog no_transcript after 15s silence
        clock.set_ms(2500 + policy.health.no_transcript_after_ms + 100)
        await ctrl.tick()
        assert ctrl.health is HealthState.NO_TRANSCRIPT
        await ctrl.stop()
        await provider.aclose()

        inv = check_conservation_invariants(store, "s-hlth", "r-hlth")
        assert inv.passed, inv.violations


async def test_manual_empty_response_and_rules_v0_controller(tmp_path: Path) -> None:
    brief = _brief("s-empty")
    policy = _policy()
    clock = ManualClock(1000)
    with SQLiteStore(tmp_path / "empty.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=create_classifier("rules_v0"),
            calibration=default_calibration_registry(),
            store=store,
            run_id="r-empty",
            clock=clock,
        )
        await ctrl.start()
        res = await ctrl.trigger_manual()
        assert res.status == "empty"
        assert "nothing useful to add" in res.message.lower()
        assert ctrl.build_ui_snapshot().manual_status == {
            "status": "empty",
            "message": "Nothing useful to add.",
        }
        await ctrl.stop()


async def test_default_clock_rules_v0_and_default_provider_adapter(tmp_path: Path) -> None:
    """BUG-1 regression: default RulesClassifier + default ProviderAdapter + default controller clock."""
    brief = _brief("s-defclk")
    base_pol = _policy(generator_enabled=True)
    policy = replace(
        base_pol,
        categories={
            **base_pol.categories,
            Category.CONFIRM_OWNER: replace(
                base_pol.categories[Category.CONFIRM_OWNER],
                route=Route.GENERATOR,
            ),
        },
    )
    http_calls = 0

    def _gen_handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        body = json.dumps(
            {
                "id": f"cmpl-{http_calls}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "question": "Who will own the database rollback tonight?",
                                    "evidence_segment_ids": ["seg-1"],
                                    "decline_reason": None,
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        )
        return httpx.Response(200, text=body)

    # Deliberately omit clock_ms on BOTH create_classifier and ProviderAdapter, and clock on controller!
    provider = ProviderAdapter(
        config=ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:9999/v1",
            model_id="qwen3:14b",
            use_streaming=False,
        ),
        transport=httpx.MockTransport(_gen_handler),
    )
    with SQLiteStore(tmp_path / "defclk.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=create_classifier("rules_v0"),
            calibration=default_calibration_registry(),
            provider=provider,
            store=store,
            run_id="r-defclk",
            telemetry_buffer=BoundedRecorderBuffer(),
        )
        await ctrl.start()
        now_ms = ctrl._clock.now_ms()
        await ctrl.submit_event(
            _seg(
                "s-defclk",
                "e1",
                "seg-1",
                "Someone needs to own the database rollback tonight.",
                rx_ms=now_ms + 100,
                start_ms=now_ms,
                end_ms=now_ms + 90,
                seq=1,
            )
        )
        await ctrl.drain()
        assert ctrl.health is HealthState.OK
        assert http_calls >= 1

        sugs = store.get_suggestions("s-defclk", "r-defclk")
        assert len(sugs) == 1
        assert sugs[0]["route"] == "generator"

        man = await ctrl.trigger_manual()
        assert man.status == "displayed"
        assert man.suggestion is not None
        assert http_calls >= 2
        await ctrl.stop()
        await provider.aclose()
        decs = store.get_decisions("s-defclk", "r-defclk")
        assert len(decs) >= 1
        assert all(d["terminal_reason"] != ReasonCode.CLASSIFIER_ERROR.value for d in decs)
        assert any(d["outcome"] == Outcome.ROUTE.value for d in decs)
        inv = check_conservation_invariants(store, "s-defclk", "r-defclk")
        assert inv.passed, inv.violations


async def test_same_tick_trigger_manual_race_and_breaker_timer_no_drift(tmp_path: Path) -> None:
    """Reviewer Finding 4: same-tick trigger_manual() returns ignored_busy; breaker wake doesn't drift."""
    brief = _brief("s-race")
    base_pol = _policy(generator_enabled=True)
    policy = replace(base_pol, health=replace(base_pol.health, no_transcript_after_ms=200_000))
    clock = ManualClock(1000)

    def _fail_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    provider = ProviderAdapter(
        config=ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:9999/v1",
            model_id="qwen3:14b",
            breaker_consecutive_failures=1,
            breaker_open_ms=60_000,
            use_streaming=False,
        ),
        transport=httpx.MockTransport(_fail_handler),
        clock_ms=clock.now_ms,
    )
    with SQLiteStore(tmp_path / "race.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_ScriptedClassifier(),
            calibration=default_calibration_registry(),
            provider=provider,
            store=store,
            run_id="r-race",
            clock=clock,
        )
        await ctrl.start()
        await ctrl.submit_event(
            _seg(
                "s-race",
                "e1",
                "seg-1",
                "Someone needs to handle the rollback plan.",
                rx_ms=1000,
                seq=1,
            )
        )
        await ctrl.drain()

        # Two same-tick trigger_manual() calls before _manual_loop dequeues the first
        t1 = asyncio.create_task(ctrl.trigger_manual())
        t2 = asyncio.create_task(ctrl.trigger_manual())
        r1, r2 = await asyncio.gather(t1, t2)
        statuses = {r1.status, r2.status}
        assert "displayed" in statuses
        assert "ignored_busy" in statuses
        assert ctrl.manual_ignored_count == 1
        shown = r1.suggestion or r2.suggestion
        assert shown is not None
        await ctrl.handle_ui_action({"action": "dismiss", "suggestion_id": shown.suggestion_id})
        await ctrl.drain()

        # Trip breaker at clock=2000 -> opened_at_ms=2000, wake must stay pinned at 62000
        clock.set_ms(2000)
        provider._record_failure(clock.now_ms())
        assert ctrl._next_display_wake_ms() == 62_000
        clock.set_ms(10_000)
        assert ctrl._next_display_wake_ms() == 62_000
        clock.set_ms(62_000)
        await ctrl.tick()
        from coach.provider_adapter import BreakerState

        assert provider.breaker_state is BreakerState.HALF_OPEN
        await ctrl.stop()
        await provider.aclose()


async def test_proactive_generator_segment_revision_during_classify_passes_inv5(
    tmp_path: Path,
) -> None:
    """In-flight segment revision during classify_batch on Route.GENERATOR satisfies Inv 5."""
    brief = _brief("s-revclf")
    base_pol = _policy(generator_enabled=True)
    policy = replace(
        base_pol,
        categories={
            **base_pol.categories,
            Category.CONFIRM_OWNER: replace(
                base_pol.categories[Category.CONFIRM_OWNER],
                route=Route.GENERATOR,
            ),
        },
    )
    clock = ManualClock(1000)
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingClassifier(_ScriptedClassifier):
        async def classify_batch(
            self,
            contexts: Sequence[Any],
            *,
            deadline_ms: int | None = None,
        ) -> list[ClassifierResult]:
            entered.set()
            await release.wait()
            return await super().classify_batch(contexts, deadline_ms=deadline_ms)

    http_calls = 0

    def _gen_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        return httpx.Response(200, text="{}")

    provider = ProviderAdapter(
        config=ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:9999/v1",
            model_id="qwen3:14b",
            use_streaming=False,
        ),
        transport=httpx.MockTransport(_gen_handler),
        clock_ms=clock.now_ms,
    )
    with SQLiteStore(tmp_path / "revclf.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_BlockingClassifier(),
            calibration=default_calibration_registry(),
            provider=provider,
            store=store,
            run_id="r-revclf",
            clock=clock,
            telemetry_buffer=BoundedRecorderBuffer(),
        )
        await ctrl.start()
        await ctrl.submit_event(
            _seg(
                "s-revclf",
                "e1",
                "seg-1",
                "Someone needs to own the database rollback tonight.",
                rx_ms=1000,
                seq=1,
                rev=1,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        clock.set_ms(1100)
        await ctrl.submit_event(
            _seg(
                "s-revclf",
                "e2",
                "seg-1",
                "I already own the database rollback tonight.",
                speaker="ME",
                rx_ms=1100,
                seq=2,
                rev=2,
            )
        )
        await ctrl._ingest_queue.join()
        release.set()
        await ctrl.drain()
        await ctrl.stop()
        await provider.aclose()

        assert http_calls == 0
        decs = store.get_decisions("s-revclf", "r-revclf")
        assert any(
            d["terminal_state"] == "withdrawn"
            and d["terminal_reason"] == ReasonCode.EVIDENCE_REVISION_CHANGED.value
            for d in decs
        )
        inv = check_conservation_invariants(store, "s-revclf", "r-revclf")
        assert inv.passed, inv.violations


async def test_proactive_multi_pending_gen_deadline_expiry_passes_inv5(tmp_path: Path) -> None:
    """2-item pending_gen where item 1 advances past item 2 display_by_ms skips LLM & passes Inv 5."""
    brief = _brief("s-multigen")
    base_pol = _policy(generator_enabled=True)
    policy = replace(
        base_pol,
        classifier=replace(base_pol.classifier, max_candidates_per_job=4),
        health=replace(base_pol.health, no_transcript_after_ms=200_000),
        categories={
            **base_pol.categories,
            Category.CONFIRM_OWNER: replace(
                base_pol.categories[Category.CONFIRM_OWNER],
                route=Route.GENERATOR,
            ),
        },
    )
    clock = ManualClock(1000)
    http_calls = 0

    def _gen_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        clock.set_ms(10_000)  # Exceeds item 2's display_by_ms (1500 + 8000 = 9500)
        body = json.dumps(
            {
                "id": f"cmpl-{http_calls}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "question": "Who owns the rollback checklist tonight?",
                                    "evidence_segment_ids": ["seg-1"],
                                    "decline_reason": None,
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        )
        return httpx.Response(200, text=body)

    provider = ProviderAdapter(
        config=ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:9999/v1",
            model_id="qwen3:14b",
            use_streaming=False,
        ),
        transport=httpx.MockTransport(_gen_handler),
        clock_ms=clock.now_ms,
    )
    with SQLiteStore(tmp_path / "multigen.db") as store:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_ScriptedClassifier(),
            calibration=default_calibration_registry(),
            provider=provider,
            store=store,
            run_id="r-multigen",
            mode=Mode.PAUSED,
            clock=clock,
            telemetry_buffer=BoundedRecorderBuffer(),
        )
        await ctrl.start()
        await ctrl.submit_event(
            _seg(
                "s-multigen",
                "e1",
                "seg-1",
                "Someone needs to handle the rollback checklist.",
                rx_ms=1000,
                start_ms=500,
                end_ms=900,
                seq=1,
            )
        )
        clock.set_ms(1500)
        await ctrl.submit_event(
            _seg(
                "s-multigen",
                "e2",
                "seg-2",
                "Someone needs to verify the cutover timeline.",
                speaker="S2",
                rx_ms=1500,
                start_ms=1100,
                end_ms=1450,
                seq=2,
            )
        )
        await ctrl.drain()
        await ctrl.set_mode(Mode.PROACTIVE)
        await ctrl.drain()
        await ctrl.stop()
        await provider.aclose()

        assert http_calls == 1
        decs = store.get_decisions("s-multigen", "r-multigen")
        assert any(
            d["terminal_state"] == "expired"
            and d["terminal_reason"] == ReasonCode.DISPLAY_BY_PASSED.value
            for d in decs
        )
        inv = check_conservation_invariants(store, "s-multigen", "r-multigen")
        assert inv.passed, inv.violations


async def test_bug5_manual_unlocked_generator_staleness_b15_b18_b04_b22(tmp_path: Path) -> None:
    """BUG-5 regression: B15, B18, B04, B22 mid-await on manual generator never display stale card."""
    base_pol = _policy(generator_enabled=True)
    policy = replace(
        base_pol,
        manual=replace(base_pol.manual, budget_ms=2000),
        categories={
            **base_pol.categories,
            Category.CONFIRM_OWNER: replace(
                base_pol.categories[Category.CONFIRM_OWNER],
                route=Route.GENERATOR,
            ),
        },
    )

    async def _run_case(case: str, *, manual_only: bool = False) -> None:
        sid = f"s-bug5-{case}"
        rid = f"r-bug5-{case}"
        clock = ManualClock(10)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _gen_handler(_request: httpx.Request) -> httpx.Response:
            entered.set()
            await release.wait()
            body = json.dumps(
                {
                    "id": f"cmpl-{case}",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "question": "Who will own the migration cutover?",
                                        "evidence_segment_ids": ["seg-1"],
                                        "decline_reason": None,
                                    }
                                )
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                }
            )
            return httpx.Response(200, text=body)

        provider = ProviderAdapter(
            config=ProviderConfig(
                provider="local",
                base_url="http://127.0.0.1:9999/v1",
                model_id="qwen3:14b",
                use_streaming=False,
            ),
            transport=httpx.MockTransport(_gen_handler),
            clock_ms=clock.now_ms,
        )
        clf = _ScriptedClassifier(fail=manual_only)
        with SQLiteStore(tmp_path / f"{case}.db") as store:
            ctrl = SuggestionController(
                brief=_brief(sid),
                policy=policy,
                classifier=clf,
                calibration=default_calibration_registry(),
                provider=provider,
                store=store,
                run_id=rid,
                mode=Mode.QUIET,
                clock=clock,
                telemetry_buffer=BoundedRecorderBuffer(),
                classifier_error_threshold=1,
            )
            await ctrl.start()
            await ctrl.submit_event(
                _seg(
                    sid,
                    "e1",
                    "seg-1",
                    "Someone needs to own the migration cutover.",
                    rx_ms=10,
                    start_ms=1,
                    end_ms=9,
                    seq=1,
                )
            )
            await ctrl.drain()
            if manual_only:
                assert ctrl.health is HealthState.MANUAL_ONLY

            clock.set_ms(11)
            man_task = asyncio.create_task(ctrl.trigger_manual())
            await asyncio.wait_for(entered.wait(), timeout=2.0)

            clock.set_ms(12)
            if case in {"b15_brief", "manual_only_b15"}:
                await ctrl.submit_event(
                    CanonicalEvent(
                        schema_version=1,
                        session_id=sid,
                        event_id="ev-brief",
                        kind=EventKind.BRIEF_UPDATE,
                        received_ms=12,
                        source_sequence=2,
                        brief_patch={
                            "objective": "New objective",
                            "do_not_suggest": ["migration"],
                        },
                    )
                )
                await ctrl._ingest_queue.join()
            elif case == "b18_otr_toggle":
                await ctrl.set_mode(Mode.OFF_THE_RECORD)
                await ctrl.set_mode(Mode.QUIET)
            elif case == "b04_rev":
                await ctrl.submit_event(
                    _seg(
                        sid,
                        "e2",
                        "seg-1",
                        "I already own the migration cutover.",
                        speaker="ME",
                        rx_ms=12,
                        start_ms=1,
                        end_ms=9,
                        seq=2,
                        rev=2,
                    )
                )
                await ctrl._ingest_queue.join()
            elif case == "b22_speaker_update":
                await ctrl.submit_event(
                    CanonicalEvent(
                        schema_version=1,
                        session_id=sid,
                        event_id="ev-spk",
                        kind=EventKind.SPEAKER_UPDATE,
                        received_ms=12,
                        source_sequence=2,
                        segment_id="seg-1",
                        speaker_id="ME",
                    )
                )
                await ctrl._ingest_queue.join()

            release.set()
            res = await asyncio.wait_for(man_task, timeout=2.0)
            assert res.status == "empty", (case, res)
            assert ctrl.active_card is None, case
            await ctrl.drain()
            assert ctrl.active_card is None, case
            await ctrl.stop()
            await provider.aclose()

            sugs = store.get_suggestions(sid, rid)
            for s in sugs:
                transitions = json.loads(s["transitions_json"])
                assert all(t["transition"] != LedgerState.DISPLAYED.value for t in transitions), (
                    case,
                    transitions,
                )
            inv = check_conservation_invariants(store, sid, rid)
            assert inv.passed, (case, inv.violations)

    await _run_case("b15_brief")
    await _run_case("b18_otr_toggle")
    await _run_case("b04_rev")
    await _run_case("b22_speaker_update")
    await _run_case("manual_only_b15", manual_only=True)


async def test_no_transcript_and_stop_retire_expired_uncovered_anchors(tmp_path: Path) -> None:
    base_pol = _policy()
    policy = replace(base_pol, display=replace(base_pol.display, max_candidate_age_ms=500))
    sid = "sess-no-tx-retire"
    rid = "run-no-tx-retire"
    clock = ManualClock(0)
    entered = asyncio.Event()
    release = asyncio.Event()
    inner = create_classifier("rules_v0", clock_ms=clock.now_ms)

    class _GatedClf:
        classifier_version = inner.classifier_version
        tokenizer = inner.tokenizer

        async def classify_batch(
            self, contexts: Sequence[Any], *, deadline_ms: int | None = None
        ) -> list[ClassifierResult]:
            entered.set()
            await release.wait()
            return await inner.classify_batch(contexts, deadline_ms=deadline_ms)

    with SQLiteStore(tmp_path / "no_tx_retire.db") as store:
        ctrl = SuggestionController(
            brief=_brief(session_id=sid),
            policy=policy,
            classifier=_GatedClf(),
            clock=clock,
            store=store,
            run_id=rid,
        )
        try:
            await ctrl.start()
            clock.set_ms(100)
            await ctrl.submit_event(
                _seg(
                    sid,
                    "e0",
                    "s0",
                    "Someone should own staging.",
                    rx_ms=100,
                    start_ms=50,
                    end_ms=90,
                    seq=1,
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            clock.set_ms(200)
            await ctrl.submit_event(
                _seg(
                    sid,
                    "e1",
                    "s1",
                    "Someone should own rollback.",
                    rx_ms=200,
                    start_ms=150,
                    end_ms=190,
                    seq=2,
                )
            )
            clock.set_ms(600)
            await ctrl.submit_event(
                _seg(
                    sid,
                    "e2",
                    "s2",
                    "Someone should own cutover.",
                    rx_ms=600,
                    start_ms=550,
                    end_ms=590,
                    seq=3,
                )
            )
            await ctrl._ingest_queue.join()
            await ctrl.report_adapter_health(HealthState.NO_TRANSCRIPT, "peecee_tunnel_lost")
            clock.set_ms(800)
            release.set()
            await ctrl.drain()
            await ctrl.tick()
            await ctrl.flush_recorder()
            decs_after_tick = {d["anchor_refs_json"]: d for d in store.get_decisions(sid, rid)}
            assert any(
                "s1" in k and v["terminal_reason"] == ReasonCode.EXPIRED_UNEVALUATED.value
                for k, v in decs_after_tick.items()
            )
        finally:
            await ctrl.stop()

        decs = {d["anchor_refs_json"]: d for d in store.get_decisions(sid, rid)}
        reasons = {d["terminal_reason"] for d in decs.values()}
        assert ReasonCode.EXPIRED_UNEVALUATED.value in reasons
        assert ReasonCode.SESSION_ENDED.value in reasons
        assert check_conservation_invariants(store, sid, rid).passed
