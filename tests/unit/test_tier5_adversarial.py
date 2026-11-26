"""Tier 5 white-box adversarial regression tests for coach/."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from coach.annotation_tool import (
    AnnotationServer,
    Pass2Session,
    load_labels_jsonl,
    load_review_queue_items,
)
from coach.candidate_validator import validate_generated_candidate
from coach.canonical_state import CanonicalState, StateSnapshot
from coach.context_builder import (
    ContextBuildResult,
    Tokenizer,
    WhitespaceTokenizer,
    build_decision_context,
)
from coach.contracts import (
    SCHEMA_VERSION,
    CanonicalEvent,
    Category,
    ClassifierResult,
    ContractValidationError,
    EventKind,
    Mode,
    Origin,
    ReasonCode,
    RecipientConfig,
    RetentionConfig,
    Route,
    SampleSource,
    ScoreSet,
    SessionBrief,
    SessionMode,
    Stability,
)
from coach.opportunity_classifier import ClassifierInput, RulesClassifier, coerce_classifier_input
from coach.provider_adapter import (
    BreakerState,
    GenerationRequest,
    ModelCapabilities,
    PromptBlocks,
    ProviderAdapter,
    ProviderConfig,
    build_openai_payload,
)
from coach.question_generator import generate_question
from coach.replay_evaluator import async_sweep_t_display, simulate_replay
from coach.routing_policy import load_policy
from coach.storage import SQLiteStore
from coach.suggestion_controller import SuggestionController
from coach.telemetry import check_conservation_invariants
from coach.transcript_adapter import ManualClock, ReplayPlayer, ReplayTimingMode
from coach.ui import CardUIServer

POLICY_PATH = Path("policies/policy-v0-a.yaml")


def _seg_ev(
    session_id: str,
    event_id: str,
    segment_id: str,
    text: str,
    *,
    speaker: str = "speaker-me",
    rx_ms: int = 1000,
    start_ms: int = 100,
    end_ms: int = 900,
    rev: int = 1,
    seq: int = 1,
    boundary: str | None = "utterance_end",
) -> CanonicalEvent:
    return CanonicalEvent(
        schema_version=SCHEMA_VERSION,
        event_id=event_id,
        session_id=session_id,
        kind=EventKind.SEGMENT,
        stability=Stability.COMMITTED,
        segment_id=segment_id,
        source_revision=rev,
        channel=0,
        speaker_id=speaker,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        received_ms=rx_ms,
        source_sequence=seq,
        boundary=boundary,
    )


@pytest.mark.asyncio
async def test_store_transcript_false_never_writes_or_reads_model_cache_text() -> None:
    policy = load_policy(POLICY_PATH)
    db = SQLiteStore(":memory:")

    def _make_state(session_id: str, store_tx: bool) -> CanonicalState:
        brief = SessionBrief(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
            objective="Secret project X",
            recipient=RecipientConfig("speaker-rec"),
            retention=RetentionConfig(
                store_transcript=store_tx, store_raw_provider_events=store_tx
            ),
        )
        st = CanonicalState(brief)
        st.apply(
            _seg_ev(
                session_id,
                f"pe-{session_id}",
                "pseg1",
                "Confidential revenue target is fifty million dollars by Q4.",
                rx_ms=3050,
            )
        )
        return st

    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "question": "Who owns the fifty million target?",
                                    "evidence_segment_ids": ["pseg1"],
                                    "decline_reason": None,
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 12},
            },
        )

    provider = ProviderAdapter(
        ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            model_id="qwen3:14b",
            local_only=True,
            use_streaming=False,
        ),
        transport=httpx.MockTransport(handler),
    )

    priv_snap = _make_state("priv-sess", False).snapshot()
    priv_built = build_decision_context(priv_snap, [priv_snap.segments[-1]], policy=policy)
    assert priv_built.context is not None
    await generate_question(
        priv_built.context,
        Category.CONFIRM_OWNER,
        priv_snap,
        policy,
        provider,
        clock_ms=lambda: 3200,
        store=db,
    )
    assert calls == 1
    assert (
        db._conn.execute("SELECT contains_text FROM model_cache WHERE contains_text = 1").fetchall()
        == []
    )

    # Even if another opt-in session populates the same prompt hash, opt-out must not read cache.
    pub_snap = _make_state("pub-sess", True).snapshot()
    pub_built = build_decision_context(pub_snap, [pub_snap.segments[-1]], policy=policy)
    assert pub_built.context is not None
    await generate_question(
        pub_built.context,
        Category.CONFIRM_OWNER,
        pub_snap,
        policy,
        provider,
        clock_ms=lambda: 3200,
        store=db,
    )
    assert calls == 2
    out_priv2 = await generate_question(
        priv_built.context,
        Category.CONFIRM_OWNER,
        priv_snap,
        policy,
        provider,
        clock_ms=lambda: 3200,
        store=db,
    )
    assert calls == 3
    assert out_priv2.from_cache is False
    await provider.aclose()
    db.close()


@pytest.mark.asyncio
async def test_delete_session_text_preserves_other_session_generator_cache() -> None:
    policy = load_policy(POLICY_PATH)
    db = SQLiteStore(":memory:")

    def _run_snap(session_id: str, text: str) -> tuple[StateSnapshot, ContextBuildResult]:
        brief = SessionBrief(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
            objective=f"Objective for {session_id}",
            recipient=RecipientConfig("speaker-rec"),
        )
        st = CanonicalState(brief)
        st.apply(_seg_ev(session_id, f"e-{session_id}", "s1", text, rx_ms=3050))
        snap = st.snapshot()
        built = build_decision_context(snap, [snap.segments[-1]], policy=policy)
        return snap, built

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content.decode("utf-8")
        q = "Who owns alpha?" if "alpha" in body else "Who owns beta?"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "question": q,
                                    "evidence_segment_ids": ["s1"],
                                    "decline_reason": None,
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10},
            },
        )

    provider = ProviderAdapter(
        ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            model_id="qwen3:14b",
            local_only=True,
            use_streaming=False,
        ),
        transport=httpx.MockTransport(handler),
    )
    snap_a, built_a = _run_snap("sess-a", "Someone should own alpha rollback.")
    snap_b, built_b = _run_snap("sess-b", "Someone should own beta rollback.")
    assert built_a.context is not None and built_b.context is not None
    await generate_question(
        built_a.context,
        Category.CONFIRM_OWNER,
        snap_a,
        policy,
        provider,
        clock_ms=lambda: 3200,
        store=db,
    )
    await generate_question(
        built_b.context,
        Category.CONFIRM_OWNER,
        snap_b,
        policy,
        provider,
        clock_ms=lambda: 3200,
        store=db,
    )
    assert (
        len(db._conn.execute("SELECT * FROM model_cache WHERE contains_text = 1").fetchall()) == 2
    )

    summary_a = db.delete_session_text("sess-a", deleted_at_ms=5000)
    assert summary_a.model_cache_purged == 1
    rem = db._conn.execute("SELECT output_json FROM model_cache WHERE contains_text = 1").fetchall()
    assert len(rem) == 1
    assert "beta" in str(rem[0]["output_json"])

    summary_b = db.delete_session_text("sess-b", deleted_at_ms=6000)
    assert summary_b.model_cache_purged == 1
    assert db._conn.execute("SELECT * FROM model_cache WHERE contains_text = 1").fetchall() == []
    await provider.aclose()
    db.close()


def test_open_tail_punctuation_free_segment_classified_before_expiry() -> None:
    policy = load_policy(POLICY_PATH)
    ev_open = _seg_ev(
        "open-sess",
        "oe1",
        "seg-open",
        "we probably need someone to check the rollout latency right",
        rx_ms=3050,
        start_ms=1000,
        end_ms=3000,
        seq=1,
        boundary=None,
    )
    ev_later = _seg_ev(
        "open-sess",
        "oe2",
        "seg-later",
        "okay moving on now.",
        rx_ms=20050,
        start_ms=19000,
        end_ms=20000,
        seq=2,
        boundary="utterance_end",
    )
    db = SQLiteStore(":memory:")
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="open-sess",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Ship v1",
        recipient=RecipientConfig("speaker-rec"),
    )
    simulate_replay(
        brief, [ev_open, ev_later], policy, RulesClassifier(), store=db, run_id="open-run"
    )
    decs = db.get_decisions("open-sess", "open-run")
    inv = check_conservation_invariants(db, "open-sess", "open-run")
    db.close()
    assert inv.passed, inv.violations
    assert decs[0]["classifier_version"] is not None
    assert decs[0]["terminal_reason"] != "expired_unevaluated"


@pytest.mark.asyncio
async def test_async_sweep_t_display_isolates_db_and_clones_provider() -> None:
    base_pol = load_policy(POLICY_PATH)
    policy = replace(
        base_pol,
        generator=replace(base_pol.generator, enabled=True),
        categories={
            **base_pol.categories,
            Category.CONFIRM_OWNER: replace(
                base_pol.categories[Category.CONFIRM_OWNER], route=Route.GENERATOR
            ),
        },
    )
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="sw-sess",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Ship v1",
        recipient=RecipientConfig("speaker-rec"),
    )
    ev = _seg_ev(
        "sw-sess", "se1", "seg1", "Someone should own the deployment rollout.", rx_ms=1000, seq=1
    )

    def err_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    provider = ProviderAdapter(
        ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            model_id="qwen3:14b",
            local_only=True,
            use_streaming=False,
        ),
        transport=httpx.MockTransport(err_handler),
    )
    caller_db = SQLiteStore(":memory:")
    curve = await async_sweep_t_display(
        [(brief, [ev])],
        [],
        policy,
        RulesClassifier(),
        [0.4, 0.8],
        store=caller_db,
        provider=provider,
        mode=Mode.PROACTIVE,
    )
    assert len(curve) == 2
    assert caller_db._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    assert caller_db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert provider.breaker_state is BreakerState.CLOSED
    await provider.aclose()
    caller_db.close()


class _FixedScoreClf:
    def __init__(self, version: str, category: Category, score: float) -> None:
        self.classifier_version = version
        self.tokenizer: Tokenizer = WhitespaceTokenizer()
        self._category = category
        self._score = score

    async def classify(
        self, context: ContextBuildResult | ClassifierInput, deadline_ms: int | None = None
    ) -> ClassifierResult:
        _ = deadline_ms
        ctx = coerce_classifier_input(context).context
        sc = ScoreSet(opportunity=self._score, categories={self._category.value: self._score})
        return ClassifierResult(
            schema_version=SCHEMA_VERSION,
            candidate_id=ctx.candidate_id,
            based_on_state_version=ctx.based_on_state_version,
            classifier_version=self.classifier_version,
            raw_scores=sc,
            calibration_version=None,
            calibrated_scores=sc,
            latency_ms=1.0,
            input_refs=ctx.anchor_refs,
        )

    async def classify_batch(
        self,
        contexts: Sequence[ContextBuildResult | ClassifierInput],
        *,
        deadline_ms: int | None = None,
    ) -> list[ClassifierResult]:
        return [await self.classify(i, deadline_ms=deadline_ms) for i in contexts]


@pytest.mark.asyncio
async def test_controller_close_pending_generic_referent_syncs_ledger_and_repeat_resets() -> None:
    base_pol = load_policy(POLICY_PATH)
    policy = replace(
        base_pol,
        display=replace(base_pol.display, settle_ms=7000, max_candidate_age_ms=8000),
    )
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="gen-ref-s",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Ship v1",
        recipient=RecipientConfig("speaker-rec"),
    )
    clk = ManualClock(1000)
    db = SQLiteStore(":memory:")
    ctrl = SuggestionController(
        brief=brief,
        policy=policy,
        classifier=_FixedScoreClf(policy.classifier.version, Category.CONFIRM_OWNER, 0.96),
        store=db,
        run_id="r-gr",
        mode=Mode.PROACTIVE,
        clock=clk,
    )
    await ctrl.start()
    # Generic referent (no specific owner/action slot, triggers generic fallback with 6s max_anchor_age_ms)
    await ctrl.submit_event(
        _seg_ev(
            "gen-ref-s",
            "ge1",
            "gseg1",
            "We really ought to make sure that happens very soon.",
            rx_ms=1000,
            seq=1,
        )
    )
    await ctrl.drain()
    # Advance past generic template max_anchor_age_ms (6000ms) -> GENERIC_REFERENT_UNCLEAR at display tick
    clk.set_ms(8000)
    await ctrl.tick()
    await ctrl.drain()

    assert ctrl.canonical_state.snapshot().ledger == {}
    # Subsequent SPEAKER_UPDATE should not leave anything stuck in READY
    clk.set_ms(8100)
    await ctrl.submit_event(
        CanonicalEvent(
            schema_version=SCHEMA_VERSION,
            event_id="gsu1",
            session_id="gen-ref-s",
            kind=EventKind.SPEAKER_UPDATE,
            received_ms=8100,
            source_sequence=2,
            segment_id="gseg1",
            speaker_id="speaker-rec",
        )
    )
    await ctrl.drain()
    assert ctrl.canonical_state.snapshot().ledger == {}
    await ctrl.stop()
    sugs = db.get_suggestions("gen-ref-s", "r-gr")
    assert any(
        s["current_state"] == "unrendered" and s["terminal_reason"] == "generic_referent_unclear"
        for s in sugs
    )
    assert check_conservation_invariants(db, "gen-ref-s", "r-gr").passed
    db.close()


@pytest.mark.asyncio
async def test_controller_resets_repeat_after_30s_gap_and_same_segment_revision() -> None:
    policy = load_policy(POLICY_PATH)
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="ctrl-s",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Ship v1",
        recipient=RecipientConfig("speaker-rec"),
    )
    clk = ManualClock(5000)
    db = SQLiteStore(":memory:")
    ctrl = SuggestionController(
        brief=brief,
        policy=policy,
        classifier=_FixedScoreClf(policy.classifier.version, Category.CONFIRM_OWNER, 0.96),
        store=db,
        run_id="r1",
        mode=Mode.PROACTIVE,
        clock=clk,
    )
    await ctrl.start()
    await ctrl.submit_event(
        _seg_ev(
            "ctrl-s",
            "de1",
            "dseg1",
            "Someone needs to own the deployment checklist.",
            start_ms=1000,
            end_ms=4000,
            rx_ms=4500,
            seq=1,
        )
    )
    await ctrl.drain()

    # Same text 40_000ms later -> NOT dropped as repeat
    clk.set_ms(45000)
    await ctrl.submit_event(
        _seg_ev(
            "ctrl-s",
            "de2",
            "dseg2",
            "Someone needs to own the deployment checklist.",
            start_ms=41000,
            end_ms=44000,
            rx_ms=44500,
            seq=2,
        )
    )
    await ctrl.drain()

    # Immediate identical text on distinct segment (< 30_000ms) -> dropped as repeat
    clk.set_ms(46000)
    await ctrl.submit_event(
        _seg_ev(
            "ctrl-s",
            "de3",
            "dseg3",
            "Someone needs to own the deployment checklist.",
            start_ms=45000,
            end_ms=45800,
            rx_ms=45900,
            seq=3,
        )
    )
    await ctrl.drain()

    # Same segment speaker_update/revision -> NOT dropped as repeat
    clk.set_ms(46500)
    await ctrl.submit_event(
        CanonicalEvent(
            schema_version=SCHEMA_VERSION,
            event_id="de4-su",
            session_id="ctrl-s",
            kind=EventKind.SPEAKER_UPDATE,
            received_ms=46500,
            source_sequence=4,
            segment_id="dseg2",
            speaker_id="speaker-other",
        )
    )
    await ctrl.drain()
    await ctrl.stop()
    decs = db.get_decisions("ctrl-s", "r1")
    by_seg = {d["anchor_refs_json"]: d for d in decs}
    assert any("dseg2" in k and d["terminal_reason"] != "repeat" for k, d in by_seg.items())
    assert any("dseg3" in k and d["terminal_reason"] == "repeat" for k, d in by_seg.items())
    assert check_conservation_invariants(db, "ctrl-s", "r1").passed
    db.close()


@pytest.mark.asyncio
async def test_controller_handle_ui_action_manual_rejects_and_manual_decision_conservation() -> (
    None
):
    base_pol = load_policy(POLICY_PATH)
    policy = replace(
        base_pol,
        generator=replace(base_pol.generator, enabled=False),
        categories={
            Category.FOLLOW_UP: replace(
                base_pol.categories[Category.CONFIRM_OWNER],
                route=Route.GENERATOR,
                template=None,
            )
        },
    )
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="man-s",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Ship v1",
        recipient=RecipientConfig("speaker-rec"),
    )
    clk = ManualClock(2000)
    db = SQLiteStore(":memory:")
    ui = CardUIServer(session_id="man-s", port=0)
    await ui.start()
    try:
        ctrl = SuggestionController(
            brief=brief,
            policy=policy,
            classifier=_FixedScoreClf(policy.classifier.version, Category.FOLLOW_UP, 0.90),
            store=db,
            run_id="r-man",
            mode=Mode.SHADOW,
            clock=clk,
            ui_server=ui,
        )
        await ctrl.start()
        await ctrl.handle_ui_action({"action": "manual"})
        assert ctrl._manual_in_flight is False
        assert ctrl._manual_queue.qsize() == 0
        snap_ui = ctrl.build_ui_snapshot()
        assert snap_ui.manual_status is not None
        assert snap_ui.manual_status["status"] == "unavailable"

        # Switch to Proactive: Route.GENERATOR with generator.enabled=False + no template -> records manual decision
        await ctrl.set_mode(Mode.PROACTIVE)
        await ctrl.submit_event(
            _seg_ev(
                "man-s",
                "me1",
                "mseg1",
                "Someone should own the cutover checklist.",
                rx_ms=2000,
                seq=1,
            )
        )
        await ctrl.drain()
        res_empty = await ctrl.trigger_manual()
        assert res_empty.status == "empty"
        await ctrl.drain()
        await ctrl.stop()
        decs = db.get_decisions("man-s", "r-man")
        man_decs = [d for d in decs if d.get("origin") == "manual"]
        assert len(man_decs) >= 1
        assert check_conservation_invariants(db, "man-s", "r-man").passed
    finally:
        await ui.stop()
        db.close()


@pytest.mark.asyncio
async def test_manual_display_does_not_eagerly_write_expired_decision() -> None:
    policy = load_policy(POLICY_PATH)
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="man-exp",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Ship v1",
        recipient=RecipientConfig("speaker-rec"),
    )
    clk = ManualClock(2000)
    db = SQLiteStore(":memory:")
    ctrl = SuggestionController(
        brief=brief,
        policy=policy,
        classifier=_FixedScoreClf(policy.classifier.version, Category.CONFIRM_OWNER, 0.95),
        store=db,
        run_id="r-exp",
        mode=Mode.QUIET,
        clock=clk,
    )
    await ctrl.start()
    await ctrl.submit_event(
        _seg_ev(
            "man-exp",
            "me2",
            "mseg2",
            "Someone needs to own the cutover plan.",
            rx_ms=2000,
            seq=1,
        )
    )
    await ctrl.drain()
    man_res = await ctrl.trigger_manual()
    assert man_res.status == "displayed"
    assert man_res.suggestion is not None
    await ctrl.flush_recorder()
    mid_decs = [d for d in db.get_decisions("man-exp", "r-exp") if d.get("origin") == "manual"]
    assert all(d["terminal_state"] != "expired" for d in mid_decs)

    await ctrl.handle_ui_action(
        {"action": "use", "suggestion_id": man_res.suggestion.suggestion_id}
    )
    await ctrl.drain()
    await ctrl.stop()
    end_decs = [d for d in db.get_decisions("man-exp", "r-exp") if d.get("origin") == "manual"]
    assert len(end_decs) == 1
    assert end_decs[0]["terminal_state"] == "used"
    assert end_decs[0]["terminal_reason"] == ReasonCode.USER_ACTION.value
    assert check_conservation_invariants(db, "man-exp", "r-exp").passed
    db.close()


def test_pass2_step1_and_unlock_cannot_overwrite_sample_source_and_labels_validate(
    tmp_path: Path,
) -> None:
    policy = load_policy(POLICY_PATH)
    ev = _seg_ev(
        "as1", "ae1", "seg1", "Someone should own this migration.", speaker="s0", rx_ms=1000
    )
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="as1",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Test",
        recipient=RecipientConfig("s1"),
    )
    items = load_review_queue_items(
        brief,
        [ev],
        [
            {
                "suggestion_id": "cid-1",
                "segment_ids": ["seg1"],
                "text": "Who owns this migration?",
                "category": "CONFIRM_OWNER",
            }
        ],
        policy=policy,
    )
    p2 = Pass2Session(items=items, annotator="a1")
    cid = items[0].candidate_id or items[0].item_id
    srv = AnnotationServer(port=0, pass2=p2)
    srv.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/api/candidates/{cid}/step1",
            data=json.dumps(
                {
                    "worthwhile": True,
                    "categories": ["CONFIRM_OWNER"],
                    "sample_source": "random",
                }
            ).encode(),
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {srv.token}")
        req.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req, timeout=3)
        assert p2._items[items[0].item_id].sample_source is SampleSource.SHADOW_REVIEW

        req2 = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/api/candidates/{cid}/step2/unlock",
            data=json.dumps(
                {
                    "step1": {
                        "worthwhile": True,
                        "categories": ["CONFIRM_OWNER"],
                        "sample_source": "random",
                    }
                }
            ).encode(),
            method="POST",
        )
        req2.add_header("Authorization", f"Bearer {srv.token}")
        req2.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req2, timeout=3)
        assert p2._items[items[0].item_id].sample_source is SampleSource.SHADOW_REVIEW
    finally:
        srv.stop()

    bad_before = tmp_path / "labels_before.jsonl"
    bad_before.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "label_id": "lbl-1",
                "session_id": "s1",
                "annotator": "a1",
                "sample_source": "random",
                "pass": "online",
                "candidate_id": None,
                "state_version_at_decision": 1,
                "worthwhile": True,
                "categories": ["CONFIRM_OWNER"],
                "example_questions": [],
                "evidence_refs": [{"segment_id": "seg1", "revision": 1}],
                "validity": {
                    "open_ms": 5000,
                    "ideal_ms": 1000,
                    "close_ms": 10000,
                    "closed_by": "topic_moved",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError):
        load_labels_jsonl(bad_before)

    bad_after = tmp_path / "labels_after.jsonl"
    bad_after.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "label_id": "lbl-2",
                "session_id": "s1",
                "annotator": "a1",
                "sample_source": "random",
                "pass": "online",
                "candidate_id": None,
                "state_version_at_decision": 1,
                "worthwhile": True,
                "categories": ["CONFIRM_OWNER"],
                "example_questions": [],
                "evidence_refs": [{"segment_id": "seg1", "revision": 1}],
                "validity": {
                    "open_ms": 5000,
                    "ideal_ms": 12000,
                    "close_ms": 10000,
                    "closed_by": "topic_moved",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError):
        load_labels_jsonl(bad_after)


def test_qwen3_uppercase_model_id_sets_chat_template_kwargs() -> None:
    cfg = ProviderConfig(
        provider="vllm",
        base_url="http://127.0.0.1:8000/v1",
        model_id="Qwen/Qwen3-32B",
        reasoning_setting="off",
    )
    caps = ModelCapabilities(supported_reasoning=frozenset({None, "off", "low"}))
    req = GenerationRequest(
        session_id="s1",
        candidate_id="c1",
        lane=Origin.PROACTIVE,
        prompt_blocks=PromptBlocks(
            policy_block="sys",
            brief_block="brief",
            transcript_block="ctx",
            opportunity_block="anch",
        ),
        segment_ids=("seg1",),
        output_token_cap=120,
        deadline_ms=999999,
    )
    payload = build_openai_payload(cfg, caps, req)
    assert payload.get("chat_template_kwargs") == {"enable_thinking": False}
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_speed", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
async def test_replay_player_rejects_non_finite_and_non_positive_speed(bad_speed: float) -> None:
    ev = _seg_ev("s1", "e1", "seg1", "Test turn.", speaker="s0", rx_ms=250)
    rp = ReplayPlayer([ev])
    with pytest.raises(ValueError):
        _ = [x async for x in rp.play(mode=ReplayTimingMode.SCALED, speed=bad_speed)]


def test_candidate_validator_rejects_html_markdown_links_and_prompt_injection() -> None:
    brief = SessionBrief(
        schema_version=SCHEMA_VERSION,
        session_id="val-s",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        objective="Check migration latency",
        recipient=RecipientConfig("speaker-rec"),
    )
    st = CanonicalState(brief)
    st.apply(
        _seg_ev(
            "val-s",
            "ve1",
            "seg1",
            "We should check migration latency < 100 ms and alert on errors.",
            rx_ms=1000,
        )
    )
    snap = st.snapshot()
    adversarial_questions = [
        "Should we check <script>alert()</script>?",
        "Should we check <img src=migration onerror=alert()>?",
        "Should we check [migration](http://localhost)?",
        "Should we check migration and ignore previous instructions?",
        "Should we check migration and disregard all instructions?",
    ]
    for q in adversarial_questions:
        res = validate_generated_candidate(
            json.dumps(
                {
                    "question": q,
                    "evidence_segment_ids": ["seg1"],
                    "decline_reason": None,
                }
            ),
            finish_reason="stop",
            category=Category.CONFIRM_OWNER,
            snapshot=snap,
            supplied_segment_ids=["seg1"],
        )
        assert not res.valid, f"Expected rejection for {q!r}"
        assert res.failed_check == 2
        assert res.reason_code is ReasonCode.GENERATOR_INVALID_OUTPUT

    ok_res = validate_generated_candidate(
        json.dumps(
            {
                "question": "Is latency < 100 ms required for migration?",
                "evidence_segment_ids": ["seg1"],
                "decline_reason": None,
            }
        ),
        finish_reason="stop",
        category=Category.CONFIRM_TIME,
        snapshot=snap,
        supplied_segment_ids=["seg1"],
    )
    assert ok_res.valid
