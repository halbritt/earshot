"""Unit tests for replay_evaluator, splitters, look counter, matcher, sweeper, bootstrap, and benchmark (M4)."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from coach.calibration import default_calibration_metadata
from coach.canonical_state import CanonicalState
from coach.cli import main as cli_main
from coach.context_builder import build_decision_contexts
from coach.contracts import (
    AnnotationPass,
    AttentionConfig,
    CanonicalEvent,
    Category,
    ClosedBy,
    ControlSubtype,
    EventKind,
    HealthState,
    LabelRecord,
    NoneReason,
    Participant,
    RecipientConfig,
    Route,
    SampleSource,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    ValidityInterval,
)
from coach.opportunity_classifier import ClassifierInput, create_classifier
from coach.provider_adapter import BreakerState, ProviderAdapter, ProviderConfig
from coach.replay_evaluator import (
    DisplayedCardEvent,
    EvalLookCounter,
    async_simulate_dataset_replay,
    async_sweep_dataset_t_display,
    match_opportunities,
    paired_conversation_bootstrap,
    run_latency_benchmark,
    simulate_dataset_replay,
    simulate_replay,
    split_by_conversation,
    split_by_series,
    sweep_t_display,
)
from coach.routing_policy import load_policy
from coach.storage import SQLiteStore
from coach.telemetry import build_session_report, check_conservation_invariants

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "policies" / "policy-v0-a.yaml"


def _brief(session_id: str = "sess-re", *, max_cards: int = 3) -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id=session_id,
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="me-1", role="reviewer"),
        objective="Leave with a clear owner and deadline for the rollback plan.",
        participants=(Participant(participant_id="spk-1", label="S1", role="engineer"),),
        agenda=("Rollback plan",),
        attention=AttentionConfig(max_cards_per_10_min=max_cards, proactive=True),
    )


def _seg(
    seg_id: str,
    text: str,
    *,
    session_id: str = "sess-re",
    speaker_id: str = "spk-1",
    start_ms: int = 1000,
    end_ms: int | None = None,
    received_ms: int | None = None,
    seq: int = 1,
    rev: int = 1,
    boundary: str | None = None,
) -> CanonicalEvent:
    end = end_ms if end_ms is not None else start_ms + 900
    rx = received_ms if received_ms is not None else end + 50
    return CanonicalEvent(
        schema_version=1,
        session_id=session_id,
        event_id=f"ev-{seg_id}-{rev}-{seq}",
        kind=EventKind.SEGMENT,
        segment_id=seg_id,
        source_revision=rev,
        channel=1,
        speaker_id=speaker_id,
        speaker_confidence=0.95,
        start_ms=start_ms,
        end_ms=end,
        received_ms=rx,
        text=text,
        stability=Stability.COMMITTED,
        source_sequence=seq,
        boundary=boundary,
    )


def _lbl(
    label_id: str,
    *,
    session_id: str = "sess-re",
    seg_id: str = "s1",
    worthwhile: bool = True,
    categories: tuple[Category, ...] = (Category.CONFIRM_OWNER,),
    open_ms: int = 1000,
    close_ms: int | None = 5000,
    closed_by: ClosedBy | None = None,
    none_reason: NoneReason | None = None,
) -> LabelRecord:
    return LabelRecord(
        schema_version=1,
        label_id=label_id,
        session_id=session_id,
        annotator="ann-1",
        sample_source=SampleSource.RANDOM,
        annotation_pass=AnnotationPass.OFFLINE,
        candidate_id=f"cand-{label_id}",
        state_version_at_decision=1,
        worthwhile=worthwhile,
        categories=categories,
        example_questions=("Who will own it?",) if worthwhile else (),
        evidence_refs=(SegmentRef(seg_id, 1),),
        validity=ValidityInterval(open_ms=open_ms, close_ms=close_ms, closed_by=closed_by),
        none_reason=none_reason,
    )


def test_splitters_and_eval_look_counter(tmp_path: Path) -> None:
    rows = [
        {"session_id": f"conv-{i}", "series_id": f"series-{i % 3}", "val": i} for i in range(12)
    ]
    conv_splits = split_by_conversation(rows, seed=42)
    assert not (set(conv_splits.train_conversations) & set(conv_splits.cal_conversations))
    assert not (set(conv_splits.train_conversations) & set(conv_splits.eval_conversations))
    assert not (set(conv_splits.cal_conversations) & set(conv_splits.eval_conversations))
    assert sum(conv_splits.to_dict()["counts"].values()) == 12

    ser_splits = split_by_series(rows, seed=7)
    assert not (set(ser_splits.train_series) & set(ser_splits.cal_series))
    assert not (set(ser_splits.train_series) & set(ser_splits.eval_series))
    assert not (set(ser_splits.cal_series) & set(ser_splits.eval_series))
    assert len(ser_splits.train_series) == 1
    assert len(ser_splits.cal_series) == 1
    assert len(ser_splits.eval_series) == 1

    counter = EvalLookCounter(tmp_path / "eval_looks.json")
    assert counter.look_count == 0
    assert counter.record_look("run-1", split_by="series", eval_conversations=["conv-2"]) == 1
    assert counter.record_look("run-2", split_by="series", eval_conversations=["conv-2"]) == 2
    assert counter.look_count == 2
    assert len(counter.history) == 2


def test_validity_matcher_exact_bounds_premature_late_distracting_and_review_queue() -> None:
    labels = [
        # (1) useful + repetition
        _lbl("l1", seg_id="s1", open_ms=1000, close_ms=2000, closed_by=ClosedBy.TOPIC_MOVED),
        # (2) lbl-em-3: open=7000, close=7120 (close - open = 120 > settle=100), display=7200 -> late
        _lbl("l3", seg_id="s3", open_ms=7000, close_ms=7120, closed_by=ClosedBy.ANSWERED),
        # (3) lbl-em-4: open=10100, close=10150 (close - open = 50 <= settle=100), display=10200 -> premature
        _lbl("l4", seg_id="s4", open_ms=10100, close_ms=10150, closed_by=ClosedBy.ANSWERED),
        # (4) negative -> distracting
        _lbl("l5", seg_id="s5", worthwhile=False, categories=(), none_reason=NoneReason.RHETORICAL),
    ]
    cards = [
        # Inclusive close_ms=2000 -> useful
        DisplayedCardEvent(
            "c1", "k1", "sess-re", Category.CONFIRM_OWNER, 2000, (SegmentRef("s1", 1),)
        ),
        # Second match on l1 -> repetition
        DisplayedCardEvent(
            "c2", "k2", "sess-re", Category.CONFIRM_OWNER, 1500, (SegmentRef("s1", 1),)
        ),
        # Late (answered)
        DisplayedCardEvent(
            "c3", "k3", "sess-re", Category.CONFIRM_OWNER, 7200, (SegmentRef("s3", 1),)
        ),
        # Premature inside settle window
        DisplayedCardEvent(
            "c4", "k4", "sess-re", Category.CONFIRM_OWNER, 10200, (SegmentRef("s4", 1),)
        ),
        # Distracting (rhetorical)
        DisplayedCardEvent(
            "c5", "k5", "sess-re", Category.CONFIRM_OWNER, 12000, (SegmentRef("s5", 1),)
        ),
        # Unlabeled anchor -> review_queue (NEVER scored as error)
        DisplayedCardEvent(
            "c6", "k6", "sess-re", Category.CONFIRM_OWNER, 15000, (SegmentRef("s-unlabeled", 1),)
        ),
    ]
    summary = match_opportunities(cards, labels, settle_ms=100)
    assert summary["useful_count"] == 1
    assert summary["repetition_count"] == 1
    assert summary["late_count"] == 1
    assert summary["late_by_closed_by"] == {"answered": 1}
    assert summary["premature_count"] == 1
    assert summary["distracting_count"] == 1
    assert summary["distracting_by_none_reason"] == {"rhetorical": 1}
    assert summary["unscored_pending_review_count"] == 1
    assert len(summary["review_queue"]) == 1
    assert summary["review_queue"][0]["suggestion_id"] == "c6"
    assert summary["recall"] == 1.0 / 3.0


def test_simulate_replay_conservation_invariants_and_timing_realism(tmp_path: Path) -> None:
    policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    clf = create_classifier("opportunity-baseline-v0")
    brief = _brief(max_cards=2)
    events = [
        _seg(
            "s1",
            "Someone should own the rollback plan.",
            start_ms=1000,
            end_ms=1900,
            received_ms=2000,
            seq=1,
        ),
        _seg("s2", "Yeah.", start_ms=4000, end_ms=4300, received_ms=4400, seq=2),
        _seg(
            "s3",
            "We must ship the billing service.",
            start_ms=30000,
            end_ms=30900,
            received_ms=31000,
            seq=3,
        ),
        _seg(
            "s4",
            "Alice will handle the canary checks.",
            start_ms=32000,
            end_ms=32400,
            received_ms=32450,
            seq=4,
        ),
        _seg(
            "s5",
            "I'll own the billing service.",
            start_ms=31500,
            end_ms=31900,
            received_ms=31950,
            seq=5,
        ),
    ]
    db_path = tmp_path / "replay.db"
    with SQLiteStore(db_path) as store:
        fast = simulate_replay(brief, events, policy, clf, store=store, run_id="r-fast")
        inv = check_conservation_invariants(store, brief.session_id, "r-fast")
        assert inv.passed, f"Conservation violations: {inv.violations}"
        assert fast.dropped_count == 1  # "Yeah." prefilter drop
        assert len(fast.displayed_cards) >= 1
        assert fast.withdrawn_count >= 1  # s3 withdrawn due to s5 resolution cue

    # Timing realism: huge simulated latency causes display_by_ms expiry -> zero displayed cards
    slow = simulate_replay(
        brief,
        events,
        policy,
        clf,
        simulated_latency_ms=15000.0,
        run_id="r-slow",
    )
    assert len(slow.displayed_cards) == 0
    assert slow.expired_count >= 1


def test_sweep_t_display_and_paired_conversation_bootstrap() -> None:
    policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    clf = create_classifier("opportunity-baseline-v0")

    sessions = []
    labels = []
    for idx in range(3):
        sid = f"sess-sw-{idx}"
        br = _brief(session_id=sid, max_cards=2)
        evs = [
            _seg(
                f"s-{idx}-1",
                "Someone should own the rollback plan.",
                session_id=sid,
                start_ms=1000,
                end_ms=1900,
                received_ms=2000,
                seq=1,
            ),
            _seg(
                f"s-{idx}-2",
                "We must ship the billing service.",
                session_id=sid,
                start_ms=30000,
                end_ms=30900,
                received_ms=31000,
                seq=2,
            ),
        ]
        sessions.append((br, evs))
        labels.append(
            _lbl(
                f"lbl-{idx}-1",
                session_id=sid,
                seg_id=f"s-{idx}-1",
                worthwhile=True,
                categories=(Category.CONFIRM_OWNER,),
                open_ms=2000,
                close_ms=10000,
            )
        )
        labels.append(
            _lbl(
                f"lbl-{idx}-2",
                session_id=sid,
                seg_id=f"s-{idx}-2",
                worthwhile=False,
                categories=(),
                open_ms=31000,
                close_ms=40000,
                none_reason=NoneReason.NOT_MATERIAL,
            )
        )

    curve = sweep_t_display(sessions, labels, policy, clf, [0.40, 0.75, 0.95])
    assert [pt["t_display"] for pt in curve] == [0.40, 0.75, 0.95]
    assert curve[0]["useful_per_10_min"] >= curve[-1]["useful_per_10_min"]
    assert all(pt["cards_per_10_min"] <= 3.01 for pt in curve)

    # Paired conversation bootstrap with low-N flagging (N=3 positives < min_positives_for_ci=10)
    cards_a = [
        DisplayedCardEvent(
            "a0", "ka0", "sess-sw-0", Category.CONFIRM_OWNER, 3500, (SegmentRef("s-0-1", 1),)
        ),
    ]
    cards_b = [
        DisplayedCardEvent(
            "b0", "kb0", "sess-sw-0", Category.CONFIRM_OWNER, 3500, (SegmentRef("s-0-1", 1),)
        ),
        DisplayedCardEvent(
            "b1", "kb1", "sess-sw-1", Category.CONFIRM_OWNER, 3500, (SegmentRef("s-1-1", 1),)
        ),
        DisplayedCardEvent(
            "b2", "kb2", "sess-sw-2", Category.CONFIRM_OWNER, 3500, (SegmentRef("s-2-1", 1),)
        ),
    ]
    boot_low = paired_conversation_bootstrap(
        cards_a, cards_b, labels, n_resamples=200, min_positives_for_ci=10, seed=1
    )
    assert boot_low["mean_delta"] > 0.0
    assert boot_low["low_n"] is True and boot_low["insufficient_n"] is True
    assert boot_low["per_category"]["CONFIRM_OWNER"]["low_n"] is True

    boot_ci = paired_conversation_bootstrap(
        cards_a, cards_b, labels, n_resamples=200, min_positives_for_ci=2, seed=1
    )
    assert boot_ci["low_n"] is False
    assert boot_ci["ci_low"] is not None and boot_ci["ci_high"] is not None
    assert boot_ci["ci_low"] <= boot_ci["mean_delta"] <= boot_ci["ci_high"]


def test_run_latency_benchmark_single_context_and_jsonl_buckets(tmp_path: Path) -> None:
    policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    clf = create_classifier("embed_logreg")
    brief = _brief()
    state = CanonicalState(brief)
    state.apply(
        _seg(
            "t1",
            "We discussed the database cluster failover details across three zones.",
            start_ms=1000,
            seq=1,
        )
    )
    state.apply(_seg("t2", "Someone should own the rollback plan.", start_ms=5000, seq=2))
    snap = state.snapshot()
    built = build_decision_contexts(snap, [snap.segments_by_id["t2"]], policy=policy)
    c_in = ClassifierInput.from_build_result(built[0])

    # Single-context batch_size=1 across all 128..2048 token buckets without NaNs
    single_report = run_latency_benchmark(
        [c_in],
        clf,
        token_buckets=(128, 256, 512, 1024, 2048),
        batch_sizes=(1, 4, 8),
        warmup_calls=1,
        timed_calls=3,
        contention="idle",
        snapshot=snap,
        policy=policy,
    )
    assert len(single_report["cells"]) == 15
    for cell in single_report["cells"]:
        assert cell["warmup_discarded"] == 1
        assert cell["timed_calls"] == 3
        assert cell["contention"] == "idle"
        assert cell["peak_rss_mb"] > 0.0
        for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
            assert math.isfinite(cell[key]) and cell[key] >= 0.0
        assert cell["p50_ms"] <= cell["p95_ms"] <= cell["p99_ms"] <= cell["max_ms"]

    # JSONL context loading
    jsonl_path = tmp_path / "contexts.jsonl"
    jsonl_path.write_text(
        json.dumps({"context": c_in.context.to_dict(), "serialized_text": c_in.serialized_text})
        + "\n",
        encoding="utf-8",
    )
    jsonl_report = run_latency_benchmark(
        jsonl_path,
        clf,
        token_buckets=(128, 512),
        batch_sizes=(1,),
        warmup_calls=1,
        timed_calls=2,
    )
    assert len(jsonl_report["cells"]) == 2


def test_simulate_replay_generator_route_conservation_and_speaker_update_withdrawal(
    tmp_path: Path,
) -> None:
    policy_txt = (
        POLICY_PATH.read_text(encoding="utf-8")
        .replace("enabled: false", "enabled: true")
        .replace("route: template", "route: generator", 1)
        .replace("template: confirm_owner_v1", "template: null", 1)
    )
    gen_policy_path = tmp_path / "gen_no_tpl.yaml"
    gen_policy_path.write_text(policy_txt, encoding="utf-8")
    gen_policy = load_policy(gen_policy_path, calibration_metadata=default_calibration_metadata())
    clf = create_classifier("opportunity-baseline-v0")
    brief = _brief(session_id="sess-gen", max_cards=2)
    events = [
        _seg(
            "s1",
            "Someone should own the rollback plan.",
            session_id="sess-gen",
            start_ms=1000,
            end_ms=1900,
            received_ms=2000,
            seq=1,
        )
    ]

    # 1. provider=None on Route.GENERATOR + template=None -> clean ABSTAIN(mode_mismatch), Inv 5 passes
    db_no_prov = tmp_path / "no_prov.db"
    with SQLiteStore(db_no_prov) as store:
        res_no_prov = simulate_replay(
            brief, events, gen_policy, clf, store=store, run_id="r-no-prov"
        )
        inv = check_conservation_invariants(store, brief.session_id, "r-no-prov")
        assert inv.passed, f"Conservation violations: {inv.violations}"
        assert len(res_no_prov.displayed_cards) == 0
        decs = store.get_decisions(brief.session_id, "r-no-prov")
        assert len(decs) == 1
        assert decs[0]["outcome"] == "ABSTAIN"
        assert decs[0]["terminal_reason"] == "mode_mismatch"

    # 2. Mock ProviderAdapter on Route.GENERATOR + template=None -> generates card, records egress, Inv 5 passes
    def _gen_handler(_request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "question": "Who will own the rollback plan?",
                "evidence_segment_ids": ["s1"],
                "decline_reason": None,
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 90, "completion_tokens": 14},
            },
        )

    provider = ProviderAdapter(
        config=ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8080/v1",
            model_id="qwen3:14b",
            use_streaming=False,
        ),
        transport=httpx.MockTransport(_gen_handler),
        validate_on_init=False,
    )
    db_with_prov = tmp_path / "with_prov.db"
    with SQLiteStore(db_with_prov) as store:
        res_prov = simulate_replay(
            brief, events, gen_policy, clf, store=store, run_id="r-prov", provider=provider
        )
        inv_prov = check_conservation_invariants(store, brief.session_id, "r-prov")
        assert inv_prov.passed, f"Conservation violations: {inv_prov.violations}"
        assert len(res_prov.displayed_cards) == 1
        assert res_prov.displayed_cards[0].route is Route.GENERATOR
        assert res_prov.displayed_cards[0].text == "Who will own the rollback plan?"
        eg_rows = store.get_egress_records(brief.session_id, "r-prov")
        assert len(eg_rows) == 1

    # 3. 7-step revalidation withdraws card when SPEAKER_UPDATE flips S1 -> ME inside settle window
    tpl_policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    brief_spk = _brief(session_id="sess-spk-upd", max_cards=2)
    spk_events = [
        _seg(
            "s1",
            "Someone should own the rollback plan.",
            session_id="sess-spk-upd",
            speaker_id="spk-1",
            start_ms=1000,
            end_ms=1900,
            received_ms=2000,
            seq=1,
        ),
        CanonicalEvent(
            schema_version=1,
            session_id="sess-spk-upd",
            event_id="ev-upd-me",
            kind=EventKind.SPEAKER_UPDATE,
            segment_id="s1",
            source_revision=1,
            speaker_id="me-1",
            speaker_uncertain=False,
            received_ms=2500,
            source_sequence=2,
        ),
    ]
    db_spk = tmp_path / "spk_upd.db"
    with SQLiteStore(db_spk) as store:
        res_spk = simulate_replay(
            brief_spk, spk_events, tpl_policy, clf, store=store, run_id="r-spk"
        )
        inv_spk = check_conservation_invariants(store, brief_spk.session_id, "r-spk")
        assert inv_spk.passed, f"Conservation violations: {inv_spk.violations}"
        assert len(res_spk.displayed_cards) == 0
        assert res_spk.withdrawn_count == 1
        decs_spk = store.get_decisions(brief_spk.session_id, "r-spk")
        assert decs_spk[0]["terminal_state"] == "withdrawn"
        assert decs_spk[0]["terminal_reason"] == "evidence_revision_changed"


def test_simulate_replay_drains_segment_ingested_during_classify_latency(tmp_path: Path) -> None:
    base_policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    policy = replace(
        base_policy,
        display=replace(
            base_policy.display,
            settle_ms=0,
            cooldown_ms=1002,
            max_candidate_age_ms=1000,
            max_cards_per_10_min=5,
        ),
    )
    clf = create_classifier("opportunity-baseline-v0")
    brief = _brief(session_id="sess-f2-lat", max_cards=5)
    events = [
        _seg(
            "seg-f2-1",
            "Someone should own the first task.",
            session_id="sess-f2-lat",
            start_ms=100,
            end_ms=180,
            received_ms=200,
            seq=1,
            boundary="utterance_end",
        ),
        _seg(
            "seg-f2-2",
            "Someone should own the second deadline task.",
            session_id="sess-f2-lat",
            start_ms=110,
            end_ms=190,
            received_ms=201,
            seq=2,
            boundary="utterance_end",
        ),
        CanonicalEvent(
            schema_version=1,
            session_id="sess-f2-lat",
            event_id="ev-f2-end",
            kind=EventKind.CONTROL,
            subtype=ControlSubtype.SESSION_END,
            received_ms=3000,
            source_sequence=3,
        ),
    ]
    for lat_ms in (0.0, 1.0, 5.0):
        run_id = f"r-lat-{int(lat_ms)}"
        with SQLiteStore(tmp_path / f"lat_{int(lat_ms)}.db") as store:
            res = simulate_replay(
                brief,
                events,
                policy,
                clf,
                store=store,
                run_id=run_id,
                simulated_latency_ms=lat_ms,
            )
            inv = check_conservation_invariants(store, brief.session_id, run_id)
            assert inv.passed, f"Conservation violations at {lat_ms}ms: {inv.violations}"
            assert len(res.displayed_cards) == 1
            sugs = store.get_suggestions(brief.session_id, run_id)
            second = [s for s in sugs if "seg-f2-2" in str(s.get("evidence_refs_json"))]
            assert len(second) == 1
            assert second[0]["current_state"] == "expired"
            assert second[0]["terminal_reason"] == "display_by_passed"


def test_simulate_replay_evaluates_trailing_segment_ingested_during_final_classify(
    tmp_path: Path,
) -> None:
    policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    clf = create_classifier("opportunity-baseline-v0")
    brief = _brief(session_id="sess-tail-ingest", max_cards=2)
    events = [
        _seg(
            "s-first",
            "We reviewed the metrics from last Tuesday afternoon.",
            session_id="sess-tail-ingest",
            start_ms=100,
            end_ms=180,
            received_ms=200,
            seq=1,
            boundary="utterance_end",
        ),
        _seg(
            "s-last",
            "Someone should own the rollback plan.",
            session_id="sess-tail-ingest",
            start_ms=190,
            end_ms=200,
            received_ms=201,
            seq=2,
            boundary="utterance_end",
        ),
    ]
    with SQLiteStore(tmp_path / "tail_ingest.db") as store:
        res = simulate_replay(
            brief,
            events,
            policy,
            clf,
            store=store,
            run_id="r-tail-ingest",
            simulated_latency_ms=1.0,
        )
        inv = check_conservation_invariants(store, brief.session_id, "r-tail-ingest")
        assert inv.passed, f"Conservation violations: {inv.violations}"
        assert res.decisions_count == 2
        assert len(res.displayed_cards) == 1
        assert res.displayed_cards[0].evidence_refs[0].segment_id == "s-last"


def test_simulate_replay_wakes_for_open_tail_anchor_display_by_before_far_future_event(
    tmp_path: Path,
) -> None:
    base_policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    policy = replace(
        base_policy,
        display=replace(base_policy.display, max_candidate_age_ms=1000),
    )
    clf = create_classifier("opportunity-baseline-v0")
    brief = _brief(session_id="sess-open-tail", max_cards=2)
    events = [
        _seg(
            "s-open",
            "Someone should own the rollback plan without punctuation",
            session_id="sess-open-tail",
            start_ms=100,
            end_ms=180,
            received_ms=200,
            seq=1,
            boundary=None,
        ),
        CanonicalEvent(
            schema_version=1,
            session_id="sess-open-tail",
            event_id="ev-open-end",
            kind=EventKind.CONTROL,
            subtype=ControlSubtype.SESSION_END,
            received_ms=3000,
            source_sequence=2,
        ),
    ]
    with SQLiteStore(tmp_path / "open_tail.db") as store:
        res = simulate_replay(brief, events, policy, clf, store=store, run_id="r-open")
        inv = check_conservation_invariants(store, brief.session_id, "r-open")
        assert inv.passed, f"Conservation violations: {inv.violations}"
        assert res.decisions_count == 1
        decs = store.get_decisions(brief.session_id, "r-open")
        assert len(decs) == 1
        assert decs[0]["terminal_state"] == "retired"
        assert decs[0]["terminal_reason"] == "expired_unevaluated"
        assert decs[0]["based_on_state_version"] == 1


def test_simulate_dataset_replay_caches_revisions_and_invariants(tmp_path: Path) -> None:
    base_policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    policy = replace(
        base_policy,
        display=replace(base_policy.display, settle_ms=100, cooldown_ms=100),
    )
    clf = create_classifier("rules_v0")
    brief = _brief(session_id="sess-ds-base", max_cards=3)
    state = CanonicalState(_brief(session_id="sess-ds-1", max_cards=3))
    state.apply(
        _seg(
            "s1",
            "Someone should own the rollback plan.",
            session_id="sess-ds-1",
            start_ms=100,
            end_ms=400,
            received_ms=500,
            seq=1,
            boundary="utterance_end",
        )
    )
    state.apply(
        _seg(
            "s1",
            "Someone should own the DNS cutover plan.",
            session_id="sess-ds-1",
            start_ms=100,
            end_ms=400,
            received_ms=600,
            seq=2,
            rev=2,
            boundary="utterance_end",
        )
    )
    built = build_decision_contexts(state.snapshot(), policy=policy)
    assert built[0].context is not None and built[0].serialized_text is not None
    flat_row = {
        **built[0].context.to_dict(),
        "serialized_text": built[0].serialized_text,
    }
    nested_row = {
        "context": {
            **built[0].context.to_dict(),
            "session_id": "sess-ds-2",
            "candidate_id": "c-ds-2",
        },
        "serialized_text": built[0].serialized_text,
    }
    with SQLiteStore(tmp_path / "ds_replay.db") as store:
        results = simulate_dataset_replay(
            [flat_row, nested_row],
            policy,
            clf,
            brief=brief,
            store=store,
            run_id="r-ds-1",
        )
        assert len(results) == 2
        assert all(len(r.displayed_cards) == 1 for r in results)
        assert (
            store.get_cached_model_output(
                built[0].context.serialized_sha256, clf.classifier_version
            )
            is not None
        )
        assert check_conservation_invariants(store, "sess-ds-1", "r-ds-1").passed
        assert check_conservation_invariants(store, "sess-ds-2", "r-ds-1").passed


def test_simulate_dataset_replay_2party_context_refs_preserves_invariants_and_grounding(
    tmp_path: Path,
) -> None:
    base_policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    policy = replace(
        base_policy,
        display=replace(base_policy.display, settle_ms=100, cooldown_ms=100),
    )
    clf = create_classifier("rules_v0")
    sid = "sess-ds-2party"
    brief = _brief(session_id=sid, max_cards=3)
    state = CanonicalState(brief)
    state.apply(
        _seg(
            "me1",
            "Let us finalize the cutover checklist today.",
            session_id=sid,
            speaker_id="ME",
            start_ms=100,
            end_ms=190,
            received_ms=200,
            seq=1,
            boundary="utterance_end",
        )
    )
    state.apply(
        _seg(
            "s_short",
            "Okay.",
            session_id=sid,
            speaker_id="S1",
            start_ms=300,
            end_ms=390,
            received_ms=400,
            seq=2,
            boundary="utterance_end",
        )
    )
    state.apply(
        _seg(
            "s_ctx_only",
            "Alice already verified staging pool A.",
            session_id=sid,
            speaker_id="S1",
            start_ms=500,
            end_ms=590,
            received_ms=600,
            seq=3,
            boundary="utterance_end",
        )
    )
    state.apply(
        _seg(
            "s_main",
            "Someone should own the rollback plan.",
            session_id=sid,
            speaker_id="S1",
            start_ms=700,
            end_ms=790,
            received_ms=800,
            seq=4,
            boundary="utterance_end",
        )
    )
    built = build_decision_contexts(state.snapshot(), policy=policy)
    kept = [
        b
        for b in built
        if not b.dropped
        and b.context is not None
        and b.serialized_text is not None
        and any(r.segment_id == "s_main" for r in b.context.anchor_refs)
    ]
    assert len(kept) == 1
    ctx_main = kept[0].context
    assert ctx_main is not None and kept[0].serialized_text is not None
    ctx_seg_ids = {r.segment_id for r in ctx_main.context_refs}
    assert {"me1", "s_short", "s_ctx_only"}.issubset(ctx_seg_ids)

    row = {"context": ctx_main.to_dict(), "serialized_text": kept[0].serialized_text}
    db_path = tmp_path / "ds_2party.db"
    with SQLiteStore(db_path) as store:
        res = simulate_dataset_replay(
            [row],
            policy,
            clf,
            brief=brief,
            store=store,
            run_id="r-2party",
        )
        assert len(res) == 1 and len(res[0].displayed_cards) == 1
        inv = check_conservation_invariants(store, sid, "r-2party")
        assert inv.passed, inv.violations
        assert build_session_report(store, sid, "r-2party").invariants.passed is True

    assert (
        cli_main(
            ["report", "--db", str(db_path), "--session", sid, "--run-id", "r-2party", "--json"]
        )
        == 0
    )

    # Route.GENERATOR + [CONTEXT] Check 4 grounding ("Alice" only appears in s_ctx_only) + Inv 5
    gen_policy_txt = (
        POLICY_PATH.read_text(encoding="utf-8")
        .replace("enabled: false", "enabled: true")
        .replace("route: template", "route: generator", 1)
        .replace("template: confirm_owner_v1", "template: null", 1)
    )
    gen_policy_path = tmp_path / "gen_2party.yaml"
    gen_policy_path.write_text(gen_policy_txt, encoding="utf-8")
    gen_policy = replace(
        load_policy(gen_policy_path, calibration_metadata=default_calibration_metadata()),
        display=replace(base_policy.display, settle_ms=100, cooldown_ms=100),
    )

    def _gen_handler(_request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "question": "Will Alice own the rollback plan?",
                "evidence_segment_ids": ["s_main"],
                "decline_reason": None,
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 90, "completion_tokens": 12},
            },
        )

    provider = ProviderAdapter(
        config=ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8080/v1",
            model_id="qwen3:14b",
            use_streaming=False,
        ),
        transport=httpx.MockTransport(_gen_handler),
        validate_on_init=False,
    )
    with SQLiteStore(tmp_path / "ds_2party_gen.db") as store_gen:
        res_gen = simulate_dataset_replay(
            [row],
            gen_policy,
            clf,
            brief=brief,
            store=store_gen,
            run_id="r-2party-gen",
            provider=provider,
        )
        assert len(res_gen) == 1 and len(res_gen[0].displayed_cards) == 1
        assert res_gen[0].displayed_cards[0].route is Route.GENERATOR
        assert res_gen[0].displayed_cards[0].text == "Will Alice own the rollback plan?"
        assert len(store_gen.get_egress_records(sid, "r-2party-gen")) == 1
        inv_gen = check_conservation_invariants(store_gen, sid, "r-2party-gen")
        assert inv_gen.passed, inv_gen.violations

        # Negative control: duplicate s_main@1 anchor across two rows triggers Inv 1 violation
        dup_row = {
            "context": {**ctx_main.to_dict(), "candidate_id": "cand-dup-neg"},
            "serialized_text": kept[0].serialized_text,
        }
        simulate_dataset_replay(
            [row, dup_row],
            policy,
            clf,
            brief=brief,
            store=store_gen,
            run_id="r-2party-neg",
        )
        inv_neg = check_conservation_invariants(store_gen, sid, "r-2party-neg")
        assert inv_neg.passed is False
        assert any(v.invariant_id == 1 for v in inv_neg.violations)

        # Breaker open syncs route_health / health_timeline to template_only
        fail_provider = ProviderAdapter(
            config=ProviderConfig(
                provider="local",
                base_url="http://127.0.0.1:8080/v1",
                model_id="qwen3:14b",
                use_streaming=False,
                breaker_consecutive_failures=1,
            ),
            transport=httpx.MockTransport(lambda _req: httpx.Response(503, text="down")),
            validate_on_init=False,
        )
        with SQLiteStore(tmp_path / "ds_2party_brk.db") as store_brk:
            simulate_dataset_replay(
                [row],
                gen_policy,
                clf,
                brief=brief,
                store=store_brk,
                run_id="r-2party-brk",
                provider=fail_provider,
            )
            sess_brk = store_brk.get_session(sid, "r-2party-brk")
            assert sess_brk is not None
            tl = json.loads(sess_brk["health_timeline_json"])
            assert [e["state"] for e in tl] == ["ok", "template_only"]


def test_splitters_support_nested_context_rows_and_objects() -> None:
    policy = load_policy(POLICY_PATH, calibration_metadata=default_calibration_metadata())
    nested_dict_rows: list[dict[str, object]] = []
    ctx_results = []
    clf_inputs = []
    for i in range(6):
        sid = f"conv-nested-{i}"
        state = CanonicalState(_brief(sid))
        state.apply(
            _seg(
                f"s-{i}",
                "Someone should own the rollback plan.",
                session_id=sid,
                speaker_id="spk-1",
                start_ms=100,
                end_ms=900,
                received_ms=950,
                seq=1,
                boundary="utterance_end",
            )
        )
        built = [
            b for b in build_decision_contexts(state.snapshot(), policy=policy) if not b.dropped
        ]
        assert len(built) == 1
        res = built[0]
        assert res.context is not None and res.serialized_text is not None
        ctx_results.append(res)
        clf_inputs.append(ClassifierInput.from_build_result(res))
        ctx_map = res.context.to_dict()
        row_dict: dict[str, object] = {
            "context": ctx_map,
            "serialized_text": res.serialized_text,
        }
        if i % 2 == 0:
            row_dict["series_id"] = f"ser-{i % 2}"
        else:
            ctx_map["series_id"] = f"ser-{i % 2}"
        nested_dict_rows.append(row_dict)

    conv_s = split_by_conversation(nested_dict_rows, seed=3)
    assert sum(conv_s.to_dict()["counts"].values()) == 6
    assert (
        len(
            set(conv_s.train_conversations)
            | set(conv_s.cal_conversations)
            | set(conv_s.eval_conversations)
        )
        == 6
    )
    assert not (set(conv_s.train_conversations) & set(conv_s.eval_conversations))

    ser_s = split_by_series(nested_dict_rows, seed=3)
    assert set(ser_s.train_series) | set(ser_s.cal_series) | set(ser_s.eval_series) == {
        "ser-0",
        "ser-1",
    }
    assert not (set(ser_s.train_series) & set(ser_s.eval_series))

    res_s = split_by_conversation(ctx_results, seed=5)
    assert sum(res_s.to_dict()["counts"].values()) == 6
    cin_s = split_by_series(clf_inputs, seed=5)
    assert sum(cin_s.to_dict()["counts"].values()) == 6

    with pytest.raises(ValueError, match="session_id/conversation_id"):
        split_by_conversation([{"context": {"candidate_id": "c1"}, "serialized_text": "x"}])
    with pytest.raises(ValueError, match="session_id/conversation_id"):
        split_by_conversation([{"context": None, "serialized_text": "x"}])


def test_async_sweep_dataset_t_display_isolates_provider_breaker_state(tmp_path: Path) -> None:
    gen_policy_txt = (
        POLICY_PATH.read_text(encoding="utf-8")
        .replace("enabled: false", "enabled: true")
        .replace("route: template", "route: generator", 1)
        .replace("template: confirm_owner_v1", "template: null", 1)
    )
    gen_policy_path = tmp_path / "gen_sweep_isolation.yaml"
    gen_policy_path.write_text(gen_policy_txt, encoding="utf-8")
    base_policy = load_policy(gen_policy_path, calibration_metadata=default_calibration_metadata())
    gen_policy = replace(
        base_policy,
        display=replace(base_policy.display, settle_ms=100, cooldown_ms=100),
    )
    clf = create_classifier("rules_v0")

    dataset_rows: list[dict[str, object]] = []
    for idx in range(3):
        sid = f"sess-sweep-iso-{idx}"
        state = CanonicalState(_brief(sid, max_cards=3))
        state.apply(
            _seg(
                f"s-{idx}",
                "Someone should own the rollback plan.",
                session_id=sid,
                speaker_id="S1",
                start_ms=100 + idx * 1000,
                end_ms=400 + idx * 1000,
                received_ms=500 + idx * 1000,
                seq=1,
                boundary="utterance_end",
            )
        )
        built = [
            b for b in build_decision_contexts(state.snapshot(), policy=gen_policy) if not b.dropped
        ]
        assert len(built) == 1
        assert built[0].context is not None and built[0].serialized_text is not None
        dataset_rows.append(
            {
                "context": built[0].context.to_dict(),
                "serialized_text": built[0].serialized_text,
            }
        )

    http_calls = 0

    def _failing_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        return httpx.Response(500, json={"error": "upstream failure"})

    transport = httpx.MockTransport(_failing_handler)

    async def _run() -> None:
        nonlocal http_calls
        async with httpx.AsyncClient(transport=transport) as shared_client:
            provider = ProviderAdapter(
                config=ProviderConfig(
                    provider="local",
                    base_url="http://127.0.0.1:8080/v1",
                    model_id="qwen3:14b",
                    use_streaming=False,
                    breaker_consecutive_failures=2,
                ),
                transport=transport,
                client=shared_client,
                validate_on_init=False,
            )

            curve = await async_sweep_dataset_t_display(
                dataset_rows,
                [],
                gen_policy,
                clf,
                [0.3, 0.6],
                provider=provider,
            )
            assert [pt["t_display"] for pt in curve] == [0.3, 0.6]
            # 2 failing calls per sweep iteration (breaker_consecutive_failures=2 trips on row 2)
            assert http_calls == 4
            assert provider.breaker_state is BreakerState.CLOSED
            assert provider.health_state is HealthState.OK
            assert not shared_client.is_closed

            # Even when caller's provider is already OPEN / TEMPLATE_ONLY from a prior replay,
            # async_sweep_dataset_t_display runs isolated and makes fresh HTTP calls.
            open_provider = ProviderAdapter(
                config=provider.config,
                transport=transport,
                client=shared_client,
                validate_on_init=False,
            )
            http_calls = 0
            await async_simulate_dataset_replay(
                dataset_rows,
                gen_policy,
                clf,
                provider=open_provider,
            )
            assert http_calls == 2
            assert open_provider.breaker_state is BreakerState.OPEN
            assert open_provider.health_state is HealthState.TEMPLATE_ONLY

            http_calls = 0
            curve_after_open = await async_sweep_dataset_t_display(
                dataset_rows,
                [],
                gen_policy,
                clf,
                [0.3, 0.6],
                provider=open_provider,
            )
            assert [pt["t_display"] for pt in curve_after_open] == [0.3, 0.6]
            assert http_calls == 4
            assert open_provider.breaker_state is BreakerState.OPEN
            assert open_provider.health_state is HealthState.TEMPLATE_ONLY
            assert not shared_client.is_closed
            await open_provider.cancel_recovery()

    asyncio.run(_run())
