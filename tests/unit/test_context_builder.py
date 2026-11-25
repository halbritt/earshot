"""Unit tests for context_builder, policies, templates, and §15 M3 boundaries."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from coach.canonical_state import CanonicalState, StateSnapshot
from coach.context_builder import (
    BUILDER_VERSION,
    PREFILTER_VERSION,
    PolicyConfigError,
    WhitespaceTokenizer,
    bucket_age_ms,
    build_decision_context,
    build_decision_contexts,
    check_generic_referent,
    coalesce_anchors,
    compute_candidate_id,
    compute_policy_sha256,
    effective_max_cards_per_10_min,
    extract_slots_span_rules_v0,
    load_policy,
    normalize_for_matching,
    run_prefilter,
    verify_grounded_slot,
)
from coach.contracts import (
    AttentionConfig,
    CanonicalEvent,
    Category,
    ControlSubtype,
    EventKind,
    GroundedSlot,
    KnownFact,
    LedgerState,
    Origin,
    Participant,
    ReasonCode,
    RecipientConfig,
    Route,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    SuggestionCandidate,
)
from coach.template_renderer import (
    load_templates_dir,
    render_generic_fallback,
    render_template,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COACH_DIR = REPO_ROOT / "coach"
POLICY_PATH = REPO_ROOT / "policies" / "policy-v0-a.yaml"
TEMPLATES_DIR = REPO_ROOT / "policies" / "templates"


def _coach_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "coach" or alias.name.startswith("coach."):
                    imported.add(alias.name.removeprefix("coach.").split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                if mod:
                    imported.add(mod.split(".")[0])
                else:
                    for alias in node.names:
                        imported.add(alias.name)
            elif mod == "coach":
                for alias in node.names:
                    imported.add(alias.name)
            elif mod.startswith("coach."):
                imported.add(mod.removeprefix("coach.").split(".")[0])
    return imported


def _make_brief(
    *,
    known_facts: tuple[KnownFact, ...] = (),
    do_not_suggest: tuple[str, ...] = (),
    max_cards: int = 3,
    participants: tuple[Participant, ...] = (),
) -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id="sess-m3",
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="me-1", role="reviewer"),
        objective="Leave with a rollback plan that has an owner and a test date.",
        participants=participants
        or (
            Participant(participant_id="spk-1", label="S1", role="service owner"),
            Participant(participant_id="spk-2", label="S2", role=None),
        ),
        agenda=("Rollback plan",),
        known_facts=known_facts,
        do_not_suggest=do_not_suggest,
        attention=AttentionConfig(max_cards_per_10_min=max_cards, proactive=True),
    )


def _seg_event(
    seg_id: str,
    *,
    speaker_id: str = "spk-1",
    text: str,
    start_ms: int,
    end_ms: int | None = None,
    received_ms: int | None = None,
    seq: int = 1,
    channel: int = 1,
    rev: int = 1,
    stability: Stability = Stability.COMMITTED,
    boundary: str | None = None,
    overlap: bool = False,
    speaker_uncertain: bool = False,
    asr_confidence: float | None = 0.98,
    asr_low_confidence: bool = False,
    low_quality_gate: bool = False,
) -> CanonicalEvent:
    end = end_ms if end_ms is not None else start_ms + 1000
    rx = received_ms if received_ms is not None else end + 50
    return CanonicalEvent(
        schema_version=1,
        session_id="sess-m3",
        event_id=f"ev-{seg_id}-{rev}-{seq}",
        kind=EventKind.SEGMENT,
        segment_id=seg_id,
        source_revision=rev,
        channel=channel,
        speaker_id=speaker_id,
        speaker_confidence=0.95,
        start_ms=start_ms,
        end_ms=end,
        received_ms=rx,
        text=text,
        stability=stability,
        source_sequence=seq,
        boundary=boundary,
        overlap=overlap,
        speaker_uncertain=speaker_uncertain,
        asr_confidence=asr_confidence,
        asr_low_confidence=asr_low_confidence,
        low_quality_gate=low_quality_gate,
    )


def test_m3_module_boundaries_upward_only() -> None:
    assert _coach_imports(COACH_DIR / "context_builder.py") <= {"contracts", "canonical_state"}
    assert _coach_imports(COACH_DIR / "template_renderer.py") <= {
        "contracts",
        "canonical_state",
        "context_builder",
    }
    assert _coach_imports(COACH_DIR / "provider_adapter.py") <= {"contracts", "storage"}
    assert _coach_imports(COACH_DIR / "candidate_validator.py") <= {
        "contracts",
        "canonical_state",
        "context_builder",
    }
    assert _coach_imports(COACH_DIR / "question_generator.py") <= {
        "contracts",
        "canonical_state",
        "storage",
        "context_builder",
        "template_renderer",
        "provider_adapter",
        "candidate_validator",
    }


def test_load_policy_v0_a_and_calibration_check(tmp_path: Path) -> None:
    policy = load_policy(
        POLICY_PATH,
        calibration_metadata={
            "calibration_version": "cal-v0-technical",
            "classifier_version": "opportunity-baseline-v0",
            "mode": "technical_working_discussion",
        },
    )
    assert policy.policy_version == "policy-v0-a"
    assert policy.mode is SessionMode.TECHNICAL_WORKING_DISCUSSION
    assert policy.classifier.input_limit_tokens == 512
    assert policy.display.max_cards_per_10_min == 3
    assert policy.generator.enabled is False
    assert policy.policy_sha256 == compute_policy_sha256(POLICY_PATH.read_bytes())
    assert set(policy.categories) == {
        Category.CONFIRM_OWNER,
        Category.CONFIRM_TIME,
        Category.SUCCESS_CRITERION,
    }

    with pytest.raises(PolicyConfigError):
        load_policy(
            POLICY_PATH,
            calibration_metadata={
                "calibration_version": "cal-v0-technical",
                "classifier_version": "different-classifier",
                "mode": "technical_working_discussion",
            },
        )
    with pytest.raises(PolicyConfigError):
        load_policy(
            POLICY_PATH,
            calibration_metadata={
                "calibration_version": "cal-v0-technical",
                "classifier_version": "opportunity-baseline-v0",
                "mode": "personal_logistics",
            },
        )

    modified = tmp_path / "policy-copy.yaml"
    modified.write_text(
        POLICY_PATH.read_text(encoding="utf-8").replace("settle_ms: 1500", "settle_ms: 1600"),
        encoding="utf-8",
    )
    assert load_policy(modified).policy_sha256 != policy.policy_sha256
    assert effective_max_cards_per_10_min(_make_brief(max_cards=2), policy) == 2
    assert effective_max_cards_per_10_min(_make_brief(max_cards=5), policy) == 3


def test_anchor_coalescing_coverage_punctuation_boundary_and_b19_overlap() -> None:
    state = CanonicalState(_make_brief())
    # Two punctuation-free segments from spk-1 coalesce, then boundary closes anchor 1.
    state.apply(
        _seg_event("s1", speaker_id="spk-1", text="We should migrate", start_ms=1000, seq=1)
    )
    state.apply(
        _seg_event(
            "s2",
            speaker_id="spk-1",
            text="the billing service soon",
            start_ms=2100,
            seq=2,
            boundary="utterance_end",
        )
    )
    # Recipient turn never becomes an anchor.
    state.apply(_seg_event("s3", speaker_id="me-1", text="When is cutover?", start_ms=3500, seq=3))
    # B19 overlapping cross-channel segments from spk-1 and spk-2 never merge.
    state.apply(
        _seg_event(
            "s4",
            speaker_id="spk-1",
            text="Maybe next Tuesday afternoon",
            start_ms=5000,
            end_ms=7000,
            channel=1,
            seq=4,
        )
    )
    state.apply(
        _seg_event(
            "s5",
            speaker_id="spk-2",
            text="Someone should own the rollback plan.",
            start_ms=5500,
            end_ms=7200,
            channel=2,
            seq=5,
        )
    )
    # Interim segment must be excluded.
    state.apply(
        _seg_event(
            "s6",
            speaker_id="spk-2",
            text="Draft interim partial",
            start_ms=8000,
            stability=Stability.INTERIM,
            seq=6,
        )
    )

    snap: StateSnapshot = state.snapshot()
    anchors = coalesce_anchors(snap)
    assert len(anchors) == 3
    assert [s.segment_id for s in anchors[0].segments] == ["s1", "s2"]
    assert [s.segment_id for s in anchors[1].segments] == ["s4"]
    assert [s.segment_id for s in anchors[2].segments] == ["s5"]
    assert anchors[2].display_by_ms == anchors[2].anchor_received_ms + 8000
    assert anchors[2].candidate_id == compute_candidate_id("sess-m3", [SegmentRef("s5", 1)])
    covered = [s.segment_id for a in anchors for s in a.segments]
    assert covered == ["s1", "s2", "s4", "s5"]
    assert not any(a.joined_text == "Draft interim partial" for a in anchors)


def test_coalesce_60_token_cap_and_pause_gap() -> None:
    state = CanonicalState(_make_brief())
    thirty = " ".join(f"word{i}" for i in range(30))
    thirty_five = " ".join(f"term{i}" for i in range(35))
    state.apply(
        _seg_event("a1", speaker_id="spk-1", text=thirty, start_ms=1000, end_ms=2000, seq=1)
    )
    state.apply(
        _seg_event("a2", speaker_id="spk-1", text=thirty_five, start_ms=2100, end_ms=3000, seq=2)
    )
    state.apply(
        _seg_event(
            "a3", speaker_id="spk-1", text="next clause here", start_ms=4500, end_ms=5000, seq=3
        )
    )
    anchors = coalesce_anchors(state.snapshot(), max_anchor_tokens=60, pause_gap_ms=1000)
    assert [[s.segment_id for s in a.segments] for a in anchors] == [["a1"], ["a2"], ["a3"]]


def test_prefilter_audited_drops_and_b21_stock_phrase() -> None:
    state = CanonicalState(_make_brief())
    state.apply(
        _seg_event("p1", speaker_id="spk-1", text="Thank you for watching.", start_ms=1000, seq=1)
    )
    state.apply(
        _seg_event("p2", speaker_id="spk-1", text="Yeah, sounds good.", start_ms=2500, seq=2)
    )
    state.apply(
        _seg_event(
            "p3", speaker_id="spk-1", text="Someone should own rollback.", start_ms=4000, seq=3
        )
    )
    state.apply(
        _seg_event(
            "p4", speaker_id="spk-1", text="Someone should own rollback.", start_ms=6000, seq=4
        )
    )
    state.apply(_seg_event("p5", speaker_id="spk-1", text="we check it", start_ms=8000, seq=5))

    results = build_decision_contexts(state.snapshot(), policy=load_policy(POLICY_PATH))
    assert [r.dropped for r in results] == [True, True, False, True, True]
    assert results[0].drop_record is not None
    assert results[0].drop_record.reason_code is ReasonCode.STOCK_PHRASE
    assert results[1].drop_record is not None
    assert results[1].drop_record.reason_code is ReasonCode.BACKCHANNEL
    assert results[2].context is not None
    assert results[2].context.builder_version == BUILDER_VERSION
    assert results[2].context.prefilter_version == PREFILTER_VERSION
    assert results[3].drop_record is not None
    assert results[3].drop_record.reason_code is ReasonCode.REPEAT
    assert results[4].drop_record is not None
    assert results[4].drop_record.reason_code is ReasonCode.TOO_SHORT


def test_serialization_layout_gaps_b06_b18_b10_truncation_and_b21_flags() -> None:
    brief = _make_brief(
        known_facts=(KnownFact(key="launch_date", value="March 3"),),
        participants=(
            Participant(participant_id="alice-id", label="Alice", role="service owner"),
            Participant(participant_id="bob-id", label="Bob", role=None),
        ),
    )
    state = CanonicalState(brief)
    state.apply(
        _seg_event(
            "t1",
            speaker_id="alice-id",
            text="Priya will handle the early canary migration setup in staging.",
            start_ms=10_000,
            end_ms=12_000,
            received_ms=12_050,
            seq=1,
        )
    )
    state.apply(
        _seg_event(
            "t2",
            speaker_id="me-1",
            text="What about the old cluster?",
            start_ms=50_000,
            end_ms=51_000,
            received_ms=51_050,
            seq=2,
        )
    )
    # Off-the-record gap (B18) between t2 and t3
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="sess-m3",
            event_id="ev-otr-start",
            kind=EventKind.CONTROL,
            received_ms=60_000,
            source_sequence=10,
            subtype=ControlSubtype.OFF_RECORD_START,
            start_ms=60_000,
            end_ms=60_000,
        )
    )
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="sess-m3",
            event_id="ev-otr-end",
            kind=EventKind.CONTROL,
            received_ms=70_000,
            source_sequence=11,
            subtype=ControlSubtype.OFF_RECORD_END,
            start_ms=60_000,
            end_ms=70_000,
        )
    )
    state.apply(
        _seg_event(
            "t3",
            speaker_id="bob-id",
            text="Someone should own the rollback plan.",
            start_ms=90_000,
            end_ms=92_000,
            received_ms=92_050,
            seq=3,
            asr_low_confidence=True,
        )
    )

    sug = SuggestionCandidate(
        schema_version=1,
        suggestion_id="sug-prev-1",
        candidate_id="cand-prev-1",
        origin=Origin.PROACTIVE,
        category=Category.CONFIRM_TIME,
        route=Route.TEMPLATE,
        text="When does the migration need to finish?",
        word_count=7,
        based_on_state_version=1,
        evidence_refs=(SegmentRef("t1", 1),),
        template_id="confirm_time_v1",
        grounded_slots={},
        ready_ms=30_000,
        display_by_ms=38_000,
    )
    state.register_suggestion(sug)
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="sess-m3",
            event_id="ev-led-1",
            kind=EventKind.LEDGER,
            received_ms=31_000,
            source_sequence=12,
            suggestion_id="sug-prev-1",
            transition=LedgerState.DISPLAYED,
            at_ms=31_000,
            cause=ReasonCode.BOUNDARY_AFTER_SETTLE,
        )
    )
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="sess-m3",
            event_id="ev-led-2",
            kind=EventKind.LEDGER,
            received_ms=35_000,
            source_sequence=13,
            suggestion_id="sug-prev-1",
            transition=LedgerState.DISMISSED,
            at_ms=35_000,
            cause=ReasonCode.USER_ACTION,
        )
    )

    policy = load_policy(POLICY_PATH)
    snap = state.snapshot()
    # Tight limit forces oldest turn t1 (end_ms=12000, 78s before anchor_start 90000 < 300s) to drop -> B10!
    res = build_decision_context(
        snap,
        [snap.segments_by_id["t3"]],
        policy,
        tokenizer=WhitespaceTokenizer(),
        input_limit_tokens=55,
        suggestion_catalog={"sug-prev-1": sug},
    )
    assert res.context is not None and res.serialized_text is not None
    txt = res.serialized_text
    assert txt.index("[OBJECTIVE]") < txt.index("[ROLE]") < txt.index("[KNOWN]")
    assert (
        txt.index("[KNOWN]")
        < txt.index("[SUGGESTED]")
        < txt.index("[CONTEXT]")
        < txt.index("[ANCHOR]")
    )
    assert "Alice:" not in txt and "Bob:" not in txt  # anonymized roles
    assert "Priya" not in txt  # t1 truncated cleanly as whole turn
    assert "[GAP: off_record]" in txt
    assert "<2m CONFIRM_TIME dismissed: When does the migration need to finish?" in txt
    assert res.context.omitted["turns"] >= 1
    assert res.context.flags["has_gap"] is True
    assert res.context.flags["has_off_record_gap"] is True
    assert res.context.flags["context_truncated_material"] is True
    assert res.context.flags["context_truncated_material:CONFIRM_OWNER"] is True
    assert res.context.flags["asr_low_confidence"] is True
    assert res.context.serialized_sha256 == hashlib.sha256(txt.encode("utf-8")).hexdigest()
    assert bucket_age_ms(5_000) == "<30s"
    assert bucket_age_ms(60_000) == "<2m"
    assert bucket_age_ms(300_000) == "<10m"
    assert bucket_age_ms(900_000) == "older"


def test_input_overflow_never_truncates_mandatory_core() -> None:
    state = CanonicalState(_make_brief())
    state.apply(
        _seg_event(
            "ov1",
            speaker_id="spk-1",
            text="Someone should own the entire multi-region database rollback automation plan.",
            start_ms=1000,
            seq=1,
        )
    )
    snap = state.snapshot()
    res = build_decision_context(snap, [snap.segments_by_id["ov1"]], input_limit_tokens=10)
    assert res.context is not None and res.serialized_text is not None
    assert res.context.flags["input_overflow"] is True
    assert res.context.context_refs == ()
    assert "[OBJECTIVE]" in res.serialized_text
    assert "[ANCHOR]" in res.serialized_text


def test_templates_appendix_a_grounding_and_generic_fallback() -> None:
    templates = load_templates_dir(TEMPLATES_DIR)
    assert set(templates) == {
        "confirm_owner_v1",
        "confirm_time_v1",
        "success_criterion_v1",
        "clarify_scope_v1",
        "request_evidence_v1",
    }

    state = CanonicalState(_make_brief())
    state.apply(
        _seg_event(
            "seg-42",
            speaker_id="spk-2",
            text="Someone should own the rollback plan.",
            start_ms=125_000,
            end_ms=129_200,
            received_ms=129_450,
            seq=1,
        )
    )
    snap = state.snapshot()
    seg42 = snap.segments_by_id["seg-42"]
    slots = extract_slots_span_rules_v0([seg42], Category.CONFIRM_OWNER)
    assert "action" in slots
    assert slots["action"].text == "the rollback plan"
    assert slots["action"].char_start == 19
    assert slots["action"].char_end == 36
    assert verify_grounded_slot(slots["action"], seg42) is True

    # Permitted edits: whitespace, trailing punct, first-char case only
    assert verify_grounded_slot(
        GroundedSlot("The rollback plan", "seg-42", 1, 19, 37, "span_rules_v0"), seg42
    )
    # Disallowed edit: mid-word modification
    assert not verify_grounded_slot(
        GroundedSlot("the rollback plans", "seg-42", 1, 19, 36, "span_rules_v0"), seg42
    )

    built = build_decision_context(snap, [seg42], load_policy(POLICY_PATH))
    assert built.context is not None
    rendered = render_template(
        built.context,
        Category.CONFIRM_OWNER,
        snap,
        templates["confirm_owner_v1"],
        now_ms=129_600,
    )
    assert rendered.candidate is not None
    assert rendered.candidate.text == "Who will own the rollback plan?"
    assert rendered.candidate.grounded_slots["action"].char_start == 19
    assert rendered.candidate.grounded_slots["action"].char_end == 36
    assert rendered.is_generic is False

    # Other 4 template categories grounded rendering
    state2 = CanonicalState(_make_brief())
    state2.apply(
        _seg_event(
            "c1", speaker_id="spk-1", text="We must ship the billing service.", start_ms=1000, seq=1
        )
    )
    state2.apply(
        _seg_event(
            "c2",
            speaker_id="spk-1",
            text="I propose we adopt canary deploys.",
            start_ms=2000,
            seq=2,
        )
    )
    state2.apply(
        _seg_event(
            "c3",
            speaker_id="spk-1",
            text="We should deploy the API gateway except legacy webhooks.",
            start_ms=3000,
            seq=3,
        )
    )
    state2.apply(
        _seg_event(
            "c4",
            speaker_id="spk-1",
            text="The benchmark shows that p99 latency drops.",
            start_ms=4000,
            seq=4,
        )
    )
    s2 = state2.snapshot()
    for seg_id, cat, tpl_id in [
        ("c1", Category.CONFIRM_TIME, "confirm_time_v1"),
        ("c2", Category.SUCCESS_CRITERION, "success_criterion_v1"),
        ("c3", Category.CLARIFY_SCOPE, "clarify_scope_v1"),
        ("c4", Category.REQUEST_EVIDENCE, "request_evidence_v1"),
    ]:
        ctx = build_decision_context(s2, [s2.segments_by_id[seg_id]]).context
        assert ctx is not None
        out = render_template(ctx, cat, s2, templates[tpl_id], now_ms=4500)
        assert out.candidate is not None
        assert not out.is_generic
        assert out.candidate.word_count <= templates[tpl_id].max_words


def test_template_slot_too_long_multi_speaker_and_generic_referent_gating() -> None:
    templates = load_templates_dir(TEMPLATES_DIR)
    state = CanonicalState(_make_brief())
    # Too-long slot (>8 words) -> never shortened, falls back to generic when referent clear
    state.apply(
        _seg_event(
            "long-1",
            speaker_id="spk-1",
            text="Someone should own the multi region active active failover database migration checklist.",
            start_ms=1000,
            end_ms=2000,
            received_ms=2100,
            seq=1,
        )
    )
    # ME speaks after long-1: ME speech does NOT break anchor_is_latest_other_speech!
    state.apply(
        _seg_event(
            "me-after",
            speaker_id="me-1",
            text="Wait a moment",
            start_ms=2500,
            end_ms=3000,
            received_ms=3050,
            seq=2,
        )
    )
    snap = state.snapshot()
    ctx = build_decision_context(snap, [snap.segments_by_id["long-1"]]).context
    assert ctx is not None

    res = render_template(
        ctx, Category.CONFIRM_OWNER, snap, templates["confirm_owner_v1"], now_ms=5000
    )
    assert res.grounded_failure_reason is ReasonCode.SLOT_TOO_LONG
    assert res.is_generic is True
    assert res.candidate is not None
    assert res.candidate.text == "Who owns that next step?"

    # Age > 6000 ms blocks generic fallback
    stale = render_template(
        ctx, Category.CONFIRM_OWNER, snap, templates["confirm_owner_v1"], now_ms=9000
    )
    assert stale.candidate is None
    assert stale.unrendered_reason is ReasonCode.GENERIC_REFERENT_UNCLEAR

    # Another non-ME turn arriving after long-1 also blocks generic fallback
    state.apply(
        _seg_event(
            "other-after",
            speaker_id="spk-2",
            text="Let us talk about logs instead.",
            start_ms=4000,
            received_ms=4100,
            seq=3,
        )
    )
    snap2 = state.snapshot()
    ok, code = check_generic_referent(snap2, ctx.anchor_refs, now_ms=4200, max_anchor_age_ms=6000)
    assert ok is False and code is ReasonCode.GENERIC_REFERENT_UNCLEAR
    fb = render_generic_fallback(
        ctx, Category.CONFIRM_OWNER, snap2, templates["confirm_owner_v1"], now_ms=4200
    )
    assert fb.candidate is None and fb.unrendered_reason is ReasonCode.GENERIC_REFERENT_UNCLEAR

    # Cross-speaker slot override must be rejected as slot_unavailable
    cross = render_template(
        ctx,
        Category.CLARIFY_SCOPE,
        snap2,
        templates["clarify_scope_v1"],
        now_ms=4200,
        slots_override={
            "proposal": GroundedSlot("the multi region", "long-1", 1, 19, 35, "span_rules_v0"),
            "explicitly_mentioned_boundary": GroundedSlot(
                "logs", "other-after", 1, 21, 25, "span_rules_v0"
            ),
        },
    )
    assert cross.grounded_failure_reason is ReasonCode.SLOT_UNAVAILABLE


def test_gap_token_overflow_clock_domain_and_empty_anchor_value_error() -> None:
    brief = _make_brief()
    state = CanonicalState(brief)
    state.apply(
        _seg_event(
            "t1",
            speaker_id="spk-1",
            text="We should migrate the billing service soon.",
            start_ms=1000,
            end_ms=2000,
            received_ms=100_050,
            seq=1,
        )
    )
    for i in range(25):
        state.apply(
            CanonicalEvent(
                schema_version=1,
                session_id="sess-m3",
                event_id=f"ev-gap-{i}",
                kind=EventKind.CONTROL,
                received_ms=100_100 + i,
                source_sequence=10 + i,
                subtype=ControlSubtype.GAP,
                start_ms=2100 + i,
                end_ms=2101 + i,
                reason="network_disconnect_long_reason_marker",
            )
        )
    # Gap without start_ms/end_ms uses received_ms vs anchor_received_ms (clock-domain separation)
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="sess-m3",
            event_id="ev-gap-mono-in",
            kind=EventKind.CONTROL,
            received_ms=100_500,
            source_sequence=50,
            subtype=ControlSubtype.GAP,
            reason="mono_before_anchor",
        )
    )
    state.apply(
        CanonicalEvent(
            schema_version=1,
            session_id="sess-m3",
            event_id="ev-gap-mono-out",
            kind=EventKind.CONTROL,
            received_ms=102_000,
            source_sequence=51,
            subtype=ControlSubtype.GAP,
            reason="mono_after_anchor",
        )
    )
    state.apply(
        _seg_event(
            "a1",
            speaker_id="spk-1",
            text="Someone should own the rollback plan.",
            start_ms=4000,
            end_ms=5000,
            received_ms=101_000,
            seq=100,
        )
    )
    snap = state.snapshot()

    with pytest.raises(ValueError):
        build_decision_context(snap, [])

    bounded = build_decision_context(
        snap, [snap.segments_by_id["a1"]], input_limit_tokens=40, tokenizer=WhitespaceTokenizer()
    )
    assert bounded.context is not None and bounded.serialized_text is not None
    assert bounded.context.token_counts["total"] <= 40
    assert bounded.context.flags["has_gap"] is True
    assert bounded.context.omitted["gaps"] > 0
    assert "[GAP: mono_before_anchor]" in bounded.serialized_text
    assert "[GAP: mono_after_anchor]" not in bounded.serialized_text

    # Unavoidable overflow above limit always sets input_overflow=True and uniform omitted keys
    overflow = build_decision_context(
        snap, [snap.segments_by_id["a1"]], input_limit_tokens=20, tokenizer=WhitespaceTokenizer()
    )
    assert overflow.context is not None
    assert overflow.context.token_counts["total"] > 20
    assert overflow.context.flags["input_overflow"] is True
    assert set(overflow.context.omitted) == {"turns", "suggestions", "known_facts", "gaps"}
    assert overflow.context.omitted["gaps"] > 0


def test_normalize_for_matching_underscores_and_prefilter_backchannel() -> None:
    assert normalize_for_matching("_salary_") == "salary"
    assert normalize_for_matching("__salary__") == "salary"
    assert normalize_for_matching("_secret project_") == "secret project"
    assert normalize_for_matching("salary_band") == "salary band"
    assert normalize_for_matching("_yeah_") == "yeah"
    assert normalize_for_matching("mm-hm") == "mm hm"
    assert normalize_for_matching("uh-huh") == "uh huh"
    assert normalize_for_matching("didn't") == "didn t"

    state = CanonicalState(_make_brief())
    state.apply(
        _seg_event(
            "u1", speaker_id="spk-1", text="_yeah_", start_ms=1000, seq=1, boundary="utterance_end"
        )
    )
    state.apply(_seg_event("u2", speaker_id="spk-1", text="mm-hm", start_ms=2000, seq=2))
    anchors = coalesce_anchors(state.snapshot())
    assert len(anchors) == 2

    pf1 = run_prefilter(anchors[0], state_version=1)
    assert pf1.passed is False
    assert pf1.reason_code is ReasonCode.BACKCHANNEL
    assert pf1.normalized_text == "yeah"

    pf2 = run_prefilter(anchors[1], state_version=1)
    assert pf2.passed is False
    assert pf2.reason_code is ReasonCode.BACKCHANNEL
    assert pf2.normalized_text == "mm hm"
