"""Routing policy, threshold enforcement, ABSTAIN/SUPPRESS checks, and route selection (§5.5, §5.7, §7.3, §15 level 6)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from coach.canonical_state import StateSnapshot
from coach.context_builder import (
    AbstainPolicy,
    BreakerPolicy,
    CategoryPolicy,
    ClassifierPolicy,
    DisplayPolicy,
    GeneratorPolicy,
    HealthPolicy,
    ManualPolicy,
    PolicyConfig,
    PolicyConfigError,
    _p_bool,
    _p_float,
    _p_int,
    _p_map,
    _p_opt_str,
    _p_str,
    compute_policy_sha256,
    effective_max_cards_per_10_min,
    normalize_for_matching,
)
from coach.context_builder import load_policy as _cb_load_policy
from coach.contracts import (
    PREDICTABLE_CATEGORIES,
    SCHEMA_VERSION,
    Category,
    ClassifierResult,
    DecisionContext,
    HealthState,
    LedgerState,
    Mode,
    Origin,
    Outcome,
    ReasonCode,
    Route,
    RoutingDecision,
    SessionBrief,
    SessionMode,
)
from coach.opportunity_classifier import ClassifierErrorResult

__all__ = [
    "AbstainPolicy",
    "BreakerPolicy",
    "CategoryPolicy",
    "ClassifierPolicy",
    "DisplayPolicy",
    "GeneratorPolicy",
    "HealthPolicy",
    "ManualPolicy",
    "PolicyConfig",
    "PolicyConfigError",
    "compute_policy_sha256",
    "effective_max_cards_per_10_min",
    "load_policy",
    "route_candidate",
    "validate_thresholds",
]

_BLOCKING_LEDGER_STATES: frozenset[LedgerState] = frozenset(
    {
        LedgerState.READY,
        LedgerState.DEFERRED,
        LedgerState.DISPLAYED,
        LedgerState.PINNED,
        LedgerState.DISMISSED,
        LedgerState.USED,
    }
)

_CONTENT_STOPWORDS: frozenset[str] = frozenset(
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
        "being",
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
    }
)


def validate_thresholds(t_low: float, t_compute: float, t_display: float) -> dict[str, float]:
    """Validate 0 <= t_low <= t_compute <= t_display <= 1 and return dict (§5.7)."""
    vals = {"t_low": t_low, "t_compute": t_compute, "t_display": t_display}
    out: dict[str, float] = {}
    for k, v in vals.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise PolicyConfigError(f"Threshold {k} must be numeric, got {v!r}")
        fv = float(v)
        if not math.isfinite(fv) or not (0.0 <= fv <= 1.0):
            raise PolicyConfigError(f"Threshold {k} must be in [0, 1], got {fv!r}")
        out[k] = fv
    if not (out["t_low"] <= out["t_compute"] <= out["t_display"]):
        raise PolicyConfigError(
            "Thresholds must satisfy t_low <= t_compute <= t_display, "
            f"got {out['t_low']}, {out['t_compute']}, {out['t_display']}"
        )
    return out


def _build_policy_with_null_calibration(
    raw_bytes: bytes, loaded: Mapping[str, Any]
) -> PolicyConfig:
    m = _p_map(loaded, "policy")
    try:
        mode = SessionMode(_p_str(m.get("mode"), "policy.mode"))
    except ValueError as exc:
        raise PolicyConfigError(f"Invalid policy.mode: {m.get('mode')!r}") from exc

    cls_m = _p_map(m.get("classifier"), "policy.classifier")
    classifier = ClassifierPolicy(
        version=_p_str(cls_m.get("version"), "classifier.version"),
        calibration="",
        input_limit_tokens=_p_int(
            cls_m.get("input_limit_tokens"), "classifier.input_limit_tokens", minimum=1
        ),
        max_candidates_per_job=_p_int(
            cls_m.get("max_candidates_per_job"), "classifier.max_candidates_per_job", minimum=1
        ),
        model_path=_p_opt_str(cls_m.get("model_path"), "classifier.model_path"),
        calibration_path=_p_opt_str(cls_m.get("calibration_path"), "classifier.calibration_path"),
    )

    cats_m = _p_map(m.get("categories"), "policy.categories")
    categories: dict[Category, CategoryPolicy] = {}
    for raw_cat, raw_cfg in cats_m.items():
        try:
            cat = Category(str(raw_cat))
        except ValueError as exc:
            raise PolicyConfigError(f"Unknown category in policy: {raw_cat!r}") from exc
        if cat not in PREDICTABLE_CATEGORIES:
            raise PolicyConfigError(f"Non-predictable category in policy: {cat!r}")
        cm = _p_map(raw_cfg, f"categories.{cat}")
        try:
            route = Route(_p_str(cm.get("route"), f"categories.{cat}.route"))
        except ValueError as exc:
            raise PolicyConfigError(f"Invalid route for {cat}") from exc
        tpl_raw = cm.get("template")
        tpl = _p_str(tpl_raw, f"categories.{cat}.template") if tpl_raw is not None else None
        categories[cat] = CategoryPolicy(
            t_low=_p_float(cm.get("t_low"), f"categories.{cat}.t_low"),
            t_compute=_p_float(cm.get("t_compute"), f"categories.{cat}.t_compute"),
            t_display=_p_float(cm.get("t_display"), f"categories.{cat}.t_display"),
            route=route,
            template=tpl,
            required_lookback_s=_p_int(
                cm.get("required_lookback_s"), f"categories.{cat}.required_lookback_s"
            ),
        )

    disp_m = _p_map(m.get("display"), "policy.display")
    display = DisplayPolicy(
        max_active_cards=_p_int(
            disp_m.get("max_active_cards"), "display.max_active_cards", minimum=1
        ),
        settle_ms=_p_int(disp_m.get("settle_ms"), "display.settle_ms"),
        max_candidate_age_ms=_p_int(
            disp_m.get("max_candidate_age_ms"), "display.max_candidate_age_ms", minimum=1
        ),
        display_ttl_ms=_p_int(disp_m.get("display_ttl_ms"), "display.display_ttl_ms", minimum=1),
        cooldown_ms=_p_int(disp_m.get("cooldown_ms"), "display.cooldown_ms"),
        max_cards_per_10_min=_p_int(
            disp_m.get("max_cards_per_10_min"), "display.max_cards_per_10_min"
        ),
    )

    man_m = _p_map(m.get("manual"), "policy.manual")
    manual = ManualPolicy(
        lookback_s=_p_int(man_m.get("lookback_s"), "manual.lookback_s", minimum=1),
        score_floor=_p_float(man_m.get("score_floor"), "manual.score_floor"),
        budget_ms=_p_int(man_m.get("budget_ms"), "manual.budget_ms", minimum=1),
    )

    gen_m = _p_map(m.get("generator"), "policy.generator")
    for bad_secret in ("api_key", "secret", "token", "password"):
        if bad_secret in gen_m:
            raise PolicyConfigError(
                f"generator.{bad_secret} is forbidden in policy YAML (§14.6); use api_key_env"
            )
    brk_m = _p_map(gen_m.get("breaker"), "generator.breaker")
    generator = GeneratorPolicy(
        enabled=_p_bool(gen_m.get("enabled"), "generator.enabled"),
        max_words=_p_int(gen_m.get("max_words"), "generator.max_words", minimum=1),
        timeout_ms=_p_int(gen_m.get("timeout_ms"), "generator.timeout_ms", minimum=1),
        breaker=BreakerPolicy(
            consecutive_failures=_p_int(
                brk_m.get("consecutive_failures"), "breaker.consecutive_failures", minimum=1
            ),
            open_ms=_p_int(brk_m.get("open_ms"), "breaker.open_ms", minimum=1),
        ),
        provider=_p_opt_str(gen_m.get("provider"), "generator.provider"),
        model_id=_p_opt_str(gen_m.get("model_id"), "generator.model_id"),
        reasoning=_p_opt_str(gen_m.get("reasoning"), "generator.reasoning"),
        api_key_env=_p_opt_str(gen_m.get("api_key_env"), "generator.api_key_env"),
        base_url=_p_opt_str(gen_m.get("base_url"), "generator.base_url"),
    )
    abs_m = _p_map(m.get("abstain"), "policy.abstain")
    floor_raw = abs_m.get("asr_confidence_floor")
    asr_floor = (
        _p_float(floor_raw, "abstain.asr_confidence_floor") if floor_raw is not None else None
    )
    abstain = AbstainPolicy(
        speaker_settle_ms=_p_int(abs_m.get("speaker_settle_ms"), "abstain.speaker_settle_ms"),
        asr_confidence_floor=asr_floor,
    )
    hlth_m = _p_map(m.get("health"), "policy.health")
    health = HealthPolicy(
        no_transcript_after_ms=_p_int(
            hlth_m.get("no_transcript_after_ms"), "health.no_transcript_after_ms", minimum=1
        )
    )
    return PolicyConfig(
        policy_version=_p_str(m.get("policy_version"), "policy.policy_version"),
        mode=mode,
        classifier=classifier,
        categories=categories,
        display=display,
        manual=manual,
        generator=generator,
        abstain=abstain,
        health=health,
        policy_sha256=compute_policy_sha256(raw_bytes),
    )


def load_policy(
    path: str | Path,
    *,
    calibration_metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    allow_null_calibration: bool = True,
) -> PolicyConfig:
    """Load policy YAML, supporting uncalibrated (`calibration: null`) policy when allowed."""
    src = Path(path)
    raw_bytes = src.read_bytes()
    loaded = yaml.safe_load(raw_bytes.decode("utf-8"))
    if (
        allow_null_calibration
        and isinstance(loaded, Mapping)
        and isinstance(loaded.get("classifier"), Mapping)
        and loaded["classifier"].get("calibration") is None
    ):
        try:
            return _build_policy_with_null_calibration(raw_bytes, loaded)
        except PolicyConfigError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyConfigError(f"Invalid policy configuration: {exc}") from exc
    return _cb_load_policy(src, calibration_metadata=calibration_metadata)


def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize_for_matching(text).split() if t and t not in _CONTENT_STOPWORDS}


def _matches_do_not_suggest(
    anchor_text: str,
    category: Category | None,
    brief: SessionBrief,
) -> str | None:
    norm_anchor = normalize_for_matching(anchor_text)
    norm_cat = normalize_for_matching(category.value) if category is not None else ""
    for term in brief.do_not_suggest:
        norm_term = normalize_for_matching(term)
        if not norm_term:
            continue
        pat = re.compile(rf"(?<![^\W_]){re.escape(norm_term)}(?![^\W_])")
        if (norm_anchor and pat.search(norm_anchor)) or (norm_cat and pat.search(norm_cat)):
            return term
    return None


def _matches_known_fact(anchor_text: str, brief: SessionBrief) -> str | None:
    norm_anchor = normalize_for_matching(anchor_text)
    if not norm_anchor:
        return None
    for fact in brief.known_facts:
        norm_val = normalize_for_matching(fact.value)
        if norm_val:
            pat = re.compile(rf"(?<![^\W_]){re.escape(norm_val)}(?![^\W_])")
            if pat.search(norm_anchor):
                return f"{fact.key}={fact.value}"
    return None


def _check_ledger_duplicate(
    anchor_text: str,
    category: Category,
    evidence_ids: Sequence[str],
    snapshot: StateSnapshot,
    *,
    ledger_texts: Mapping[str, str] | None,
    ledger_categories: Mapping[str, Category] | None,
    prior_anchor_texts: Mapping[str, str] | None,
    near_dup_threshold: float,
) -> ReasonCode | None:
    norm_anchor = normalize_for_matching(anchor_text)
    anchor_tokens = _content_tokens(anchor_text)
    ev_set = set(evidence_ids)

    for entry in snapshot.ledger.values():
        if entry.state not in _BLOCKING_LEDGER_STATES:
            continue
        entry_cat = ledger_categories.get(entry.suggestion_id) if ledger_categories else None
        same_or_unknown_cat = entry_cat is None or entry_cat is category
        entry_ev = {r.segment_id for r in entry.evidence_refs}
        if same_or_unknown_cat and (ev_set & entry_ev):
            return ReasonCode.DUPLICATE

        candidates_to_compare: list[str] = []
        if ledger_texts and entry.suggestion_id in ledger_texts:
            candidates_to_compare.append(ledger_texts[entry.suggestion_id])
        if prior_anchor_texts and entry.candidate_id in prior_anchor_texts:
            candidates_to_compare.append(prior_anchor_texts[entry.candidate_id])
        if same_or_unknown_cat:
            seg_parts = [
                snapshot.segments_by_id[r.segment_id].text
                for r in entry.evidence_refs
                if r.segment_id in snapshot.segments_by_id
            ]
            if seg_parts:
                candidates_to_compare.append(" ".join(seg_parts))

        # Pass 1: exact normalized match across all candidates -> DUPLICATE
        for prev_text in candidates_to_compare:
            if (
                prev_text
                and norm_anchor
                and normalize_for_matching(prev_text) == norm_anchor
                and same_or_unknown_cat
            ):
                return ReasonCode.DUPLICATE

        # Pass 2: token Jaccard overlap across all candidates -> NEAR_DUPLICATE
        for prev_text in candidates_to_compare:
            if not prev_text:
                continue
            prev_tokens = _content_tokens(prev_text)
            if anchor_tokens and prev_tokens:
                union_sz = len(anchor_tokens | prev_tokens)
                jaccard = len(anchor_tokens & prev_tokens) / union_sz if union_sz else 0.0
                if jaccard >= near_dup_threshold and same_or_unknown_cat:
                    return ReasonCode.NEAR_DUPLICATE
    return None


def _resolve_anchor_text(
    context: DecisionContext,
    snapshot: StateSnapshot,
    anchor_text: str | None,
) -> str:
    if anchor_text is not None:
        return anchor_text
    parts: list[str] = []
    for ref in context.anchor_refs:
        seg = snapshot.segments_by_id.get(ref.segment_id)
        if seg is not None:
            parts.append(seg.text.strip())
    return " ".join(p for p in parts if p)


def _count_recent_proactive_displays(
    snapshot: StateSnapshot,
    decided_ms: int,
    recent_displayed_ms: Sequence[int] | None,
) -> int:
    window_start = decided_ms - 600_000
    if recent_displayed_ms is not None:
        return sum(1 for ts in recent_displayed_ms if window_start <= ts <= decided_ms)
    count = 0
    for entry in snapshot.ledger.values():
        for tr in entry.history:
            if tr.to_state is LedgerState.DISPLAYED and window_start <= tr.at_ms <= decided_ms:
                count += 1
                break
    return count


def _make_decision(
    context: DecisionContext,
    policy: PolicyConfig,
    *,
    outcome: Outcome,
    decided_ms: int,
    route: Route | None = None,
    category: Category | None = None,
    reason_codes: tuple[ReasonCode, ...] = (),
    abstain_reasons: tuple[ReasonCode, ...] = (),
    thresholds: dict[str, float] | None = None,
) -> RoutingDecision:
    return RoutingDecision(
        schema_version=SCHEMA_VERSION,
        candidate_id=context.candidate_id,
        policy_version=policy.policy_version,
        policy_sha256=policy.policy_sha256,
        based_on_state_version=context.based_on_state_version,
        outcome=outcome,
        route=route,
        category=category,
        reason_codes=reason_codes,
        abstain_reasons=abstain_reasons,
        thresholds=thresholds,
        decided_ms=decided_ms,
    )


def _detect_input_abstain_reasons(
    context: DecisionContext,
    snapshot: StateSnapshot,
    policy: PolicyConfig,
    *,
    decided_ms: int,
    speaker_update_times_ms: Mapping[str, int] | None,
) -> list[ReasonCode]:
    reasons: list[ReasonCode] = []
    spk_uncertain = bool(context.flags.get("speaker_uncertain", False))
    asr_low = bool(context.flags.get("asr_low_confidence", False))
    settle_ms = policy.abstain.speaker_settle_ms
    asr_floor = policy.abstain.asr_confidence_floor
    recipient_ids = {"ME", snapshot.brief.recipient.participant_id}

    for ref in context.anchor_refs:
        seg = snapshot.segments_by_id.get(ref.segment_id)
        if seg is not None:
            if (
                seg.speaker_uncertain
                or seg.speaker_id in recipient_ids
                or seg.speaker_id in {"S?", "unknown", ""}
                or (seg.speaker_confidence is not None and seg.speaker_confidence < 0.5)
            ):
                spk_uncertain = True
            if (
                seg.asr_low_confidence
                or seg.low_quality_gate
                or (
                    asr_floor is not None
                    and seg.asr_confidence is not None
                    and seg.asr_confidence < asr_floor
                )
            ):
                asr_low = True
        if speaker_update_times_ms and ref.segment_id in speaker_update_times_ms:
            upd_ms = speaker_update_times_ms[ref.segment_id]
            if 0 <= (decided_ms - upd_ms) <= settle_ms:
                spk_uncertain = True

    if spk_uncertain:
        reasons.append(ReasonCode.SPEAKER_UNCERTAIN)
    if asr_low:
        reasons.append(ReasonCode.ASR_LOW_CONFIDENCE)
    return reasons


def route_candidate(
    context: DecisionContext,
    classifier_result: ClassifierResult | None,
    snapshot: StateSnapshot,
    policy: PolicyConfig,
    *,
    mode: Mode = Mode.PROACTIVE,
    health: HealthState = HealthState.OK,
    now_ms: int | None = None,
    speaker_update_times_ms: Mapping[str, int] | None = None,
    ledger_texts: Mapping[str, str] | None = None,
    ledger_categories: Mapping[str, Category] | None = None,
    prior_anchor_texts: Mapping[str, str] | None = None,
    anchor_text: str | None = None,
    near_dup_threshold: float = 0.75,
    classifier_error: bool = False,
    recent_displayed_ms: Sequence[int] | None = None,
) -> RoutingDecision:
    """Apply §5.5, §5.7, §7.3, and §16 routing rules to produce a RoutingDecision."""
    decided_ms = context.anchor_received_ms if now_ms is None else int(now_ms)
    is_manual = context.origin is Origin.MANUAL
    anchor_text_eff = _resolve_anchor_text(context, snapshot, anchor_text)

    # 1. Proactive disabled (Quiet / Paused / Off-the-record / brief.attention.proactive=False)
    if not is_manual and (
        mode in {Mode.QUIET, Mode.PAUSED, Mode.OFF_THE_RECORD}
        or not snapshot.brief.attention.proactive
    ):
        return _make_decision(
            context,
            policy,
            outcome=Outcome.SUPPRESS,
            decided_ms=decided_ms,
            reason_codes=(ReasonCode.PROACTIVE_DISABLED,),
        )

    # 2. Classifier error / mode_mismatch / input_overflow
    if (
        classifier_error
        or classifier_result is None
        or isinstance(classifier_result, ClassifierErrorResult)
    ):
        return _make_decision(
            context,
            policy,
            outcome=Outcome.ABSTAIN,
            decided_ms=decided_ms,
            reason_codes=(ReasonCode.CLASSIFIER_ERROR,),
            abstain_reasons=(ReasonCode.CLASSIFIER_ERROR,),
        )

    if (
        snapshot.brief.mode != policy.mode
        or not policy.classifier.calibration
        or classifier_result.calibrated_scores is None
        or classifier_result.calibration_version is None
        or not policy.categories
    ):
        return _make_decision(
            context,
            policy,
            outcome=Outcome.ABSTAIN,
            decided_ms=decided_ms,
            reason_codes=(ReasonCode.MODE_MISMATCH,),
            abstain_reasons=(ReasonCode.MODE_MISMATCH,),
        )

    if context.flags.get("input_overflow", False):
        return _make_decision(
            context,
            policy,
            outcome=Outcome.ABSTAIN,
            decided_ms=decided_ms,
            reason_codes=(ReasonCode.INPUT_OVERFLOW,),
            abstain_reasons=(ReasonCode.INPUT_OVERFLOW,),
        )

    # 3. Anchor-level do_not_suggest & known_fact suppression
    if _matches_do_not_suggest(anchor_text_eff, None, snapshot.brief) is not None:
        return _make_decision(
            context,
            policy,
            outcome=Outcome.SUPPRESS,
            decided_ms=decided_ms,
            reason_codes=(ReasonCode.DO_NOT_SUGGEST,),
        )
    if not is_manual and _matches_known_fact(anchor_text_eff, snapshot.brief) is not None:
        return _make_decision(
            context,
            policy,
            outcome=Outcome.SUPPRESS,
            decided_ms=decided_ms,
            reason_codes=(ReasonCode.KNOWN_FACT,),
        )

    # 4. Input-quality abstentions (proactive) + recipient-flip guard (proactive & manual)
    recipient_ids = {"ME", snapshot.brief.recipient.participant_id}
    if not is_manual:
        input_abstain = _detect_input_abstain_reasons(
            context,
            snapshot,
            policy,
            decided_ms=decided_ms,
            speaker_update_times_ms=speaker_update_times_ms,
        )
        if input_abstain:
            return _make_decision(
                context,
                policy,
                outcome=Outcome.ABSTAIN,
                decided_ms=decided_ms,
                reason_codes=tuple(input_abstain),
                abstain_reasons=tuple(input_abstain),
            )
    elif any(
        (seg := snapshot.segments_by_id.get(r.segment_id)) is not None
        and seg.speaker_id in recipient_ids
        for r in context.anchor_refs
    ):
        return _make_decision(
            context,
            policy,
            outcome=Outcome.ABSTAIN,
            decided_ms=decided_ms,
            reason_codes=(ReasonCode.SPEAKER_UNCERTAIN,),
            abstain_reasons=(ReasonCode.SPEAKER_UNCERTAIN,),
        )

    # 5. Score selection & threshold checks
    scores = classifier_result.calibrated_scores
    active_cats = [c for c in policy.categories if c in PREDICTABLE_CATEGORIES]
    cat_scores = {c: float(scores.categories.get(c.value, 0.0)) for c in active_cats}
    best_cat = max(
        active_cats,
        key=lambda c: (cat_scores[c] - policy.categories[c].t_display, cat_scores[c], c.value),
    )
    best_cpol = policy.categories[best_cat]
    best_thresh = validate_thresholds(best_cpol.t_low, best_cpol.t_compute, best_cpol.t_display)

    desired_route: Route
    if is_manual:
        if max(float(scores.opportunity), cat_scores[best_cat]) < policy.manual.score_floor:
            return _make_decision(
                context,
                policy,
                outcome=Outcome.NONE,
                decided_ms=decided_ms,
                category=best_cat,
                reason_codes=(ReasonCode.BELOW_T_LOW,),
                thresholds=best_thresh,
            )
        chosen_cat = best_cat
        chosen_thresh = best_thresh
        desired_route = (
            Route.GENERATOR
            if (
                cat_scores[best_cat] < best_cpol.t_low
                and policy.generator.enabled
                and health is HealthState.OK
            )
            else best_cpol.route
        )
    else:

        def _is_truncated(cat: Category) -> bool:
            key = f"context_truncated_material:{cat.value}"
            if key in context.flags:
                return bool(context.flags[key])
            return bool(context.flags.get("context_truncated_material", False))

        eligible = [
            c
            for c in active_cats
            if cat_scores[c] >= policy.categories[c].t_display and not _is_truncated(c)
        ]
        if eligible:
            chosen_cat = max(
                eligible,
                key=lambda c: (
                    cat_scores[c] - policy.categories[c].t_display,
                    cat_scores[c],
                    c.value,
                ),
            )
            cpol = policy.categories[chosen_cat]
            chosen_thresh = validate_thresholds(cpol.t_low, cpol.t_compute, cpol.t_display)
            desired_route = cpol.route
        else:
            blocked_by_trunc = any(
                cat_scores[c] >= policy.categories[c].t_display and _is_truncated(c)
                for c in active_cats
            ) or (
                _is_truncated(best_cat)
                and max(float(scores.opportunity), cat_scores[best_cat]) >= best_cpol.t_low
            )
            if blocked_by_trunc:
                return _make_decision(
                    context,
                    policy,
                    outcome=Outcome.ABSTAIN,
                    decided_ms=decided_ms,
                    category=best_cat,
                    reason_codes=(ReasonCode.CONTEXT_TRUNCATED_MATERIAL,),
                    abstain_reasons=(ReasonCode.CONTEXT_TRUNCATED_MATERIAL,),
                    thresholds=best_thresh,
                )
            min_t_low = min(policy.categories[c].t_low for c in active_cats)
            if float(scores.opportunity) < min_t_low and all(
                cat_scores[c] < policy.categories[c].t_low for c in active_cats
            ):
                return _make_decision(
                    context,
                    policy,
                    outcome=Outcome.NONE,
                    decided_ms=decided_ms,
                    category=best_cat,
                    reason_codes=(ReasonCode.BELOW_T_LOW,),
                    thresholds=best_thresh,
                )
            return _make_decision(
                context,
                policy,
                outcome=Outcome.ABSTAIN,
                decided_ms=decided_ms,
                category=None
                if all(cat_scores[c] < policy.categories[c].t_low for c in active_cats)
                else best_cat,
                reason_codes=(ReasonCode.SCORE_IN_BAND,),
                abstain_reasons=(ReasonCode.SCORE_IN_BAND,),
                thresholds=best_thresh,
            )

    # 6. Category-aware suppressions (DNS, dedup, attention budget cap)
    if _matches_do_not_suggest(anchor_text_eff, chosen_cat, snapshot.brief) is not None:
        return _make_decision(
            context,
            policy,
            outcome=Outcome.SUPPRESS,
            decided_ms=decided_ms,
            category=chosen_cat,
            reason_codes=(ReasonCode.DO_NOT_SUGGEST,),
            thresholds=chosen_thresh,
        )

    if not is_manual:
        ev_ids = [r.segment_id for r in context.anchor_refs]
        dup_code = _check_ledger_duplicate(
            anchor_text_eff,
            chosen_cat,
            ev_ids,
            snapshot,
            ledger_texts=ledger_texts,
            ledger_categories=ledger_categories,
            prior_anchor_texts=prior_anchor_texts,
            near_dup_threshold=near_dup_threshold,
        )
        if dup_code is not None:
            return _make_decision(
                context,
                policy,
                outcome=Outcome.SUPPRESS,
                decided_ms=decided_ms,
                category=chosen_cat,
                reason_codes=(dup_code,),
                thresholds=chosen_thresh,
            )

        cap = effective_max_cards_per_10_min(snapshot.brief, policy)
        displayed_10m = _count_recent_proactive_displays(snapshot, decided_ms, recent_displayed_ms)
        if cap <= 0 or displayed_10m >= cap:
            return _make_decision(
                context,
                policy,
                outcome=Outcome.SUPPRESS,
                decided_ms=decided_ms,
                category=chosen_cat,
                reason_codes=(ReasonCode.PROACTIVE_DISABLED,),
                thresholds=chosen_thresh,
            )

    # 7. Route realization (generator fallback to template if disabled/template_only)
    chosen_cpol = policy.categories[chosen_cat]
    selected_route = desired_route
    if desired_route is Route.GENERATOR and (
        not policy.generator.enabled or health is HealthState.TEMPLATE_ONLY
    ):
        if chosen_cpol.template is not None:
            selected_route = Route.TEMPLATE
        else:
            return _make_decision(
                context,
                policy,
                outcome=Outcome.ABSTAIN,
                decided_ms=decided_ms,
                category=chosen_cat,
                reason_codes=(ReasonCode.MODE_MISMATCH,),
                abstain_reasons=(ReasonCode.MODE_MISMATCH,),
                thresholds=chosen_thresh,
            )

    return _make_decision(
        context,
        policy,
        outcome=Outcome.ROUTE,
        decided_ms=decided_ms,
        route=selected_route,
        category=chosen_cat,
        reason_codes=(ReasonCode.MANUAL_REQUEST if is_manual else ReasonCode.ABOVE_T_DISPLAY,),
        thresholds=chosen_thresh,
    )
