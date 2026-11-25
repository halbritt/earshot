"""Replay evaluator, dataset splitters, validity matcher, threshold sweeper, bootstrap, and latency benchmark (§10.2, §12.5–§12.8, §15 level 12)."""

from __future__ import annotations

import asyncio
import importlib
import json
import random
import resource
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from coach.calibration import CalibrationModel, CalibrationRegistry, default_calibration_registry
from coach.canonical_state import CanonicalState, StateSnapshot
from coach.context_builder import (
    CategoryPolicy,
    ContextBuildResult,
    PolicyConfig,
    WhitespaceTokenizer,
    build_decision_context,
    build_decision_contexts,
    effective_max_cards_per_10_min,
)
from coach.contracts import (
    SCHEMA_VERSION,
    CanonicalEvent,
    Category,
    ClosedBy,
    DecisionContext,
    EgressRecord,
    EventKind,
    HealthState,
    LabelRecord,
    LedgerEvent,
    LedgerState,
    Mode,
    Origin,
    Outcome,
    ReasonCode,
    Route,
    SegmentRef,
    SessionBrief,
    Stability,
    SuggestionCandidate,
)
from coach.opportunity_classifier import (
    ClassifierErrorResult,
    ClassifierInput,
    OpportunityClassifier,
    classify_batch_cached,
)
from coach.provider_adapter import ProviderAdapter
from coach.question_generator import generate_question
from coach.routing_policy import route_candidate
from coach.storage import SQLiteStore
from coach.template_renderer import TemplateSpec, load_templates_dir, render_template

_RESOLUTION_CUE_TERMS = (
    "i'll own",
    "i will own",
    "i'll handle",
    "i will handle",
    "i'll take",
    "assigned to",
    "let's do friday",
    "by friday",
    "deadline is",
    "never mind",
    "already decided",
)


@dataclass(frozen=True, slots=True)
class DatasetSplits[T]:
    """Disjoint conversation/series dataset splits (§12.5)."""

    train: tuple[T, ...]
    cal: tuple[T, ...]
    eval: tuple[T, ...]
    train_conversations: tuple[str, ...]
    cal_conversations: tuple[str, ...]
    eval_conversations: tuple[str, ...]
    train_series: tuple[str, ...]
    cal_series: tuple[str, ...]
    eval_series: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_conversations": list(self.train_conversations),
            "cal_conversations": list(self.cal_conversations),
            "eval_conversations": list(self.eval_conversations),
            "train_series": list(self.train_series),
            "cal_series": list(self.cal_series),
            "eval_series": list(self.eval_series),
            "counts": {
                "train": len(self.train),
                "cal": len(self.cal),
                "eval": len(self.eval),
            },
        }


def _get_field(item: Any, *names: str) -> str | None:
    if isinstance(item, Mapping):
        for n in names:
            v = item.get(n)
            if v is not None and str(v):
                return str(v)
    else:
        for n in names:
            v = getattr(item, n, None)
            if v is not None and str(v):
                return str(v)
    return None


def _default_session_id(item: Any) -> str:
    val = _get_field(item, "session_id", "conversation_id")
    if val is None:
        raise ValueError(f"Item {item!r} lacks session_id/conversation_id")
    return val


def _default_series_id(item: Any) -> str:
    val = _get_field(item, "series_id")
    if val is not None:
        return val
    return _default_session_id(item)


def _split_groups[T](
    items: Sequence[T],
    *,
    group_fn: Callable[[T], str],
    session_fn: Callable[[T], str],
    series_fn: Callable[[T], str],
    train_frac: float,
    cal_frac: float,
    eval_frac: float,
    seed: int,
) -> DatasetSplits[T]:
    total_frac = train_frac + cal_frac + eval_frac
    if total_frac <= 0.0:
        raise ValueError("Sum of split fractions must be > 0")
    t_f = train_frac / total_frac
    c_f = cal_frac / total_frac

    grouped: dict[str, list[T]] = {}
    for item in items:
        grouped.setdefault(group_fn(item), []).append(item)

    group_keys = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = round(n * t_f)
    n_cal = round(n * c_f)
    if n >= 3 and train_frac > 0 and cal_frac > 0 and eval_frac > 0:
        n_train = max(1, min(n - 2, n_train))
        n_cal = max(1, min(n - n_train - 1, n_cal))
    n_train = min(n, max(0, n_train))
    n_cal = min(n - n_train, max(0, n_cal))

    train_keys = set(group_keys[:n_train])
    cal_keys = set(group_keys[n_train : n_train + n_cal])

    train_items: list[T] = []
    cal_items: list[T] = []
    eval_items: list[T] = []
    for k in group_keys:
        bucket = train_items if k in train_keys else (cal_items if k in cal_keys else eval_items)
        bucket.extend(grouped[k])

    return DatasetSplits(
        train=tuple(train_items),
        cal=tuple(cal_items),
        eval=tuple(eval_items),
        train_conversations=tuple(sorted({session_fn(x) for x in train_items})),
        cal_conversations=tuple(sorted({session_fn(x) for x in cal_items})),
        eval_conversations=tuple(sorted({session_fn(x) for x in eval_items})),
        train_series=tuple(sorted({series_fn(x) for x in train_items})),
        cal_series=tuple(sorted({series_fn(x) for x in cal_items})),
        eval_series=tuple(sorted({series_fn(x) for x in eval_items})),
    )


def split_by_conversation[T](
    items: Sequence[T],
    *,
    session_id_fn: Callable[[T], str] | None = None,
    series_id_fn: Callable[[T], str] | None = None,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    eval_frac: float = 0.2,
    seed: int = 0,
) -> DatasetSplits[T]:
    """Split items by complete conversation (zero session_id leakage across splits, §12.5)."""
    s_fn = session_id_fn or _default_session_id
    ser_fn = series_id_fn or _default_series_id
    return _split_groups(
        items,
        group_fn=s_fn,
        session_fn=s_fn,
        series_fn=ser_fn,
        train_frac=train_frac,
        cal_frac=cal_frac,
        eval_frac=eval_frac,
        seed=seed,
    )


def split_by_series[T](
    items: Sequence[T],
    *,
    series_id_fn: Callable[[T], str] | None = None,
    session_id_fn: Callable[[T], str] | None = None,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    eval_frac: float = 0.2,
    seed: int = 0,
) -> DatasetSplits[T]:
    """Split items keeping every recurring meeting series on one side of the split (§12.5)."""
    s_fn = session_id_fn or _default_session_id
    ser_fn = series_id_fn or _default_series_id
    return _split_groups(
        items,
        group_fn=ser_fn,
        session_fn=s_fn,
        series_fn=ser_fn,
        train_frac=train_frac,
        cal_frac=cal_frac,
        eval_frac=eval_frac,
        seed=seed,
    )


class EvalLookCounter:
    """Persistent JSON counter logging every evaluation run against the frozen eval set (§12.5)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"look_count": 0, "history": []}
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return {"look_count": 0, "history": []}
        return {
            "look_count": int(loaded.get("look_count", 0)),
            "history": list(loaded.get("history", [])),
        }

    @property
    def look_count(self) -> int:
        return int(self._read()["look_count"])

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._read()["history"])

    def record_look(
        self,
        run_id: str,
        *,
        split_by: str = "conversation",
        eval_conversations: Sequence[str] = (),
        notes: str | None = None,
        timestamp_ms: int | None = None,
    ) -> int:
        state = self._read()
        new_count = int(state["look_count"]) + 1
        ts = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
        entry: dict[str, Any] = {
            "look": new_count,
            "run_id": run_id,
            "split_by": split_by,
            "eval_conversations": list(eval_conversations),
            "timestamp_ms": ts,
        }
        if notes is not None:
            entry["notes"] = notes
        history = list(state["history"])
        history.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"look_count": new_count, "history": history}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return new_count


@dataclass(frozen=True, slots=True)
class DisplayedCardEvent:
    """Displayed card record for replay evaluation & opportunity matching (§12.6)."""

    suggestion_id: str
    candidate_id: str
    session_id: str
    category: Category
    display_ms: int
    evidence_refs: tuple[SegmentRef, ...]
    equivalent_categories: tuple[Category, ...] = ()
    text: str = ""
    route: Route = Route.TEMPLATE
    origin: Origin = Origin.PROACTIVE


def _coerce_displayed_card(card: DisplayedCardEvent | Mapping[str, Any]) -> DisplayedCardEvent:
    if isinstance(card, DisplayedCardEvent):
        return card
    ev_raw = card.get("evidence_refs") or ()
    refs: list[SegmentRef] = []
    for r in ev_raw:
        if isinstance(r, SegmentRef):
            refs.append(r)
        elif isinstance(r, Mapping):
            refs.append(SegmentRef.from_dict(r))
    if not refs and card.get("segment_ids"):
        refs = [SegmentRef(str(s), 1) for s in card["segment_ids"]]
    eq_raw = card.get("equivalent_categories") or ()
    eq_cats = tuple(Category(str(c)) for c in eq_raw)
    return DisplayedCardEvent(
        suggestion_id=str(card["suggestion_id"]),
        candidate_id=str(card.get("candidate_id", card["suggestion_id"])),
        session_id=str(card.get("session_id", "")),
        category=Category(str(card["category"])),
        display_ms=int(card["display_ms"]),
        evidence_refs=tuple(refs),
        equivalent_categories=eq_cats,
        text=str(card.get("text", "")),
        route=Route(str(card.get("route", Route.TEMPLATE.value))),
        origin=Origin(str(card.get("origin", Origin.PROACTIVE.value))),
    )


def match_opportunities(
    displayed_cards: Sequence[DisplayedCardEvent | Mapping[str, Any]],
    labels: Sequence[LabelRecord],
    *,
    settle_ms: int = 1500,
    premature_tolerance_ms: int = 200,
    duration_ms: int | None = None,
    min_positives_for_ci: int = 10,
) -> dict[str, Any]:
    """Match displayed cards against LabelRecord validity intervals (§12.6)."""
    cards = sorted(
        (_coerce_displayed_card(c) for c in displayed_cards),
        key=lambda c: (c.display_ms, c.suggestion_id),
    )
    positives = [lbl for lbl in labels if lbl.worthwhile]
    negatives = [lbl for lbl in labels if not lbl.worthwhile]

    matched_pos_ids: set[str] = set()
    useful_count = 0
    repetition_count = 0
    late_count = 0
    late_by_closed_by: Counter[str] = Counter()
    premature_count = 0
    distracting_count = 0
    distracting_by_none_reason: Counter[str] = Counter()
    review_queue: list[dict[str, Any]] = []
    cat_useful: Counter[str] = Counter()

    for card in cards:
        card_segs = {r.segment_id for r in card.evidence_refs}
        pos_matches: list[LabelRecord] = []
        for lbl in positives:
            if card.session_id and lbl.session_id and card.session_id != lbl.session_id:
                continue
            lbl_segs = {r.segment_id for r in lbl.evidence_refs}
            if not (card_segs & lbl_segs):
                continue
            cat_ok = (
                card.category in lbl.categories
                or Category.OTHER in lbl.categories
                or any(eq in lbl.categories for eq in card.equivalent_categories)
            )
            if cat_ok:
                pos_matches.append(lbl)

        if pos_matches:

            def _pos_sort_key(
                lbl: LabelRecord, disp_ms: int = card.display_ms
            ) -> tuple[int, int, str]:
                in_win = (
                    0
                    if (
                        disp_ms >= lbl.validity.open_ms
                        and (lbl.validity.close_ms is None or disp_ms <= lbl.validity.close_ms)
                    )
                    else 1
                )
                return (in_win, lbl.validity.open_ms, lbl.label_id)

            lbl = min(pos_matches, key=_pos_sort_key)
            v = lbl.validity
            is_premature = card.display_ms < v.open_ms or (
                v.closed_by is ClosedBy.ANSWERED
                and v.close_ms is not None
                and (v.close_ms - v.open_ms) <= settle_ms
                and (
                    card.display_ms < v.close_ms
                    or card.display_ms <= v.open_ms + settle_ms + premature_tolerance_ms
                )
            )
            if is_premature:
                premature_count += 1
            elif v.close_ms is not None and card.display_ms > v.close_ms:
                late_count += 1
                cb = str(v.closed_by) if v.closed_by is not None else "unspecified"
                late_by_closed_by[cb] += 1
            else:
                if lbl.label_id not in matched_pos_ids:
                    matched_pos_ids.add(lbl.label_id)
                    useful_count += 1
                    cat_useful[card.category.value] += 1
                else:
                    repetition_count += 1
            continue

        neg_match: LabelRecord | None = None
        for lbl in negatives:
            if card.session_id and lbl.session_id and card.session_id != lbl.session_id:
                continue
            lbl_segs = {r.segment_id for r in lbl.evidence_refs}
            if card_segs & lbl_segs:
                neg_match = lbl
                break

        if neg_match is not None:
            distracting_count += 1
            nr = str(neg_match.none_reason) if neg_match.none_reason is not None else "unspecified"
            distracting_by_none_reason[nr] += 1
        else:
            review_queue.append(
                {
                    "suggestion_id": card.suggestion_id,
                    "candidate_id": card.candidate_id,
                    "session_id": card.session_id,
                    "category": str(card.category),
                    "display_ms": card.display_ms,
                    "segment_ids": sorted(card_segs),
                }
            )

    if duration_ms is not None and duration_ms > 0:
        eff_dur_ms = duration_ms
    else:
        all_ts = [c.display_ms for c in cards] + [lbl.validity.open_ms for lbl in labels]
        eff_dur_ms = max(600_000, (max(all_ts) - min(all_ts)) if len(all_ts) >= 2 else 600_000)
    ten_min_units = float(eff_dur_ms) / 600_000.0

    pos_by_cat: Counter[str] = Counter()
    for lbl in positives:
        for cat in lbl.categories:
            pos_by_cat[cat.value] += 1
    per_category: dict[str, dict[str, Any]] = {}
    for cat_name in sorted(set(pos_by_cat) | set(cat_useful)):
        n_pos = int(pos_by_cat[cat_name])
        u_cnt = int(cat_useful[cat_name])
        low_n = n_pos < min_positives_for_ci
        per_category[cat_name] = {
            "n_positives": n_pos,
            "useful_count": u_cnt,
            "recall": (float(u_cnt) / float(n_pos)) if n_pos > 0 else None,
            "low_n": low_n,
            "insufficient_n": low_n,
        }

    total_pos = len(positives)
    return {
        "useful_count": useful_count,
        "repetition_count": repetition_count,
        "late_count": late_count,
        "late_by_closed_by": dict(sorted(late_by_closed_by.items())),
        "premature_count": premature_count,
        "distracting_count": distracting_count,
        "distracting_by_none_reason": dict(sorted(distracting_by_none_reason.items())),
        "unscored_pending_review_count": len(review_queue),
        "review_queue": review_queue,
        "useful_per_10_min": float(useful_count) / ten_min_units,
        "distracting_per_10_min": float(distracting_count) / ten_min_units,
        "cards_per_10_min": float(len(cards)) / ten_min_units,
        "recall": (float(len(matched_pos_ids)) / float(total_pos)) if total_pos > 0 else None,
        "displayed_count": len(cards),
        "positive_label_count": total_pos,
        "per_category": per_category,
    }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Output of deterministic policy-level replay simulation (§12.6)."""

    session_id: str
    run_id: str
    duration_ms: int
    displayed_cards: tuple[DisplayedCardEvent, ...]
    decisions_count: int
    dropped_count: int
    routed_count: int
    withdrawn_count: int
    expired_count: int
    latencies_ms: tuple[float, ...]


def _sample_stage_latency_ms(
    measured_ms: float,
    simulated_latency_ms: float | None,
    latency_distribution_ms: Sequence[float] | None,
    rng: random.Random,
) -> int:
    if simulated_latency_ms is not None:
        return max(0, int(round(simulated_latency_ms)))
    if latency_distribution_ms:
        return max(0, int(round(float(rng.choice(latency_distribution_ms)))))
    return max(0, int(round(measured_ms)))


def _has_resolution_cue(text: str, template: TemplateSpec | None) -> bool:
    low = text.lower()
    cues = list(_RESOLUTION_CUE_TERMS)
    if template is not None:
        cues.extend(c.lower() for c in template.resolution_cues)
    return any(cue in low for cue in cues)


def _store_decision_from_routing(
    store: SQLiteStore,
    *,
    run_id: str,
    context: DecisionContext,
    clf_res: Any,
    routing: Any,
    serialized_text: str | None,
) -> None:
    if routing.outcome is Outcome.NONE:
        t_state = "no_opportunity"
    elif routing.outcome is Outcome.ABSTAIN:
        t_state = "abstained"
    elif routing.outcome is Outcome.SUPPRESS:
        t_state = "suppressed"
    else:
        t_state = "ready"
    t_reason = routing.reason_codes[0] if routing.reason_codes else None
    store.record_decision(
        run_id=run_id,
        context=context,
        terminal_state=t_state,
        terminal_reason=t_reason,
        classifier=clf_res,
        routing=routing,
        serialized_text=serialized_text,
    )


async def async_simulate_replay(
    brief: SessionBrief,
    events: Sequence[CanonicalEvent],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    *,
    calibration: CalibrationModel | CalibrationRegistry | None = None,
    templates: Mapping[str, TemplateSpec] | None = None,
    store: SQLiteStore | None = None,
    run_id: str = "replay",
    mode: Mode = Mode.PROACTIVE,
    provider: ProviderAdapter | None = None,
    health: HealthState | None = None,
    simulated_latency_ms: float | None = None,
    latency_distribution_ms: Sequence[float] | None = None,
    rng_seed: int = 0,
    cache_classifier: bool = True,
) -> ReplayResult:
    """Run deterministic policy-level replay simulation with honest latency accounting (§12.6)."""
    ordered = sorted(events, key=lambda ev: (ev.received_ms, ev.source_sequence, ev.event_id))
    state = CanonicalState(brief)
    tpl_map = dict(templates) if templates is not None else load_templates_dir()
    cal_obj: CalibrationModel | CalibrationRegistry | None = (
        calibration
        if calibration is not None
        else (default_calibration_registry() if policy.classifier.calibration else None)
    )
    eff_health = (
        health
        if health is not None
        else (provider.health_state if provider is not None else HealthState.TEMPLATE_ONLY)
    )
    rng = random.Random(rng_seed)

    if store is not None:
        store.upsert_session(
            brief=brief,
            run_id=run_id,
            runtime_mode=mode,
            effective_max_cards_per_10_min=effective_max_cards_per_10_min(brief, policy),
        )

    speaker_update_times: dict[str, int] = {}
    speaker_update_keys: dict[str, tuple[int, int, str]] = {}
    boundary_events: list[tuple[int, str | None]] = []
    assigned_segments: set[str] = set()
    displayed_cards: list[DisplayedCardEvent] = []
    displayed_times: list[int] = []
    latencies_ms: list[float] = []
    decisions_count = 0
    dropped_count = 0
    routed_count = 0
    withdrawn_count = 0
    expired_count = 0
    ledger_texts: dict[str, str] = {}
    ledger_categories: dict[str, Category] = {}
    prior_anchor_texts: dict[str, str] = {}
    last_display_ms: int | None = None
    cap = effective_max_cards_per_10_min(brief, policy)
    recipient_ids = {"ME", brief.recipient.participant_id}

    # Pre-index future boundaries & events by received_ms for lookahead-bounded revalidation
    for ev in ordered:
        if ev.kind is EventKind.BOUNDARY:
            boundary_events.append((ev.received_ms, ev.speaker_id))

    for ev_idx, ev in enumerate(ordered):
        apply_res = state.apply(ev)
        prefix_snap = state.snapshot()
        if store is not None and apply_res.accepted:
            can_rev: int | None = None
            if ev.segment_id and ev.segment_id in prefix_snap.segments_by_id:
                can_rev = prefix_snap.segments_by_id[ev.segment_id].revision
            store.record_canonical_event(
                run_id=run_id,
                event=ev,
                retention=brief.retention,
                canonical_revision=can_rev,
            )
        if apply_res.accepted and ev.kind is EventKind.SPEAKER_UPDATE and ev.segment_id:
            speaker_update_times[ev.segment_id] = ev.received_ms
            speaker_update_keys[ev.segment_id] = (
                ev.source_revision if ev.source_revision is not None else 0,
                ev.source_sequence,
                ev.event_id,
            )
        if (
            ev.kind is not EventKind.SEGMENT
            or ev.stability is not Stability.COMMITTED
            or not ev.segment_id
            or ev.segment_id in assigned_segments
        ):
            continue
        seg = prefix_snap.segments_by_id.get(ev.segment_id)
        if seg is None or seg.speaker_id in recipient_ids:
            continue

        assigned_segments.add(seg.segment_id)
        built_items = build_decision_contexts(
            prefix_snap,
            anchor_segments=[seg],
            policy=policy,
            tokenizer=classifier.tokenizer,
        )
        for built in built_items:
            decisions_count += 1
            if built.dropped:
                dropped_count += 1
                if store is not None and built.drop_record is not None:
                    drop = built.drop_record
                    dummy_ctx = DecisionContext(
                        schema_version=SCHEMA_VERSION,
                        candidate_id=drop.candidate_id,
                        session_id=drop.session_id,
                        origin=Origin.PROACTIVE,
                        based_on_state_version=drop.based_on_state_version,
                        builder_version=policy.classifier.version,
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
                    store.record_decision(
                        run_id=run_id,
                        context=dummy_ctx,
                        terminal_state="dropped",
                        terminal_reason=drop.reason_code,
                    )
                continue

            assert built.context is not None and built.serialized_text is not None
            ctx = built.context
            c_in = ClassifierInput(context=ctx, serialized_text=built.serialized_text)
            t0 = time.perf_counter()
            if cache_classifier and store is not None:
                clf_list = await classify_batch_cached(classifier, [c_in], store=store)
            else:
                clf_list = await classifier.classify_batch([c_in])
            measured_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
            clf_raw = clf_list[0]
            if cal_obj is not None and not isinstance(clf_raw, ClassifierErrorResult):
                clf_res = cal_obj.calibrate(ctx, clf_raw, mode=prefix_snap.brief.mode)
            else:
                clf_res = clf_raw
            stage_ms = _sample_stage_latency_ms(
                measured_ms, simulated_latency_ms, latency_distribution_ms, rng
            )
            latencies_ms.append(float(stage_ms))

            routing = route_candidate(
                ctx,
                clf_res,
                prefix_snap,
                policy,
                mode=mode,
                health=eff_health,
                now_ms=ctx.anchor_received_ms + stage_ms,
                speaker_update_times_ms=speaker_update_times,
                ledger_texts=ledger_texts,
                ledger_categories=ledger_categories,
                prior_anchor_texts=prior_anchor_texts,
                recent_displayed_ms=displayed_times,
            )
            if routing.outcome is not Outcome.ROUTE or routing.category is None:
                if store is not None:
                    _store_decision_from_routing(
                        store,
                        run_id=run_id,
                        context=ctx,
                        clf_res=clf_res,
                        routing=routing,
                        serialized_text=built.serialized_text,
                    )
                continue

            routed_count += 1
            effective_routing = routing
            cpol = policy.categories[routing.category]
            tpl = tpl_map.get(cpol.template) if cpol.template else None
            ready_ms = ctx.anchor_received_ms + stage_ms
            cand: SuggestionCandidate
            if routing.route is Route.GENERATOR:
                if provider is not None:
                    orig_sink = provider.egress_sink
                    orig_clock = provider.clock_ms
                    egress_recorded = False

                    def _replay_clock_ms(_ready_ms: int = ready_ms) -> int:
                        return _ready_ms

                    def _on_egress(
                        rec: EgressRecord,
                        _cand_id: str = ctx.candidate_id,
                        _orig_sink: Callable[[EgressRecord], None] | None = orig_sink,
                    ) -> None:
                        nonlocal egress_recorded
                        if rec.candidate_id == _cand_id:
                            egress_recorded = True
                        if store is not None:
                            store.record_egress(run_id=run_id, record=rec)
                        if _orig_sink is not None:
                            _orig_sink(rec)

                    provider.egress_sink = _on_egress
                    provider.clock_ms = _replay_clock_ms
                    try:
                        gen_out = await generate_question(
                            ctx,
                            routing.category,
                            prefix_snap,
                            policy,
                            provider,
                            clock_ms=_replay_clock_ms,
                            store=store,
                            templates=tpl_map,
                            ledger_texts=ledger_texts,
                            ledger_categories=ledger_categories,
                        )
                    finally:
                        provider.egress_sink = orig_sink
                        provider.clock_ms = orig_clock

                    if gen_out.candidate is None:
                        if store is not None:
                            store.record_decision(
                                run_id=run_id,
                                context=ctx,
                                terminal_state="unrendered",
                                terminal_reason=(
                                    gen_out.terminal_reason or ReasonCode.GENERATOR_INVALID_OUTPUT
                                ),
                                classifier=clf_res,
                                routing=effective_routing,
                                serialized_text=built.serialized_text,
                            )
                        continue
                    cand = gen_out.candidate
                    if gen_out.fallback_used and not egress_recorded:
                        effective_routing = replace(routing, route=cand.route)
                elif tpl is not None:
                    effective_routing = replace(routing, route=Route.TEMPLATE)
                    rendered = render_template(
                        ctx,
                        routing.category,
                        prefix_snap,
                        tpl,
                        now_ms=ready_ms,
                        ready_ms=ready_ms,
                        display_ttl_ms=policy.display.display_ttl_ms,
                    )
                    if rendered.candidate is None:
                        if store is not None:
                            store.record_decision(
                                run_id=run_id,
                                context=ctx,
                                terminal_state="unrendered",
                                terminal_reason=(
                                    rendered.unrendered_reason or ReasonCode.SLOT_UNAVAILABLE
                                ),
                                classifier=clf_res,
                                routing=effective_routing,
                                serialized_text=built.serialized_text,
                            )
                        continue
                    cand = rendered.candidate
                else:
                    if store is not None:
                        store.record_decision(
                            run_id=run_id,
                            context=ctx,
                            terminal_state="unrendered",
                            terminal_reason=ReasonCode.INSUFFICIENT_BUDGET,
                            classifier=clf_res,
                            routing=effective_routing,
                            serialized_text=built.serialized_text,
                        )
                    continue
            else:
                if tpl is None:
                    if store is not None:
                        store.record_decision(
                            run_id=run_id,
                            context=ctx,
                            terminal_state="unrendered",
                            terminal_reason=ReasonCode.SLOT_UNAVAILABLE,
                            classifier=clf_res,
                            routing=effective_routing,
                            serialized_text=built.serialized_text,
                        )
                    continue

                rendered = render_template(
                    ctx,
                    routing.category,
                    prefix_snap,
                    tpl,
                    now_ms=ready_ms,
                    ready_ms=ready_ms,
                    display_ttl_ms=policy.display.display_ttl_ms,
                )
                if rendered.candidate is None:
                    if store is not None:
                        store.record_decision(
                            run_id=run_id,
                            context=ctx,
                            terminal_state="unrendered",
                            terminal_reason=(
                                rendered.unrendered_reason or ReasonCode.SLOT_UNAVAILABLE
                            ),
                            classifier=clf_res,
                            routing=effective_routing,
                            serialized_text=built.serialized_text,
                        )
                    continue
                cand = rendered.candidate
            settle_end = ctx.anchor_received_ms + policy.display.settle_ms
            bnd_hits = [
                ts
                for ts, b_spk in boundary_events
                if ctx.anchor_received_ms < ts <= settle_end
                and (b_spk is None or b_spk == seg.speaker_id)
            ]
            if bnd_hits:
                target_ms = max(ready_ms, min(bnd_hits))
                disp_cause = ReasonCode.BOUNDARY_AFTER_SETTLE
            else:
                target_ms = max(ready_ms, settle_end)
                disp_cause = ReasonCode.SETTLE_TIMEOUT

            if (
                last_display_ms is not None
                and target_ms < last_display_ms + policy.display.cooldown_ms
            ):
                target_ms = last_display_ms + policy.display.cooldown_ms

            if target_ms > ctx.display_by_ms:
                expired_count += 1
                if store is not None:
                    store.record_decision(
                        run_id=run_id,
                        context=ctx,
                        terminal_state="expired",
                        terminal_reason=ReasonCode.DISPLAY_BY_PASSED,
                        classifier=clf_res,
                        routing=effective_routing,
                        serialized_text=built.serialized_text,
                    )
                continue

            in_win = [t for t in displayed_times if (target_ms - 600_000) <= t <= target_ms]
            if cap <= 0 or len(in_win) >= cap:
                if store is not None:
                    store.record_decision(
                        run_id=run_id,
                        context=ctx,
                        terminal_state="suppressed",
                        terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                        classifier=clf_res,
                        routing=effective_routing,
                        serialized_text=built.serialized_text,
                    )
                continue

            # 7-step revalidation against events occurring in (anchor_received_ms, target_ms]
            ev_seg_ids = {r.segment_id for r in cand.evidence_refs}
            withdraw_reason: ReasonCode | None = None
            for fut_ev in ordered[ev_idx + 1 :]:
                if fut_ev.received_ms > target_ms:
                    break
                if (
                    fut_ev.kind is EventKind.SEGMENT
                    and fut_ev.stability is Stability.COMMITTED
                    and fut_ev.segment_id in ev_seg_ids
                    and fut_ev.source_revision is not None
                ):
                    ref_seg = prefix_snap.segments_by_id.get(fut_ev.segment_id)
                    if ref_seg is not None and fut_ev.source_revision > ref_seg.source_revision:
                        withdraw_reason = ReasonCode.EVIDENCE_REVISION_CHANGED
                        break
                if fut_ev.kind is EventKind.SPEAKER_UPDATE and fut_ev.segment_id in ev_seg_ids:
                    fut_key = (
                        fut_ev.source_revision if fut_ev.source_revision is not None else 0,
                        fut_ev.source_sequence,
                        fut_ev.event_id,
                    )
                    prev_key = speaker_update_keys.get(fut_ev.segment_id)
                    ref_seg = prefix_snap.segments_by_id.get(fut_ev.segment_id)
                    if (
                        (prev_key is None or fut_key > prev_key)
                        and ref_seg is not None
                        and (
                            (
                                fut_ev.speaker_id is not None
                                and fut_ev.speaker_id != ref_seg.speaker_id
                            )
                            or fut_ev.speaker_uncertain != ref_seg.speaker_uncertain
                        )
                    ):
                        withdraw_reason = ReasonCode.EVIDENCE_REVISION_CHANGED
                        break
                if (
                    fut_ev.kind is EventKind.BRIEF_UPDATE
                    and fut_ev.brief_patch is not None
                    and "objective" in fut_ev.brief_patch
                    and str(fut_ev.brief_patch["objective"]) != prefix_snap.brief.objective
                ):
                    withdraw_reason = ReasonCode.OBJECTIVE_CHANGED
                    break
                if (
                    fut_ev.kind is EventKind.SEGMENT
                    and fut_ev.stability is Stability.COMMITTED
                    and fut_ev.text
                    and _has_resolution_cue(fut_ev.text, tpl)
                ):
                    withdraw_reason = ReasonCode.POSSIBLY_RESOLVED
                    break

            if withdraw_reason is not None:
                withdrawn_count += 1
                if store is not None:
                    store.record_decision(
                        run_id=run_id,
                        context=ctx,
                        terminal_state="withdrawn",
                        terminal_reason=withdraw_reason,
                        classifier=clf_res,
                        routing=effective_routing,
                        serialized_text=built.serialized_text,
                    )
                continue

            displayed_times.append(target_ms)
            last_display_ms = target_ms
            ledger_texts[cand.suggestion_id] = cand.text
            ledger_categories[cand.suggestion_id] = cand.category
            anchor_parts: list[str] = []
            for a_ref in ctx.anchor_refs:
                a_seg = prefix_snap.segments_by_id.get(a_ref.segment_id)
                if a_seg is not None and a_seg.text.strip():
                    anchor_parts.append(a_seg.text.strip())
            if anchor_parts:
                prior_anchor_texts[cand.candidate_id] = " ".join(anchor_parts)
            state.register_suggestion(cand)
            state.apply(
                LedgerEvent(
                    schema_version=SCHEMA_VERSION,
                    event_id=f"ev-disp-{cand.suggestion_id}",
                    suggestion_id=cand.suggestion_id,
                    transition=LedgerState.DISPLAYED,
                    cause=disp_cause,
                    at_ms=target_ms,
                    evidence_refs=cand.evidence_refs,
                ).to_canonical_event(brief.session_id, received_ms=target_ms)
            )
            displayed_cards.append(
                DisplayedCardEvent(
                    suggestion_id=cand.suggestion_id,
                    candidate_id=cand.candidate_id,
                    session_id=brief.session_id,
                    category=cand.category,
                    display_ms=target_ms,
                    evidence_refs=cand.evidence_refs,
                    text=cand.text,
                    route=cand.route,
                    origin=cand.origin,
                )
            )
            if store is not None:
                store.record_decision(
                    run_id=run_id,
                    context=ctx,
                    terminal_state="expired",
                    terminal_reason=ReasonCode.DISPLAY_TTL_PASSED,
                    classifier=clf_res,
                    routing=effective_routing,
                    serialized_text=built.serialized_text,
                )
                tr_disp = LedgerEvent(
                    schema_version=SCHEMA_VERSION,
                    event_id=f"ev-disp-{cand.suggestion_id}",
                    suggestion_id=cand.suggestion_id,
                    transition=LedgerState.DISPLAYED,
                    cause=disp_cause,
                    at_ms=target_ms,
                    evidence_refs=cand.evidence_refs,
                )
                tr_exp = LedgerEvent(
                    schema_version=SCHEMA_VERSION,
                    event_id=f"ev-exp-{cand.suggestion_id}",
                    suggestion_id=cand.suggestion_id,
                    transition=LedgerState.EXPIRED,
                    cause=ReasonCode.DISPLAY_TTL_PASSED,
                    at_ms=target_ms + policy.display.display_ttl_ms,
                    evidence_refs=cand.evidence_refs,
                )
                store.record_suggestion(
                    session_id=brief.session_id,
                    run_id=run_id,
                    suggestion=cand,
                    current_state="expired",
                    terminal_reason=ReasonCode.DISPLAY_TTL_PASSED,
                    transitions=[tr_disp.to_dict(), tr_exp.to_dict()],
                )

    final_snap = state.snapshot()
    if store is not None:
        store.upsert_session(
            brief=final_snap.brief,
            run_id=run_id,
            runtime_mode=mode,
            effective_max_cards_per_10_min=effective_max_cards_per_10_min(final_snap.brief, policy),
            reducer_counters=final_snap.counters,
            revision_history=final_snap.revision_history,
        )

    rx_times = [e.received_ms for e in ordered]
    dur_ms = max(1, (max(rx_times) - min(rx_times)) if len(rx_times) >= 2 else 1)
    return ReplayResult(
        session_id=brief.session_id,
        run_id=run_id,
        duration_ms=dur_ms,
        displayed_cards=tuple(displayed_cards),
        decisions_count=decisions_count,
        dropped_count=dropped_count,
        routed_count=routed_count,
        withdrawn_count=withdrawn_count,
        expired_count=expired_count,
        latencies_ms=tuple(latencies_ms),
    )


def simulate_replay(
    brief: SessionBrief,
    events: Sequence[CanonicalEvent],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    **kwargs: Any,
) -> ReplayResult:
    """Synchronous wrapper for async_simulate_replay."""
    return asyncio.run(async_simulate_replay(brief, events, policy, classifier, **kwargs))


def _policy_with_t_display(policy: PolicyConfig, t_display: float) -> PolicyConfig:
    t_disp = min(1.0, max(0.0, float(t_display)))
    new_cats: dict[Category, CategoryPolicy] = {}
    for cat, cpol in policy.categories.items():
        t_comp = min(cpol.t_compute, t_disp)
        t_low = min(cpol.t_low, t_comp)
        new_cats[cat] = replace(cpol, t_low=t_low, t_compute=t_comp, t_display=t_disp)
    return replace(policy, categories=new_cats)


async def async_sweep_t_display(
    sessions: Sequence[tuple[SessionBrief, Sequence[CanonicalEvent]]],
    labels: Sequence[LabelRecord],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    t_display_values: Sequence[float],
    *,
    calibration: CalibrationModel | CalibrationRegistry | None = None,
    templates: Mapping[str, TemplateSpec] | None = None,
    store: SQLiteStore | None = None,
    simulated_latency_ms: float | None = None,
    latency_distribution_ms: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Sweep `t_display` through policy-level replay under effective_max_cards_per_10_min (§12.8)."""
    if not t_display_values:
        raise ValueError("t_display_values must be non-empty")
    curve: list[dict[str, Any]] = []
    for t_disp in sorted(float(v) for v in t_display_values):
        swept_policy = _policy_with_t_display(policy, t_disp)
        all_cards: list[DisplayedCardEvent] = []
        total_dur_ms = 0
        max_cap_10m = 0
        for idx, (brief, events) in enumerate(sessions):
            res = await async_simulate_replay(
                brief,
                events,
                swept_policy,
                classifier,
                calibration=calibration,
                templates=templates,
                store=store,
                run_id=f"sweep-{t_disp:.4f}-{idx}",
                simulated_latency_ms=simulated_latency_ms,
                latency_distribution_ms=latency_distribution_ms,
            )
            all_cards.extend(res.displayed_cards)
            total_dur_ms += max(600_000, res.duration_ms)
            max_cap_10m = max(max_cap_10m, effective_max_cards_per_10_min(brief, swept_policy))

        summary = match_opportunities(
            all_cards,
            labels,
            settle_ms=swept_policy.display.settle_ms,
            duration_ms=total_dur_ms if sessions else None,
        )
        curve.append(
            {
                "t_display": t_disp,
                "useful_count": summary["useful_count"],
                "distracting_count": summary["distracting_count"],
                "displayed_count": summary["displayed_count"],
                "useful_per_10_min": summary["useful_per_10_min"],
                "distracting_per_10_min": summary["distracting_per_10_min"],
                "cards_per_10_min": summary["cards_per_10_min"],
                "effective_max_cards_per_10_min": max_cap_10m,
            }
        )
    return curve


def sweep_t_display(
    sessions: Sequence[tuple[SessionBrief, Sequence[CanonicalEvent]]],
    labels: Sequence[LabelRecord],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    t_display_values: Sequence[float],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for async_sweep_t_display."""
    return asyncio.run(
        async_sweep_t_display(sessions, labels, policy, classifier, t_display_values, **kwargs)
    )


def paired_conversation_bootstrap(
    cards_a: Sequence[DisplayedCardEvent | Mapping[str, Any]],
    cards_b: Sequence[DisplayedCardEvent | Mapping[str, Any]],
    labels: Sequence[LabelRecord],
    *,
    metric: str = "useful_count",
    settle_ms: int = 1500,
    n_resamples: int = 1000,
    min_positives_for_ci: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired whole-conversation bootstrap comparison with low-N flagging (§12.7)."""
    coerced_a = [_coerce_displayed_card(c) for c in cards_a]
    coerced_b = [_coerce_displayed_card(c) for c in cards_b]
    session_ids = sorted(
        {c.session_id for c in coerced_a if c.session_id}
        | {c.session_id for c in coerced_b if c.session_id}
        | {lbl.session_id for lbl in labels if lbl.session_id}
    )
    if not session_ids:
        return {
            "metric": metric,
            "n_conversations": 0,
            "n_resamples": n_resamples,
            "mean_delta": 0.0,
            "ci_low": None,
            "ci_high": None,
            "low_n": True,
            "insufficient_n": True,
            "per_category": {},
        }

    per_sess_deltas: list[float] = []
    all_cats: set[str] = {c.value for lbl in labels if lbl.worthwhile for c in lbl.categories}
    all_cats |= {c.category.value for c in coerced_a} | {c.category.value for c in coerced_b}
    cat_deltas: dict[str, list[float]] = {cat: [] for cat in sorted(all_cats)}
    pos_counts: Counter[str] = Counter()
    for lbl in labels:
        if lbl.worthwhile:
            for c in lbl.categories:
                pos_counts[c.value] += 1

    for sid in session_ids:
        ca = [c for c in coerced_a if c.session_id == sid]
        cb = [c for c in coerced_b if c.session_id == sid]
        sl = [lbl for lbl in labels if lbl.session_id == sid]
        sum_a = match_opportunities(
            ca, sl, settle_ms=settle_ms, min_positives_for_ci=min_positives_for_ci
        )
        sum_b = match_opportunities(
            cb, sl, settle_ms=settle_ms, min_positives_for_ci=min_positives_for_ci
        )
        val_a = float(sum_a.get(metric) or 0.0)
        val_b = float(sum_b.get(metric) or 0.0)
        per_sess_deltas.append(val_b - val_a)
        for cat in cat_deltas:
            ua = float((sum_a["per_category"].get(cat) or {}).get("useful_count", 0))
            ub = float((sum_b["per_category"].get(cat) or {}).get("useful_count", 0))
            cat_deltas[cat].append(ub - ua)

    rng = random.Random(seed)
    n_conv = len(session_ids)
    boot_means: list[float] = []
    for _ in range(max(1, n_resamples)):
        sample = [per_sess_deltas[rng.randrange(n_conv)] for _ in range(n_conv)]
        boot_means.append(sum(sample) / float(n_conv))
    arr = np.asarray(boot_means, dtype=np.float64)
    ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
    total_pos = sum(1 for lbl in labels if lbl.worthwhile)
    overall_low_n = total_pos < min_positives_for_ci or n_conv < 2

    per_category: dict[str, dict[str, Any]] = {}
    for idx, cat in enumerate(sorted(cat_deltas), start=1):
        n_pos = int(pos_counts[cat])
        is_low = n_pos < min_positives_for_ci or n_conv < 2
        deltas = cat_deltas[cat]
        mean_c = sum(deltas) / float(n_conv)
        if is_low:
            per_category[cat] = {
                "n_positives": n_pos,
                "mean_delta": mean_c,
                "ci_low": None,
                "ci_high": None,
                "low_n": True,
                "insufficient_n": True,
            }
        else:
            c_rng = random.Random(seed + idx)
            c_boot = [
                sum(deltas[c_rng.randrange(n_conv)] for _ in range(n_conv)) / float(n_conv)
                for _ in range(max(1, n_resamples))
            ]
            c_lo, c_hi = np.percentile(np.asarray(c_boot, dtype=np.float64), [2.5, 97.5])
            per_category[cat] = {
                "n_positives": n_pos,
                "mean_delta": mean_c,
                "ci_low": float(c_lo),
                "ci_high": float(c_hi),
                "low_n": False,
                "insufficient_n": False,
            }

    return {
        "metric": metric,
        "n_conversations": n_conv,
        "n_resamples": n_resamples,
        "mean_delta": sum(per_sess_deltas) / float(n_conv),
        "ci_low": None if overall_low_n else float(ci_lo),
        "ci_high": None if overall_low_n else float(ci_hi),
        "low_n": overall_low_n,
        "insufficient_n": overall_low_n,
        "per_category": per_category,
    }


def load_benchmark_inputs_jsonl(
    path: str | Path,
    *,
    brief: SessionBrief | None = None,
    policy: PolicyConfig | None = None,
) -> tuple[list[ClassifierInput], StateSnapshot | None]:
    """Load real serialized DecisionContexts (or build from CanonicalEvent JSONL via context_builder)."""
    src = Path(path)
    inputs: list[ClassifierInput] = []
    events: list[CanonicalEvent] = []
    loaded_brief = brief

    with src.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, Mapping):
                raise ValueError(f"JSONL line {line_no} must be a JSON object")
            if "context" in obj and "serialized_text" in obj:
                ctx = DecisionContext.from_dict(obj["context"])
                inputs.append(
                    ClassifierInput(context=ctx, serialized_text=str(obj["serialized_text"]))
                )
            elif "objective" in obj and "recipient" in obj and "kind" not in obj:
                loaded_brief = SessionBrief.from_dict(obj)
            elif "kind" in obj:
                events.append(CanonicalEvent.from_dict(obj))
            else:
                raise ValueError(
                    f"Unsupported benchmark JSONL record at line {line_no}; expected "
                    "{context, serialized_text} or CanonicalEvent"
                )

    if inputs:
        return (inputs, None)
    if not events:
        raise ValueError(f"No benchmark contexts or events found in {src}")
    if loaded_brief is None:
        raise ValueError("Benchmark event JSONL requires a SessionBrief header or brief parameter")
    state = CanonicalState(loaded_brief)
    for ev in sorted(events, key=lambda e: (e.received_ms, e.source_sequence, e.event_id)):
        state.apply(ev)
    snap = state.snapshot()
    built = build_decision_contexts(snap, policy=policy)
    inputs = [ClassifierInput.from_build_result(b) for b in built if not b.dropped]
    if not inputs:
        raise ValueError("All contexts were dropped by prefilter in benchmark JSONL")
    return (inputs, snap)


def _bucket_context_to_tokens(
    item: ClassifierInput,
    target_tokens: int,
    *,
    tokenizer: Any,
    snapshot: StateSnapshot | None,
    policy: PolicyConfig | None,
) -> ClassifierInput:
    tok = tokenizer or WhitespaceTokenizer()
    if snapshot is not None:
        anchor_segs = [
            snapshot.segments_by_id[r.segment_id]
            for r in item.context.anchor_refs
            if r.segment_id in snapshot.segments_by_id
        ]
        if anchor_segs:
            res = build_decision_context(
                snapshot,
                anchor_segs,
                policy=policy,
                origin=item.context.origin,
                tokenizer=tok,
                input_limit_tokens=target_tokens,
            )
            return ClassifierInput.from_build_result(res)

    # Fallback when only serialized ClassifierInput rows are supplied: trim [CONTEXT] turns oldest-first
    lines = item.serialized_text.splitlines()
    if tok.count_tokens(item.serialized_text) <= target_tokens:
        return item
    ctx_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "[CONTEXT]"), None)
    anch_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("[ANCHOR]")), None)
    if ctx_idx is not None and anch_idx is not None and ctx_idx + 1 < anch_idx:
        head = lines[: ctx_idx + 1]
        turns = lines[ctx_idx + 1 : anch_idx]
        tail = lines[anch_idx:]
        while turns and tok.count_tokens("\n".join([*head, *turns, *tail])) > target_tokens:
            turns.pop(0)
        new_lines = [*head, *turns, *tail] if turns else [*lines[:ctx_idx], *tail]
        new_text = "\n".join(new_lines)
        import hashlib

        new_sha = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
        new_ctx = replace(
            item.context,
            serialized_sha256=new_sha,
            token_counts={
                **item.context.token_counts,
                "total": tok.count_tokens(new_text),
                "limit": target_tokens,
            },
        )
        return ClassifierInput(context=new_ctx, serialized_text=new_text)
    return item


def _peak_rss_mb() -> float:
    kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max(0.001, kb / 1024.0)


def _peak_vram_mb() -> float | None:
    try:
        torch_mod = importlib.import_module("torch")
        if torch_mod.cuda.is_available():
            return float(torch_mod.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


async def async_run_latency_benchmark(
    contexts: Sequence[ContextBuildResult | ClassifierInput] | str | Path,
    classifier: OpportunityClassifier,
    *,
    token_buckets: Sequence[int] = (128, 256, 512, 1024, 2048),
    batch_sizes: Sequence[int] = (1, 4, 8),
    warmup_calls: int = 2,
    timed_calls: int = 20,
    contention: str = "idle",
    placement: str = "in_process",
    precision: str = "fp32",
    snapshot: StateSnapshot | None = None,
    policy: PolicyConfig | None = None,
    brief: SessionBrief | None = None,
) -> dict[str, Any]:
    """Run §10.2 latency & memory benchmark over real serialized contexts."""
    if warmup_calls < 1:
        raise ValueError("warmup_calls must be >= 1")
    if timed_calls < 1:
        raise ValueError("timed_calls must be >= 1")

    snap = snapshot
    if isinstance(contexts, (str, Path)):
        raw_inputs, loaded_snap = load_benchmark_inputs_jsonl(contexts, brief=brief, policy=policy)
        if snap is None:
            snap = loaded_snap
    else:
        raw_inputs = [
            item if isinstance(item, ClassifierInput) else ClassifierInput.from_build_result(item)
            for item in contexts
        ]
    if not raw_inputs:
        raise ValueError("Benchmark requires at least one real DecisionContext")

    cells: list[dict[str, Any]] = []
    for tok_budget in token_buckets:
        bucketed = [
            _bucket_context_to_tokens(
                it,
                int(tok_budget),
                tokenizer=classifier.tokenizer,
                snapshot=snap,
                policy=policy,
            )
            for it in raw_inputs
        ]
        for bsz in batch_sizes:
            batch_sz = max(1, int(bsz))
            batch = [bucketed[i % len(bucketed)] for i in range(batch_sz)]

            for _ in range(warmup_calls):
                await classifier.classify_batch(batch)

            samples_ms: list[float] = []
            for _ in range(timed_calls):
                t0 = time.perf_counter()
                # Include tokenization + SHA-256 verification + inference in timed lane call (§10.2)
                verified = [
                    ClassifierInput(context=it.context, serialized_text=it.serialized_text)
                    for it in batch
                ]
                for v_it in verified:
                    classifier.tokenizer.count_tokens(v_it.serialized_text)
                await classifier.classify_batch(verified)
                elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
                samples_ms.append(elapsed)

            arr = np.asarray(samples_ms, dtype=np.float64)
            p50, p95, p99 = np.percentile(arr, [50.0, 95.0, 99.0])
            max_v = float(np.max(arr))
            # Guard float rounding monotonicity p50 <= p95 <= p99 <= max
            p95 = max(p50, p95)
            p99 = max(p95, p99)
            max_v = max(p99, max_v)
            cells.append(
                {
                    "tokens": int(tok_budget),
                    "batch_size": batch_sz,
                    "p50_ms": float(p50),
                    "p95_ms": float(p95),
                    "p99_ms": float(p99),
                    "max_ms": float(max_v),
                    "peak_rss_mb": _peak_rss_mb(),
                    "peak_vram_mb": _peak_vram_mb(),
                    "warmup_discarded": int(warmup_calls),
                    "timed_calls": int(timed_calls),
                    "contention": contention,
                    "placement": placement,
                    "precision": precision,
                }
            )

    return {
        "classifier_version": classifier.classifier_version,
        "contention": contention,
        "placement": placement,
        "precision": precision,
        "cells": cells,
    }


def run_latency_benchmark(
    contexts: Sequence[ContextBuildResult | ClassifierInput] | str | Path,
    classifier: OpportunityClassifier,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synchronous wrapper for async_run_latency_benchmark."""
    return asyncio.run(async_run_latency_benchmark(contexts, classifier, **kwargs))
