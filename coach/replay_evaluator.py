"""Replay evaluator, dataset splitters, validity matcher, threshold sweeper, bootstrap, and latency benchmark (§10.2, §12.5–§12.8, §15 level 12)."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import random
import re
import resource
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from coach.calibration import CalibrationModel, CalibrationRegistry, default_calibration_registry
from coach.candidate_validator import validate_template_candidate
from coach.canonical_state import CanonicalState, StateSnapshot
from coach.context_builder import (
    _SENTENCE_FINAL_RE,
    DEFAULT_ANCHOR_MAX_TOKENS,
    PREFILTER_VERSION,
    AnchorSpan,
    CategoryPolicy,
    ContextBuildResult,
    CoverageCursor,
    PolicyConfig,
    WhitespaceTokenizer,
    build_decision_context,
    build_decision_contexts,
    coalesce_anchors,
    effective_max_cards_per_10_min,
    normalize_for_matching,
    run_prefilter,
)
from coach.contracts import (
    SCHEMA_VERSION,
    SUPPRESS_REASON_CODES,
    AttentionConfig,
    CanonicalEvent,
    Category,
    ClassifierResult,
    ClosedBy,
    ControlSubtype,
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
    RecipientConfig,
    RetentionConfig,
    Route,
    RoutingDecision,
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
from coach.provider_adapter import BreakerState, ProviderAdapter
from coach.question_generator import generate_question
from coach.routing_policy import is_equivalent_question, matches_resolution_cue, route_candidate
from coach.storage import SQLiteStore
from coach.template_renderer import (
    TemplateSpec,
    load_templates_dir,
    render_template,
)

_SHOWN_OR_ACTED_STATES: frozenset[LedgerState] = frozenset(
    {
        LedgerState.DISPLAYED,
        LedgerState.PINNED,
        LedgerState.DISMISSED,
        LedgerState.USED,
    }
)
_TERMINAL_LEDGER_STATES: frozenset[LedgerState] = frozenset(
    {
        LedgerState.DISMISSED,
        LedgerState.RESOLVED,
        LedgerState.WITHDRAWN,
        LedgerState.EXPIRED,
        LedgerState.OUTDATED,
    }
)
_REPLAY_STOPWORDS: frozenset[str] = frozenset(
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


@dataclass(slots=True)
class _ReplayPendingItem:
    cand: SuggestionCandidate
    context: DecisionContext
    classifier: Any
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
class _ReplayActiveItem:
    cand: SuggestionCandidate
    context: DecisionContext
    classifier: Any
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


def _replay_content_tokens(text: str) -> set[str]:
    return {t for t in normalize_for_matching(text).split() if t and t not in _REPLAY_STOPWORDS}


def _replay_brief_material_key(brief: SessionBrief) -> tuple[Any, ...]:
    return (
        brief.objective,
        tuple((f.key, f.value) for f in brief.known_facts),
        brief.do_not_suggest,
        brief.mode,
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


def _get_direct_field(item: Any, *names: str) -> str | None:
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


def _get_field(item: Any, *names: str) -> str | None:
    direct = _get_direct_field(item, *names)
    if direct is not None:
        return direct
    ctx = item.get("context") if isinstance(item, Mapping) else getattr(item, "context", None)
    if ctx is not None:
        return _get_direct_field(ctx, *names)
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


def _replay_anchor_sort_key(
    ctx: DecisionContext, final_snap: StateSnapshot
) -> tuple[int, int, int, int, str]:
    if not ctx.anchor_refs:
        return (ctx.anchor_received_ms, 0, 0, ctx.anchor_received_ms, ctx.candidate_id)
    min_start = None
    min_chan = None
    min_seq = None
    for ref in ctx.anchor_refs:
        seg = final_snap.segments_by_id.get(ref.segment_id)
        if seg is None:
            continue
        k = (seg.start_ms, seg.channel, seg.source_sequence)
        if min_start is None or k < (min_start, min_chan or 0, min_seq or 0):
            min_start, min_chan, min_seq = k
    if min_start is None:
        return (ctx.anchor_received_ms, 0, 0, ctx.anchor_received_ms, ctx.candidate_id)
    assert min_chan is not None and min_seq is not None
    return (min_start, min_chan, min_seq, ctx.anchor_received_ms, ctx.candidate_id)


class _ReplaySim:
    """Causal discrete-event simulator matching SuggestionController semantics."""

    def __init__(
        self,
        *,
        brief: SessionBrief,
        ordered: list[CanonicalEvent],
        policy: PolicyConfig,
        classifier: OpportunityClassifier,
        cal_obj: CalibrationModel | CalibrationRegistry | None,
        tpl_map: Mapping[str, TemplateSpec],
        store: SQLiteStore | None,
        run_id: str,
        mode: Mode,
        provider: ProviderAdapter | None,
        health: HealthState | None,
        simulated_latency_ms: float | None,
        latency_distribution_ms: Sequence[float] | None,
        rng: random.Random,
        cache_classifier: bool,
    ) -> None:
        self.brief = brief
        self.ordered = ordered
        self.policy = policy
        self.classifier = classifier
        self.cal_obj = cal_obj
        self.tpl_map = tpl_map
        self.store = store
        self.run_id = run_id
        self.mode = mode
        self.provider = provider
        self.health_override = health
        self.simulated_latency_ms = simulated_latency_ms
        self.latency_distribution_ms = latency_distribution_ms
        self.rng = rng
        self.cache_classifier = cache_classifier
        self.state = CanonicalState(brief)
        self.cursor = CoverageCursor()
        self.tok = classifier.tokenizer or WhitespaceTokenizer()
        self.recipient_ids = {"ME", brief.recipient.participant_id}
        self.local_blocked = (
            provider is not None
            and provider.config.local_only
            and provider.config.effective_is_hosted
        )
        if health is not None:
            self.eff_health = health
        elif self.local_blocked or provider is None:
            self.eff_health = HealthState.TEMPLATE_ONLY
        else:
            self.eff_health = provider.health_state
        self.sim_now_ms = ordered[0].received_ms if ordered else 0
        self.next_ev_idx = 0
        self.health_timeline: list[dict[str, Any]] = [
            {
                "at_ms": self.sim_now_ms,
                "state": self.eff_health.value,
                "health": self.eff_health.value,
                "reason": (
                    "local_only"
                    if self.local_blocked
                    else ("init" if provider is not None else "no_provider")
                ),
            }
        ]
        self.speaker_update_times: dict[str, int] = {}
        self.speaker_boundary_end_ms: dict[str, int] = {}
        self.speaker_boundary_rx_ms: dict[str, int] = {}
        self.global_boundary_rx_ms: int = -1
        self.last_gap_ms: int | None = None
        self.recipient_speaking = False
        self.recipient_speaking_until_ms = 0
        self.prev_anchor_norm: str | None = None
        self.prev_anchor_seg_ids: set[str] = set()
        self.prev_anchor_rx_ms: int | None = None
        self.prev_anchor_order_key: tuple[int, tuple[int, int, int, str]] | None = None
        self.settling: list[_ReplayPendingItem] = []
        self.deferred: _ReplayPendingItem | None = None
        self.active: _ReplayActiveItem | None = None
        self.used_active_items: dict[str, _ReplayActiveItem] = {}
        self.last_proactive_display_ms: int | None = None
        self.displayed_times: list[int] = []
        self.displayed_cards: list[DisplayedCardEvent] = []
        self.latencies_ms: list[float] = []
        self.decisions_count = 0
        self.dropped_count = 0
        self.routed_count = 0
        self.withdrawn_count = 0
        self.expired_count = 0
        self.consecutive_clf_errors = 0
        self.classifier_error_threshold = 3
        self.suggestions_by_id: dict[str, SuggestionCandidate] = {}
        self.ledger_texts: dict[str, str] = {}
        self.ledger_categories: dict[str, Category] = {}
        self.prior_anchor_texts: dict[str, str] = {}
        self.pending_decisions: list[dict[str, Any]] = []
        self.decision_idx_by_cand: dict[str, int] = {}
        self.evt_seq = 0

    def next_evt_id(self, prefix: str) -> str:
        self.evt_seq += 1
        return f"{prefix}-rep-{self.evt_seq:06d}"

    def record_health(self, new_state: HealthState, at_ms: int, reason: str) -> None:
        self.eff_health = new_state
        if self.health_timeline[-1]["state"] != new_state.value:
            self.health_timeline.append(
                {
                    "at_ms": at_ms,
                    "state": new_state.value,
                    "health": new_state.value,
                    "reason": reason,
                }
            )

    def upsert_decision(
        self,
        *,
        context: DecisionContext,
        terminal_state: str,
        terminal_reason: ReasonCode | None,
        classifier: Any = None,
        routing: RoutingDecision | None = None,
        serialized_text: str | None = None,
    ) -> None:
        rec = {
            "context": context,
            "terminal_state": terminal_state,
            "terminal_reason": terminal_reason,
            "classifier": classifier,
            "routing": routing,
            "serialized_text": serialized_text,
        }
        if context.candidate_id in self.decision_idx_by_cand:
            self.pending_decisions[self.decision_idx_by_cand[context.candidate_id]] = rec
        else:
            self.decision_idx_by_cand[context.candidate_id] = len(self.pending_decisions)
            self.pending_decisions.append(rec)

    def record_non_routed(
        self,
        ctx: DecisionContext,
        clf_res: Any,
        routing: RoutingDecision,
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
        self.upsert_decision(
            context=ctx,
            terminal_state=t_state,
            terminal_reason=t_reason,
            classifier=clf_res,
            routing=routing,
            serialized_text=serialized_text,
        )

    def retire_anchor(self, anchor: AnchorSpan, *, state_version: int, reason: ReasonCode) -> None:
        self.decisions_count += 1
        ctx = _replay_make_dummy_context(
            candidate_id=anchor.candidate_id,
            session_id=self.brief.session_id,
            state_version=state_version,
            builder_version=self.policy.classifier.version,
            prefilter_version=PREFILTER_VERSION,
            anchor_refs=anchor.anchor_refs,
            anchor_received_ms=anchor.anchor_received_ms,
            display_by_ms=anchor.display_by_ms,
            sha="retired",
        )
        self.upsert_decision(context=ctx, terminal_state="retired", terminal_reason=reason)

    def retire_uncovered_segments(self, snap: StateSnapshot, reason: ReasonCode) -> None:
        unproc = self.cursor.unprocessed_segments(snap)
        if not unproc:
            return
        anchors = coalesce_anchors(
            snap,
            max_candidate_age_ms=self.policy.display.max_candidate_age_ms,
            tokenizer=self.tok,
            speaker_boundary_end_ms=self.speaker_boundary_end_ms,
            segments_subset=unproc,
        )
        for anchor in anchors:
            eff_reason = (
                ReasonCode.EXPIRED_UNEVALUATED
                if reason is ReasonCode.SESSION_ENDED and anchor.display_by_ms <= self.sim_now_ms
                else reason
            )
            self.retire_anchor(anchor, state_version=snap.state_version, reason=eff_reason)
            self.cursor.mark_covered(anchor.segments)

    def close_open_items_for_halt(self, *, at_ms: int) -> None:
        if self.active is not None:
            if self.active.pinned:
                self.transition_active(LedgerState.OUTDATED, ReasonCode.RESOLUTION_CUE, at_ms=at_ms)
            else:
                self.transition_active(
                    LedgerState.EXPIRED, ReasonCode.DISPLAY_TTL_PASSED, at_ms=at_ms
                )
        if self.deferred is not None:
            d = self.deferred
            self.deferred = None
            self.close_pending(d, ReasonCode.DISPLAY_BY_PASSED, at_ms=at_ms)
        if self.settling:
            items = list(self.settling)
            self.settling.clear()
            for item in items:
                self.close_pending(item, ReasonCode.DISPLAY_BY_PASSED, at_ms=at_ms)

    def _apply_external_ledger(self, ev: CanonicalEvent) -> None:
        assert ev.suggestion_id is not None and ev.transition is not None and ev.cause is not None
        if self.store is not None:
            self.store.record_ledger_event(
                session_id=self.brief.session_id,
                run_id=self.run_id,
                event=LedgerEvent(
                    schema_version=SCHEMA_VERSION,
                    event_id=ev.event_id,
                    suggestion_id=ev.suggestion_id,
                    transition=ev.transition,
                    cause=ev.cause,
                    at_ms=ev.at_ms or ev.received_ms,
                    evidence_refs=ev.evidence_refs,
                    feedback=ev.feedback,
                ),
            )
        is_closing = ev.transition in _TERMINAL_LEDGER_STATES or ev.transition is LedgerState.USED
        if is_closing:
            self.prev_anchor_norm = None
        if self.active is not None and self.active.cand.suggestion_id == ev.suggestion_id:
            if ev.transition is LedgerState.PINNED:
                self.active.pinned = True
            elif is_closing:
                act = self.active
                self.active = None
                if ev.transition is LedgerState.USED:
                    self.used_active_items[ev.suggestion_id] = act
                self.upsert_decision(
                    context=act.context,
                    terminal_state=str(ev.transition),
                    terminal_reason=ev.cause,
                    classifier=act.classifier,
                    routing=act.routing,
                    serialized_text=act.serialized_text,
                )
        elif ev.transition is LedgerState.RESOLVED and ev.suggestion_id in self.used_active_items:
            used_act = self.used_active_items.pop(ev.suggestion_id)
            self.upsert_decision(
                context=used_act.context,
                terminal_state=str(ev.transition),
                terminal_reason=ev.cause,
                classifier=used_act.classifier,
                routing=used_act.routing,
                serialized_text=used_act.serialized_text,
            )
        if (
            self.deferred is not None
            and self.deferred.cand.suggestion_id == ev.suggestion_id
            and is_closing
        ):
            d = self.deferred
            self.deferred = None
            self.upsert_decision(
                context=d.context,
                terminal_state=str(ev.transition),
                terminal_reason=ev.cause,
                classifier=d.classifier,
                routing=d.routing,
                serialized_text=d.serialized_text,
            )
        for i, item in enumerate(self.settling):
            if item.cand.suggestion_id == ev.suggestion_id and is_closing:
                self.settling.pop(i)
                self.upsert_decision(
                    context=item.context,
                    terminal_state=str(ev.transition),
                    terminal_reason=ev.cause,
                    classifier=item.classifier,
                    routing=item.routing,
                    serialized_text=item.serialized_text,
                )
                break

    def ingest_event(self, ev: CanonicalEvent) -> None:
        if (
            ev.kind is EventKind.CONTROL
            and ev.subtype is ControlSubtype.OFF_RECORD_START
            and not self.state.off_record
            and not self.state.has_applied_event(ev.source_id, ev.event_id)
        ):
            pre_snap = self.state.snapshot()
            self.retire_uncovered_segments(pre_snap, ReasonCode.OFF_RECORD)
            self.close_open_items_for_halt(at_ms=ev.received_ms)
            self.prev_anchor_norm = None
            self.prev_anchor_seg_ids.clear()
            self.prev_anchor_rx_ms = None
            self.prev_anchor_order_key = None

        apply_res = self.state.apply(ev)
        snap = self.state.snapshot()
        if not apply_res.accepted:
            if apply_res.drop_reason == "off_record":
                self.decisions_count += 1
                otr_ctx = _replay_make_dummy_context(
                    candidate_id=f"otr-{ev.event_id}",
                    session_id=self.brief.session_id,
                    state_version=snap.state_version,
                    builder_version=self.policy.classifier.version,
                    prefilter_version=PREFILTER_VERSION,
                    anchor_refs=(),
                    anchor_received_ms=ev.received_ms,
                    display_by_ms=ev.received_ms,
                    flags={"has_off_record_gap": True},
                    sha="off_record",
                )
                self.upsert_decision(
                    context=otr_ctx,
                    terminal_state="retired",
                    terminal_reason=ReasonCode.OFF_RECORD,
                )
            return

        if self.store is not None:
            can_rev: int | None = None
            if ev.segment_id and ev.segment_id in snap.segments_by_id:
                can_rev = snap.segments_by_id[ev.segment_id].revision
            self.store.record_canonical_event(
                run_id=self.run_id,
                event=ev,
                retention=self.brief.retention,
                canonical_revision=can_rev,
            )

        if ev.kind is EventKind.CONTROL:
            if ev.subtype in {ControlSubtype.GAP, ControlSubtype.RECONNECT}:
                self.last_gap_ms = ev.received_ms
                self.recipient_speaking = False
                self.prev_anchor_norm = None
            elif ev.subtype is ControlSubtype.OFF_RECORD_START:
                self.prev_anchor_norm = None
                self.prev_anchor_seg_ids.clear()
                self.prev_anchor_rx_ms = None
                self.prev_anchor_order_key = None
        elif ev.kind is EventKind.BOUNDARY:
            if ev.speaker_id is not None and ev.end_ms is not None:
                prev_b = self.speaker_boundary_end_ms.get(ev.speaker_id)
                if prev_b is None or ev.end_ms > prev_b:
                    self.speaker_boundary_end_ms[ev.speaker_id] = ev.end_ms
            if ev.speaker_id is not None:
                prev_rx = self.speaker_boundary_rx_ms.get(ev.speaker_id, -1)
                if ev.received_ms > prev_rx:
                    self.speaker_boundary_rx_ms[ev.speaker_id] = ev.received_ms
            elif ev.received_ms > self.global_boundary_rx_ms:
                self.global_boundary_rx_ms = ev.received_ms
            if ev.speaker_id is None or ev.speaker_id in self.recipient_ids:
                self.recipient_speaking = False
        elif ev.kind is EventKind.SPEAKER_UPDATE and ev.segment_id:
            self.speaker_update_times[ev.segment_id] = ev.received_ms
        elif ev.kind is EventKind.SEGMENT and ev.speaker_id in self.recipient_ids:
            if ev.stability is Stability.INTERIM:
                self.recipient_speaking = True
            elif ev.stability is Stability.COMMITTED and ev.segment_id:
                self.recipient_speaking = False
                self.recipient_speaking_until_ms = max(
                    self.recipient_speaking_until_ms, ev.received_ms + 1400
                )
        elif ev.kind is EventKind.LEDGER and ev.suggestion_id:
            self._apply_external_ledger(ev)

        if apply_res.material_change and apply_res.affected_segment_id is not None:
            self.cursor.note_material_segment(apply_res.affected_segment_id)

    def _open_tail_wake_ms(self, anchor: AnchorSpan) -> int:
        lat_margin = int(self.simulated_latency_ms or 0)
        if self.latency_distribution_ms:
            lat_margin = max(lat_margin, int(max(self.latency_distribution_ms)))
        classify_margin = max(50, lat_margin + 1)
        latest_eval_ms = anchor.display_by_ms - self.policy.display.settle_ms - classify_margin
        if latest_eval_ms <= anchor.anchor_received_ms:
            return anchor.anchor_received_ms
        return min(anchor.anchor_received_ms + 2000, latest_eval_ms)

    def is_open_tail_anchor(self, anchor: AnchorSpan, snap: StateSnapshot) -> bool:
        if self.next_ev_idx >= len(self.ordered) or snap.off_record:
            return False
        if anchor.display_by_ms <= self.sim_now_ms:
            return False
        if self.sim_now_ms >= self._open_tail_wake_ms(anchor):
            return False
        if self.tok.count_tokens(anchor.joined_text) >= DEFAULT_ANCHOR_MAX_TOKENS:
            return False
        for seg in anchor.segments:
            if seg.boundary is not None or _SENTENCE_FINAL_RE.search(seg.text) is not None:
                return False
        last_s = anchor.segments[-1]
        bnd_end = self.speaker_boundary_end_ms.get(last_s.speaker_id)
        if bnd_end is not None and bnd_end >= last_s.end_ms:
            return False
        bnd_rx = max(
            self.speaker_boundary_rx_ms.get(last_s.speaker_id, -1),
            self.global_boundary_rx_ms,
        )
        if bnd_rx >= last_s.received_ms:
            return False
        if self.recipient_speaking:
            return False
        last_key = CoverageCursor.seg_key(last_s)
        for seg in snap.segments:
            if seg.speaker_id in self.recipient_ids and CoverageCursor.seg_key(seg) > last_key:
                return False
        return True

    def display_moment_block(self, now_ms: int, snap: StateSnapshot) -> ReasonCode | None:
        if self.recipient_speaking or now_ms < self.recipient_speaking_until_ms:
            return ReasonCode.RECIPIENT_SPEAKING
        if self.active is not None:
            return ReasonCode.CARD_ACTIVE
        if self.last_proactive_display_ms is not None:
            cooldown_end = self.last_proactive_display_ms + self.policy.display.cooldown_ms
            if now_ms < cooldown_end:
                return ReasonCode.COOLDOWN
        cap = effective_max_cards_per_10_min(snap.brief, self.policy)
        recent = [t for t in self.displayed_times if (now_ms - t) <= 600_000]
        if cap <= 0 or len(recent) >= cap:
            return ReasonCode.CARD_ACTIVE
        return None

    def revalidate_item(
        self,
        item: _ReplayPendingItem | _ReplayActiveItem,
        snap: StateSnapshot,
        now_ms: int,
        *,
        is_active: bool,
    ) -> tuple[ReasonCode | None, ReasonCode | None]:
        cand = item.cand
        if is_active:
            assert isinstance(item, _ReplayActiveItem)
            if not item.pinned and now_ms >= item.expires_at_ms:
                return (ReasonCode.DISPLAY_TTL_PASSED, None)
        else:
            if now_ms >= cand.display_by_ms:
                return (ReasonCode.DISPLAY_BY_PASSED, None)
            if self.last_gap_ms is not None and item.anchor_received_ms < self.last_gap_ms:
                return (ReasonCode.DISPLAY_BY_PASSED, None)

        for ref in cand.evidence_refs:
            seg = snap.segments_by_id.get(ref.segment_id)
            if seg is None or seg.revision != ref.revision or seg.speaker_id in self.recipient_ids:
                return (ReasonCode.EVIDENCE_REVISION_CHANGED, None)

        if _replay_brief_material_key(snap.brief) != _replay_brief_material_key(
            item.brief_snapshot
        ):
            return (ReasonCode.OBJECTIVE_CHANGED, None)

        if not is_active and (
            self.mode not in {Mode.PROACTIVE, Mode.SHADOW}
            or snap.off_record
            or not snap.brief.attention.proactive
        ):
            return (ReasonCode.DISPLAY_BY_PASSED, None)

        ev_ids = {r.segment_id for r in cand.evidence_refs}
        norm_q = normalize_for_matching(cand.text)
        q_toks = _replay_content_tokens(cand.text)
        for entry in snap.ledger.values():
            if (
                entry.suggestion_id == cand.suggestion_id
                or entry.state not in _SHOWN_OR_ACTED_STATES
            ):
                continue
            e_cat = self.ledger_categories.get(entry.suggestion_id)
            if e_cat is None and entry.suggestion_id in self.suggestions_by_id:
                e_cat = self.suggestions_by_id[entry.suggestion_id].category
            e_ev = {r.segment_id for r in entry.evidence_refs}
            if e_cat is cand.category and (ev_ids & e_ev):
                return (ReasonCode.DUPLICATE, None)
            prev_txt = self.ledger_texts.get(entry.suggestion_id)
            if prev_txt is None and entry.suggestion_id in self.suggestions_by_id:
                prev_txt = self.suggestions_by_id[entry.suggestion_id].text
            if prev_txt:
                if normalize_for_matching(prev_txt) == norm_q:
                    return (ReasonCode.DUPLICATE, None)
                p_toks = _replay_content_tokens(prev_txt)
                if q_toks and p_toks and (len(q_toks & p_toks) / len(q_toks | p_toks)) >= 0.75:
                    return (ReasonCode.NEAR_DUPLICATE, None)

        cpol = self.policy.categories.get(cand.category)
        tpl = self.tpl_map.get(cpol.template) if cpol and cpol.template else None
        anc_toks = _replay_content_tokens(item.anchor_text)
        for seg in snap.segments:
            if CoverageCursor.seg_key(seg) > item.anchor_last_key and (
                matches_resolution_cue(seg.text, cand.category, tpl)
                or (
                    cand.category is Category.CONFIRM_TIME
                    and matches_resolution_cue(seg.text, Category.CONFIRM_OWNER, None)
                    and bool(anc_toks & _replay_content_tokens(seg.text))
                )
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
                    and matches_resolution_cue(int_seg.text, cand.category, tpl)
                ):
                    return (None, ReasonCode.INTERIM_RESOLUTION_HINT)

        for seg in snap.segments:
            if (
                seg.speaker_id in self.recipient_ids
                and CoverageCursor.seg_key(seg) > item.anchor_last_key
                and is_equivalent_question(seg.text, cand.text, cand.category)
            ):
                return (ReasonCode.EQUIVALENT_QUESTION_ASKED, None)

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

    def offer_deferred(self, item: _ReplayPendingItem, reason: ReasonCode, *, at_ms: int) -> None:
        if self.deferred is not None:
            if item.score > self.deferred.score:
                old = self.deferred
                self.deferred = None
                self.close_pending(old, ReasonCode.DISPLACED_BY_BETTER, at_ms=at_ms)
            else:
                self.close_pending(item, ReasonCode.DISPLACED_BY_BETTER, at_ms=at_ms)
                return

        entry = self.state.snapshot().ledger.get(item.cand.suggestion_id)
        if entry is not None and entry.state is LedgerState.READY:
            def_ev = LedgerEvent(
                schema_version=SCHEMA_VERSION,
                event_id=self.next_evt_id("def"),
                suggestion_id=item.cand.suggestion_id,
                transition=LedgerState.DEFERRED,
                cause=reason,
                at_ms=at_ms,
                evidence_refs=item.cand.evidence_refs,
            )
            self.state.apply(def_ev.to_canonical_event(self.brief.session_id, received_ms=at_ms))
            if self.store is not None:
                self.store.record_ledger_event(
                    session_id=self.brief.session_id, run_id=self.run_id, event=def_ev
                )

        item.deferred_reason = reason
        if reason is ReasonCode.COOLDOWN and self.last_proactive_display_ms is not None:
            item.cooldown_end_ms = self.last_proactive_display_ms + self.policy.display.cooldown_ms
        else:
            item.cooldown_end_ms = None
        self.deferred = item

    def display_pending(self, item: _ReplayPendingItem, *, cause: ReasonCode, at_ms: int) -> None:
        snap = self.state.snapshot()
        if (
            self.mode not in {Mode.PROACTIVE, Mode.SHADOW}
            or snap.off_record
            or not snap.brief.attention.proactive
        ):
            self.close_pending(item, ReasonCode.DISPLAY_BY_PASSED, at_ms=at_ms)
            return
        disp_ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=self.next_evt_id("disp"),
            suggestion_id=item.cand.suggestion_id,
            transition=LedgerState.DISPLAYED,
            cause=cause,
            at_ms=at_ms,
            evidence_refs=item.cand.evidence_refs,
        )
        self.state.apply(disp_ev.to_canonical_event(self.brief.session_id, received_ms=at_ms))
        if self.store is not None:
            self.store.record_ledger_event(
                session_id=self.brief.session_id, run_id=self.run_id, event=disp_ev
            )
        self.ledger_texts[item.cand.suggestion_id] = item.cand.text
        self.ledger_categories[item.cand.suggestion_id] = item.cand.category
        if item.anchor_text:
            self.prior_anchor_texts[item.cand.candidate_id] = item.anchor_text
        self.prev_anchor_norm = None
        self.last_proactive_display_ms = at_ms
        self.displayed_times.append(at_ms)
        self.displayed_cards.append(
            DisplayedCardEvent(
                suggestion_id=item.cand.suggestion_id,
                candidate_id=item.cand.candidate_id,
                session_id=self.brief.session_id,
                category=item.cand.category,
                display_ms=at_ms,
                evidence_refs=item.cand.evidence_refs,
                text=item.cand.text,
                route=item.cand.route,
                origin=item.cand.origin,
            )
        )
        self.active = _ReplayActiveItem(
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
        )

    def close_pending(self, item: _ReplayPendingItem, reason: ReasonCode, *, at_ms: int) -> None:
        if reason is ReasonCode.GENERIC_REFERENT_UNCLEAR:
            self.state.unregister_suggestion(item.cand.suggestion_id)
            self.suggestions_by_id.pop(item.cand.suggestion_id, None)
            if self.store is not None:
                self.store.record_suggestion(
                    session_id=self.brief.session_id,
                    run_id=self.run_id,
                    suggestion=item.cand,
                    current_state="unrendered",
                    terminal_reason=ReasonCode.GENERIC_REFERENT_UNCLEAR,
                    transitions=(),
                    retention=self.brief.retention,
                )
            self.upsert_decision(
                context=item.context,
                terminal_state="unrendered",
                terminal_reason=ReasonCode.GENERIC_REFERENT_UNCLEAR,
                classifier=item.classifier,
                routing=item.routing,
                serialized_text=item.serialized_text,
            )
            return

        if reason in {ReasonCode.DISPLAY_BY_PASSED, ReasonCode.DISPLAY_TTL_PASSED}:
            t_state_enum = LedgerState.EXPIRED
            t_reason = ReasonCode.DISPLAY_BY_PASSED
            dec_state = "expired"
            self.expired_count += 1
        else:
            t_state_enum = LedgerState.WITHDRAWN
            t_reason = reason
            dec_state = "withdrawn"
            self.withdrawn_count += 1

        ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=self.next_evt_id("cls"),
            suggestion_id=item.cand.suggestion_id,
            transition=t_state_enum,
            cause=t_reason,
            at_ms=at_ms,
            evidence_refs=item.cand.evidence_refs,
        )
        self.state.apply(ev.to_canonical_event(self.brief.session_id, received_ms=at_ms))
        if self.store is not None:
            self.store.record_ledger_event(
                session_id=self.brief.session_id, run_id=self.run_id, event=ev
            )
        self.upsert_decision(
            context=item.context,
            terminal_state=dec_state,
            terminal_reason=t_reason,
            classifier=item.classifier,
            routing=item.routing,
            serialized_text=item.serialized_text,
        )

    def transition_active(
        self, target_state: LedgerState, cause: ReasonCode, *, at_ms: int
    ) -> None:
        act = self.active
        if act is None:
            return
        self.active = None
        if target_state is LedgerState.EXPIRED:
            self.expired_count += 1
        elif target_state is LedgerState.WITHDRAWN:
            self.withdrawn_count += 1
        ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=self.next_evt_id("act"),
            suggestion_id=act.cand.suggestion_id,
            transition=target_state,
            cause=cause,
            at_ms=at_ms,
            evidence_refs=act.cand.evidence_refs,
        )
        self.state.apply(ev.to_canonical_event(self.brief.session_id, received_ms=at_ms))
        if self.store is not None:
            self.store.record_ledger_event(
                session_id=self.brief.session_id, run_id=self.run_id, event=ev
            )
        self.upsert_decision(
            context=act.context,
            terminal_state=str(target_state),
            terminal_reason=cause,
            classifier=act.classifier,
            routing=act.routing,
            serialized_text=act.serialized_text,
        )

    def advance_display_timers(self, now_ms: int) -> None:
        snap = self.state.snapshot()
        if self.active is not None:
            fail_reason, _ = self.revalidate_item(self.active, snap, now_ms, is_active=True)
            if fail_reason is not None:
                if fail_reason is ReasonCode.DISPLAY_TTL_PASSED:
                    self.transition_active(
                        LedgerState.EXPIRED, ReasonCode.DISPLAY_TTL_PASSED, at_ms=now_ms
                    )
                elif fail_reason in {
                    ReasonCode.POSSIBLY_RESOLVED,
                    ReasonCode.EQUIVALENT_QUESTION_ASKED,
                }:
                    self.transition_active(
                        LedgerState.RESOLVED, ReasonCode.RESOLUTION_CUE, at_ms=now_ms
                    )
                elif self.active.pinned:
                    self.transition_active(
                        LedgerState.OUTDATED, ReasonCode.RESOLUTION_CUE, at_ms=now_ms
                    )
                else:
                    self.transition_active(LedgerState.WITHDRAWN, fail_reason, at_ms=now_ms)

        if self.deferred is not None:
            def_item = self.deferred
            fail_reason, defer_reason = self.revalidate_item(
                def_item, snap, now_ms, is_active=False
            )
            if fail_reason is not None:
                self.deferred = None
                self.close_pending(def_item, fail_reason, at_ms=now_ms)
            elif defer_reason is None:
                gate = self.display_moment_block(now_ms, snap)
                if gate is None:
                    self.deferred = None
                    self.display_pending(def_item, cause=ReasonCode.DEFERRED_RELEASED, at_ms=now_ms)
                elif gate is ReasonCode.COOLDOWN and self.last_proactive_display_ms is not None:
                    def_item.cooldown_end_ms = (
                        self.last_proactive_display_ms + self.policy.display.cooldown_ms
                    )

        snap = self.state.snapshot()
        still_settling: list[_ReplayPendingItem] = []
        for item in self.settling:
            fail_reason, defer_reason = self.revalidate_item(item, snap, now_ms, is_active=False)
            if fail_reason is not None:
                self.close_pending(item, fail_reason, at_ms=now_ms)
                continue
            bnd_rx = max(
                self.speaker_boundary_rx_ms.get(item.anchor_speaker_id, -1),
                self.global_boundary_rx_ms,
            )
            if bnd_rx > item.anchor_received_ms and now_ms < item.settle_end_ms:
                item.settled = True
                item.settle_cause = ReasonCode.BOUNDARY_AFTER_SETTLE
            elif now_ms >= item.settle_end_ms:
                item.settled = True
                item.settle_cause = ReasonCode.SETTLE_TIMEOUT
            if not item.settled:
                still_settling.append(item)
                continue
            if defer_reason is not None:
                self.offer_deferred(item, defer_reason, at_ms=now_ms)
                continue
            gate = self.display_moment_block(now_ms, snap)
            if gate is not None:
                self.offer_deferred(item, gate, at_ms=now_ms)
                continue
            self.display_pending(item, cause=item.settle_cause, at_ms=now_ms)
        self.settling = still_settling

    async def _render_and_enqueue(
        self,
        *,
        ctx: DecisionContext,
        clf_res: Any,
        routing: RoutingDecision,
        snap: StateSnapshot,
        serialized_text: str,
        anchor: AnchorSpan,
    ) -> None:
        assert routing.category is not None
        eff_routing = routing
        cpol = self.policy.categories[routing.category]
        tpl = self.tpl_map.get(cpol.template) if cpol.template else None
        ready_ms = self.sim_now_ms
        cand: SuggestionCandidate | None = None
        fail_reason: ReasonCode | None = None

        if routing.route is Route.GENERATOR:
            if self.provider is not None:
                orig_sink = self.provider.egress_sink
                orig_clock = self.provider.clock_ms
                egress_recorded = False

                def _replay_clock_ms(_r: int = ready_ms) -> int:
                    return _r

                def _on_egress(
                    rec: EgressRecord,
                    _cid: str = ctx.candidate_id,
                    _orig: Callable[[EgressRecord], None] | None = orig_sink,
                ) -> None:
                    nonlocal egress_recorded
                    if rec.candidate_id == _cid:
                        egress_recorded = True
                    if self.store is not None:
                        self.store.record_egress(run_id=self.run_id, record=rec)
                    if _orig is not None:
                        _orig(rec)

                self.provider.egress_sink = _on_egress
                self.provider.clock_ms = _replay_clock_ms
                try:
                    gen_out = await generate_question(
                        ctx,
                        routing.category,
                        snap,
                        self.policy,
                        self.provider,
                        clock_ms=_replay_clock_ms,
                        store=self.store,
                        templates=self.tpl_map,
                        ledger_texts=self.ledger_texts,
                        ledger_categories=self.ledger_categories,
                    )
                finally:
                    self.provider.egress_sink = orig_sink
                    self.provider.clock_ms = orig_clock
                    if self.health_override is None and not self.local_blocked:
                        new_h = self.provider.health_state
                        if new_h is not self.eff_health:
                            self.record_health(new_h, ready_ms, "generator_breaker")
                cand = gen_out.candidate
                if cand is None:
                    fail_reason = gen_out.terminal_reason or ReasonCode.GENERATOR_INVALID_OUTPUT
                elif gen_out.fallback_used and not egress_recorded:
                    eff_routing = replace(routing, route=cand.route)
            elif tpl is not None:
                eff_routing = replace(routing, route=Route.TEMPLATE)
                rendered = render_template(
                    ctx,
                    routing.category,
                    snap,
                    tpl,
                    now_ms=ready_ms,
                    ready_ms=ready_ms,
                    display_ttl_ms=self.policy.display.display_ttl_ms,
                )
                cand = rendered.candidate
                if cand is None:
                    fail_reason = rendered.unrendered_reason or ReasonCode.SLOT_UNAVAILABLE
            else:
                fail_reason = ReasonCode.INSUFFICIENT_BUDGET
        else:
            if tpl is None:
                fail_reason = ReasonCode.SLOT_UNAVAILABLE
            else:
                rendered = render_template(
                    ctx,
                    routing.category,
                    snap,
                    tpl,
                    now_ms=ready_ms,
                    ready_ms=ready_ms,
                    display_ttl_ms=self.policy.display.display_ttl_ms,
                )
                if rendered.candidate is None:
                    fail_reason = rendered.unrendered_reason or ReasonCode.SLOT_UNAVAILABLE
                else:
                    val = validate_template_candidate(
                        rendered.candidate,
                        snap,
                        max_words=tpl.max_words,
                        ledger_texts=self.ledger_texts,
                        ledger_categories=self.ledger_categories,
                    )
                    if val.valid:
                        cand = rendered.candidate
                    else:
                        fail_reason = val.reason_code or ReasonCode.SLOT_UNAVAILABLE

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
            self.upsert_decision(
                context=ctx,
                terminal_state=t_state,
                terminal_reason=t_reason,
                classifier=clf_res,
                routing=eff_routing,
                serialized_text=serialized_text,
            )
            return

        self.routed_count += 1
        self.state.register_suggestion(cand)
        self.suggestions_by_id[cand.suggestion_id] = cand
        if self.store is not None:
            self.store.record_suggestion(
                session_id=self.brief.session_id,
                run_id=self.run_id,
                suggestion=cand,
                current_state="ready",
                terminal_reason=None,
                transitions=(),
                retention=self.brief.retention,
            )
        self.upsert_decision(
            context=ctx,
            terminal_state="ready",
            terminal_reason=eff_routing.reason_codes[0] if eff_routing.reason_codes else None,
            classifier=clf_res,
            routing=eff_routing,
            serialized_text=serialized_text,
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
        self.settling.append(
            _ReplayPendingItem(
                cand=cand,
                context=ctx,
                classifier=clf_res,
                routing=eff_routing,
                serialized_text=serialized_text,
                score=score,
                brief_snapshot=snap.brief,
                anchor_text=anchor.joined_text,
                anchor_speaker_id=anchor_spk,
                anchor_last_key=last_key,
                anchor_start_ms=anchor_start,
                anchor_received_ms=ctx.anchor_received_ms,
                settle_end_ms=ctx.anchor_received_ms + self.policy.display.settle_ms,
            )
        )

    def _ready_proactive_anchors(
        self, snap: StateSnapshot | None = None
    ) -> tuple[StateSnapshot, list[AnchorSpan], int | None]:
        eff_snap = snap if snap is not None else self.state.snapshot()
        if self.mode not in {Mode.PROACTIVE, Mode.QUIET, Mode.SHADOW}:
            return (eff_snap, [], None)
        if self.eff_health in {
            HealthState.NO_TRANSCRIPT,
            HealthState.PAUSED,
            HealthState.MANUAL_ONLY,
        }:
            return (eff_snap, [], None)
        if eff_snap.off_record:
            return (eff_snap, [], None)
        unprocessed = self.cursor.unprocessed_segments(eff_snap)
        if not unprocessed:
            return (eff_snap, [], None)
        anchors = coalesce_anchors(
            eff_snap,
            max_candidate_age_ms=self.policy.display.max_candidate_age_ms,
            tokenizer=self.tok,
            speaker_boundary_end_ms=self.speaker_boundary_end_ms,
            segments_subset=unprocessed,
        )
        open_tail_wake_ms: int | None = None
        if anchors and self.is_open_tail_anchor(anchors[-1], eff_snap):
            open_tail_wake_ms = self._open_tail_wake_ms(anchors[-1])
            anchors = anchors[:-1]
        return (eff_snap, anchors, open_tail_wake_ms)

    async def run_proactive_once(self) -> bool:
        snap, anchors, _open_tail_wake_ms = self._ready_proactive_anchors()
        if not anchors:
            return False

        live_anchors: list[AnchorSpan] = []
        for anchor in anchors:
            if anchor.display_by_ms <= self.sim_now_ms:
                self.retire_anchor(
                    anchor, state_version=snap.state_version, reason=ReasonCode.EXPIRED_UNEVALUATED
                )
                self.cursor.mark_covered(anchor.segments)
            else:
                live_anchors.append(anchor)
        if not live_anchors:
            return False

        max_per_job = self.policy.classifier.max_candidates_per_job
        has_overflow = len(live_anchors) > max_per_job
        selected = live_anchors[-max_per_job:]

        batch_prev_norm = self.prev_anchor_norm
        batch_prev_seg_ids = set(self.prev_anchor_seg_ids)
        batch_prev_rx_ms = self.prev_anchor_rx_ms
        batch_prev_order_key = self.prev_anchor_order_key
        surviving: list[tuple[ContextBuildResult, str]] = []
        for anchor in selected:
            seg_ids = {s.segment_id for s in anchor.segments}
            anchor_order_key = (
                anchor.anchor_received_ms,
                max(CoverageCursor.seg_key(s) for s in anchor.segments),
            )
            same_seg_rev = bool(seg_ids & batch_prev_seg_ids)
            out_of_order = (
                batch_prev_order_key is not None and anchor_order_key <= batch_prev_order_key
            )
            gap_exceeded = (
                batch_prev_rx_ms is not None
                and (anchor.anchor_received_ms - batch_prev_rx_ms) > 30_000
            )
            eff_prev = None if (same_seg_rev or out_of_order or gap_exceeded) else batch_prev_norm
            pf = run_prefilter(
                anchor,
                state_version=snap.state_version,
                previous_anchor_norm_text=eff_prev,
            )
            batch_prev_norm = pf.normalized_text
            batch_prev_seg_ids = seg_ids
            batch_prev_rx_ms = anchor.anchor_received_ms
            batch_prev_order_key = anchor_order_key
            if self.prev_anchor_order_key is None or anchor_order_key >= self.prev_anchor_order_key:
                self.prev_anchor_norm = pf.normalized_text
                self.prev_anchor_seg_ids = seg_ids
                self.prev_anchor_rx_ms = anchor.anchor_received_ms
                self.prev_anchor_order_key = anchor_order_key
            if not pf.passed and pf.drop_record is not None:
                self.decisions_count += 1
                self.dropped_count += 1
                drop = pf.drop_record
                dummy_ctx = _replay_make_dummy_context(
                    candidate_id=drop.candidate_id,
                    session_id=drop.session_id,
                    state_version=drop.based_on_state_version,
                    builder_version=self.policy.classifier.version,
                    prefilter_version=drop.prefilter_version,
                    anchor_refs=drop.anchor_refs,
                    anchor_received_ms=drop.anchor_received_ms,
                    display_by_ms=drop.display_by_ms,
                    sha="dropped",
                )
                self.upsert_decision(
                    context=dummy_ctx, terminal_state="dropped", terminal_reason=drop.reason_code
                )
                self.cursor.mark_covered(anchor.segments)
            else:
                built = build_decision_context(
                    snap,
                    anchor,
                    self.policy,
                    origin=Origin.PROACTIVE,
                    tokenizer=self.tok,
                    suggestion_catalog=self.suggestions_by_id,
                )
                surviving.append((built, pf.normalized_text))

        if not surviving:
            _, ready_anchors, _ = self._ready_proactive_anchors()
            return has_overflow or bool(ready_anchors)
        return await self._classify_and_route_batch(surviving, has_overflow=has_overflow)

    async def _classify_and_route_batch(
        self,
        surviving: list[tuple[ContextBuildResult, str]],
        *,
        has_overflow: bool,
    ) -> bool:
        c_inputs = [ClassifierInput.from_build_result(b) for b, _ in surviving]
        t0 = time.perf_counter()
        if self.cache_classifier and self.store is not None:
            clf_results = await classify_batch_cached(self.classifier, c_inputs, store=self.store)
        else:
            clf_results = await self.classifier.classify_batch(c_inputs)
        measured_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
        stage_ms = _sample_stage_latency_ms(
            measured_ms, self.simulated_latency_ms, self.latency_distribution_ms, self.rng
        )
        all_err = all(isinstance(r, ClassifierErrorResult) for r in clf_results)
        if all_err:
            self.consecutive_clf_errors += 1
            if self.consecutive_clf_errors >= self.classifier_error_threshold:
                self.record_health(
                    HealthState.MANUAL_ONLY, self.sim_now_ms + stage_ms, "classifier_errors"
                )
            elif stage_ms == 0 and any(
                isinstance(r, ClassifierErrorResult) and r.error == "timeout" for r in clf_results
            ):
                stage_ms = 1
        else:
            self.consecutive_clf_errors = 0

        self.sim_now_ms += stage_ms
        self.latencies_ms.append(float(stage_ms))
        while (
            self.next_ev_idx < len(self.ordered)
            and self.ordered[self.next_ev_idx].received_ms <= self.sim_now_ms
        ):
            self.ingest_event(self.ordered[self.next_ev_idx])
            self.next_ev_idx += 1
        self.advance_display_timers(self.sim_now_ms)

        current_snap = self.state.snapshot()
        retry_pending = False
        for (built, _norm), clf_raw in zip(surviving, clf_results, strict=True):
            assert built.context is not None and built.serialized_text is not None
            ctx = built.context
            if self.cal_obj is not None and not isinstance(clf_raw, ClassifierErrorResult):
                if isinstance(self.cal_obj, CalibrationRegistry):
                    cal_ver = (
                        self.policy.classifier.calibration
                        if self.policy.classifier.calibration
                        and clf_raw.classifier_version == self.policy.classifier.version
                        else None
                    )
                    clf_res = self.cal_obj.calibrate(
                        ctx, clf_raw, mode=current_snap.brief.mode, calibration_version=cal_ver
                    )
                else:
                    clf_res = self.cal_obj.calibrate(ctx, clf_raw, mode=current_snap.brief.mode)
            else:
                clf_res = clf_raw

            stale = any(
                (cur_seg := current_snap.segments_by_id.get(ref.segment_id)) is None
                or cur_seg.revision != ref.revision
                or cur_seg.speaker_id in self.recipient_ids
                for ref in ctx.anchor_refs
            )
            if (
                not stale
                and isinstance(clf_raw, ClassifierErrorResult)
                and clf_raw.error in {"timeout", "deadline_exceeded"}
            ):
                if (
                    clf_raw.error == "timeout"
                    and self.consecutive_clf_errors < self.classifier_error_threshold
                    and self.sim_now_ms < ctx.display_by_ms
                    and self.eff_health is not HealthState.MANUAL_ONLY
                ):
                    retry_pending = True
                    continue
                if (
                    self.sim_now_ms >= ctx.display_by_ms
                    or clf_raw.error == "deadline_exceeded"
                    or self.eff_health is HealthState.MANUAL_ONLY
                ):
                    self.retire_anchor(
                        built.anchor,
                        state_version=current_snap.state_version,
                        reason=ReasonCode.EXPIRED_UNEVALUATED,
                    )
                    self.cursor.mark_covered(built.anchor.segments)
                    continue

            self.decisions_count += 1
            if stale:
                self.withdrawn_count += 1
                self.upsert_decision(
                    context=ctx,
                    terminal_state="withdrawn",
                    terminal_reason=ReasonCode.EVIDENCE_REVISION_CHANGED,
                    classifier=clf_res,
                    serialized_text=built.serialized_text,
                )
                unchanged = [
                    s
                    for s in built.anchor.segments
                    if (cur := current_snap.segments_by_id.get(s.segment_id)) is not None
                    and cur.revision == s.revision
                ]
                if unchanged:
                    self.cursor.mark_covered(unchanged)
                continue

            self.cursor.mark_covered(built.anchor.segments)
            route_health = self.eff_health
            if (
                self.health_override is None
                and self.provider is not None
                and not self.local_blocked
            ):
                brk_state = self.provider.advance_clock(self.sim_now_ms)
                if brk_state is BreakerState.HALF_OPEN and self.provider.half_open_probe_available:
                    route_health = HealthState.OK
                else:
                    route_health = self.provider.health_state
                    if route_health is not self.eff_health:
                        self.record_health(route_health, self.sim_now_ms, "breaker")

            routing = route_candidate(
                ctx,
                clf_res,
                current_snap,
                self.policy,
                mode=self.mode,
                health=route_health,
                now_ms=self.sim_now_ms,
                speaker_update_times_ms=self.speaker_update_times,
                ledger_texts=self.ledger_texts,
                ledger_categories=self.ledger_categories,
                prior_anchor_texts=self.prior_anchor_texts,
                recent_displayed_ms=self.displayed_times,
            )
            if routing.outcome is not Outcome.ROUTE or routing.category is None:
                self.record_non_routed(ctx, clf_res, routing, built.serialized_text)
                continue

            await self._render_and_enqueue(
                ctx=ctx,
                clf_res=clf_res,
                routing=routing,
                snap=current_snap,
                serialized_text=built.serialized_text,
                anchor=built.anchor,
            )
        self.advance_display_timers(self.sim_now_ms)
        _, ready_anchors, _ = self._ready_proactive_anchors()
        return has_overflow or retry_pending or bool(ready_anchors)

    def next_wake_ms(self) -> int | None:
        candidates: list[int] = []
        if self.next_ev_idx < len(self.ordered):
            candidates.append(self.ordered[self.next_ev_idx].received_ms)
        _, ready_anchors, open_tail_wake_ms = self._ready_proactive_anchors()
        if ready_anchors:
            candidates.append(self.sim_now_ms)
        elif open_tail_wake_ms is not None and open_tail_wake_ms > self.sim_now_ms:
            candidates.append(open_tail_wake_ms)
        for item in self.settling:
            if not item.settled and item.settle_end_ms > self.sim_now_ms:
                candidates.append(item.settle_end_ms)
            if item.cand.display_by_ms > self.sim_now_ms:
                candidates.append(item.cand.display_by_ms)
        if self.deferred is not None:
            if self.deferred.cand.display_by_ms > self.sim_now_ms:
                candidates.append(self.deferred.cand.display_by_ms)
            if (
                self.deferred.cooldown_end_ms is not None
                and self.deferred.cooldown_end_ms > self.sim_now_ms
            ):
                candidates.append(self.deferred.cooldown_end_ms)
        if (
            self.settling or self.deferred is not None
        ) and self.recipient_speaking_until_ms > self.sim_now_ms:
            candidates.append(self.recipient_speaking_until_ms)
        if (
            self.active is not None
            and not self.active.pinned
            and self.active.expires_at_ms > self.sim_now_ms
        ):
            candidates.append(self.active.expires_at_ms)
        return min(candidates) if candidates else None

    async def run(self) -> ReplayResult:
        if self.store is not None:
            self.store.upsert_session(
                brief=self.brief,
                run_id=self.run_id,
                runtime_mode=self.mode,
                effective_max_cards_per_10_min=effective_max_cards_per_10_min(
                    self.brief, self.policy
                ),
                health_timeline=self.health_timeline,
            )

        while (
            self.next_ev_idx < len(self.ordered)
            or self.settling
            or self.deferred is not None
            or self.active is not None
            or bool(self._ready_proactive_anchors()[1])
        ):
            while (
                self.next_ev_idx < len(self.ordered)
                and self.ordered[self.next_ev_idx].received_ms <= self.sim_now_ms
            ):
                self.ingest_event(self.ordered[self.next_ev_idx])
                self.next_ev_idx += 1
            self.advance_display_timers(self.sim_now_ms)

            while await self.run_proactive_once():
                while (
                    self.next_ev_idx < len(self.ordered)
                    and self.ordered[self.next_ev_idx].received_ms <= self.sim_now_ms
                ):
                    self.ingest_event(self.ordered[self.next_ev_idx])
                    self.next_ev_idx += 1
                self.advance_display_timers(self.sim_now_ms)

            wake = self.next_wake_ms()
            if wake is None:
                break
            self.sim_now_ms = max(self.sim_now_ms, wake)

        end_snap = self.state.snapshot()
        self.retire_uncovered_segments(end_snap, ReasonCode.SESSION_ENDED)
        self.close_open_items_for_halt(at_ms=self.sim_now_ms)
        final_snap = self.state.snapshot()
        if self.store is not None:
            sorted_decs = sorted(
                self.pending_decisions,
                key=lambda rec: _replay_anchor_sort_key(rec["context"], final_snap),
            )
            for rec in sorted_decs:
                self.store.record_decision(
                    run_id=self.run_id,
                    context=rec["context"],
                    terminal_state=rec["terminal_state"],
                    terminal_reason=rec["terminal_reason"],
                    classifier=rec["classifier"],
                    routing=rec["routing"],
                    serialized_text=rec["serialized_text"],
                    retention=self.brief.retention,
                )
            self.store.upsert_session(
                brief=final_snap.brief,
                run_id=self.run_id,
                runtime_mode=self.mode,
                effective_max_cards_per_10_min=effective_max_cards_per_10_min(
                    final_snap.brief, self.policy
                ),
                health_timeline=self.health_timeline,
                reducer_counters=final_snap.counters,
                revision_history=final_snap.revision_history,
            )

        rx_times = [e.received_ms for e in self.ordered]
        dur_ms = max(1, (max(rx_times) - min(rx_times)) if len(rx_times) >= 2 else 1)
        return ReplayResult(
            session_id=self.brief.session_id,
            run_id=self.run_id,
            duration_ms=dur_ms,
            displayed_cards=tuple(self.displayed_cards),
            decisions_count=self.decisions_count,
            dropped_count=self.dropped_count,
            routed_count=self.routed_count,
            withdrawn_count=self.withdrawn_count,
            expired_count=self.expired_count,
            latencies_ms=tuple(self.latencies_ms),
        )


def _replay_make_dummy_context(
    *,
    candidate_id: str,
    session_id: str,
    state_version: int,
    builder_version: str,
    prefilter_version: str,
    anchor_refs: tuple[SegmentRef, ...],
    anchor_received_ms: int,
    display_by_ms: int,
    flags: dict[str, bool] | None = None,
    sha: str = "retired",
) -> DecisionContext:
    return DecisionContext(
        schema_version=SCHEMA_VERSION,
        candidate_id=candidate_id,
        session_id=session_id,
        origin=Origin.PROACTIVE,
        based_on_state_version=state_version,
        builder_version=builder_version,
        prefilter_version=prefilter_version,
        anchor_refs=anchor_refs,
        context_refs=(),
        anchor_received_ms=anchor_received_ms,
        display_by_ms=display_by_ms,
        token_counts={"total": 0},
        omitted={"turns": 0, "suggestions": 0, "known_facts": 0, "gaps": 0},
        flags=flags or {},
        serialized_sha256=sha,
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
    tpl_map = dict(templates) if templates is not None else load_templates_dir()
    cal_obj: CalibrationModel | CalibrationRegistry | None = (
        calibration
        if calibration is not None
        else (default_calibration_registry() if policy.classifier.calibration else None)
    )
    prev_rx_for_seg: dict[str, int] = {}
    ordered_tagged: list[tuple[int, int, CanonicalEvent]] = []
    for orig_idx, ev in enumerate(events):
        if ev.segment_id is not None:
            eff_rx = max(prev_rx_for_seg.get(ev.segment_id, -1), ev.received_ms)
            prev_rx_for_seg[ev.segment_id] = eff_rx
        else:
            eff_rx = ev.received_ms
        ordered_tagged.append((eff_rx, orig_idx, ev))
    ordered_tagged.sort(key=lambda item: (item[0], item[1]))
    sim = _ReplaySim(
        brief=brief,
        ordered=[item[2] for item in ordered_tagged],
        policy=policy,
        classifier=classifier,
        cal_obj=cal_obj,
        tpl_map=tpl_map,
        store=store,
        run_id=run_id,
        mode=mode,
        provider=provider,
        health=health,
        simulated_latency_ms=simulated_latency_ms,
        latency_distribution_ms=latency_distribution_ms,
        rng=random.Random(rng_seed),
        cache_classifier=cache_classifier,
    )
    return await sim.run()


def simulate_replay(
    brief: SessionBrief,
    events: Sequence[CanonicalEvent],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    **kwargs: Any,
) -> ReplayResult:
    """Synchronous wrapper for async_simulate_replay."""
    return asyncio.run(async_simulate_replay(brief, events, policy, classifier, **kwargs))


def _extract_anchor_text_from_serialized(serialized_text: str) -> str:
    lines: list[str] = []
    in_anchor = False
    for raw_line in serialized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "[ANCHOR]" or line.startswith("[ANCHOR] "):
            in_anchor = True
            rest = line[len("[ANCHOR]") :].strip()
            if rest:
                lines.append(rest)
            continue
        if in_anchor:
            if re.match(r"^\[[A-Z_]+\](?:\s|$)", line):
                break
            lines.append(line)
    joined = " ".join(lines).strip()
    if not joined:
        return ""
    if ":" in joined:
        prefix, rest = joined.split(":", 1)
        if re.fullmatch(r"(?:S\d+|S\?|ME|[A-Za-z0-9_-]{1,32})", prefix.strip()):
            return rest.strip()
    return joined


_CONTEXT_TURN_RE = re.compile(
    r"^(?:(?:<\d+[sm]|older)\s+)?(ME|S\d+|S\?|[A-Za-z0-9_-]{1,32}):\s*(.*)$"
)


def _extract_context_turns_from_serialized(serialized_text: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    in_context = False
    for raw_line in serialized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "[CONTEXT]":
            in_context = True
            continue
        if not in_context:
            continue
        if line.startswith("[GAP:"):
            continue
        if re.match(r"^\[[A-Z_]+\](?:\s|$)", line):
            break
        m = _CONTEXT_TURN_RE.match(line)
        if m is not None:
            turns.append((m.group(1), m.group(2).strip()))
        elif turns:
            prev_role, prev_text = turns[-1]
            turns[-1] = (prev_role, f"{prev_text} {line}".strip())
    return turns


def _parse_dataset_classifier_input(row: Mapping[str, Any]) -> ClassifierInput:
    if "context" in row and "serialized_text" in row:
        ctx = DecisionContext.from_dict(row["context"])
        return ClassifierInput(context=ctx, serialized_text=str(row["serialized_text"]))
    if "candidate_id" in row and "serialized_text" in row:
        ctx_payload = {k: v for k, v in row.items() if k in _DECISION_CONTEXT_KEYS}
        ctx = DecisionContext.from_dict(ctx_payload)
        return ClassifierInput(context=ctx, serialized_text=str(row["serialized_text"]))
    raise ValueError(
        "Dataset row must contain (context, serialized_text) or DecisionContext + serialized_text"
    )


def _calibrate_dataset_result(
    cal_obj: CalibrationModel | CalibrationRegistry | None,
    ctx: DecisionContext,
    clf_raw: ClassifierResult,
    policy: PolicyConfig,
    mode: Any,
) -> ClassifierResult:
    if cal_obj is None or isinstance(clf_raw, ClassifierErrorResult):
        return clf_raw
    if isinstance(cal_obj, CalibrationRegistry):
        cal_ver = (
            policy.classifier.calibration
            if policy.classifier.calibration
            and clf_raw.classifier_version == policy.classifier.version
            else None
        )
        return cal_obj.calibrate(ctx, clf_raw, mode=mode, calibration_version=cal_ver)
    return cal_obj.calibrate(ctx, clf_raw, mode=mode)


def _dataset_non_routed_state(routing: RoutingDecision) -> tuple[str, ReasonCode | None]:
    if routing.outcome is Outcome.NONE:
        t_state = "no_opportunity"
    elif routing.outcome is Outcome.ABSTAIN:
        t_state = "abstained"
    elif routing.outcome is Outcome.SUPPRESS:
        t_state = "suppressed"
    else:
        t_state = "ready"
    t_reason = routing.reason_codes[0] if routing.reason_codes else None
    return (t_state, t_reason)


async def _render_dataset_candidate(
    *,
    ctx: DecisionContext,
    routing: RoutingDecision,
    snap: StateSnapshot,
    policy: PolicyConfig,
    tpl_map: Mapping[str, TemplateSpec],
    provider: ProviderAdapter | None,
    store: SQLiteStore | None,
    run_id: str,
    ledger_texts: Mapping[str, str],
    ledger_categories: Mapping[str, Category],
) -> tuple[SuggestionCandidate | None, RoutingDecision, ReasonCode | None]:
    assert routing.category is not None
    eff_routing = routing
    cpol = policy.categories[routing.category]
    tpl = tpl_map.get(cpol.template) if cpol.template else None
    ready_ms = ctx.anchor_received_ms
    cand: SuggestionCandidate | None = None
    fail_reason: ReasonCode | None = None

    if routing.route is Route.GENERATOR:
        if provider is not None:
            orig_sink = provider.egress_sink
            orig_clock = provider.clock_ms
            egress_recorded = False

            def _ds_clock_ms(_r: int = ready_ms) -> int:
                return _r

            def _on_egress(
                rec: EgressRecord,
                _cid: str = ctx.candidate_id,
                _orig: Callable[[EgressRecord], None] | None = orig_sink,
            ) -> None:
                nonlocal egress_recorded
                if rec.candidate_id == _cid:
                    egress_recorded = True
                if store is not None:
                    store.record_egress(run_id=run_id, record=rec)
                if _orig is not None:
                    _orig(rec)

            provider.egress_sink = _on_egress
            provider.clock_ms = _ds_clock_ms
            try:
                gen_out = await generate_question(
                    ctx,
                    routing.category,
                    snap,
                    policy,
                    provider,
                    clock_ms=_ds_clock_ms,
                    store=store,
                    templates=tpl_map,
                    ledger_texts=ledger_texts,
                    ledger_categories=ledger_categories,
                )
            finally:
                provider.egress_sink = orig_sink
                provider.clock_ms = orig_clock
            cand = gen_out.candidate
            if cand is None:
                fail_reason = gen_out.terminal_reason or ReasonCode.GENERATOR_INVALID_OUTPUT
            elif gen_out.fallback_used and not egress_recorded:
                eff_routing = replace(routing, route=cand.route)
            return (cand, eff_routing, fail_reason)
        if tpl is not None:
            eff_routing = replace(routing, route=Route.TEMPLATE)
        else:
            return (None, eff_routing, ReasonCode.INSUFFICIENT_BUDGET)

    if tpl is None:
        return (None, eff_routing, ReasonCode.SLOT_UNAVAILABLE)
    rendered = render_template(
        ctx,
        routing.category,
        snap,
        tpl,
        now_ms=ready_ms,
        ready_ms=ready_ms,
        display_ttl_ms=policy.display.display_ttl_ms,
    )
    if rendered.candidate is None:
        return (None, eff_routing, rendered.unrendered_reason or ReasonCode.SLOT_UNAVAILABLE)
    val = validate_template_candidate(
        rendered.candidate,
        snap,
        max_words=tpl.max_words,
        ledger_texts=ledger_texts,
        ledger_categories=ledger_categories,
    )
    if val.valid:
        return (rendered.candidate, eff_routing, None)
    return (None, eff_routing, val.reason_code or ReasonCode.SLOT_UNAVAILABLE)


def _ensure_dataset_segment_revs(
    *,
    state: CanonicalState,
    applied_seg_revs: dict[str, int],
    seq_counter: int,
    refs: Sequence[SegmentRef],
    text: str,
    rx_ms: int,
    session_id: str,
    run_id: str,
    retention: RetentionConfig,
    store: SQLiteStore | None,
    is_context: bool,
    anchor_seg_revs: set[tuple[str, int]] | None = None,
    anchor_text_by_ref: Mapping[tuple[str, int], str] | None = None,
    context_turns: Sequence[tuple[str, str]] | None = None,
) -> int:
    use_ctx_turns = is_context and context_turns is not None and len(context_turns) == len(refs)
    for idx, ref in enumerate(refs):
        cur_rev = applied_seg_revs.get(ref.segment_id, 0)
        if cur_rev >= ref.revision:
            continue
        role_label, turn_text = (
            context_turns[idx] if (use_ctx_turns and context_turns is not None) else ("S1", text)
        )
        for rev in range(cur_rev + 1, ref.revision + 1):
            seq_counter += 1
            if is_context:
                is_anchor_rev = (
                    anchor_seg_revs is not None and (ref.segment_id, rev) in anchor_seg_revs
                )
                if is_anchor_rev:
                    eff_speaker = role_label if re.fullmatch(r"S\d+", role_label) else "S1"
                    eff_channel = 1
                    resolved_text = (
                        anchor_text_by_ref.get((ref.segment_id, rev))
                        if anchor_text_by_ref is not None
                        else None
                    ) or turn_text
                else:
                    eff_speaker = (
                        "ME"
                        if role_label == "ME"
                        else (role_label if re.fullmatch(r"S\d+", role_label) else "S1")
                    )
                    eff_channel = 0 if eff_speaker == "ME" else 1
                    resolved_text = turn_text
                seg_text = (
                    resolved_text
                    if (rev == ref.revision and resolved_text)
                    else f"__rev_{rev}__ {resolved_text or ref.segment_id}"
                )
                should_record = is_anchor_rev
            else:
                eff_speaker = "S1"
                eff_channel = 1
                seg_text = (
                    text
                    if (rev == ref.revision and text)
                    else f"__rev_{rev}__ {text or ref.segment_id}"
                )
                should_record = True
            start_ms = max(0, rx_ms - (1000 if is_context else 500))
            end_ms = max(start_ms + 1, rx_ms - (500 if is_context else 0))
            ev = CanonicalEvent(
                schema_version=SCHEMA_VERSION,
                event_id=f"ds-{run_id}-{session_id}-{ref.segment_id}-r{rev}-{seq_counter}",
                session_id=session_id,
                source_id="dataset_replay",
                kind=EventKind.SEGMENT,
                received_ms=rx_ms,
                source_sequence=seq_counter,
                segment_id=ref.segment_id,
                source_revision=rev,
                stability=Stability.COMMITTED,
                channel=eff_channel,
                speaker_id=eff_speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                text=seg_text,
            )
            res_apply = state.apply(ev)
            if res_apply.accepted:
                snap_now = state.snapshot()
                actual_rev = (
                    snap_now.segments_by_id[ref.segment_id].revision
                    if ref.segment_id in snap_now.segments_by_id
                    else rev
                )
                applied_seg_revs[ref.segment_id] = actual_rev
                if store is not None and should_record:
                    store.record_canonical_event(
                        run_id=run_id,
                        event=ev,
                        retention=retention,
                        canonical_revision=actual_rev,
                    )
    return seq_counter


async def _simulate_one_dataset_session(
    *,
    sid: str,
    items: list[tuple[ClassifierInput, ClassifierResult]],
    sess_brief: SessionBrief,
    policy: PolicyConfig,
    cal_obj: CalibrationModel | CalibrationRegistry | None,
    tpl_map: Mapping[str, TemplateSpec],
    store: SQLiteStore | None,
    run_id: str,
    mode: Mode,
    provider: ProviderAdapter | None,
    eff_health: HealthState,
    stage_latency_ms: float,
    health_override: HealthState | None = None,
    local_blocked: bool = False,
) -> ReplayResult:
    state = CanonicalState(sess_brief)
    applied_seg_revs: dict[str, int] = {}
    seq_counter = 0
    evt_seq = 0
    ledger_texts: dict[str, str] = {}
    ledger_categories: dict[str, Category] = {}
    prior_anchor_texts: dict[str, str] = {}
    displayed_times: list[int] = []
    displayed_cards: list[DisplayedCardEvent] = []
    active_until_ms: int | None = None
    last_proactive_display_ms: int | None = None
    decisions_count = 0
    routed_count = 0
    withdrawn_count = 0
    expired_count = 0

    anchor_seg_revs: set[tuple[str, int]] = set()
    anchor_text_by_ref: dict[tuple[str, int], str] = {}
    for cin_item, _ in items:
        item_anchor_text = _extract_anchor_text_from_serialized(cin_item.serialized_text)
        for r in cin_item.context.anchor_refs:
            key = (r.segment_id, r.revision)
            anchor_seg_revs.add(key)
            if item_anchor_text and key not in anchor_text_by_ref:
                anchor_text_by_ref[key] = item_anchor_text

    rx_times = [c.context.anchor_received_ms for c, _ in items]
    started_ms = min(rx_times, default=0)
    cur_health = eff_health
    health_timeline: list[dict[str, Any]] = [
        {
            "at_ms": started_ms,
            "state": cur_health.value,
            "health": cur_health.value,
            "reason": "dataset_replay",
        }
    ]

    for cin, clf_raw in items:
        ctx = cin.context
        anchor_text = _extract_anchor_text_from_serialized(cin.serialized_text)
        ctx_turns = _extract_context_turns_from_serialized(cin.serialized_text)
        seq_counter = _ensure_dataset_segment_revs(
            state=state,
            applied_seg_revs=applied_seg_revs,
            seq_counter=seq_counter,
            refs=ctx.context_refs,
            text="",
            rx_ms=max(0, ctx.anchor_received_ms - 500),
            session_id=sid,
            run_id=run_id,
            retention=sess_brief.retention,
            store=store,
            is_context=True,
            anchor_seg_revs=anchor_seg_revs,
            anchor_text_by_ref=anchor_text_by_ref,
            context_turns=ctx_turns,
        )
        seq_counter = _ensure_dataset_segment_revs(
            state=state,
            applied_seg_revs=applied_seg_revs,
            seq_counter=seq_counter,
            refs=ctx.anchor_refs,
            text=anchor_text,
            rx_ms=ctx.anchor_received_ms,
            session_id=sid,
            run_id=run_id,
            retention=sess_brief.retention,
            store=store,
            is_context=False,
        )
        snap = state.snapshot()
        clf_res = _calibrate_dataset_result(cal_obj, ctx, clf_raw, policy, snap.brief.mode)
        decisions_count += 1
        route_health = cur_health
        if health_override is None and provider is not None and not local_blocked:
            brk_state = provider.advance_clock(ctx.anchor_received_ms)
            if brk_state is BreakerState.HALF_OPEN and provider.half_open_probe_available:
                route_health = HealthState.OK
            else:
                route_health = provider.health_state
                if route_health is not cur_health:
                    cur_health = route_health
                    health_timeline.append(
                        {
                            "at_ms": ctx.anchor_received_ms,
                            "state": cur_health.value,
                            "health": cur_health.value,
                            "reason": "breaker",
                        }
                    )
        routing = route_candidate(
            ctx,
            clf_res,
            snap,
            policy,
            mode=mode,
            health=route_health,
            now_ms=ctx.anchor_received_ms,
            ledger_texts=ledger_texts,
            ledger_categories=ledger_categories,
            prior_anchor_texts=prior_anchor_texts,
            anchor_text=anchor_text or None,
            recent_displayed_ms=displayed_times,
        )
        if routing.outcome is not Outcome.ROUTE or routing.category is None:
            t_state, t_reason = _dataset_non_routed_state(routing)
            if store is not None:
                store.record_decision(
                    run_id=run_id,
                    context=ctx,
                    terminal_state=t_state,
                    terminal_reason=t_reason,
                    classifier=clf_res,
                    routing=routing,
                    serialized_text=cin.serialized_text,
                    retention=sess_brief.retention,
                )
            continue

        cand, eff_routing, fail_reason = await _render_dataset_candidate(
            ctx=ctx,
            routing=routing,
            snap=snap,
            policy=policy,
            tpl_map=tpl_map,
            provider=provider,
            store=store,
            run_id=run_id,
            ledger_texts=ledger_texts,
            ledger_categories=ledger_categories,
        )
        if health_override is None and provider is not None and not local_blocked:
            post_health = provider.health_state
            if post_health is not cur_health:
                cur_health = post_health
                health_timeline.append(
                    {
                        "at_ms": ctx.anchor_received_ms,
                        "state": cur_health.value,
                        "health": cur_health.value,
                        "reason": "breaker",
                    }
                )
        if cand is None:
            t_reason = fail_reason or ReasonCode.SLOT_UNAVAILABLE
            t_state = "suppressed" if t_reason in SUPPRESS_REASON_CODES else "unrendered"
            if t_state == "suppressed":
                eff_routing = replace(
                    eff_routing, outcome=Outcome.SUPPRESS, route=None, reason_codes=(t_reason,)
                )
            if store is not None:
                store.record_decision(
                    run_id=run_id,
                    context=ctx,
                    terminal_state=t_state,
                    terminal_reason=t_reason,
                    classifier=clf_res,
                    routing=eff_routing,
                    serialized_text=cin.serialized_text,
                    retention=sess_brief.retention,
                )
            continue

        routed_count += 1
        is_manual = ctx.origin is Origin.MANUAL
        display_ms = (
            ctx.anchor_received_ms
            if is_manual
            else ctx.anchor_received_ms + policy.display.settle_ms
        )
        if not is_manual:
            if active_until_ms is not None:
                display_ms = max(display_ms, active_until_ms)
            if last_proactive_display_ms is not None:
                display_ms = max(display_ms, last_proactive_display_ms + policy.display.cooldown_ms)

        if display_ms > ctx.display_by_ms:
            expired_count += 1
            if store is not None:
                store.record_decision(
                    run_id=run_id,
                    context=ctx,
                    terminal_state="expired",
                    terminal_reason=ReasonCode.DISPLAY_BY_PASSED,
                    classifier=clf_res,
                    routing=eff_routing,
                    serialized_text=cin.serialized_text,
                    retention=sess_brief.retention,
                )
            continue

        cap = effective_max_cards_per_10_min(snap.brief, policy)
        recent = [t for t in displayed_times if (display_ms - t) <= 600_000]
        if not is_manual and (cap <= 0 or len(recent) >= cap):
            withdrawn_count += 1
            sup_rt = replace(
                eff_routing,
                outcome=Outcome.SUPPRESS,
                route=None,
                reason_codes=(ReasonCode.PROACTIVE_DISABLED,),
            )
            if store is not None:
                store.record_decision(
                    run_id=run_id,
                    context=ctx,
                    terminal_state="suppressed",
                    terminal_reason=ReasonCode.PROACTIVE_DISABLED,
                    classifier=clf_res,
                    routing=sup_rt,
                    serialized_text=cin.serialized_text,
                    retention=sess_brief.retention,
                )
            continue

        state.register_suggestion(cand)
        ledger_texts[cand.suggestion_id] = cand.text
        ledger_categories[cand.suggestion_id] = cand.category
        if anchor_text:
            prior_anchor_texts[cand.candidate_id] = anchor_text
        evt_seq += 1
        disp_ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=f"ds-{run_id}-disp-{evt_seq:06d}",
            suggestion_id=cand.suggestion_id,
            transition=LedgerState.DISPLAYED,
            cause=ReasonCode.MANUAL_REQUEST if is_manual else ReasonCode.SETTLE_TIMEOUT,
            at_ms=display_ms,
            evidence_refs=cand.evidence_refs,
        )
        state.apply(disp_ev.to_canonical_event(sid, received_ms=display_ms))
        exp_ms = display_ms + cand.display_ttl_ms
        evt_seq += 1
        exp_ev = LedgerEvent(
            schema_version=SCHEMA_VERSION,
            event_id=f"ds-{run_id}-exp-{evt_seq:06d}",
            suggestion_id=cand.suggestion_id,
            transition=LedgerState.EXPIRED,
            cause=ReasonCode.DISPLAY_TTL_PASSED,
            at_ms=exp_ms,
            evidence_refs=cand.evidence_refs,
        )
        state.apply(exp_ev.to_canonical_event(sid, received_ms=exp_ms))
        expired_count += 1
        if not is_manual:
            last_proactive_display_ms = display_ms
        active_until_ms = exp_ms
        displayed_times.append(display_ms)
        displayed_cards.append(
            DisplayedCardEvent(
                suggestion_id=cand.suggestion_id,
                candidate_id=cand.candidate_id,
                session_id=sid,
                category=cand.category,
                display_ms=display_ms,
                evidence_refs=cand.evidence_refs,
                text=cand.text,
                route=cand.route,
                origin=cand.origin,
            )
        )
        if store is not None:
            store.record_suggestion(
                session_id=sid,
                run_id=run_id,
                suggestion=cand,
                current_state="ready",
                terminal_reason=None,
                transitions=(),
                retention=sess_brief.retention,
            )
            store.record_ledger_event(session_id=sid, run_id=run_id, event=disp_ev)
            store.record_ledger_event(session_id=sid, run_id=run_id, event=exp_ev)
            store.record_decision(
                run_id=run_id,
                context=ctx,
                terminal_state="expired",
                terminal_reason=ReasonCode.DISPLAY_TTL_PASSED,
                classifier=clf_res,
                routing=eff_routing,
                serialized_text=cin.serialized_text,
                retention=sess_brief.retention,
            )

    final_snap = state.snapshot()
    dur_ms = max(1, (max(rx_times) - min(rx_times)) if len(rx_times) >= 2 else 1)
    ended_ms = max([*rx_times, *displayed_times], default=started_ms)
    if store is not None:
        store.upsert_session(
            brief=final_snap.brief,
            run_id=run_id,
            runtime_mode=mode,
            effective_max_cards_per_10_min=effective_max_cards_per_10_min(final_snap.brief, policy),
            health_timeline=health_timeline,
            reducer_counters=final_snap.counters,
            revision_history=final_snap.revision_history,
            started_ms=started_ms,
            ended_ms=ended_ms,
        )
    return ReplayResult(
        session_id=sid,
        run_id=run_id,
        duration_ms=dur_ms,
        displayed_cards=tuple(displayed_cards),
        decisions_count=decisions_count,
        dropped_count=0,
        routed_count=routed_count,
        withdrawn_count=withdrawn_count,
        expired_count=expired_count,
        latencies_ms=(stage_latency_ms,),
    )


async def async_simulate_dataset_replay(
    dataset_rows: Sequence[Mapping[str, Any]],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    *,
    brief: SessionBrief | None = None,
    calibration: CalibrationModel | CalibrationRegistry | None = None,
    templates: Mapping[str, TemplateSpec] | None = None,
    store: SQLiteStore | None = None,
    run_id: str = "replay",
    mode: Mode = Mode.PROACTIVE,
    provider: ProviderAdapter | None = None,
    health: HealthState | None = None,
    simulated_latency_ms: float | None = None,
    cache_classifier: bool = True,
) -> list[ReplayResult]:
    """Run policy-level replay evaluation over pre-built DecisionContext dataset rows."""
    inputs = [_parse_dataset_classifier_input(row) for row in dataset_rows]
    if not inputs:
        return []
    tpl_map = dict(templates) if templates is not None else load_templates_dir()
    cal_obj: CalibrationModel | CalibrationRegistry | None = (
        calibration
        if calibration is not None
        else (default_calibration_registry() if policy.classifier.calibration else None)
    )
    local_blocked = (
        provider is not None and provider.config.local_only and provider.config.effective_is_hosted
    )

    t0 = time.perf_counter()
    if cache_classifier and store is not None:
        clf_results = await classify_batch_cached(classifier, inputs, store=store)
    else:
        clf_results = await classifier.classify_batch(inputs)
    measured_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
    stage_ms = float(
        max(0, int(round(simulated_latency_ms)))
        if simulated_latency_ms is not None
        else measured_ms
    )

    by_session: dict[str, list[tuple[ClassifierInput, ClassifierResult]]] = {}
    for cin, clf_raw in zip(inputs, clf_results, strict=True):
        by_session.setdefault(cin.context.session_id, []).append((cin, clf_raw))

    results: list[ReplayResult] = []
    for sid in sorted(by_session):
        sess_brief = (
            replace(brief, session_id=sid)
            if brief is not None
            else SessionBrief(
                schema_version=SCHEMA_VERSION,
                session_id=sid,
                objective="Evaluate dataset replay candidates.",
                recipient=RecipientConfig(participant_id="ME", role="participant"),
                mode=policy.mode,
                retention=RetentionConfig(),
                attention=AttentionConfig(max_cards_per_10_min=3, proactive=True),
            )
        )
        items = sorted(
            by_session[sid],
            key=lambda pair: (pair[0].context.anchor_received_ms, pair[0].context.candidate_id),
        )
        iter_provider = provider.clone_isolated() if provider is not None else None
        sess_eff_health = (
            health
            if health is not None
            else (
                HealthState.TEMPLATE_ONLY
                if (local_blocked or iter_provider is None)
                else iter_provider.health_state
            )
        )
        try:
            results.append(
                await _simulate_one_dataset_session(
                    sid=sid,
                    items=items,
                    sess_brief=sess_brief,
                    policy=policy,
                    cal_obj=cal_obj,
                    tpl_map=tpl_map,
                    store=store,
                    run_id=run_id,
                    mode=mode,
                    provider=iter_provider,
                    eff_health=sess_eff_health,
                    stage_latency_ms=stage_ms,
                    health_override=health,
                    local_blocked=local_blocked,
                )
            )
        finally:
            if iter_provider is not None:
                await iter_provider.cancel_recovery()
    return results


def simulate_dataset_replay(
    dataset_rows: Sequence[Mapping[str, Any]],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    **kwargs: Any,
) -> list[ReplayResult]:
    """Synchronous wrapper for async_simulate_dataset_replay."""
    return asyncio.run(async_simulate_dataset_replay(dataset_rows, policy, classifier, **kwargs))


def _validate_t_display(value: float) -> float:
    t = float(value)
    if not math.isfinite(t) or t < 0.0 or t > 1.0:
        raise ValueError(f"t_display must be finite and within [0.0, 1.0], got {value!r}")
    return t


def _policy_with_t_display(policy: PolicyConfig, t_display: float) -> PolicyConfig:
    t_disp = _validate_t_display(t_display)
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
    mode: Mode = Mode.PROACTIVE,
    provider: ProviderAdapter | None = None,
    health: HealthState | None = None,
    simulated_latency_ms: float | None = None,
    latency_distribution_ms: Sequence[float] | None = None,
    cache_classifier: bool = True,
) -> list[dict[str, Any]]:
    """Sweep `t_display` through policy-level replay using an isolated store (§12.8)."""
    if not t_display_values:
        raise ValueError("t_display_values must be non-empty")
    validated_t = sorted(_validate_t_display(v) for v in t_display_values)
    _ = store  # Isolation: sweep runs never pollute caller's persistent DB.
    curve: list[dict[str, Any]] = []
    with SQLiteStore(":memory:") as sweep_store:
        for t_disp in validated_t:
            swept_policy = _policy_with_t_display(policy, t_disp)
            all_cards: list[DisplayedCardEvent] = []
            total_dur_ms = 0
            max_cap_10m = 0
            for idx, (brief, events) in enumerate(sessions):
                iter_provider = provider.clone_isolated() if provider is not None else None
                try:
                    res = await async_simulate_replay(
                        brief,
                        events,
                        swept_policy,
                        classifier,
                        calibration=calibration,
                        templates=templates,
                        store=sweep_store if cache_classifier else None,
                        run_id=f"sweep-{t_disp:.4f}-{idx}",
                        mode=mode,
                        provider=iter_provider,
                        health=health,
                        simulated_latency_ms=simulated_latency_ms,
                        latency_distribution_ms=latency_distribution_ms,
                        cache_classifier=cache_classifier,
                    )
                finally:
                    if iter_provider is not None:
                        await iter_provider.cancel_recovery()
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


async def async_sweep_dataset_t_display(
    dataset_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[LabelRecord],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    t_display_values: Sequence[float],
    *,
    brief: SessionBrief | None = None,
    calibration: CalibrationModel | CalibrationRegistry | None = None,
    templates: Mapping[str, TemplateSpec] | None = None,
    mode: Mode = Mode.PROACTIVE,
    provider: ProviderAdapter | None = None,
    health: HealthState | None = None,
    simulated_latency_ms: float | None = None,
    cache_classifier: bool = True,
) -> list[dict[str, Any]]:
    """Sweep `t_display` through policy-level dataset replay using an isolated store (§12.8)."""
    if not t_display_values:
        raise ValueError("t_display_values must be non-empty")
    validated_t = sorted(_validate_t_display(v) for v in t_display_values)
    curve: list[dict[str, Any]] = []
    with SQLiteStore(":memory:") as sweep_store:
        for idx, t_disp in enumerate(validated_t):
            swept_policy = _policy_with_t_display(policy, t_disp)
            iter_provider = provider.clone_isolated() if provider is not None else None
            try:
                session_results = await async_simulate_dataset_replay(
                    dataset_rows,
                    swept_policy,
                    classifier,
                    brief=brief,
                    calibration=calibration,
                    templates=templates,
                    store=sweep_store if cache_classifier else None,
                    run_id=f"sweep-{idx}-{t_disp:.4f}",
                    mode=mode,
                    provider=iter_provider,
                    health=health,
                    simulated_latency_ms=simulated_latency_ms,
                    cache_classifier=cache_classifier,
                )
            finally:
                if iter_provider is not None:
                    await iter_provider.cancel_recovery()
            all_cards = [c for res in session_results for c in res.displayed_cards]
            total_dur_ms = sum(max(600_000, res.duration_ms) for res in session_results)
            max_cap_10m = 0
            if session_results:
                eff_brief = (
                    brief
                    if brief is not None
                    else SessionBrief(
                        schema_version=SCHEMA_VERSION,
                        session_id="sweep-default",
                        objective="Evaluate dataset replay candidates.",
                        recipient=RecipientConfig(participant_id="ME", role="participant"),
                        mode=swept_policy.mode,
                        retention=RetentionConfig(),
                        attention=AttentionConfig(max_cards_per_10_min=3, proactive=True),
                    )
                )
                max_cap_10m = effective_max_cards_per_10_min(eff_brief, swept_policy)

            summary = match_opportunities(
                all_cards,
                labels,
                settle_ms=swept_policy.display.settle_ms,
                duration_ms=total_dur_ms if session_results else None,
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


def sweep_dataset_t_display(
    dataset_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[LabelRecord],
    policy: PolicyConfig,
    classifier: OpportunityClassifier,
    t_display_values: Sequence[float],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for async_sweep_dataset_t_display."""
    return asyncio.run(
        async_sweep_dataset_t_display(
            dataset_rows, labels, policy, classifier, t_display_values, **kwargs
        )
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


_DECISION_CONTEXT_KEYS = frozenset(
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
            elif "candidate_id" in obj and "serialized_text" in obj and "kind" not in obj:
                ctx_payload = {k: v for k, v in obj.items() if k in _DECISION_CONTEXT_KEYS}
                ctx = DecisionContext.from_dict(ctx_payload)
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
