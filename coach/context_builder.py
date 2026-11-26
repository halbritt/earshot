"""Context builder, policy loader, prefilter, and span rules (§5.2–§5.3, §5.6, §15, §16)."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from coach.canonical_state import (
    GapRecord,
    SegmentRecord,
    StateSnapshot,
)
from coach.contracts import (
    PREDICTABLE_CATEGORIES,
    Category,
    ContractValidationError,
    ControlSubtype,
    DecisionContext,
    GroundedSlot,
    Origin,
    ReasonCode,
    Route,
    SegmentRef,
    SessionBrief,
    SessionMode,
    SuggestionCandidate,
)

BUILDER_VERSION = "ctx-v0"
PREFILTER_VERSION = "pre-v0"
SPAN_EXTRACTOR_VERSION = "span_rules_v0"
DEFAULT_ANCHOR_MAX_TOKENS = 60
DEFAULT_MAX_CANDIDATE_AGE_MS = 8000
DEFAULT_GENERIC_MAX_ANCHOR_AGE_MS = 6000


class PolicyConfigError(ContractValidationError):
    """Raised when policy configuration or calibration matching fails."""


def _p_str(val: Any, name: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise PolicyConfigError(f"{name} must be a non-empty string")
    return val


def _p_int(val: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise PolicyConfigError(f"{name} must be an int >= {minimum}, got {val!r}")
    return val


def _p_float(val: Any, name: str, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise PolicyConfigError(f"{name} must be a float, got {val!r}")
    f = float(val)
    if not (minimum <= f <= maximum):
        raise PolicyConfigError(f"{name} must be in [{minimum}, {maximum}], got {f}")
    return f


def _p_bool(val: Any, name: str) -> bool:
    if not isinstance(val, bool):
        raise PolicyConfigError(f"{name} must be a bool, got {val!r}")
    return val


def _p_map(val: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(val, Mapping):
        raise PolicyConfigError(f"{name} must be a mapping")
    return val


def _p_opt_str(val: Any, name: str) -> str | None:
    if val is None:
        return None
    return _p_str(val, name)


@dataclass(frozen=True, slots=True)
class ClassifierPolicy:
    version: str
    calibration: str
    input_limit_tokens: int
    max_candidates_per_job: int
    model_path: str | None = None
    calibration_path: str | None = None


@dataclass(frozen=True, slots=True)
class CategoryPolicy:
    t_low: float
    t_compute: float
    t_display: float
    route: Route
    template: str | None
    required_lookback_s: int

    def __post_init__(self) -> None:
        if not (self.t_low <= self.t_compute <= self.t_display):
            raise PolicyConfigError(
                f"Thresholds must satisfy t_low <= t_compute <= t_display, "
                f"got {self.t_low}, {self.t_compute}, {self.t_display}"
            )
        if self.route is Route.TEMPLATE and not self.template:
            raise PolicyConfigError("CategoryPolicy with route='template' requires template")


@dataclass(frozen=True, slots=True)
class DisplayPolicy:
    max_active_cards: int
    settle_ms: int
    max_candidate_age_ms: int
    display_ttl_ms: int
    cooldown_ms: int
    max_cards_per_10_min: int


@dataclass(frozen=True, slots=True)
class ManualPolicy:
    lookback_s: int
    score_floor: float
    budget_ms: int


@dataclass(frozen=True, slots=True)
class BreakerPolicy:
    consecutive_failures: int
    open_ms: int


@dataclass(frozen=True, slots=True)
class GeneratorPolicy:
    enabled: bool
    max_words: int
    timeout_ms: int
    breaker: BreakerPolicy
    provider: str | None = None
    model_id: str | None = None
    reasoning: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class AbstainPolicy:
    speaker_settle_ms: int
    asr_confidence_floor: float | None


@dataclass(frozen=True, slots=True)
class HealthPolicy:
    no_transcript_after_ms: int


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    policy_version: str
    mode: SessionMode
    classifier: ClassifierPolicy
    categories: dict[Category, CategoryPolicy]
    display: DisplayPolicy
    manual: ManualPolicy
    generator: GeneratorPolicy
    abstain: AbstainPolicy
    health: HealthPolicy
    policy_sha256: str


def compute_policy_sha256(raw_bytes: bytes) -> str:
    """Compute immutable SHA-256 hex digest of raw policy file bytes (§16)."""
    return hashlib.sha256(raw_bytes).hexdigest()


def _verify_calibration_match(
    policy: PolicyConfig,
    calibration_metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> None:
    cal_name = policy.classifier.calibration
    entry: Mapping[str, Any] | None = None
    if isinstance(calibration_metadata, Mapping):
        if "calibration_version" in calibration_metadata:
            if calibration_metadata.get("calibration_version") == cal_name:
                entry = calibration_metadata
        elif cal_name in calibration_metadata:
            candidate = calibration_metadata[cal_name]
            if isinstance(candidate, Mapping):
                entry = candidate
    else:
        for item in calibration_metadata:
            if item.get("calibration_version") == cal_name:
                entry = item
                break
    if entry is None:
        raise PolicyConfigError(f"Calibration metadata missing entry for {cal_name!r}")
    cls_ver = entry.get("classifier_version")
    cal_mode = entry.get("mode")
    if cls_ver != policy.classifier.version:
        raise PolicyConfigError(
            f"Calibration {cal_name!r} classifier_version {cls_ver!r} "
            f"does not match policy {policy.classifier.version!r}"
        )
    if str(cal_mode) != str(policy.mode):
        raise PolicyConfigError(
            f"Calibration {cal_name!r} mode {cal_mode!r} does not match policy {policy.mode!r}"
        )


def load_policy(
    path: str | Path,
    *,
    calibration_metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> PolicyConfig:
    """Load, validate, hash, and optionally calibration-verify a YAML policy (§16)."""
    raw_bytes = Path(path).read_bytes()
    policy_sha256 = compute_policy_sha256(raw_bytes)
    loaded = yaml.safe_load(raw_bytes.decode("utf-8"))
    m = _p_map(loaded, "policy")
    try:
        mode = SessionMode(_p_str(m.get("mode"), "policy.mode"))
    except ValueError as exc:
        raise PolicyConfigError(f"Invalid policy.mode: {m.get('mode')!r}") from exc

    cls_m = _p_map(m.get("classifier"), "policy.classifier")
    classifier = ClassifierPolicy(
        version=_p_str(cls_m.get("version"), "classifier.version"),
        calibration=_p_str(cls_m.get("calibration"), "classifier.calibration"),
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
    policy = PolicyConfig(
        policy_version=_p_str(m.get("policy_version"), "policy.policy_version"),
        mode=mode,
        classifier=classifier,
        categories=categories,
        display=display,
        manual=manual,
        generator=generator,
        abstain=abstain,
        health=health,
        policy_sha256=policy_sha256,
    )
    if calibration_metadata is not None:
        _verify_calibration_match(policy, calibration_metadata)
    return policy


def effective_max_cards_per_10_min(brief: SessionBrief, policy: PolicyConfig) -> int:
    """Apply the stricter attention cap across SessionBrief and PolicyConfig (§16)."""
    return min(brief.attention.max_cards_per_10_min, policy.display.max_cards_per_10_min)


@runtime_checkable
class Tokenizer(Protocol):
    """Pluggable token counter matching the active classifier tokenizer (§5.3, §10.2)."""

    def count_tokens(self, text: str) -> int: ...


class WhitespaceTokenizer:
    """Deterministic fallback tokenizer counting whitespace-delimited tokens."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True, slots=True)
class AnchorSpan:
    """Coalesced non-recipient committed anchor span (§5.2)."""

    candidate_id: str
    session_id: str
    speaker_id: str
    segments: tuple[SegmentRecord, ...]
    anchor_refs: tuple[SegmentRef, ...]
    anchor_received_ms: int
    display_by_ms: int
    joined_text: str


@dataclass(frozen=True, slots=True)
class PrefilterDropRecord:
    """Audited prefilter drop record (§5.2, Appendix C)."""

    candidate_id: str
    session_id: str
    based_on_state_version: int
    anchor_refs: tuple[SegmentRef, ...]
    anchor_received_ms: int
    display_by_ms: int
    reason_code: ReasonCode
    prefilter_version: str = PREFILTER_VERSION


@dataclass(frozen=True, slots=True)
class PrefilterResult:
    """Outcome of running the deterministic audited prefilter."""

    passed: bool
    reason_code: ReasonCode | None = None
    drop_record: PrefilterDropRecord | None = None
    normalized_text: str = ""


def compute_candidate_id(session_id: str, anchor_refs: Sequence[SegmentRef]) -> str:
    """Deterministic SHA-256 of (session_id, [(segment_id, revision), ...]) (§5.2)."""
    payload = {
        "session_id": session_id,
        "anchor_refs": [[r.segment_id, r.revision] for r in anchor_refs],
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_SENTENCE_FINAL_RE = re.compile(r'[.?!]["\'”’\)]*\s*$')


def _make_anchor_span(
    session_id: str,
    buf: Sequence[SegmentRecord],
    max_candidate_age_ms: int,
) -> AnchorSpan:
    if not buf:
        raise ValueError("Anchor must contain at least one segment")
    segs = tuple(buf)
    refs = tuple(SegmentRef(s.segment_id, s.revision) for s in segs)
    cid = compute_candidate_id(session_id, refs)
    rx_ms = max(s.received_ms for s in segs)
    return AnchorSpan(
        candidate_id=cid,
        session_id=session_id,
        speaker_id=segs[0].speaker_id,
        segments=segs,
        anchor_refs=refs,
        anchor_received_ms=rx_ms,
        display_by_ms=rx_ms + max_candidate_age_ms,
        joined_text=" ".join(s.text.strip() for s in segs if s.text.strip()),
    )


def coalesce_anchors(
    snapshot: StateSnapshot,
    *,
    max_anchor_tokens: int = DEFAULT_ANCHOR_MAX_TOKENS,
    max_candidate_age_ms: int = DEFAULT_MAX_CANDIDATE_AGE_MS,
    tokenizer: Tokenizer | None = None,
    speaker_boundary_end_ms: Mapping[str, int] | None = None,
    pause_gap_ms: int | None = None,
    segments_subset: Sequence[SegmentRecord] | None = None,
) -> list[AnchorSpan]:
    """Coalesce non-recipient committed segments into candidate anchors (§5.2, B19)."""
    tok = tokenizer or WhitespaceTokenizer()
    recipient_ids = {"ME", snapshot.brief.recipient.participant_id}
    source = (
        tuple(
            sorted(
                (s for s in segments_subset if s.committed),
                key=lambda s: (s.start_ms, s.channel, s.source_sequence, s.segment_id),
            )
        )
        if segments_subset is not None
        else snapshot.segments
    )
    anchors: list[AnchorSpan] = []
    buf: list[SegmentRecord] = []

    def _flush() -> None:
        if buf:
            anchors.append(_make_anchor_span(snapshot.session_id, buf, max_candidate_age_ms))
            buf.clear()

    for idx, seg in enumerate(source):
        if seg.speaker_id in recipient_ids:
            _flush()
            continue
        if buf:
            prev = buf[-1]
            speaker_changed = prev.speaker_id != seg.speaker_id
            overlap_cross_channel = (prev.overlap or seg.overlap) and prev.channel != seg.channel
            joined_if_added = " ".join(s.text.strip() for s in [*buf, seg] if s.text.strip())
            exceeds_tokens = tok.count_tokens(joined_if_added) > max_anchor_tokens
            if speaker_changed or overlap_cross_channel or exceeds_tokens:
                _flush()
        buf.append(seg)
        next_seg = source[idx + 1] if idx + 1 < len(source) else None
        has_punct = _SENTENCE_FINAL_RE.search(seg.text) is not None
        has_seg_boundary = seg.boundary is not None and seg.boundary != "segment"
        bnd_end = (
            speaker_boundary_end_ms.get(seg.speaker_id)
            if speaker_boundary_end_ms is not None
            else None
        )
        has_speaker_boundary = (
            bnd_end is not None
            and bnd_end >= seg.end_ms
            and (
                next_seg is None
                or next_seg.speaker_id != seg.speaker_id
                or bnd_end < next_seg.start_ms
            )
        )
        has_pause_gap = (
            pause_gap_ms is not None
            and next_seg is not None
            and next_seg.speaker_id == seg.speaker_id
            and (next_seg.start_ms - seg.end_ms) >= pause_gap_ms
        )
        cur_text = " ".join(s.text.strip() for s in buf if s.text.strip())
        at_cap = tok.count_tokens(cur_text) >= max_anchor_tokens
        if has_punct or has_seg_boundary or has_speaker_boundary or has_pause_gap or at_cap:
            _flush()
    _flush()
    return anchors


DEFAULT_STOCK_PHRASES: frozenset[str] = frozenset(
    {
        "thank you",
        "thank you very much",
        "thank you for watching",
        "thanks for watching",
        "thanks for listening",
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "goodbye",
        "bye",
        "bye bye",
        "subtitles by the amara org community",
    }
)

_BACKCHANNELS: frozenset[str] = frozenset(
    {
        "yeah",
        "yes",
        "yep",
        "yup",
        "no",
        "nope",
        "nah",
        "right",
        "exactly",
        "true",
        "mmhm",
        "mm hm",
        "mhm",
        "uh huh",
        "uhhuh",
        "okay",
        "ok",
        "sure",
        "got it",
        "gotcha",
        "makes sense",
        "i see",
        "hmm",
        "hm",
        "aha",
        "uh",
        "um",
        "cool",
        "nice",
        "great",
        "sounds good",
        "yeah right",
        "oh okay",
        "alright",
        "all right",
    }
)

_FILLER_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "so",
        "if",
        "then",
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
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "they",
        "them",
        "their",
        "he",
        "she",
        "uh",
        "um",
        "like",
        "just",
        "well",
        "yeah",
        "okay",
        "ok",
        "right",
    }
)

_PUNCT_STRIP_RE = re.compile(r"(?:[^\w\s]|_)", flags=re.UNICODE)


def normalize_for_matching(text: str) -> str:
    """NFC normalize, lowercase, strip punctuation, and collapse whitespace."""
    nfc = unicodedata.normalize("NFC", text).lower()
    cleaned = _PUNCT_STRIP_RE.sub(" ", nfc)
    return " ".join(cleaned.split())


def run_prefilter(
    anchor: AnchorSpan,
    *,
    state_version: int = 0,
    previous_anchor_norm_text: str | None = None,
    stock_phrases: frozenset[str] = DEFAULT_STOCK_PHRASES,
) -> PrefilterResult:
    """Audited deterministic prefilter (§5.2, Appendix C, B21)."""
    norm = normalize_for_matching(anchor.joined_text)
    seg_norms = [normalize_for_matching(s.text) for s in anchor.segments]

    def _drop(reason: ReasonCode) -> PrefilterResult:
        rec = PrefilterDropRecord(
            candidate_id=anchor.candidate_id,
            session_id=anchor.session_id,
            based_on_state_version=state_version,
            anchor_refs=anchor.anchor_refs,
            anchor_received_ms=anchor.anchor_received_ms,
            display_by_ms=anchor.display_by_ms,
            reason_code=reason,
        )
        return PrefilterResult(
            passed=False,
            reason_code=reason,
            drop_record=rec,
            normalized_text=norm,
        )

    if norm in stock_phrases or (seg_norms and all(sn in stock_phrases for sn in seg_norms)):
        return _drop(ReasonCode.STOCK_PHRASE)
    clauses = [
        c_norm
        for part in re.split(r"[,;.!?]+", unicodedata.normalize("NFC", anchor.joined_text))
        if (c_norm := normalize_for_matching(part))
    ]
    if norm in _BACKCHANNELS or (clauses and all(c in _BACKCHANNELS for c in clauses)):
        return _drop(ReasonCode.BACKCHANNEL)
    if previous_anchor_norm_text is not None and norm == previous_anchor_norm_text:
        return _drop(ReasonCode.REPEAT)

    content_tokens = [tok for tok in norm.split() if tok not in _FILLER_STOPWORDS]
    if len(content_tokens) < 4:
        return _drop(ReasonCode.TOO_SHORT)

    return PrefilterResult(passed=True, reason_code=None, drop_record=None, normalized_text=norm)


_ROLE_LABEL_RE = re.compile(r"^S(\d+)$")


def assign_speaker_roles(snapshot: StateSnapshot) -> dict[str, str]:
    """Map speaker_ids deterministically to 'ME', 'S1'..'Sn', or 'S?' (§5.3)."""
    recipient_ids = {"ME", snapshot.brief.recipient.participant_id}
    mapping: dict[str, str] = {rid: "ME" for rid in recipient_ids}
    used_labels: set[str] = set()

    for p in snapshot.brief.participants:
        if p.participant_id in recipient_ids:
            continue
        if _ROLE_LABEL_RE.match(p.label) and p.label not in used_labels:
            mapping[p.participant_id] = p.label
            used_labels.add(p.label)
        elif _ROLE_LABEL_RE.match(p.participant_id) and p.participant_id not in used_labels:
            mapping[p.participant_id] = p.participant_id
            used_labels.add(p.participant_id)

    next_idx = 1

    def _alloc() -> str:
        nonlocal next_idx
        while f"S{next_idx}" in used_labels:
            next_idx += 1
        lbl = f"S{next_idx}"
        used_labels.add(lbl)
        next_idx += 1
        return lbl

    for p in snapshot.brief.participants:
        if p.participant_id not in mapping:
            mapping[p.participant_id] = _alloc()

    has_declared_non_recipient = any(
        p.participant_id not in recipient_ids for p in snapshot.brief.participants
    )
    for seg in snapshot.segments:
        spk = seg.speaker_id
        if spk in recipient_ids or spk in {"S?", "unknown", ""}:
            continue
        if spk not in mapping:
            if _ROLE_LABEL_RE.match(spk):
                mapping[spk] = spk
                used_labels.add(spk)
            elif not has_declared_non_recipient:
                mapping[spk] = _alloc()
    return mapping


def bucket_age_ms(age_ms: int) -> str:
    """Bucket an age in milliseconds into §5.3 discrete labels."""
    clamped = max(0, age_ms)
    if clamped < 30_000:
        return "<30s"
    if clamped < 120_000:
        return "<2m"
    if clamped < 600_000:
        return "<10m"
    return "older"


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Single-serializer output for one candidate anchor (§5.3, §15)."""

    anchor: AnchorSpan
    dropped: bool
    drop_record: PrefilterDropRecord | None
    context: DecisionContext | None
    serialized_text: str | None


def _speaker_label_for_segment(
    seg: SegmentRecord,
    role_map: Mapping[str, str],
) -> str:
    if seg.speaker_uncertain or seg.speaker_id in {"S?", "unknown", ""}:
        return "S?"
    return role_map.get(seg.speaker_id, "S?")


def _build_role_line(
    snapshot: StateSnapshot,
    role_map: Mapping[str, str],
    used_roles: set[str],
) -> str:
    participant_roles: dict[str, str] = {}
    for p in snapshot.brief.participants:
        lbl = role_map.get(p.participant_id)
        if lbl and lbl != "ME":
            participant_roles[lbl] = p.role or "unknown"
    for lbl in used_roles:
        if lbl not in {"ME", "S?"} and lbl not in participant_roles:
            participant_roles[lbl] = "unknown"

    def _sort_key(lbl: str) -> tuple[int, str]:
        m = _ROLE_LABEL_RE.match(lbl)
        return (int(m.group(1)), lbl) if m else (10_000, lbl)

    parts = [f"ME={snapshot.brief.recipient.role or 'unknown'}"]
    for lbl in sorted(participant_roles, key=_sort_key):
        parts.append(f"{lbl}={participant_roles[lbl]}")
    if "S?" in used_roles:
        parts.append("S?=unknown")
    return "[ROLE] " + " | ".join(parts)


def _format_suggestion_line(
    entry_state: str,
    category_str: str,
    text: str,
    age_ms: int,
) -> str:
    return f"[SUGGESTED] {bucket_age_ms(age_ms)} {category_str} {entry_state}: {text}"


def _gap_included(
    gap: GapRecord,
    anchor_end_ms: int,
    anchor_received_ms: int,
) -> bool:
    if gap.end_ms is not None:
        return gap.end_ms <= anchor_end_ms
    if gap.start_ms is not None:
        return gap.start_ms <= anchor_end_ms
    return gap.received_ms <= anchor_received_ms


def _gap_sort_audio_ms(
    gap: GapRecord,
    prior_turns: Sequence[SegmentRecord],
    anchor_start_ms: int,
) -> int:
    if gap.end_ms is not None:
        return gap.end_ms
    if gap.start_ms is not None:
        return gap.start_ms
    for turn in prior_turns:
        if turn.received_ms > gap.received_ms:
            return turn.start_ms
    return anchor_start_ms


def _render_context_lines(
    retained_turns: Sequence[SegmentRecord],
    relevant_gaps: Sequence[tuple[int, str]],
    role_map: Mapping[str, str],
    anchor_received_ms: int | None = None,
) -> list[str]:
    if not retained_turns and not relevant_gaps:
        return []
    lines: list[str] = ["[CONTEXT]"]
    gap_idx = 0
    for turn in retained_turns:
        while gap_idx < len(relevant_gaps) and relevant_gaps[gap_idx][0] <= turn.start_ms:
            lines.append(relevant_gaps[gap_idx][1])
            gap_idx += 1
        lbl = _speaker_label_for_segment(turn, role_map)
        if anchor_received_ms is not None:
            age_tag = bucket_age_ms(max(0, anchor_received_ms - turn.end_ms))
            lines.append(f"{age_tag} {lbl}: {turn.text.strip()}")
        else:
            lines.append(f"{lbl}: {turn.text.strip()}")
    while gap_idx < len(relevant_gaps):
        lines.append(relevant_gaps[gap_idx][1])
        gap_idx += 1
    return lines


def build_decision_context(
    snapshot: StateSnapshot,
    anchor: AnchorSpan | Sequence[SegmentRecord],
    policy: PolicyConfig | None = None,
    *,
    origin: Origin = Origin.PROACTIVE,
    tokenizer: Tokenizer | None = None,
    input_limit_tokens: int | None = None,
    max_candidate_age_ms: int | None = None,
    suggestion_catalog: Mapping[str, SuggestionCandidate] | None = None,
    default_required_lookback_s: int = 300,
) -> ContextBuildResult:
    """Serialize one anchor into DecisionContext using the sole §5.3 serializer."""
    tok = tokenizer or WhitespaceTokenizer()
    age_cap = (
        max_candidate_age_ms
        if max_candidate_age_ms is not None
        else (policy.display.max_candidate_age_ms if policy else DEFAULT_MAX_CANDIDATE_AGE_MS)
    )
    if isinstance(anchor, AnchorSpan):
        if not anchor.segments:
            raise ValueError("Anchor must contain at least one segment")
        anchor_span = anchor
    else:
        anchor_span = _make_anchor_span(snapshot.session_id, anchor, age_cap)
    limit = (
        input_limit_tokens
        if input_limit_tokens is not None
        else (policy.classifier.input_limit_tokens if policy else 512)
    )
    if limit <= 512:
        header_max, suggested_max = 96, 64
    else:
        header_max, suggested_max = 128, 128

    role_map = assign_speaker_roles(snapshot)
    anchor_ids = {s.segment_id for s in anchor_span.segments}
    first_anchor = min(
        anchor_span.segments,
        key=lambda s: (s.start_ms, s.channel, s.source_sequence, s.segment_id),
    )
    first_anchor_key = (
        first_anchor.start_ms,
        first_anchor.channel,
        first_anchor.source_sequence,
        first_anchor.segment_id,
    )
    prior_turns = [
        s
        for s in snapshot.segments
        if s.segment_id not in anchor_ids
        and (s.start_ms, s.channel, s.source_sequence, s.segment_id) < first_anchor_key
    ]
    anchor_start_ms = min(s.start_ms for s in anchor_span.segments)
    anchor_end_ms = max(s.end_ms for s in anchor_span.segments)

    relevant_gaps: list[tuple[int, str]] = []
    has_gap = False
    has_off_record_gap = False
    for g in snapshot.gaps:
        if _gap_included(g, anchor_end_ms, anchor_span.anchor_received_ms):
            has_gap = True
            is_off = g.subtype is ControlSubtype.OFF_RECORD_END or g.reason == "off_record"
            if is_off:
                has_off_record_gap = True
            label = "off_record" if is_off else (g.reason or str(g.subtype))
            gts = _gap_sort_audio_ms(g, prior_turns, anchor_start_ms)
            relevant_gaps.append((gts, f"[GAP: {label}]"))
    relevant_gaps.sort(key=lambda item: item[0])

    anchor_roles = {_speaker_label_for_segment(s, role_map) for s in anchor_span.segments}
    anchor_role = next(iter(anchor_roles)) if len(anchor_roles) == 1 else "S?"
    prior_roles = {_speaker_label_for_segment(t, role_map) for t in prior_turns}

    obj_line = f"[OBJECTIVE] {snapshot.brief.objective}"
    anchor_line = f"[ANCHOR] {anchor_role}: {anchor_span.joined_text}"
    full_role_line = _build_role_line(snapshot, role_map, anchor_roles | prior_roles)
    core_only_role_line = _build_role_line(snapshot, role_map, anchor_roles)
    core_text = "\n".join([obj_line, core_only_role_line, anchor_line])
    core_tokens = tok.count_tokens(core_text)

    asr_floor = policy.abstain.asr_confidence_floor if policy else None
    speaker_uncertain = any(
        s.speaker_uncertain
        or _speaker_label_for_segment(s, role_map) == "S?"
        or (s.speaker_confidence is not None and s.speaker_confidence < 0.5)
        for s in anchor_span.segments
    )
    asr_low_confidence = any(
        s.asr_low_confidence
        or s.low_quality_gate
        or (asr_floor is not None and s.asr_confidence is not None and s.asr_confidence < asr_floor)
        for s in anchor_span.segments
    )

    all_suggestions = list(snapshot.ledger.values())
    all_facts = list(snapshot.brief.known_facts)
    flags: dict[str, bool] = {
        "input_overflow": False,
        "context_truncated_material": False,
        "speaker_uncertain": speaker_uncertain,
        "asr_low_confidence": asr_low_confidence,
        "overlap": any(s.overlap for s in anchor_span.segments),
        "has_gap": has_gap,
        "has_off_record_gap": has_off_record_gap,
    }
    if policy is not None:
        for cat in policy.categories:
            flags[f"context_truncated_material:{cat.value}"] = False

    if core_tokens > limit:
        flags["input_overflow"] = True
        omitted = {
            "turns": len(prior_turns),
            "suggestions": len(all_suggestions),
            "known_facts": len(all_facts),
            "gaps": len(relevant_gaps),
        }
        if prior_turns:
            max_omit_end = max(t.end_ms for t in prior_turns)
            if policy is not None and policy.categories:
                for cat, cpol in policy.categories.items():
                    hit = (anchor_start_ms - max_omit_end) <= cpol.required_lookback_s * 1000
                    flags[f"context_truncated_material:{cat.value}"] = hit
                flags["context_truncated_material"] = any(
                    flags[f"context_truncated_material:{c.value}"] for c in policy.categories
                )
            else:
                flags["context_truncated_material"] = (
                    anchor_start_ms - max_omit_end
                ) <= default_required_lookback_s * 1000
        sha = hashlib.sha256(core_text.encode("utf-8")).hexdigest()
        token_counts = {
            "total": core_tokens,
            "limit": limit,
            "objective": tok.count_tokens(obj_line),
            "role": tok.count_tokens(core_only_role_line),
            "known_facts": 0,
            "suggested": 0,
            "context": 0,
            "anchor": tok.count_tokens(anchor_line),
        }
        ctx = DecisionContext(
            schema_version=1,
            candidate_id=anchor_span.candidate_id,
            session_id=snapshot.session_id,
            origin=origin,
            based_on_state_version=snapshot.state_version,
            builder_version=BUILDER_VERSION,
            prefilter_version=PREFILTER_VERSION,
            anchor_refs=anchor_span.anchor_refs,
            context_refs=(),
            anchor_received_ms=anchor_span.anchor_received_ms,
            display_by_ms=anchor_span.display_by_ms,
            token_counts=token_counts,
            omitted=omitted,
            flags=flags,
            serialized_sha256=sha,
        )
        return ContextBuildResult(
            anchor=anchor_span,
            dropped=False,
            drop_record=None,
            context=ctx,
            serialized_text=core_text,
        )

    # Step 2: Fit [KNOWN] within header_max
    core_header_tokens = tok.count_tokens(f"{obj_line}\n{full_role_line}")
    retained_facts: list[str] = []
    for fact in all_facts:
        item = f"{fact.key}: {fact.value}"
        cand_line = "[KNOWN] " + " | ".join([*retained_facts, item])
        if (
            core_header_tokens + tok.count_tokens(cand_line) <= header_max
            and tok.count_tokens("\n".join([obj_line, full_role_line, cand_line, anchor_line]))
            <= limit
        ):
            retained_facts.append(item)
        else:
            break
    known_line = "[KNOWN] " + " | ".join(retained_facts) if retained_facts else ""
    omitted_facts = len(all_facts) - len(retained_facts)

    # Step 3: Fit [SUGGESTED] (newest first) within suggested_max
    sug_entries: list[tuple[int, str]] = []
    for entry in all_suggestions:
        cat_item = suggestion_catalog.get(entry.suggestion_id) if suggestion_catalog else None
        item_ms = (
            entry.history[-1].at_ms
            if entry.history
            else (cat_item.ready_ms if cat_item else anchor_span.anchor_received_ms)
        )
        cat_str = str(cat_item.category) if cat_item else "SUGGESTION"
        text = cat_item.text if cat_item else entry.suggestion_id
        line = _format_suggestion_line(
            str(entry.state), cat_str, text, anchor_span.anchor_received_ms - item_ms
        )
        sug_entries.append((item_ms, line))
    sug_entries.sort(key=lambda x: x[0], reverse=True)

    retained_sug_lines: list[str] = []
    sug_tok_sum = 0
    for _, sline in sug_entries:
        ltok = tok.count_tokens(sline)
        base_parts = [obj_line, full_role_line]
        if known_line:
            base_parts.append(known_line)
        base_parts.extend([*retained_sug_lines, sline, anchor_line])
        if sug_tok_sum + ltok <= suggested_max and tok.count_tokens("\n".join(base_parts)) <= limit:
            retained_sug_lines.append(sline)
            sug_tok_sum += ltok
        else:
            break
    retained_sug_lines.reverse()
    omitted_suggestions = len(sug_entries) - len(retained_sug_lines)
    if not retained_sug_lines and not sug_entries:
        none_line = "[SUGGESTED] none"
        base_parts = [obj_line, full_role_line]
        if known_line:
            base_parts.append(known_line)
        base_parts.extend([none_line, anchor_line])
        if tok.count_tokens("\n".join(base_parts)) <= limit:
            retained_sug_lines.append(none_line)

    # Step 4: Fit [CONTEXT] whole turns oldest-first dropped (fill newest first)
    retained_rev: list[SegmentRecord] = []
    for turn in reversed(prior_turns):
        candidate_turns = list(reversed([*retained_rev, turn]))
        ctx_lines = _render_context_lines(
            candidate_turns, relevant_gaps, role_map, anchor_span.anchor_received_ms
        )
        parts = [obj_line, full_role_line]
        if known_line:
            parts.append(known_line)
        parts.extend(retained_sug_lines)
        parts.extend(ctx_lines)
        parts.append(anchor_line)
        if tok.count_tokens("\n".join(parts)) <= limit:
            retained_rev.append(turn)
        else:
            break

    retained_turns = list(reversed(retained_rev))
    omitted_count = len(prior_turns) - len(retained_turns)
    omitted_turns = prior_turns[:omitted_count]
    retained_gaps = list(relevant_gaps)
    omitted_gaps = 0

    def _assemble_parts() -> tuple[str, list[str], str]:
        used = anchor_roles | {_speaker_label_for_segment(t, role_map) for t in retained_turns}
        r_line = _build_role_line(snapshot, role_map, used)
        c_lines = _render_context_lines(
            retained_turns, retained_gaps, role_map, anchor_span.anchor_received_ms
        )
        parts_out = [obj_line, r_line]
        if known_line:
            parts_out.append(known_line)
        parts_out.extend(retained_sug_lines)
        parts_out.extend(c_lines)
        parts_out.append(anchor_line)
        return (r_line, c_lines, "\n".join(parts_out))

    final_role_line, ctx_lines, serialized_text = _assemble_parts()
    while retained_turns and tok.count_tokens(serialized_text) > limit:
        omitted_turns.append(retained_turns.pop(0))
        final_role_line, ctx_lines, serialized_text = _assemble_parts()
    omitted_count = len(omitted_turns)

    while retained_gaps and tok.count_tokens(serialized_text) > limit:
        retained_gaps.pop(0)
        omitted_gaps += 1
        final_role_line, ctx_lines, serialized_text = _assemble_parts()

    while retained_sug_lines and tok.count_tokens(serialized_text) > limit:
        retained_sug_lines.pop(0)
        omitted_suggestions += 1
        final_role_line, ctx_lines, serialized_text = _assemble_parts()

    while retained_facts and tok.count_tokens(serialized_text) > limit:
        retained_facts.pop()
        omitted_facts += 1
        known_line = "[KNOWN] " + " | ".join(retained_facts) if retained_facts else ""
        final_role_line, ctx_lines, serialized_text = _assemble_parts()

    if omitted_turns:
        max_omit_end = max(t.end_ms for t in omitted_turns)
        if policy is not None and policy.categories:
            for cat, cpol in policy.categories.items():
                hit = (anchor_start_ms - max_omit_end) <= cpol.required_lookback_s * 1000
                flags[f"context_truncated_material:{cat.value}"] = hit
            flags["context_truncated_material"] = any(
                flags[f"context_truncated_material:{c.value}"] for c in policy.categories
            )
        else:
            flags["context_truncated_material"] = (
                anchor_start_ms - max_omit_end
            ) <= default_required_lookback_s * 1000

    sha = hashlib.sha256(serialized_text.encode("utf-8")).hexdigest()
    total_tokens = tok.count_tokens(serialized_text)
    if total_tokens > limit:
        flags["input_overflow"] = True
    token_counts = {
        "total": total_tokens,
        "limit": limit,
        "objective": tok.count_tokens(obj_line),
        "role": tok.count_tokens(final_role_line),
        "known_facts": tok.count_tokens(known_line) if known_line else 0,
        "suggested": sum(tok.count_tokens(s) for s in retained_sug_lines),
        "context": sum(tok.count_tokens(c) for c in ctx_lines),
        "anchor": tok.count_tokens(anchor_line),
    }
    ctx = DecisionContext(
        schema_version=1,
        candidate_id=anchor_span.candidate_id,
        session_id=snapshot.session_id,
        origin=origin,
        based_on_state_version=snapshot.state_version,
        builder_version=BUILDER_VERSION,
        prefilter_version=PREFILTER_VERSION,
        anchor_refs=anchor_span.anchor_refs,
        context_refs=tuple(SegmentRef(t.segment_id, t.revision) for t in retained_turns),
        anchor_received_ms=anchor_span.anchor_received_ms,
        display_by_ms=anchor_span.display_by_ms,
        token_counts=token_counts,
        omitted={
            "turns": omitted_count,
            "suggestions": omitted_suggestions,
            "known_facts": omitted_facts,
            "gaps": omitted_gaps,
        },
        flags=flags,
        serialized_sha256=sha,
    )
    return ContextBuildResult(
        anchor=anchor_span,
        dropped=False,
        drop_record=None,
        context=ctx,
        serialized_text=serialized_text,
    )


def build_decision_contexts(
    snapshot: StateSnapshot,
    anchor_segments: Sequence[SegmentRecord] | None = None,
    policy: PolicyConfig | None = None,
    tokenizer: Tokenizer | None = None,
    *,
    origin: Origin = Origin.PROACTIVE,
    max_anchor_tokens: int = DEFAULT_ANCHOR_MAX_TOKENS,
    max_candidate_age_ms: int | None = None,
    speaker_boundary_end_ms: Mapping[str, int] | None = None,
    pause_gap_ms: int | None = None,
    suggestion_catalog: Mapping[str, SuggestionCandidate] | None = None,
    stock_phrases: frozenset[str] = DEFAULT_STOCK_PHRASES,
    input_limit_tokens: int | None = None,
    initial_previous_anchor_norm_text: str | None = None,
) -> list[ContextBuildResult]:
    """Sole context-building pipeline across live, replay, and training (§5.2–§5.3, §15)."""
    tok = tokenizer or WhitespaceTokenizer()
    age_cap = (
        max_candidate_age_ms
        if max_candidate_age_ms is not None
        else (policy.display.max_candidate_age_ms if policy else DEFAULT_MAX_CANDIDATE_AGE_MS)
    )
    anchors = coalesce_anchors(
        snapshot,
        max_anchor_tokens=max_anchor_tokens,
        max_candidate_age_ms=age_cap,
        tokenizer=tok,
        speaker_boundary_end_ms=speaker_boundary_end_ms,
        pause_gap_ms=pause_gap_ms,
        segments_subset=anchor_segments,
    )
    results: list[ContextBuildResult] = []
    prev_norm: str | None = initial_previous_anchor_norm_text
    for anchor in anchors:
        pf = run_prefilter(
            anchor,
            state_version=snapshot.state_version,
            previous_anchor_norm_text=prev_norm,
            stock_phrases=stock_phrases,
        )
        prev_norm = pf.normalized_text
        if not pf.passed:
            results.append(
                ContextBuildResult(
                    anchor=anchor,
                    dropped=True,
                    drop_record=pf.drop_record,
                    context=None,
                    serialized_text=None,
                )
            )
            continue
        results.append(
            build_decision_context(
                snapshot,
                anchor,
                policy,
                origin=origin,
                tokenizer=tok,
                input_limit_tokens=input_limit_tokens,
                max_candidate_age_ms=age_cap,
                suggestion_catalog=suggestion_catalog,
            )
        )
    return results


_BARE_PRONOUNS: frozenset[str] = frozenset(
    {
        "it",
        "that",
        "this",
        "them",
        "those",
        "these",
        "something",
        "anything",
        "everything",
        "nothing",
        "stuff",
        "things",
    }
)

_TRAILING_PUNCT = ".,!?;:"

_OWNER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:will|should|needs?\s+to|has\s+to|have\s+to|going\s+to|gonna|can|could|must)?\s*"
        r"(?:also\s+)?(?:own|take\s+on|take\s+care\s+of|take|handle|follow\s+up\s+on|lead|drive|manage|pick\s+up|run|verify)\s+"
        r"(?P<span>.+?)(?=[.,!?;:]|\s+(?:before|by|until|after|because|so|and\s+then|if|when)\b|$)",
        flags=re.IGNORECASE,
    ),
)

_TIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:deadline|target\s+date|finish\s+date|due\s+date|timeline|eta)\s+(?:for|on)\s+"
        r"(?P<span>.+?)(?=[.,!?;:]|\s+(?:before|by|until|soon|later|eventually|when|if|so|because)\b|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:finish|ship|land|complete|deploy|cut\s+over|deliver|launch|wrap\s+up)\s+"
        r"(?P<span>.+?)(?=[.,!?;:]|\s+(?:before|by|until|soon|later|eventually|when|if|so|because)\b|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[.,!?;:]\s*)(?P<span>[A-Za-z0-9][A-Za-z0-9\s_-]{1,60}?)\s+"
        r"(?:needs?\s+to\s+be|should\s+be|will\s+be|has\s+to\s+be|is)\s+"
        r"(?:finished|shipped|landed|done|completed|deployed|delivered|launched)\b",
        flags=re.IGNORECASE,
    ),
)

_PROPOSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:define|confirm|check|agree\s+on|set)\s+(?:a\s+)?(?:(?:test|success|acceptance)\s+criteri(?:on|a)|metric)\s+(?:for|on)\s+"
        r"(?P<span>.+?)(?=[.,!?;:]|\s+(?:except|excluding|including|only\s+for|up\s+to|within|outside\s+of|without|not\s+including|for\s+now|so|because|to\s+make|when|if)\b|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:propose|suggest|recommend|plan\s+to|try|use|roll\s+out|switch\s+to|cut\s+over\s+to|migrate\s+to|adopt|add|build|deploy|review)\s+"
        r"(?P<span>.+?)(?=[.,!?;:]|\s+(?:except|excluding|including|only\s+for|up\s+to|within|outside\s+of|without|not\s+including|for\s+now|so|because|to\s+make|when|if)\b|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[.,!?;:]\s*)(?P<span>[A-Za-z0-9][A-Za-z0-9\s_-]{1,60}?)\s+"
        r"(?:will\s+work|should\s+work|works|fixes|improves|helps|includes?|covers?|handles?|supports?|applies\s+to)\b",
        flags=re.IGNORECASE,
    ),
)

_BOUNDARY_PATTERN = re.compile(
    r"\b(?:except\s+for|except|excluding|including|only\s+for|up\s+to|within|outside\s+of|without|not\s+including|for\s+now)\s+"
    r"(?P<span>[^.,!?;:]+)",
    flags=re.IGNORECASE,
)

_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:shows?|means?|proves?|indicates?|demonstrates?|confirms?|think|believe|expect|assume|heard)\s+"
        r"(?:that\s+)?(?P<span>[^.,!?;:]+)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^(?:actually,?\s+|basically,?\s+|honestly,?\s+)?(?P<span>[^.,!?;:]{8,})",
        flags=re.IGNORECASE,
    ),
)


def _make_slot_from_span(
    seg: SegmentRecord,
    start: int,
    end: int,
    *,
    extractor: str = SPAN_EXTRACTOR_VERSION,
) -> GroundedSlot | None:
    text = seg.text
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _TRAILING_PUNCT):
        end -= 1
    if end <= start:
        return None
    span_text = text[start:end]
    if normalize_for_matching(span_text) in _BARE_PRONOUNS:
        return None
    return GroundedSlot(
        text=span_text,
        segment_id=seg.segment_id,
        revision=seg.revision,
        char_start=start,
        char_end=end,
        extractor=extractor,
    )


def verify_grounded_slot(slot: GroundedSlot, segment: SegmentRecord) -> bool:
    """Enforce §5.6 strict contiguous-substring grounding invariant."""
    if slot.segment_id != segment.segment_id or slot.revision != segment.revision:
        return False
    if not (0 <= slot.char_start < slot.char_end <= len(segment.text)):
        return False
    raw = segment.text[slot.char_start : slot.char_end]
    trimmed = raw.strip()
    if not trimmed:
        return False
    candidates = [trimmed]
    cur = trimmed
    while cur and cur[-1] in _TRAILING_PUNCT:
        cur = cur[:-1].rstrip()
        if cur:
            candidates.append(cur)
    for cand in candidates:
        if slot.text == cand:
            return True
        if (
            len(slot.text) == len(cand)
            and len(cand) >= 1
            and slot.text[0].lower() == cand[0].lower()
            and slot.text[1:] == cand[1:]
        ):
            return True
    return False


def _first_matching_slot(
    seg: SegmentRecord,
    patterns: Sequence[re.Pattern[str]],
) -> GroundedSlot | None:
    for pat in patterns:
        m = pat.search(seg.text)
        if m is not None:
            slot = _make_slot_from_span(seg, m.start("span"), m.end("span"))
            if slot is not None and verify_grounded_slot(slot, seg):
                return slot
    return None


def extract_slots_span_rules_v0(
    anchor_segments: Sequence[SegmentRecord],
    category: Category,
) -> dict[str, GroundedSlot]:
    """Deterministic per-segment slot extraction for §9.1 templates (`span_rules_v0`)."""
    if not anchor_segments:
        return {}
    if category is Category.CONFIRM_OWNER:
        for seg in anchor_segments:
            slot = _first_matching_slot(seg, _OWNER_PATTERNS)
            if slot is not None:
                return {"action": slot}
        return {}
    if category is Category.CONFIRM_TIME:
        for seg in anchor_segments:
            slot = _first_matching_slot(seg, _TIME_PATTERNS)
            if slot is not None:
                return {"action": slot}
        return {}
    if category is Category.SUCCESS_CRITERION:
        for seg in anchor_segments:
            slot = _first_matching_slot(seg, _PROPOSAL_PATTERNS)
            if slot is not None:
                return {"proposal": slot}
        return {}
    if category is Category.CLARIFY_SCOPE:
        for seg in anchor_segments:
            prop = _first_matching_slot(seg, _PROPOSAL_PATTERNS)
            bnd = _first_matching_slot(seg, (_BOUNDARY_PATTERN,))
            if prop is not None and bnd is not None:
                return {"proposal": prop, "explicitly_mentioned_boundary": bnd}
        prop_by_spk: dict[str, GroundedSlot] = {}
        bnd_by_spk: dict[str, GroundedSlot] = {}
        for seg in anchor_segments:
            if seg.speaker_id not in prop_by_spk:
                p_slot = _first_matching_slot(seg, _PROPOSAL_PATTERNS)
                if p_slot is not None:
                    prop_by_spk[seg.speaker_id] = p_slot
            if seg.speaker_id not in bnd_by_spk:
                b_slot = _first_matching_slot(seg, (_BOUNDARY_PATTERN,))
                if b_slot is not None:
                    bnd_by_spk[seg.speaker_id] = b_slot
            if seg.speaker_id in prop_by_spk and seg.speaker_id in bnd_by_spk:
                return {
                    "proposal": prop_by_spk[seg.speaker_id],
                    "explicitly_mentioned_boundary": bnd_by_spk[seg.speaker_id],
                }
        return {}
    if category is Category.REQUEST_EVIDENCE:
        for seg in anchor_segments:
            claim = _first_matching_slot(seg, _CLAIM_PATTERNS)
            if claim is not None:
                return {"stated_claim": claim}
        return {}
    return {}


def check_generic_referent(
    snapshot: StateSnapshot,
    anchor_refs: Sequence[SegmentRef],
    now_ms: int,
    max_anchor_age_ms: int = DEFAULT_GENERIC_MAX_ANCHOR_AGE_MS,
) -> tuple[bool, ReasonCode | None]:
    """Check §9.1 generic-template referent rule (latest other speech & age cap)."""
    if not anchor_refs:
        return (False, ReasonCode.GENERIC_REFERENT_UNCLEAR)
    anchor_segs: list[SegmentRecord] = []
    for ref in anchor_refs:
        seg = snapshot.segments_by_id.get(ref.segment_id)
        if seg is None or seg.revision != ref.revision:
            return (False, ReasonCode.GENERIC_REFERENT_UNCLEAR)
        anchor_segs.append(seg)

    anchor_last = max(
        anchor_segs,
        key=lambda s: (s.start_ms, s.channel, s.source_sequence, s.segment_id),
    )
    anchor_last_key = (
        anchor_last.start_ms,
        anchor_last.channel,
        anchor_last.source_sequence,
        anchor_last.segment_id,
    )
    anchor_rx_ms = max(s.received_ms for s in anchor_segs)
    if now_ms < anchor_rx_ms or (now_ms - anchor_rx_ms) > max_anchor_age_ms:
        return (False, ReasonCode.GENERIC_REFERENT_UNCLEAR)

    recipient_ids = {"ME", snapshot.brief.recipient.participant_id}
    for seg in snapshot.segments:
        if seg.speaker_id in recipient_ids:
            continue
        seg_key = (seg.start_ms, seg.channel, seg.source_sequence, seg.segment_id)
        if seg_key > anchor_last_key:
            return (False, ReasonCode.GENERIC_REFERENT_UNCLEAR)
    return (True, None)


@dataclass(slots=True)
class CoverageCursor:
    """High-water + exception-set cursor over committed non-recipient segments (§8.3)."""

    high_water_key: tuple[int, int, int, str] | None = None
    covered_revisions: dict[str, int] = field(default_factory=dict)
    pending_exceptions: set[str] = field(default_factory=set)

    @staticmethod
    def seg_key(seg: SegmentRecord) -> tuple[int, int, int, str]:
        return (seg.start_ms, seg.channel, seg.source_sequence, seg.segment_id)

    def note_material_segment(self, seg_id: str | None) -> None:
        if seg_id is not None:
            self.pending_exceptions.add(seg_id)

    def unprocessed_segments(self, snapshot: StateSnapshot) -> list[SegmentRecord]:
        recipient_ids = {"ME", snapshot.brief.recipient.participant_id}
        out: list[SegmentRecord] = []
        for seg in snapshot.segments:
            if not seg.committed or seg.speaker_id in recipient_ids:
                continue
            prev_rev = self.covered_revisions.get(seg.segment_id, 0)
            if prev_rev < seg.revision or seg.segment_id in self.pending_exceptions:
                out.append(seg)
        return out

    def mark_covered(self, segments: Iterable[SegmentRecord]) -> None:
        for seg in segments:
            prev = self.covered_revisions.get(seg.segment_id, 0)
            if seg.revision > prev:
                self.covered_revisions[seg.segment_id] = seg.revision
            self.pending_exceptions.discard(seg.segment_id)
            key = self.seg_key(seg)
            if self.high_water_key is None or key > self.high_water_key:
                self.high_water_key = key
