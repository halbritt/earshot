"""Versioned schema-v1 contracts and closed enums for the Live Conversation Coach."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = 1


class ContractValidationError(ValueError):
    """Raised when a contract payload fails schema-v1 validation."""


class SessionMode(StrEnum):
    """Domain conversation modes in SessionBrief (§3.2, §3.8)."""

    TECHNICAL_WORKING_DISCUSSION = "technical_working_discussion"
    PERSONAL_LOGISTICS = "personal_logistics"


class Mode(StrEnum):
    """Runtime coach operating modes (§3.6)."""

    PROACTIVE = "Proactive"
    QUIET = "Quiet"
    SHADOW = "Shadow"
    PAUSED = "Paused"
    OFF_THE_RECORD = "Off the record"


class HealthState(StrEnum):
    """Runtime health indicator states (§8.6)."""

    OK = "ok"
    TEMPLATE_ONLY = "template_only"
    MANUAL_ONLY = "manual_only"
    NO_TRANSCRIPT = "no_transcript"
    PAUSED = "paused"


class Stability(StrEnum):
    """Segment stability levels (§6.1, §6.2)."""

    INTERIM = "interim"
    COMMITTED = "committed"


class EventKind(StrEnum):
    """Canonical event kinds (§6.1)."""

    SEGMENT = "segment"
    SPEAKER_UPDATE = "speaker_update"
    BOUNDARY = "boundary"
    BRIEF_UPDATE = "brief_update"
    LEDGER = "ledger"
    CONTROL = "control"


class ControlSubtype(StrEnum):
    """Subtypes for control canonical events (§6.1)."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    RECONNECT = "reconnect"
    GAP = "gap"
    OFF_RECORD_START = "off_record_start"
    OFF_RECORD_END = "off_record_end"


class Origin(StrEnum):
    """Candidate request origin (§3.7, §7.1)."""

    PROACTIVE = "proactive"
    MANUAL = "manual"


class Outcome(StrEnum):
    """Routing decision outcomes (§7.3)."""

    NONE = "NONE"
    ABSTAIN = "ABSTAIN"
    SUPPRESS = "SUPPRESS"
    ROUTE = "ROUTE"


class Route(StrEnum):
    """Candidate realization route (§7.3, §7.4)."""

    TEMPLATE = "template"
    GENERATOR = "generator"


class LedgerState(StrEnum):
    """Suggestion lifecycle states (§7.5)."""

    READY = "ready"
    DEFERRED = "deferred"
    DISPLAYED = "displayed"
    PINNED = "pinned"
    USED = "used"
    DISMISSED = "dismissed"
    RESOLVED = "resolved"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    OUTDATED = "outdated"


class FeedbackReason(StrEnum):
    """Optional dismissal feedback reasons (§7.5, Appendix C)."""

    TOO_LATE = "too_late"
    ALREADY_ANSWERED = "already_answered"
    NOT_RELEVANT = "not_relevant"
    BADLY_PHRASED = "badly_phrased"


class Category(StrEnum):
    """Closed coaching category taxonomy (§5.4)."""

    CLARIFY_REFERENCE = "CLARIFY_REFERENCE"
    CLARIFY_SCOPE = "CLARIFY_SCOPE"
    CONFIRM_OWNER = "CONFIRM_OWNER"
    CONFIRM_TIME = "CONFIRM_TIME"
    SUCCESS_CRITERION = "SUCCESS_CRITERION"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    PROBE_TRADEOFF = "PROBE_TRADEOFF"
    FOLLOW_UP = "FOLLOW_UP"
    OTHER = "OTHER"


PREDICTABLE_CATEGORIES: frozenset[Category] = frozenset(
    c for c in Category if c is not Category.OTHER
)


class SampleSource(StrEnum):
    """Sampling origin for LabelRecord (§12.3, §12.4)."""

    RANDOM = "random"
    PASS1_MARK = "pass1_mark"
    MODEL_POSITIVE = "model_positive"
    SCORE_BAND = "score_band"
    MANUAL_REQUEST = "manual_request"
    SHADOW_REVIEW = "shadow_review"
    LIVE_FEEDBACK = "live_feedback"


class ClosedBy(StrEnum):
    """How a validity window closed (§12.3)."""

    ANSWERED = "answered"
    TOPIC_MOVED = "topic_moved"
    TOO_LATE_TO_ASK = "too_late_to_ask"
    SESSION_END = "session_end"


class NoneReason(StrEnum):
    """Why a labeled anchor has worthwhile=false (§12.3)."""

    NO_GAP = "no_gap"
    ALREADY_KNOWN = "already_known"
    NOT_MATERIAL = "not_material"
    NOT_MY_QUESTION = "not_my_question"
    ALREADY_SUGGESTED = "already_suggested"
    RHETORICAL = "rhetorical"
    TENTATIVE = "tentative"
    BETTER_TO_LISTEN = "better_to_listen"


class ConsentBasis(StrEnum):
    """Consent basis (§3.2, §14.1)."""

    PRESUMED = "presumed"
    STATED = "stated"


class TelemetryLevel(StrEnum):
    """Recorder event priority levels (§4.2)."""

    DEBUG = "debug"
    LIFECYCLE = "lifecycle"
    DECISION = "decision"


class AnnotationPass(StrEnum):
    """Annotation passes (§12.3, §12.4)."""

    ONLINE = "online"
    OFFLINE = "offline"


class ReasonCode(StrEnum):
    """Closed Appendix C reason codes (47 distinct codes; unknown codes rejected)."""

    # Prefilter (dropped, terminal)
    BACKCHANNEL = "backchannel"
    TOO_SHORT = "too_short"
    REPEAT = "repeat"
    STOCK_PHRASE = "stock_phrase"

    # Scheduling (retired, terminal)
    EXPIRED_UNEVALUATED = "expired_unevaluated"
    SESSION_ENDED = "session_ended"
    OFF_RECORD = "off_record"

    # Routing NONE (terminal)
    BELOW_T_LOW = "below_t_low"

    # Routing ABSTAIN (terminal)
    INPUT_OVERFLOW = "input_overflow"
    CONTEXT_TRUNCATED_MATERIAL = "context_truncated_material"
    SPEAKER_UNCERTAIN = "speaker_uncertain"
    ASR_LOW_CONFIDENCE = "asr_low_confidence"
    SCORE_IN_BAND = "score_in_band"
    MODE_MISMATCH = "mode_mismatch"
    CLASSIFIER_ERROR = "classifier_error"

    # Routing SUPPRESS (terminal)
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    DO_NOT_SUGGEST = "do_not_suggest"
    KNOWN_FACT = "known_fact"
    PROACTIVE_DISABLED = "proactive_disabled"

    # Routing ROUTE (non-terminal; manual_request also used in Display displayed)
    ABOVE_T_DISPLAY = "above_t_display"
    MANUAL_REQUEST = "manual_request"

    # Rendering (unrendered, terminal)
    SLOT_UNAVAILABLE = "slot_unavailable"
    SLOT_TOO_LONG = "slot_too_long"
    GENERIC_REFERENT_UNCLEAR = "generic_referent_unclear"
    GENERATOR_DECLINED = "generator_declined"
    GENERATOR_INVALID_OUTPUT = "generator_invalid_output"
    GENERATOR_TIMEOUT = "generator_timeout"
    INSUFFICIENT_BUDGET = "insufficient_budget"

    # Display deferred (non-terminal)
    RECIPIENT_SPEAKING = "recipient_speaking"
    CARD_ACTIVE = "card_active"
    COOLDOWN = "cooldown"
    SETTLING = "settling"
    INTERIM_RESOLUTION_HINT = "interim_resolution_hint"

    # Display displayed (non-terminal)
    BOUNDARY_AFTER_SETTLE = "boundary_after_settle"
    SETTLE_TIMEOUT = "settle_timeout"
    DEFERRED_RELEASED = "deferred_released"

    # Display withdrawn (terminal; duplicate/near_duplicate shared with SUPPRESS)
    EVIDENCE_REVISION_CHANGED = "evidence_revision_changed"
    OBJECTIVE_CHANGED = "objective_changed"
    POSSIBLY_RESOLVED = "possibly_resolved"
    EQUIVALENT_QUESTION_ASKED = "equivalent_question_asked"
    DISPLACED_BY_BETTER = "displaced_by_better"
    TOPIC_MOVED = "topic_moved"

    # Display expired (terminal)
    DISPLAY_BY_PASSED = "display_by_passed"
    DISPLAY_TTL_PASSED = "display_ttl_passed"

    # Closing (used, dismissed, resolved, outdated)
    USER_ACTION = "user_action"
    RESOLUTION_CUE = "resolution_cue"


PREFILTER_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.BACKCHANNEL,
        ReasonCode.TOO_SHORT,
        ReasonCode.REPEAT,
        ReasonCode.STOCK_PHRASE,
    }
)
SCHEDULING_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.EXPIRED_UNEVALUATED,
        ReasonCode.SESSION_ENDED,
        ReasonCode.OFF_RECORD,
    }
)
ABSTAIN_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.INPUT_OVERFLOW,
        ReasonCode.CONTEXT_TRUNCATED_MATERIAL,
        ReasonCode.SPEAKER_UNCERTAIN,
        ReasonCode.ASR_LOW_CONFIDENCE,
        ReasonCode.SCORE_IN_BAND,
        ReasonCode.MODE_MISMATCH,
        ReasonCode.CLASSIFIER_ERROR,
    }
)
SUPPRESS_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.DUPLICATE,
        ReasonCode.NEAR_DUPLICATE,
        ReasonCode.DO_NOT_SUGGEST,
        ReasonCode.KNOWN_FACT,
        ReasonCode.PROACTIVE_DISABLED,
    }
)
ROUTE_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.ABOVE_T_DISPLAY,
        ReasonCode.MANUAL_REQUEST,
    }
)
RENDER_FAIL_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.SLOT_UNAVAILABLE,
        ReasonCode.SLOT_TOO_LONG,
        ReasonCode.GENERIC_REFERENT_UNCLEAR,
        ReasonCode.GENERATOR_DECLINED,
        ReasonCode.GENERATOR_INVALID_OUTPUT,
        ReasonCode.GENERATOR_TIMEOUT,
        ReasonCode.INSUFFICIENT_BUDGET,
    }
)
DEFERRED_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.RECIPIENT_SPEAKING,
        ReasonCode.CARD_ACTIVE,
        ReasonCode.COOLDOWN,
        ReasonCode.SETTLING,
        ReasonCode.INTERIM_RESOLUTION_HINT,
    }
)
DISPLAYED_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.BOUNDARY_AFTER_SETTLE,
        ReasonCode.SETTLE_TIMEOUT,
        ReasonCode.DEFERRED_RELEASED,
        ReasonCode.MANUAL_REQUEST,
    }
)
WITHDRAW_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.EVIDENCE_REVISION_CHANGED,
        ReasonCode.OBJECTIVE_CHANGED,
        ReasonCode.POSSIBLY_RESOLVED,
        ReasonCode.EQUIVALENT_QUESTION_ASKED,
        ReasonCode.DUPLICATE,
        ReasonCode.NEAR_DUPLICATE,
        ReasonCode.DISPLACED_BY_BETTER,
        ReasonCode.TOPIC_MOVED,
    }
)
EXPIRE_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.DISPLAY_BY_PASSED,
        ReasonCode.DISPLAY_TTL_PASSED,
    }
)
CLOSING_CAUSE_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.USER_ACTION,
        ReasonCode.RESOLUTION_CUE,
    }
)
TERMINAL_REASON_CODES: frozenset[ReasonCode] = (
    PREFILTER_REASON_CODES
    | SCHEDULING_REASON_CODES
    | frozenset({ReasonCode.BELOW_T_LOW})
    | ABSTAIN_REASON_CODES
    | SUPPRESS_REASON_CODES
    | RENDER_FAIL_REASON_CODES
    | WITHDRAW_REASON_CODES
    | EXPIRE_REASON_CODES
    | CLOSING_CAUSE_CODES
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be str, got {type(value).__name__}")
    if not allow_empty and not value:
        raise ContractValidationError(f"{name} must be non-empty")
    return value


def _optional_str(value: Any, name: str, *, allow_empty: bool = True) -> str | None:
    if value is None:
        return None
    return _require_str(value, name, allow_empty=allow_empty)


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{name} must be int (not bool), got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _optional_int(value: Any, name: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _require_int(value, name, minimum=minimum)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(
            f"{name} must be float/int (not bool), got {type(value).__name__}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise ContractValidationError(f"{name} must be finite, got {value!r}")
    return out


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, name)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{name} must be bool, got {type(value).__name__}")
    return value


def _require_schema_version(data: Mapping[str, Any], name: str) -> int:
    if "schema_version" not in data:
        raise ContractValidationError(f"{name} missing required field 'schema_version'")
    version = _require_int(data["schema_version"], f"{name}.schema_version")
    if version != SCHEMA_VERSION:
        raise ContractValidationError(
            f"{name}.schema_version must be {SCHEMA_VERSION}, got {version}"
        )
    return version


def _require_enum[E: StrEnum](enum_cls: type[E], value: Any, name: str) -> E:
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise ContractValidationError(
            f"{name} must be {enum_cls.__name__} str, got {type(value).__name__}"
        )
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ContractValidationError(f"Unknown {enum_cls.__name__} for {name}: {value!r}") from exc


def _require_str_seq(value: Any, name: str, *, allow_empty_items: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{name} must be a list or tuple")
    return tuple(
        _require_str(item, f"{name}[{i}]", allow_empty=allow_empty_items)
        for i, item in enumerate(value)
    )


def _check_keys(data: Mapping[str, Any], required: set[str], allowed: set[str], name: str) -> None:
    missing = required - set(data.keys())
    if missing:
        raise ContractValidationError(f"{name} missing required fields: {sorted(missing)}")
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ContractValidationError(f"{name} has unknown fields: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class SegmentRef:
    """Canonical segment evidence reference (§7.1)."""

    segment_id: str
    revision: int

    def __post_init__(self) -> None:
        _require_str(self.segment_id, "SegmentRef.segment_id")
        _require_int(self.revision, "SegmentRef.revision", minimum=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SegmentRef:
        m = _require_mapping(data, "SegmentRef")
        _check_keys(m, {"segment_id", "revision"}, {"segment_id", "revision"}, "SegmentRef")
        return cls(
            segment_id=_require_str(m["segment_id"], "SegmentRef.segment_id"),
            revision=_require_int(m["revision"], "SegmentRef.revision", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "revision": self.revision}


def _parse_segment_refs(value: Any, name: str) -> tuple[SegmentRef, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{name} must be a list or tuple")
    return tuple(SegmentRef.from_dict(item) for item in value)


@dataclass(frozen=True, slots=True)
class RecipientConfig:
    """Recipient spec inside SessionBrief (§3.2)."""

    participant_id: str
    role: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.participant_id, "RecipientConfig.participant_id")
        if self.role is not None:
            _require_str(self.role, "RecipientConfig.role")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecipientConfig:
        m = _require_mapping(data, "RecipientConfig")
        _check_keys(m, {"participant_id"}, {"participant_id", "role"}, "RecipientConfig")
        return cls(
            participant_id=_require_str(m["participant_id"], "RecipientConfig.participant_id"),
            role=_optional_str(m.get("role"), "RecipientConfig.role", allow_empty=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"participant_id": self.participant_id, "role": self.role}


@dataclass(frozen=True, slots=True)
class Participant:
    """Other participant spec inside SessionBrief (§3.2)."""

    participant_id: str
    label: str
    role: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.participant_id, "Participant.participant_id")
        _require_str(self.label, "Participant.label")
        if self.role is not None:
            _require_str(self.role, "Participant.role")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Participant:
        m = _require_mapping(data, "Participant")
        _check_keys(
            m, {"participant_id", "label"}, {"participant_id", "label", "role"}, "Participant"
        )
        return cls(
            participant_id=_require_str(m["participant_id"], "Participant.participant_id"),
            label=_require_str(m["label"], "Participant.label"),
            role=_optional_str(m.get("role"), "Participant.role", allow_empty=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"participant_id": self.participant_id, "label": self.label, "role": self.role}


@dataclass(frozen=True, slots=True)
class KnownFact:
    """Explicit user-provided fact in SessionBrief (§3.2)."""

    key: str
    value: str
    source: str = "brief"

    def __post_init__(self) -> None:
        _require_str(self.key, "KnownFact.key")
        _require_str(self.value, "KnownFact.value")
        _require_str(self.source, "KnownFact.source")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KnownFact:
        m = _require_mapping(data, "KnownFact")
        _check_keys(m, {"key", "value"}, {"key", "value", "source"}, "KnownFact")
        return cls(
            key=_require_str(m["key"], "KnownFact.key"),
            value=_require_str(m["value"], "KnownFact.value"),
            source=_require_str(m.get("source", "brief"), "KnownFact.source"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "source": self.source}


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Attention budget settings (§3.2)."""

    max_cards_per_10_min: int = 3
    proactive: bool = True

    def __post_init__(self) -> None:
        _require_int(self.max_cards_per_10_min, "AttentionConfig.max_cards_per_10_min", minimum=0)
        _require_bool(self.proactive, "AttentionConfig.proactive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttentionConfig:
        m = _require_mapping(data, "AttentionConfig")
        _check_keys(m, set(), {"max_cards_per_10_min", "proactive"}, "AttentionConfig")
        return cls(
            max_cards_per_10_min=_require_int(
                m.get("max_cards_per_10_min", 3), "AttentionConfig.max_cards_per_10_min", minimum=0
            ),
            proactive=_require_bool(m.get("proactive", True), "AttentionConfig.proactive"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"max_cards_per_10_min": self.max_cards_per_10_min, "proactive": self.proactive}


@dataclass(frozen=True, slots=True)
class ConsentConfig:
    """Session consent basis (§3.2, §14.1)."""

    basis: ConsentBasis = ConsentBasis.PRESUMED
    note: str | None = None

    def __post_init__(self) -> None:
        _require_enum(ConsentBasis, self.basis, "ConsentConfig.basis")
        if self.note is not None:
            _require_str(self.note, "ConsentConfig.note")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConsentConfig:
        m = _require_mapping(data, "ConsentConfig")
        _check_keys(m, set(), {"basis", "note"}, "ConsentConfig")
        return cls(
            basis=_require_enum(
                ConsentBasis, m.get("basis", ConsentBasis.PRESUMED), "ConsentConfig.basis"
            ),
            note=_optional_str(m.get("note"), "ConsentConfig.note"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"basis": str(self.basis), "note": self.note}


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    """Session transcript/raw retention configuration (§3.2, §14.2)."""

    store_transcript: bool = True
    store_raw_provider_events: bool = True

    def __post_init__(self) -> None:
        _require_bool(self.store_transcript, "RetentionConfig.store_transcript")
        _require_bool(self.store_raw_provider_events, "RetentionConfig.store_raw_provider_events")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetentionConfig:
        m = _require_mapping(data, "RetentionConfig")
        _check_keys(m, set(), {"store_transcript", "store_raw_provider_events"}, "RetentionConfig")
        return cls(
            store_transcript=_require_bool(
                m.get("store_transcript", True), "RetentionConfig.store_transcript"
            ),
            store_raw_provider_events=_require_bool(
                m.get("store_raw_provider_events", True),
                "RetentionConfig.store_raw_provider_events",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_transcript": self.store_transcript,
            "store_raw_provider_events": self.store_raw_provider_events,
        }


@dataclass(frozen=True, slots=True)
class SessionBrief:
    """Explicit session configuration (§3.2)."""

    schema_version: int
    session_id: str
    mode: SessionMode
    recipient: RecipientConfig
    objective: str
    participants: tuple[Participant, ...] = ()
    agenda: tuple[str, ...] = ()
    known_facts: tuple[KnownFact, ...] = ()
    do_not_suggest: tuple[str, ...] = ()
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    consent: ConsentConfig = field(default_factory=ConsentConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"SessionBrief.schema_version must be {SCHEMA_VERSION}")
        _require_str(self.session_id, "SessionBrief.session_id")
        _require_enum(SessionMode, self.mode, "SessionBrief.mode")
        _require_str(self.objective, "SessionBrief.objective")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionBrief:
        m = _require_mapping(data, "SessionBrief")
        version = _require_schema_version(m, "SessionBrief")
        required = {"schema_version", "session_id", "mode", "recipient", "objective"}
        allowed = required | {
            "participants",
            "agenda",
            "known_facts",
            "do_not_suggest",
            "attention",
            "consent",
            "retention",
        }
        _check_keys(m, required, allowed, "SessionBrief")
        participants_raw = m.get("participants", [])
        if not isinstance(participants_raw, (list, tuple)):
            raise ContractValidationError("SessionBrief.participants must be a list or tuple")
        facts_raw = m.get("known_facts", [])
        if not isinstance(facts_raw, (list, tuple)):
            raise ContractValidationError("SessionBrief.known_facts must be a list or tuple")
        return cls(
            schema_version=version,
            session_id=_require_str(m["session_id"], "SessionBrief.session_id"),
            mode=_require_enum(SessionMode, m["mode"], "SessionBrief.mode"),
            recipient=RecipientConfig.from_dict(
                _require_mapping(m["recipient"], "SessionBrief.recipient")
            ),
            objective=_require_str(m["objective"], "SessionBrief.objective"),
            participants=tuple(Participant.from_dict(p) for p in participants_raw),
            agenda=_require_str_seq(m.get("agenda", []), "SessionBrief.agenda"),
            known_facts=tuple(KnownFact.from_dict(f) for f in facts_raw),
            do_not_suggest=_require_str_seq(
                m.get("do_not_suggest", []), "SessionBrief.do_not_suggest"
            ),
            attention=AttentionConfig.from_dict(m["attention"])
            if "attention" in m
            else AttentionConfig(),
            consent=ConsentConfig.from_dict(m["consent"]) if "consent" in m else ConsentConfig(),
            retention=RetentionConfig.from_dict(m["retention"])
            if "retention" in m
            else RetentionConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "mode": str(self.mode),
            "recipient": self.recipient.to_dict(),
            "objective": self.objective,
            "participants": [p.to_dict() for p in self.participants],
            "agenda": list(self.agenda),
            "known_facts": [f.to_dict() for f in self.known_facts],
            "do_not_suggest": list(self.do_not_suggest),
            "attention": self.attention.to_dict(),
            "consent": self.consent.to_dict(),
            "retention": self.retention.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """Normalized canonical event covering all six §6.1 kinds."""

    schema_version: int
    session_id: str
    event_id: str
    kind: EventKind
    received_ms: int
    source_sequence: int
    source_id: str = "default"
    # segment & speaker_update fields
    segment_id: str | None = None
    source_revision: int | None = None
    channel: int | None = None
    speaker_id: str | None = None
    speaker_confidence: float | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    text: str | None = None
    stability: Stability | None = None
    boundary: str | None = None
    asr_confidence: float | None = None
    low_quality_gate: bool = False
    asr_low_confidence: bool = False
    speaker_uncertain: bool = False
    overlap: bool = False
    # brief_update fields
    brief_patch: dict[str, Any] | None = None
    # ledger fields
    suggestion_id: str | None = None
    transition: LedgerState | None = None
    cause: ReasonCode | None = None
    at_ms: int | None = None
    evidence_refs: tuple[SegmentRef, ...] = ()
    feedback: FeedbackReason | None = None
    # control fields
    subtype: ControlSubtype | None = None
    seq_start: int | None = None
    seq_end: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"CanonicalEvent.schema_version must be {SCHEMA_VERSION}")
        _require_str(self.session_id, "CanonicalEvent.session_id")
        _require_str(self.event_id, "CanonicalEvent.event_id")
        kind = _require_enum(EventKind, self.kind, "CanonicalEvent.kind")
        _require_int(self.received_ms, "CanonicalEvent.received_ms", minimum=0)
        _require_int(self.source_sequence, "CanonicalEvent.source_sequence", minimum=0)
        _require_str(self.source_id, "CanonicalEvent.source_id")
        _require_bool(self.low_quality_gate, "CanonicalEvent.low_quality_gate")
        _require_bool(self.asr_low_confidence, "CanonicalEvent.asr_low_confidence")
        _require_bool(self.speaker_uncertain, "CanonicalEvent.speaker_uncertain")
        _require_bool(self.overlap, "CanonicalEvent.overlap")
        _optional_float(self.speaker_confidence, "CanonicalEvent.speaker_confidence")
        _optional_float(self.asr_confidence, "CanonicalEvent.asr_confidence")
        _optional_int(self.seq_start, "CanonicalEvent.seq_start", minimum=0)
        _optional_int(self.seq_end, "CanonicalEvent.seq_end", minimum=0)
        _optional_int(self.source_revision, "CanonicalEvent.source_revision", minimum=1)
        _optional_int(self.channel, "CanonicalEvent.channel", minimum=0)
        _optional_int(self.start_ms, "CanonicalEvent.start_ms", minimum=0)
        _optional_int(self.end_ms, "CanonicalEvent.end_ms", minimum=0)
        if kind is EventKind.SEGMENT:
            _require_str(self.segment_id, "segment.segment_id")
            _require_int(self.source_revision, "segment.source_revision", minimum=1)
            _require_int(self.channel, "segment.channel", minimum=0)
            _require_str(self.speaker_id, "segment.speaker_id")
            start = _require_int(self.start_ms, "segment.start_ms", minimum=0)
            end = _require_int(self.end_ms, "segment.end_ms", minimum=0)
            if end < start:
                raise ContractValidationError(f"segment.end_ms ({end}) < start_ms ({start})")
            _require_str(self.text, "segment.text", allow_empty=True)
            _require_enum(Stability, self.stability, "segment.stability")
        elif kind is EventKind.SPEAKER_UPDATE:
            _require_str(self.segment_id, "speaker_update.segment_id")
            _require_str(self.speaker_id, "speaker_update.speaker_id")
            if self.source_revision is not None:
                _require_int(self.source_revision, "speaker_update.source_revision", minimum=1)
        elif kind is EventKind.BOUNDARY:
            _require_int(self.end_ms, "boundary.end_ms", minimum=0)
            if self.channel is None and self.speaker_id is None:
                raise ContractValidationError("boundary requires channel or speaker_id")
            if self.speaker_id is not None:
                _require_str(self.speaker_id, "boundary.speaker_id", allow_empty=False)
        elif kind is EventKind.BRIEF_UPDATE:
            if not isinstance(self.brief_patch, Mapping) or not self.brief_patch:
                raise ContractValidationError("brief_update.brief_patch must be a non-empty dict")
        elif kind is EventKind.LEDGER:
            _require_str(self.suggestion_id, "ledger.suggestion_id")
            _require_enum(LedgerState, self.transition, "ledger.transition")
            _require_enum(ReasonCode, self.cause, "ledger.cause")
            _require_int(self.at_ms, "ledger.at_ms", minimum=0)
            if self.feedback is not None:
                _require_enum(FeedbackReason, self.feedback, "ledger.feedback")
        elif kind is EventKind.CONTROL:
            subtype = _require_enum(ControlSubtype, self.subtype, "control.subtype")
            if subtype is ControlSubtype.GAP:
                if (
                    self.seq_start is not None
                    and self.seq_end is not None
                    and self.seq_end < self.seq_start
                ):
                    raise ContractValidationError(
                        f"control.gap.seq_end ({self.seq_end}) < seq_start ({self.seq_start})"
                    )
                if (
                    self.start_ms is not None
                    and self.end_ms is not None
                    and self.end_ms < self.start_ms
                ):
                    raise ContractValidationError(
                        f"control.gap.end_ms ({self.end_ms}) < start_ms ({self.start_ms})"
                    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CanonicalEvent:
        m = _require_mapping(data, "CanonicalEvent")
        version = _require_schema_version(m, "CanonicalEvent")
        required = {
            "schema_version",
            "session_id",
            "event_id",
            "kind",
            "received_ms",
            "source_sequence",
        }
        allowed = required | {
            "source_id",
            "segment_id",
            "source_revision",
            "channel",
            "speaker_id",
            "speaker_confidence",
            "start_ms",
            "end_ms",
            "text",
            "stability",
            "boundary",
            "asr_confidence",
            "low_quality_gate",
            "asr_low_confidence",
            "speaker_uncertain",
            "overlap",
            "brief_patch",
            "suggestion_id",
            "transition",
            "cause",
            "at_ms",
            "evidence_refs",
            "feedback",
            "subtype",
            "seq_start",
            "seq_end",
            "reason",
        }
        _check_keys(m, required, allowed, "CanonicalEvent")
        kind = _require_enum(EventKind, m["kind"], "CanonicalEvent.kind")
        stability_raw = m.get("stability")
        transition_raw = m.get("transition")
        cause_raw = m.get("cause")
        feedback_raw = m.get("feedback")
        subtype_raw = m.get("subtype")
        brief_patch_raw = m.get("brief_patch")
        return cls(
            schema_version=version,
            session_id=_require_str(m["session_id"], "CanonicalEvent.session_id"),
            event_id=_require_str(m["event_id"], "CanonicalEvent.event_id"),
            kind=kind,
            received_ms=_require_int(m["received_ms"], "CanonicalEvent.received_ms", minimum=0),
            source_sequence=_require_int(
                m["source_sequence"], "CanonicalEvent.source_sequence", minimum=0
            ),
            source_id=_require_str(m.get("source_id", "default"), "CanonicalEvent.source_id"),
            segment_id=_optional_str(
                m.get("segment_id"), "CanonicalEvent.segment_id", allow_empty=False
            ),
            source_revision=_optional_int(
                m.get("source_revision"), "CanonicalEvent.source_revision", minimum=1
            ),
            channel=_optional_int(m.get("channel"), "CanonicalEvent.channel", minimum=0),
            speaker_id=_optional_str(
                m.get("speaker_id"), "CanonicalEvent.speaker_id", allow_empty=False
            ),
            speaker_confidence=_optional_float(
                m.get("speaker_confidence"), "CanonicalEvent.speaker_confidence"
            ),
            start_ms=_optional_int(m.get("start_ms"), "CanonicalEvent.start_ms", minimum=0),
            end_ms=_optional_int(m.get("end_ms"), "CanonicalEvent.end_ms", minimum=0),
            text=_optional_str(m.get("text"), "CanonicalEvent.text", allow_empty=True),
            stability=_require_enum(Stability, stability_raw, "CanonicalEvent.stability")
            if stability_raw is not None
            else None,
            boundary=_optional_str(m.get("boundary"), "CanonicalEvent.boundary", allow_empty=False),
            asr_confidence=_optional_float(
                m.get("asr_confidence"), "CanonicalEvent.asr_confidence"
            ),
            low_quality_gate=_require_bool(
                m.get("low_quality_gate", False), "CanonicalEvent.low_quality_gate"
            ),
            asr_low_confidence=_require_bool(
                m.get("asr_low_confidence", False), "CanonicalEvent.asr_low_confidence"
            ),
            speaker_uncertain=_require_bool(
                m.get("speaker_uncertain", False), "CanonicalEvent.speaker_uncertain"
            ),
            overlap=_require_bool(m.get("overlap", False), "CanonicalEvent.overlap"),
            brief_patch=dict(brief_patch_raw) if isinstance(brief_patch_raw, Mapping) else None,
            suggestion_id=_optional_str(
                m.get("suggestion_id"), "CanonicalEvent.suggestion_id", allow_empty=False
            ),
            transition=_require_enum(LedgerState, transition_raw, "CanonicalEvent.transition")
            if transition_raw is not None
            else None,
            cause=_require_enum(ReasonCode, cause_raw, "CanonicalEvent.cause")
            if cause_raw is not None
            else None,
            at_ms=_optional_int(m.get("at_ms"), "CanonicalEvent.at_ms", minimum=0),
            evidence_refs=_parse_segment_refs(
                m.get("evidence_refs", []), "CanonicalEvent.evidence_refs"
            ),
            feedback=_require_enum(FeedbackReason, feedback_raw, "CanonicalEvent.feedback")
            if feedback_raw is not None
            else None,
            subtype=_require_enum(ControlSubtype, subtype_raw, "CanonicalEvent.subtype")
            if subtype_raw is not None
            else None,
            seq_start=_optional_int(m.get("seq_start"), "CanonicalEvent.seq_start", minimum=0),
            seq_end=_optional_int(m.get("seq_end"), "CanonicalEvent.seq_end", minimum=0),
            reason=_optional_str(m.get("reason"), "CanonicalEvent.reason", allow_empty=True),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "kind": str(self.kind),
            "received_ms": self.received_ms,
            "source_sequence": self.source_sequence,
            "source_id": self.source_id,
        }
        if self.segment_id is not None:
            out["segment_id"] = self.segment_id
        if self.source_revision is not None:
            out["source_revision"] = self.source_revision
        if self.channel is not None:
            out["channel"] = self.channel
        if self.speaker_id is not None:
            out["speaker_id"] = self.speaker_id
        if self.speaker_confidence is not None:
            out["speaker_confidence"] = self.speaker_confidence
        if self.start_ms is not None:
            out["start_ms"] = self.start_ms
        if self.end_ms is not None:
            out["end_ms"] = self.end_ms
        if self.text is not None:
            out["text"] = self.text
        if self.stability is not None:
            out["stability"] = str(self.stability)
        if self.boundary is not None:
            out["boundary"] = self.boundary
        if self.asr_confidence is not None:
            out["asr_confidence"] = self.asr_confidence
        if self.low_quality_gate:
            out["low_quality_gate"] = True
        if self.asr_low_confidence:
            out["asr_low_confidence"] = True
        if self.speaker_uncertain:
            out["speaker_uncertain"] = True
        if self.overlap:
            out["overlap"] = True
        if self.brief_patch is not None:
            out["brief_patch"] = dict(self.brief_patch)
        if self.suggestion_id is not None:
            out["suggestion_id"] = self.suggestion_id
        if self.transition is not None:
            out["transition"] = str(self.transition)
        if self.cause is not None:
            out["cause"] = str(self.cause)
        if self.at_ms is not None:
            out["at_ms"] = self.at_ms
        if self.evidence_refs:
            out["evidence_refs"] = [r.to_dict() for r in self.evidence_refs]
        if self.feedback is not None:
            out["feedback"] = str(self.feedback)
        if self.subtype is not None:
            out["subtype"] = str(self.subtype)
        if self.seq_start is not None:
            out["seq_start"] = self.seq_start
        if self.seq_end is not None:
            out["seq_end"] = self.seq_end
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def _require_int_dict(value: Any, name: str) -> dict[str, int]:
    m = _require_mapping(value, name)
    return {
        _require_str(k, f"{name}.key"): _require_int(v, f"{name}[{k!r}]", minimum=0)
        for k, v in m.items()
    }


def _require_bool_dict(value: Any, name: str) -> dict[str, bool]:
    m = _require_mapping(value, name)
    return {
        _require_str(k, f"{name}.key"): _require_bool(v, f"{name}[{k!r}]") for k, v in m.items()
    }


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Decision context handed to classifier (§7.1)."""

    schema_version: int
    candidate_id: str
    session_id: str
    origin: Origin
    based_on_state_version: int
    builder_version: str
    prefilter_version: str
    anchor_refs: tuple[SegmentRef, ...]
    context_refs: tuple[SegmentRef, ...]
    anchor_received_ms: int
    display_by_ms: int
    token_counts: dict[str, int]
    omitted: dict[str, int]
    flags: dict[str, bool]
    serialized_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"DecisionContext.schema_version must be {SCHEMA_VERSION}"
            )
        _require_str(self.candidate_id, "DecisionContext.candidate_id")
        _require_str(self.session_id, "DecisionContext.session_id")
        _require_enum(Origin, self.origin, "DecisionContext.origin")
        _require_int(
            self.based_on_state_version, "DecisionContext.based_on_state_version", minimum=0
        )
        _require_str(self.builder_version, "DecisionContext.builder_version")
        _require_str(self.prefilter_version, "DecisionContext.prefilter_version")
        _require_int(self.anchor_received_ms, "DecisionContext.anchor_received_ms", minimum=0)
        _require_int(self.display_by_ms, "DecisionContext.display_by_ms", minimum=0)
        if self.display_by_ms < self.anchor_received_ms:
            raise ContractValidationError("DecisionContext.display_by_ms < anchor_received_ms")
        _require_int_dict(self.token_counts, "DecisionContext.token_counts")
        _require_int_dict(self.omitted, "DecisionContext.omitted")
        _require_bool_dict(self.flags, "DecisionContext.flags")
        _require_str(self.serialized_sha256, "DecisionContext.serialized_sha256")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionContext:
        m = _require_mapping(data, "DecisionContext")
        version = _require_schema_version(m, "DecisionContext")
        required = {
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
        _check_keys(m, required, required, "DecisionContext")
        return cls(
            schema_version=version,
            candidate_id=_require_str(m["candidate_id"], "DecisionContext.candidate_id"),
            session_id=_require_str(m["session_id"], "DecisionContext.session_id"),
            origin=_require_enum(Origin, m["origin"], "DecisionContext.origin"),
            based_on_state_version=_require_int(
                m["based_on_state_version"], "DecisionContext.based_on_state_version", minimum=0
            ),
            builder_version=_require_str(m["builder_version"], "DecisionContext.builder_version"),
            prefilter_version=_require_str(
                m["prefilter_version"], "DecisionContext.prefilter_version"
            ),
            anchor_refs=_parse_segment_refs(m["anchor_refs"], "DecisionContext.anchor_refs"),
            context_refs=_parse_segment_refs(m["context_refs"], "DecisionContext.context_refs"),
            anchor_received_ms=_require_int(
                m["anchor_received_ms"], "DecisionContext.anchor_received_ms", minimum=0
            ),
            display_by_ms=_require_int(
                m["display_by_ms"], "DecisionContext.display_by_ms", minimum=0
            ),
            token_counts=_require_int_dict(m["token_counts"], "DecisionContext.token_counts"),
            omitted=_require_int_dict(m["omitted"], "DecisionContext.omitted"),
            flags=_require_bool_dict(m["flags"], "DecisionContext.flags"),
            serialized_sha256=_require_str(
                m["serialized_sha256"], "DecisionContext.serialized_sha256"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "session_id": self.session_id,
            "origin": str(self.origin),
            "based_on_state_version": self.based_on_state_version,
            "builder_version": self.builder_version,
            "prefilter_version": self.prefilter_version,
            "anchor_refs": [r.to_dict() for r in self.anchor_refs],
            "context_refs": [r.to_dict() for r in self.context_refs],
            "anchor_received_ms": self.anchor_received_ms,
            "display_by_ms": self.display_by_ms,
            "token_counts": dict(self.token_counts),
            "omitted": dict(self.omitted),
            "flags": dict(self.flags),
            "serialized_sha256": self.serialized_sha256,
        }


@dataclass(frozen=True, slots=True)
class ScoreSet:
    """Opportunity + per-category score bundle (§7.2)."""

    opportunity: float
    categories: dict[str, float]

    def __post_init__(self) -> None:
        _require_float(self.opportunity, "ScoreSet.opportunity")
        for k, v in self.categories.items():
            cat = _require_enum(Category, k, f"ScoreSet.categories[{k!r}]")
            if cat is Category.OTHER:
                raise ContractValidationError(
                    "ScoreSet.categories must not contain annotation-only 'OTHER'"
                )
            _require_float(v, f"ScoreSet.categories[{k!r}]")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScoreSet:
        m = _require_mapping(data, "ScoreSet")
        _check_keys(m, {"opportunity", "categories"}, {"opportunity", "categories"}, "ScoreSet")
        cats_raw = _require_mapping(m["categories"], "ScoreSet.categories")
        cats: dict[str, float] = {}
        for k, v in cats_raw.items():
            cat = _require_enum(Category, k, f"ScoreSet.categories[{k!r}]")
            if cat is Category.OTHER:
                raise ContractValidationError("ScoreSet.categories must not contain 'OTHER'")
            cats[str(cat)] = _require_float(v, f"ScoreSet.categories[{k!r}]")
        return cls(
            opportunity=_require_float(m["opportunity"], "ScoreSet.opportunity"),
            categories=cats,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"opportunity": self.opportunity, "categories": dict(self.categories)}


@dataclass(frozen=True, slots=True)
class ClassifierResult:
    """Classifier output record (§7.2)."""

    schema_version: int
    candidate_id: str
    based_on_state_version: int
    classifier_version: str
    raw_scores: ScoreSet
    calibration_version: str | None
    calibrated_scores: ScoreSet | None
    latency_ms: float
    input_refs: tuple[SegmentRef, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"ClassifierResult.schema_version must be {SCHEMA_VERSION}"
            )
        _require_str(self.candidate_id, "ClassifierResult.candidate_id")
        _require_int(
            self.based_on_state_version, "ClassifierResult.based_on_state_version", minimum=0
        )
        _require_str(self.classifier_version, "ClassifierResult.classifier_version")
        if self.calibration_version is not None:
            _require_str(self.calibration_version, "ClassifierResult.calibration_version")
        lat = _require_float(self.latency_ms, "ClassifierResult.latency_ms")
        if lat < 0.0:
            raise ContractValidationError(f"ClassifierResult.latency_ms must be >= 0, got {lat}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClassifierResult:
        m = _require_mapping(data, "ClassifierResult")
        version = _require_schema_version(m, "ClassifierResult")
        required = {
            "schema_version",
            "candidate_id",
            "based_on_state_version",
            "classifier_version",
            "raw_scores",
            "calibration_version",
            "calibrated_scores",
            "latency_ms",
            "input_refs",
        }
        _check_keys(m, required, required, "ClassifierResult")
        cal_scores_raw = m["calibrated_scores"]
        return cls(
            schema_version=version,
            candidate_id=_require_str(m["candidate_id"], "ClassifierResult.candidate_id"),
            based_on_state_version=_require_int(
                m["based_on_state_version"], "ClassifierResult.based_on_state_version", minimum=0
            ),
            classifier_version=_require_str(
                m["classifier_version"], "ClassifierResult.classifier_version"
            ),
            raw_scores=ScoreSet.from_dict(m["raw_scores"]),
            calibration_version=_optional_str(
                m["calibration_version"], "ClassifierResult.calibration_version", allow_empty=False
            ),
            calibrated_scores=ScoreSet.from_dict(cal_scores_raw)
            if cal_scores_raw is not None
            else None,
            latency_ms=_require_float(m["latency_ms"], "ClassifierResult.latency_ms"),
            input_refs=_parse_segment_refs(m["input_refs"], "ClassifierResult.input_refs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "based_on_state_version": self.based_on_state_version,
            "classifier_version": self.classifier_version,
            "raw_scores": self.raw_scores.to_dict(),
            "calibration_version": self.calibration_version,
            "calibrated_scores": self.calibrated_scores.to_dict()
            if self.calibrated_scores is not None
            else None,
            "latency_ms": self.latency_ms,
            "input_refs": [r.to_dict() for r in self.input_refs],
        }


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Routing decision record (§7.3, §16)."""

    schema_version: int
    candidate_id: str
    policy_version: str
    policy_sha256: str
    based_on_state_version: int
    outcome: Outcome
    route: Route | None
    category: Category | None
    reason_codes: tuple[ReasonCode, ...]
    abstain_reasons: tuple[ReasonCode, ...]
    thresholds: dict[str, float] | None
    decided_ms: int

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"RoutingDecision.schema_version must be {SCHEMA_VERSION}"
            )
        _require_str(self.candidate_id, "RoutingDecision.candidate_id")
        _require_str(self.policy_version, "RoutingDecision.policy_version")
        _require_str(self.policy_sha256, "RoutingDecision.policy_sha256")
        _require_int(
            self.based_on_state_version, "RoutingDecision.based_on_state_version", minimum=0
        )
        outcome = _require_enum(Outcome, self.outcome, "RoutingDecision.outcome")
        if self.route is not None:
            _require_enum(Route, self.route, "RoutingDecision.route")
        if self.category is not None:
            cat = _require_enum(Category, self.category, "RoutingDecision.category")
            if cat is Category.OTHER:
                raise ContractValidationError("RoutingDecision.category must never be 'OTHER'")
        if outcome is Outcome.ROUTE and (self.route is None or self.category is None):
            raise ContractValidationError(
                "RoutingDecision with outcome ROUTE requires route and category"
            )
        if outcome is not Outcome.ROUTE and self.route is not None:
            raise ContractValidationError(
                f"RoutingDecision with outcome {outcome} must have route=None, got {self.route!r}"
            )
        for rc in self.reason_codes:
            _require_enum(ReasonCode, rc, "RoutingDecision.reason_codes")
        for ar in self.abstain_reasons:
            code = _require_enum(ReasonCode, ar, "RoutingDecision.abstain_reasons")
            if code not in ABSTAIN_REASON_CODES:
                raise ContractValidationError(f"Invalid abstain reason: {code}")
        if self.thresholds is not None:
            for k in ("t_low", "t_compute", "t_display"):
                if k not in self.thresholds:
                    raise ContractValidationError(f"RoutingDecision.thresholds missing '{k}'")
                _require_float(self.thresholds[k], f"RoutingDecision.thresholds[{k!r}]")
            if not (
                self.thresholds["t_low"]
                <= self.thresholds["t_compute"]
                <= self.thresholds["t_display"]
            ):
                raise ContractValidationError(
                    "Thresholds must satisfy t_low <= t_compute <= t_display"
                )
        _require_int(self.decided_ms, "RoutingDecision.decided_ms", minimum=0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoutingDecision:
        m = _require_mapping(data, "RoutingDecision")
        version = _require_schema_version(m, "RoutingDecision")
        required = {
            "schema_version",
            "candidate_id",
            "policy_version",
            "policy_sha256",
            "based_on_state_version",
            "outcome",
            "route",
            "category",
            "reason_codes",
            "abstain_reasons",
            "thresholds",
            "decided_ms",
        }
        _check_keys(m, required, required, "RoutingDecision")
        rc_raw = m["reason_codes"]
        ar_raw = m["abstain_reasons"]
        if not isinstance(rc_raw, (list, tuple)) or not isinstance(ar_raw, (list, tuple)):
            raise ContractValidationError("reason_codes and abstain_reasons must be lists/tuples")
        thresh_raw = m["thresholds"]
        thresholds: dict[str, float] | None = None
        if thresh_raw is not None:
            tm = _require_mapping(thresh_raw, "RoutingDecision.thresholds")
            thresholds = {
                _require_str(k, "RoutingDecision.thresholds.key"): _require_float(
                    v, f"RoutingDecision.thresholds[{k!r}]"
                )
                for k, v in tm.items()
            }
        return cls(
            schema_version=version,
            candidate_id=_require_str(m["candidate_id"], "RoutingDecision.candidate_id"),
            policy_version=_require_str(m["policy_version"], "RoutingDecision.policy_version"),
            policy_sha256=_require_str(m["policy_sha256"], "RoutingDecision.policy_sha256"),
            based_on_state_version=_require_int(
                m["based_on_state_version"], "RoutingDecision.based_on_state_version", minimum=0
            ),
            outcome=_require_enum(Outcome, m["outcome"], "RoutingDecision.outcome"),
            route=_require_enum(Route, m["route"], "RoutingDecision.route")
            if m["route"] is not None
            else None,
            category=_require_enum(Category, m["category"], "RoutingDecision.category")
            if m["category"] is not None
            else None,
            reason_codes=tuple(
                _require_enum(ReasonCode, c, "RoutingDecision.reason_codes") for c in rc_raw
            ),
            abstain_reasons=tuple(
                _require_enum(ReasonCode, c, "RoutingDecision.abstain_reasons") for c in ar_raw
            ),
            thresholds=thresholds,
            decided_ms=_require_int(m["decided_ms"], "RoutingDecision.decided_ms", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "based_on_state_version": self.based_on_state_version,
            "outcome": str(self.outcome),
            "route": str(self.route) if self.route is not None else None,
            "category": str(self.category) if self.category is not None else None,
            "reason_codes": [str(c) for c in self.reason_codes],
            "abstain_reasons": [str(c) for c in self.abstain_reasons],
            "thresholds": dict(self.thresholds) if self.thresholds is not None else None,
            "decided_ms": self.decided_ms,
        }


@dataclass(frozen=True, slots=True)
class GroundedSlot:
    """Extracted slot with canonical segment revision and character span (§7.4, §5.6)."""

    text: str
    segment_id: str
    revision: int
    char_start: int
    char_end: int
    extractor: str

    def __post_init__(self) -> None:
        _require_str(self.text, "GroundedSlot.text")
        _require_str(self.segment_id, "GroundedSlot.segment_id")
        _require_int(self.revision, "GroundedSlot.revision", minimum=1)
        start = _require_int(self.char_start, "GroundedSlot.char_start", minimum=0)
        end = _require_int(self.char_end, "GroundedSlot.char_end", minimum=0)
        if end <= start:
            raise ContractValidationError(f"GroundedSlot.char_end ({end}) <= char_start ({start})")
        _require_str(self.extractor, "GroundedSlot.extractor")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundedSlot:
        m = _require_mapping(data, "GroundedSlot")
        required = {"text", "segment_id", "revision", "char_start", "char_end", "extractor"}
        _check_keys(m, required, required, "GroundedSlot")
        return cls(
            text=_require_str(m["text"], "GroundedSlot.text"),
            segment_id=_require_str(m["segment_id"], "GroundedSlot.segment_id"),
            revision=_require_int(m["revision"], "GroundedSlot.revision", minimum=1),
            char_start=_require_int(m["char_start"], "GroundedSlot.char_start", minimum=0),
            char_end=_require_int(m["char_end"], "GroundedSlot.char_end", minimum=0),
            extractor=_require_str(m["extractor"], "GroundedSlot.extractor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "segment_id": self.segment_id,
            "revision": self.revision,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "extractor": self.extractor,
        }


@dataclass(frozen=True, slots=True)
class SuggestionCandidate:
    """Validated suggestion candidate ready for display scheduling (§7.4)."""

    schema_version: int
    suggestion_id: str
    candidate_id: str
    origin: Origin
    category: Category
    route: Route
    text: str
    word_count: int
    based_on_state_version: int
    evidence_refs: tuple[SegmentRef, ...]
    template_id: str | None
    grounded_slots: dict[str, GroundedSlot]
    ready_ms: int
    display_by_ms: int
    display_ttl_ms: int = 15000
    invalidate_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"SuggestionCandidate.schema_version must be {SCHEMA_VERSION}"
            )
        _require_str(self.suggestion_id, "SuggestionCandidate.suggestion_id")
        _require_str(self.candidate_id, "SuggestionCandidate.candidate_id")
        _require_enum(Origin, self.origin, "SuggestionCandidate.origin")
        cat = _require_enum(Category, self.category, "SuggestionCandidate.category")
        if cat is Category.OTHER:
            raise ContractValidationError("SuggestionCandidate.category cannot be 'OTHER'")
        _require_enum(Route, self.route, "SuggestionCandidate.route")
        _require_str(self.text, "SuggestionCandidate.text")
        _require_int(self.word_count, "SuggestionCandidate.word_count", minimum=1)
        _require_int(
            self.based_on_state_version, "SuggestionCandidate.based_on_state_version", minimum=0
        )
        if self.template_id is not None:
            _require_str(self.template_id, "SuggestionCandidate.template_id")
        _require_int(self.ready_ms, "SuggestionCandidate.ready_ms", minimum=0)
        _require_int(self.display_by_ms, "SuggestionCandidate.display_by_ms", minimum=0)
        _require_int(self.display_ttl_ms, "SuggestionCandidate.display_ttl_ms", minimum=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SuggestionCandidate:
        m = _require_mapping(data, "SuggestionCandidate")
        version = _require_schema_version(m, "SuggestionCandidate")
        required = {
            "schema_version",
            "suggestion_id",
            "candidate_id",
            "origin",
            "category",
            "route",
            "text",
            "word_count",
            "based_on_state_version",
            "evidence_refs",
            "template_id",
            "grounded_slots",
            "ready_ms",
            "display_by_ms",
            "display_ttl_ms",
            "invalidate_on",
        }
        _check_keys(m, required, required, "SuggestionCandidate")
        slots_raw = _require_mapping(m["grounded_slots"], "SuggestionCandidate.grounded_slots")
        slots = {
            _require_str(k, "SuggestionCandidate.grounded_slots.key"): GroundedSlot.from_dict(v)
            for k, v in slots_raw.items()
        }
        return cls(
            schema_version=version,
            suggestion_id=_require_str(m["suggestion_id"], "SuggestionCandidate.suggestion_id"),
            candidate_id=_require_str(m["candidate_id"], "SuggestionCandidate.candidate_id"),
            origin=_require_enum(Origin, m["origin"], "SuggestionCandidate.origin"),
            category=_require_enum(Category, m["category"], "SuggestionCandidate.category"),
            route=_require_enum(Route, m["route"], "SuggestionCandidate.route"),
            text=_require_str(m["text"], "SuggestionCandidate.text"),
            word_count=_require_int(m["word_count"], "SuggestionCandidate.word_count", minimum=1),
            based_on_state_version=_require_int(
                m["based_on_state_version"], "SuggestionCandidate.based_on_state_version", minimum=0
            ),
            evidence_refs=_parse_segment_refs(
                m["evidence_refs"], "SuggestionCandidate.evidence_refs"
            ),
            template_id=_optional_str(
                m["template_id"], "SuggestionCandidate.template_id", allow_empty=False
            ),
            grounded_slots=slots,
            ready_ms=_require_int(m["ready_ms"], "SuggestionCandidate.ready_ms", minimum=0),
            display_by_ms=_require_int(
                m["display_by_ms"], "SuggestionCandidate.display_by_ms", minimum=0
            ),
            display_ttl_ms=_require_int(
                m["display_ttl_ms"], "SuggestionCandidate.display_ttl_ms", minimum=1
            ),
            invalidate_on=_require_str_seq(m["invalidate_on"], "SuggestionCandidate.invalidate_on"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suggestion_id": self.suggestion_id,
            "candidate_id": self.candidate_id,
            "origin": str(self.origin),
            "category": str(self.category),
            "route": str(self.route),
            "text": self.text,
            "word_count": self.word_count,
            "based_on_state_version": self.based_on_state_version,
            "evidence_refs": [r.to_dict() for r in self.evidence_refs],
            "template_id": self.template_id,
            "grounded_slots": {k: v.to_dict() for k, v in self.grounded_slots.items()},
            "ready_ms": self.ready_ms,
            "display_by_ms": self.display_by_ms,
            "display_ttl_ms": self.display_ttl_ms,
            "invalidate_on": list(self.invalidate_on),
        }


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """Lifecycle and feedback event (§7.5)."""

    schema_version: int
    event_id: str
    suggestion_id: str
    transition: LedgerState
    cause: ReasonCode
    at_ms: int
    kind: str = "ledger"
    evidence_refs: tuple[SegmentRef, ...] = ()
    feedback: FeedbackReason | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"LedgerEvent.schema_version must be {SCHEMA_VERSION}")
        _require_str(self.event_id, "LedgerEvent.event_id")
        if self.kind != EventKind.LEDGER:
            raise ContractValidationError(f"LedgerEvent.kind must be 'ledger', got {self.kind!r}")
        _require_str(self.suggestion_id, "LedgerEvent.suggestion_id")
        _require_enum(LedgerState, self.transition, "LedgerEvent.transition")
        _require_enum(ReasonCode, self.cause, "LedgerEvent.cause")
        _require_int(self.at_ms, "LedgerEvent.at_ms", minimum=0)
        if self.feedback is not None:
            _require_enum(FeedbackReason, self.feedback, "LedgerEvent.feedback")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LedgerEvent:
        m = _require_mapping(data, "LedgerEvent")
        version = _require_schema_version(m, "LedgerEvent")
        required = {
            "schema_version",
            "event_id",
            "kind",
            "suggestion_id",
            "transition",
            "cause",
            "at_ms",
        }
        allowed = required | {"evidence_refs", "feedback"}
        _check_keys(m, required, allowed, "LedgerEvent")
        fb_raw = m.get("feedback")
        return cls(
            schema_version=version,
            event_id=_require_str(m["event_id"], "LedgerEvent.event_id"),
            kind=_require_str(m["kind"], "LedgerEvent.kind"),
            suggestion_id=_require_str(m["suggestion_id"], "LedgerEvent.suggestion_id"),
            transition=_require_enum(LedgerState, m["transition"], "LedgerEvent.transition"),
            cause=_require_enum(ReasonCode, m["cause"], "LedgerEvent.cause"),
            at_ms=_require_int(m["at_ms"], "LedgerEvent.at_ms", minimum=0),
            evidence_refs=_parse_segment_refs(
                m.get("evidence_refs", []), "LedgerEvent.evidence_refs"
            ),
            feedback=_require_enum(FeedbackReason, fb_raw, "LedgerEvent.feedback")
            if fb_raw is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "kind": self.kind,
            "suggestion_id": self.suggestion_id,
            "transition": str(self.transition),
            "cause": str(self.cause),
            "at_ms": self.at_ms,
        }
        if self.evidence_refs:
            out["evidence_refs"] = [r.to_dict() for r in self.evidence_refs]
        if self.feedback is not None:
            out["feedback"] = str(self.feedback)
        return out

    def to_canonical_event(
        self,
        session_id: str,
        *,
        received_ms: int | None = None,
        source_sequence: int = 0,
        source_id: str = "controller",
    ) -> CanonicalEvent:
        return CanonicalEvent(
            schema_version=self.schema_version,
            session_id=session_id,
            event_id=self.event_id,
            kind=EventKind.LEDGER,
            received_ms=self.at_ms if received_ms is None else received_ms,
            source_sequence=source_sequence,
            source_id=source_id,
            suggestion_id=self.suggestion_id,
            transition=self.transition,
            cause=self.cause,
            at_ms=self.at_ms,
            evidence_refs=self.evidence_refs,
            feedback=self.feedback,
        )


@dataclass(frozen=True, slots=True)
class ValidityInterval:
    """Opportunity validity interval inside LabelRecord (§12.3)."""

    open_ms: int
    ideal_ms: int | None = None
    close_ms: int | None = None
    closed_by: ClosedBy | None = None

    def __post_init__(self) -> None:
        open_ms = _require_int(self.open_ms, "ValidityInterval.open_ms", minimum=0)
        if self.ideal_ms is not None:
            ideal = _require_int(self.ideal_ms, "ValidityInterval.ideal_ms", minimum=0)
            if ideal < open_ms:
                raise ContractValidationError("ValidityInterval.ideal_ms < open_ms")
        if self.close_ms is not None:
            close = _require_int(self.close_ms, "ValidityInterval.close_ms", minimum=0)
            if close < open_ms:
                raise ContractValidationError("ValidityInterval.close_ms < open_ms")
        if self.closed_by is not None:
            _require_enum(ClosedBy, self.closed_by, "ValidityInterval.closed_by")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidityInterval:
        m = _require_mapping(data, "ValidityInterval")
        _check_keys(
            m, {"open_ms"}, {"open_ms", "ideal_ms", "close_ms", "closed_by"}, "ValidityInterval"
        )
        cb_raw = m.get("closed_by")
        return cls(
            open_ms=_require_int(m["open_ms"], "ValidityInterval.open_ms", minimum=0),
            ideal_ms=_optional_int(m.get("ideal_ms"), "ValidityInterval.ideal_ms", minimum=0),
            close_ms=_optional_int(m.get("close_ms"), "ValidityInterval.close_ms", minimum=0),
            closed_by=_require_enum(ClosedBy, cb_raw, "ValidityInterval.closed_by")
            if cb_raw is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_ms": self.open_ms,
            "ideal_ms": self.ideal_ms,
            "close_ms": self.close_ms,
            "closed_by": str(self.closed_by) if self.closed_by is not None else None,
        }


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """Offline/online human annotation record (§12.3)."""

    schema_version: int
    label_id: str
    session_id: str
    annotator: str
    sample_source: SampleSource
    annotation_pass: AnnotationPass
    candidate_id: str | None
    state_version_at_decision: int
    worthwhile: bool
    categories: tuple[Category, ...]
    example_questions: tuple[str, ...]
    evidence_refs: tuple[SegmentRef, ...]
    validity: ValidityInterval
    none_reason: NoneReason | None = None
    confidence: str = "sure"
    notes: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"LabelRecord.schema_version must be {SCHEMA_VERSION}")
        _require_str(self.label_id, "LabelRecord.label_id")
        _require_str(self.session_id, "LabelRecord.session_id")
        _require_str(self.annotator, "LabelRecord.annotator")
        _require_enum(SampleSource, self.sample_source, "LabelRecord.sample_source")
        _require_enum(AnnotationPass, self.annotation_pass, "LabelRecord.pass")
        if self.candidate_id is not None:
            _require_str(self.candidate_id, "LabelRecord.candidate_id")
        _require_int(
            self.state_version_at_decision, "LabelRecord.state_version_at_decision", minimum=0
        )
        _require_bool(self.worthwhile, "LabelRecord.worthwhile")
        for cat in self.categories:
            _require_enum(Category, cat, "LabelRecord.categories")
        if self.none_reason is not None:
            _require_enum(NoneReason, self.none_reason, "LabelRecord.none_reason")
        _require_str(self.confidence, "LabelRecord.confidence")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LabelRecord:
        m = _require_mapping(data, "LabelRecord")
        version = _require_schema_version(m, "LabelRecord")
        required = {
            "schema_version",
            "label_id",
            "session_id",
            "annotator",
            "sample_source",
            "candidate_id",
            "state_version_at_decision",
            "worthwhile",
            "categories",
            "example_questions",
            "evidence_refs",
            "validity",
        }
        allowed = required | {"pass", "annotation_pass", "none_reason", "confidence", "notes"}
        _check_keys(m, required, allowed, "LabelRecord")
        pass_val = m.get("pass", m.get("annotation_pass"))
        if pass_val is None:
            raise ContractValidationError("LabelRecord missing required field 'pass'")
        cats_raw = m["categories"]
        if not isinstance(cats_raw, (list, tuple)):
            raise ContractValidationError("LabelRecord.categories must be list or tuple")
        nr_raw = m.get("none_reason")
        return cls(
            schema_version=version,
            label_id=_require_str(m["label_id"], "LabelRecord.label_id"),
            session_id=_require_str(m["session_id"], "LabelRecord.session_id"),
            annotator=_require_str(m["annotator"], "LabelRecord.annotator"),
            sample_source=_require_enum(
                SampleSource, m["sample_source"], "LabelRecord.sample_source"
            ),
            annotation_pass=_require_enum(AnnotationPass, pass_val, "LabelRecord.pass"),
            candidate_id=_optional_str(
                m["candidate_id"], "LabelRecord.candidate_id", allow_empty=False
            ),
            state_version_at_decision=_require_int(
                m["state_version_at_decision"], "LabelRecord.state_version_at_decision", minimum=0
            ),
            worthwhile=_require_bool(m["worthwhile"], "LabelRecord.worthwhile"),
            categories=tuple(
                _require_enum(Category, c, "LabelRecord.categories") for c in cats_raw
            ),
            example_questions=_require_str_seq(
                m["example_questions"], "LabelRecord.example_questions"
            ),
            evidence_refs=_parse_segment_refs(m["evidence_refs"], "LabelRecord.evidence_refs"),
            validity=ValidityInterval.from_dict(m["validity"]),
            none_reason=_require_enum(NoneReason, nr_raw, "LabelRecord.none_reason")
            if nr_raw is not None
            else None,
            confidence=_require_str(m.get("confidence", "sure"), "LabelRecord.confidence"),
            notes=_optional_str(m.get("notes"), "LabelRecord.notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_id": self.label_id,
            "session_id": self.session_id,
            "annotator": self.annotator,
            "sample_source": str(self.sample_source),
            "pass": str(self.annotation_pass),
            "candidate_id": self.candidate_id,
            "state_version_at_decision": self.state_version_at_decision,
            "worthwhile": self.worthwhile,
            "categories": [str(c) for c in self.categories],
            "example_questions": list(self.example_questions),
            "evidence_refs": [r.to_dict() for r in self.evidence_refs],
            "validity": self.validity.to_dict(),
            "none_reason": str(self.none_reason) if self.none_reason is not None else None,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Structured telemetry event (§4.2, §13.1)."""

    schema_version: int
    record_id: str
    session_id: str
    at_ms: int
    level: TelemetryLevel
    kind: str
    candidate_id: str | None = None
    suggestion_id: str | None = None
    stage: str | None = None
    reason_code: ReasonCode | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"TelemetryRecord.schema_version must be {SCHEMA_VERSION}"
            )
        _require_str(self.record_id, "TelemetryRecord.record_id")
        _require_str(self.session_id, "TelemetryRecord.session_id")
        _require_int(self.at_ms, "TelemetryRecord.at_ms", minimum=0)
        _require_enum(TelemetryLevel, self.level, "TelemetryRecord.level")
        _require_str(self.kind, "TelemetryRecord.kind")
        if self.reason_code is not None:
            _require_enum(ReasonCode, self.reason_code, "TelemetryRecord.reason_code")
        if not isinstance(self.payload, Mapping):
            raise ContractValidationError("TelemetryRecord.payload must be a mapping")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TelemetryRecord:
        m = _require_mapping(data, "TelemetryRecord")
        version = _require_schema_version(m, "TelemetryRecord")
        required = {"schema_version", "record_id", "session_id", "at_ms", "level", "kind"}
        allowed = required | {"candidate_id", "suggestion_id", "stage", "reason_code", "payload"}
        _check_keys(m, required, allowed, "TelemetryRecord")
        rc_raw = m.get("reason_code")
        payload_raw = m.get("payload", {})
        if not isinstance(payload_raw, Mapping):
            raise ContractValidationError("TelemetryRecord.payload must be a mapping")
        return cls(
            schema_version=version,
            record_id=_require_str(m["record_id"], "TelemetryRecord.record_id"),
            session_id=_require_str(m["session_id"], "TelemetryRecord.session_id"),
            at_ms=_require_int(m["at_ms"], "TelemetryRecord.at_ms", minimum=0),
            level=_require_enum(TelemetryLevel, m["level"], "TelemetryRecord.level"),
            kind=_require_str(m["kind"], "TelemetryRecord.kind"),
            candidate_id=_optional_str(
                m.get("candidate_id"), "TelemetryRecord.candidate_id", allow_empty=False
            ),
            suggestion_id=_optional_str(
                m.get("suggestion_id"), "TelemetryRecord.suggestion_id", allow_empty=False
            ),
            stage=_optional_str(m.get("stage"), "TelemetryRecord.stage", allow_empty=False),
            reason_code=_require_enum(ReasonCode, rc_raw, "TelemetryRecord.reason_code")
            if rc_raw is not None
            else None,
            payload=dict(payload_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "session_id": self.session_id,
            "at_ms": self.at_ms,
            "level": str(self.level),
            "kind": self.kind,
            "candidate_id": self.candidate_id,
            "suggestion_id": self.suggestion_id,
            "stage": self.stage,
            "reason_code": str(self.reason_code) if self.reason_code is not None else None,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class EgressRecord:
    """Hosted/external generator egress accounting record (§9.3, §13.2 #5, §14.3)."""

    schema_version: int
    egress_id: str
    session_id: str
    candidate_id: str | None
    lane: Origin
    provider: str
    model_id: str
    segment_ids: tuple[str, ...]
    sent_at_ms: int
    first_byte_ms: int | None = None
    complete_ms: int | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    discarded: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"EgressRecord.schema_version must be {SCHEMA_VERSION}")
        _require_str(self.egress_id, "EgressRecord.egress_id")
        _require_str(self.session_id, "EgressRecord.session_id")
        if self.candidate_id is not None:
            _require_str(self.candidate_id, "EgressRecord.candidate_id")
        _require_enum(Origin, self.lane, "EgressRecord.lane")
        _require_str(self.provider, "EgressRecord.provider")
        _require_str(self.model_id, "EgressRecord.model_id")
        _require_str_seq(self.segment_ids, "EgressRecord.segment_ids")
        _require_int(self.sent_at_ms, "EgressRecord.sent_at_ms", minimum=0)
        if self.first_byte_ms is not None:
            _require_int(self.first_byte_ms, "EgressRecord.first_byte_ms", minimum=0)
        if self.complete_ms is not None:
            _require_int(self.complete_ms, "EgressRecord.complete_ms", minimum=0)
        _require_int_dict(self.usage, "EgressRecord.usage")
        _require_bool(self.discarded, "EgressRecord.discarded")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EgressRecord:
        m = _require_mapping(data, "EgressRecord")
        version = _require_schema_version(m, "EgressRecord")
        required = {
            "schema_version",
            "egress_id",
            "session_id",
            "candidate_id",
            "lane",
            "provider",
            "model_id",
            "segment_ids",
            "sent_at_ms",
        }
        allowed = required | {"first_byte_ms", "complete_ms", "finish_reason", "usage", "discarded"}
        _check_keys(m, required, allowed, "EgressRecord")
        return cls(
            schema_version=version,
            egress_id=_require_str(m["egress_id"], "EgressRecord.egress_id"),
            session_id=_require_str(m["session_id"], "EgressRecord.session_id"),
            candidate_id=_optional_str(
                m["candidate_id"], "EgressRecord.candidate_id", allow_empty=False
            ),
            lane=_require_enum(Origin, m["lane"], "EgressRecord.lane"),
            provider=_require_str(m["provider"], "EgressRecord.provider"),
            model_id=_require_str(m["model_id"], "EgressRecord.model_id"),
            segment_ids=_require_str_seq(m["segment_ids"], "EgressRecord.segment_ids"),
            sent_at_ms=_require_int(m["sent_at_ms"], "EgressRecord.sent_at_ms", minimum=0),
            first_byte_ms=_optional_int(
                m.get("first_byte_ms"), "EgressRecord.first_byte_ms", minimum=0
            ),
            complete_ms=_optional_int(m.get("complete_ms"), "EgressRecord.complete_ms", minimum=0),
            finish_reason=_optional_str(m.get("finish_reason"), "EgressRecord.finish_reason"),
            usage=_require_int_dict(m.get("usage", {}), "EgressRecord.usage"),
            discarded=_require_bool(m.get("discarded", False), "EgressRecord.discarded"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "egress_id": self.egress_id,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "lane": str(self.lane),
            "provider": self.provider,
            "model_id": self.model_id,
            "segment_ids": list(self.segment_ids),
            "sent_at_ms": self.sent_at_ms,
            "first_byte_ms": self.first_byte_ms,
            "complete_ms": self.complete_ms,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "discarded": self.discarded,
        }
