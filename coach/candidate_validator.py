"""Deterministic candidate validation (§8.4, §9.1, §9.2, §5.6, B16)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from coach.canonical_state import SegmentRecord, StateSnapshot
from coach.context_builder import normalize_for_matching, verify_grounded_slot
from coach.contracts import (
    Category,
    LedgerState,
    ReasonCode,
    SessionBrief,
    SuggestionCandidate,
)

DEFAULT_GENERATOR_MAX_WORDS = 20
DEFAULT_NEAR_DUP_THRESHOLD = 0.75

BLOCKING_LEDGER_STATES: frozenset[LedgerState] = frozenset(
    {
        LedgerState.READY,
        LedgerState.DEFERRED,
        LedgerState.DISPLAYED,
        LedgerState.PINNED,
        LedgerState.DISMISSED,
        LedgerState.USED,
    }
)

_EXPECTED_JSON_KEYS = frozenset({"question", "evidence_segment_ids", "decline_reason"})
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_BENIGN_DOT_RE = re.compile(r"(?:\d+\.\d+|v\d+(?:\.\d+)*|\be\.g\.|\bi\.e\.|\bU\.S\.)", re.I)

_NUMBER_TOKEN_RE = re.compile(r"[$€£]?\d+(?:[.,]\d+)?[kKmMbB%]?(?:st|nd|rd|th)?")
_NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
    }
)
_MONTHS = (
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_WEEKDAYS = (
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
)
_DATE_EXPR_RE = re.compile(
    rf"\b(?:(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|"
    rf"(?:in|by|before|after|until|on|since)\s+(?:{_MONTHS})\b|"
    rf"(?:{_WEEKDAYS})\b|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?|"
    rf"q[1-4](?:\s+\d{{4}})?|\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm))\b",
    flags=re.IGNORECASE,
)
_CAP_WORD_RE = re.compile(r"^[A-Z][A-Za-z0-9_-]*$")
_INITIAL_STOPLIST: frozenset[str] = frozenset(
    {
        "Who",
        "What",
        "When",
        "Where",
        "Why",
        "How",
        "Which",
        "Whose",
        "Whom",
        "Is",
        "Are",
        "Was",
        "Were",
        "Do",
        "Does",
        "Did",
        "Can",
        "Could",
        "Would",
        "Will",
        "Should",
        "Has",
        "Have",
        "Had",
        "May",
        "Might",
        "Must",
        "If",
        "In",
        "On",
        "At",
        "By",
        "For",
        "To",
        "With",
        "From",
        "Any",
        "The",
        "A",
        "An",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Here",
        "Before",
        "After",
        "During",
        "Since",
        "Until",
        "While",
        "Unless",
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
        "need",
        "needs",
    }
)


@dataclass(frozen=True, slots=True)
class ParsedGeneratorOutput:
    """Strictly parsed §9.2 generator JSON contract."""

    question: str | None
    evidence_segment_ids: tuple[str, ...]
    decline_reason: str | None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of deterministic candidate validation with check ID (1..6)."""

    valid: bool
    failed_check: int | None = None
    reason: str | None = None
    reason_code: ReasonCode | None = None
    parsed: ParsedGeneratorOutput | None = None


def parse_generator_json(
    raw_text: str,
    *,
    finish_reason: str | None = None,
) -> tuple[ParsedGeneratorOutput | None, str | None]:
    """Check 1: Parse strict §9.2 JSON object with no extra text or truncation."""
    if finish_reason == "length":
        return (None, "Output truncated by token cap (finish_reason='length')")
    stripped = raw_text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return (None, "Response must be a single JSON object with no surrounding text")
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return (None, f"Invalid JSON: {exc}")
    if not isinstance(obj, dict):
        return (None, "Top-level JSON value must be an object")
    if set(obj) != _EXPECTED_JSON_KEYS:
        return (None, f"JSON keys {sorted(obj)} do not match required contract")

    q_raw = obj["question"]
    ev_raw = obj["evidence_segment_ids"]
    dec_raw = obj["decline_reason"]
    if q_raw is not None and not isinstance(q_raw, str):
        return (None, "question must be a string or null")
    if dec_raw is not None and not isinstance(dec_raw, str):
        return (None, "decline_reason must be a string or null")
    if not isinstance(ev_raw, list) or not all(isinstance(x, str) and x.strip() for x in ev_raw):
        return (None, "evidence_segment_ids must be a list of non-empty strings")

    if q_raw is None:
        if dec_raw is None or not dec_raw.strip():
            return (None, "Declined response (question=null) requires non-empty decline_reason")
        return (
            ParsedGeneratorOutput(
                question=None,
                evidence_segment_ids=tuple(ev_raw),
                decline_reason=dec_raw.strip(),
            ),
            None,
        )

    q_clean = q_raw.strip()
    if not q_clean:
        return (None, "question string must not be empty")
    return (
        ParsedGeneratorOutput(
            question=q_clean,
            evidence_segment_ids=tuple(ev_raw),
            decline_reason=dec_raw.strip()
            if isinstance(dec_raw, str) and dec_raw.strip()
            else None,
        ),
        None,
    )


def _check_single_question_format(question: str, max_words: int) -> str | None:
    q = question.strip()
    if not q:
        return "Empty question"
    if "\n" in q or "\r" in q:
        return "Question must be a single line"
    if _LIST_PREFIX_RE.match(q):
        return "Question must not be formatted as a list item"
    if not q.endswith("?"):
        return "Question must end with '?'"
    cap = min(max_words, DEFAULT_GENERATOR_MAX_WORDS)
    words = q.split()
    if len(words) > cap:
        return f"Question exceeds word cap ({len(words)} > {cap})"
    masked = _BENIGN_DOT_RE.sub("TOK", q[:-1])
    if any(ch in masked for ch in ".!?"):
        return "Question must be a single sentence"
    return None


def _build_grounding_corpus(
    snapshot: StateSnapshot,
    supplied_segment_ids: Sequence[str],
    brief: SessionBrief,
) -> tuple[str, set[str]]:
    parts: list[str] = [
        brief.objective,
        *brief.agenda,
        *(f"{f.key} {f.value}" for f in brief.known_facts),
        *(f"{p.participant_id} {p.label} {p.role or ''}" for p in brief.participants),
        brief.recipient.participant_id,
        brief.recipient.role or "",
        "ME",
        *(f"S{i}" for i in range(1, 10)),
    ]
    for sid in supplied_segment_ids:
        seg = snapshot.segments_by_id.get(sid)
        if seg is not None:
            parts.append(seg.text)
    raw_corpus = " ".join(parts)
    tokens: set[str] = set()
    for raw_tok in raw_corpus.split():
        cleaned = raw_tok.strip(".,!?;:\"'”’()[]{}").lower()
        if cleaned.endswith("'s") or cleaned.endswith("’s"):
            cleaned = cleaned[:-2]
        if cleaned:
            tokens.add(cleaned)
    return (raw_corpus.lower(), tokens)


def _check_premise_grounding(
    question: str,
    snapshot: StateSnapshot,
    supplied_segment_ids: Sequence[str],
    brief: SessionBrief,
) -> str | None:
    corpus_lower, corpus_tokens = _build_grounding_corpus(snapshot, supplied_segment_ids, brief)

    # 1. Numbers & number words
    for m in _NUMBER_TOKEN_RE.finditer(question):
        num_tok = m.group(0).lower()
        if num_tok not in corpus_tokens and num_tok not in corpus_lower:
            return f"Ungrounded number {m.group(0)!r}"
    for raw_tok in question.split():
        norm_tok = raw_tok.strip(".,!?;:\"'”’()[]{}").lower()
        if norm_tok in _NUMBER_WORDS and norm_tok not in corpus_tokens:
            return f"Ungrounded number word {norm_tok!r}"

    # 2. Date / time expressions
    for m in _DATE_EXPR_RE.finditer(question):
        expr = " ".join(m.group(0).lower().split())
        if expr not in corpus_lower:
            return f"Ungrounded date/time expression {m.group(0)!r}"

    # 3. Capitalized names / acronyms
    for idx, raw_tok in enumerate(question.split()):
        stripped = raw_tok.strip(".,!?;:\"'”’()[]{}")
        if stripped.endswith("'s") or stripped.endswith("’s"):
            stripped = stripped[:-2]
        if not stripped or not _CAP_WORD_RE.match(stripped):
            continue
        if idx == 0 and stripped in _INITIAL_STOPLIST:
            continue
        if stripped.lower() not in corpus_tokens:
            return f"Ungrounded capitalized name/acronym {stripped!r}"
    return None


def _check_do_not_suggest(
    question: str,
    category: Category,
    brief: SessionBrief,
) -> str | None:
    norm_q = normalize_for_matching(question)
    norm_cat = normalize_for_matching(category.value)
    for term in brief.do_not_suggest:
        norm_term = normalize_for_matching(term)
        if not norm_term:
            continue
        pat = re.compile(rf"(?<!\w){re.escape(norm_term)}(?!\w)")
        if pat.search(norm_q) or pat.search(norm_cat):
            return f"Matches do_not_suggest term {term!r}"
    return None


def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize_for_matching(text).split() if t and t not in _CONTENT_STOPWORDS}


def _check_ledger_dedup(
    question: str,
    category: Category,
    evidence_ids: Sequence[str],
    snapshot: StateSnapshot,
    *,
    ledger_texts: Mapping[str, str] | None = None,
    ledger_categories: Mapping[str, Category] | None = None,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> tuple[ReasonCode, str] | None:
    norm_q = normalize_for_matching(question)
    q_tokens = _content_tokens(question)
    ev_set = set(evidence_ids)
    for entry in snapshot.ledger.values():
        if entry.state not in BLOCKING_LEDGER_STATES:
            continue
        entry_cat = ledger_categories.get(entry.suggestion_id) if ledger_categories else None
        entry_ev = {r.segment_id for r in entry.evidence_refs}
        if entry_cat is category and (ev_set & entry_ev):
            return (
                ReasonCode.DUPLICATE,
                f"Duplicate category {category} and overlapping evidence with {entry.suggestion_id}",
            )
        prev_text = ledger_texts.get(entry.suggestion_id) if ledger_texts else None
        if prev_text:
            if normalize_for_matching(prev_text) == norm_q:
                return (
                    ReasonCode.DUPLICATE,
                    f"Exact duplicate question of {entry.suggestion_id}",
                )
            prev_tokens = _content_tokens(prev_text)
            if q_tokens and prev_tokens:
                union_sz = len(q_tokens | prev_tokens)
                jaccard = len(q_tokens & prev_tokens) / union_sz if union_sz else 0.0
                if jaccard >= near_dup_threshold:
                    return (
                        ReasonCode.NEAR_DUPLICATE,
                        f"Near-duplicate question of {entry.suggestion_id} (jaccard={jaccard:.2f})",
                    )
    return None


def validate_generated_candidate(
    raw_text: str,
    *,
    finish_reason: str | None,
    category: Category,
    snapshot: StateSnapshot,
    supplied_segment_ids: Sequence[str],
    brief: SessionBrief | None = None,
    max_words: int = DEFAULT_GENERATOR_MAX_WORDS,
    ledger_texts: Mapping[str, str] | None = None,
    ledger_categories: Mapping[str, Category] | None = None,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> ValidationResult:
    """Apply all 6 §9.2 deterministic validation checks before ready transition."""
    active_brief = brief or snapshot.brief

    # Check 1: strict JSON contract
    parsed, parse_err = parse_generator_json(raw_text, finish_reason=finish_reason)
    if parsed is None:
        return ValidationResult(
            valid=False,
            failed_check=1,
            reason=parse_err,
            reason_code=ReasonCode.GENERATOR_INVALID_OUTPUT,
        )
    if parsed.question is None:
        return ValidationResult(
            valid=False,
            failed_check=None,
            reason=parsed.decline_reason or "declined",
            reason_code=ReasonCode.GENERATOR_DECLINED,
            parsed=parsed,
        )

    # Check 2: single sentence, ends '?', within word cap
    fmt_err = _check_single_question_format(parsed.question, max_words=max_words)
    if fmt_err is not None:
        return ValidationResult(
            valid=False,
            failed_check=2,
            reason=fmt_err,
            reason_code=ReasonCode.GENERATOR_INVALID_OUTPUT,
            parsed=parsed,
        )

    # Check 3: non-empty subset of supplied segment_ids resolving in snapshot
    supplied_set = set(supplied_segment_ids)
    if not parsed.evidence_segment_ids:
        return ValidationResult(
            valid=False,
            failed_check=3,
            reason="evidence_segment_ids must be non-empty",
            reason_code=ReasonCode.GENERATOR_INVALID_OUTPUT,
            parsed=parsed,
        )
    if not set(parsed.evidence_segment_ids).issubset(supplied_set):
        return ValidationResult(
            valid=False,
            failed_check=3,
            reason="evidence_segment_ids contains IDs not supplied in prompt",
            reason_code=ReasonCode.GENERATOR_INVALID_OUTPUT,
            parsed=parsed,
        )
    for sid in parsed.evidence_segment_ids:
        if sid not in snapshot.segments_by_id:
            return ValidationResult(
                valid=False,
                failed_check=3,
                reason=f"evidence_segment_id {sid!r} missing from canonical snapshot",
                reason_code=ReasonCode.GENERATOR_INVALID_OUTPUT,
                parsed=parsed,
            )

    # Check 4: numbers, dates, capitalized names grounded in transcript or brief
    premise_err = _check_premise_grounding(
        parsed.question, snapshot, supplied_segment_ids, active_brief
    )
    if premise_err is not None:
        return ValidationResult(
            valid=False,
            failed_check=4,
            reason=premise_err,
            reason_code=ReasonCode.GENERATOR_INVALID_OUTPUT,
            parsed=parsed,
        )

    # Check 5: do_not_suggest check
    dns_err = _check_do_not_suggest(parsed.question, category, active_brief)
    if dns_err is not None:
        return ValidationResult(
            valid=False,
            failed_check=5,
            reason=dns_err,
            reason_code=ReasonCode.DO_NOT_SUGGEST,
            parsed=parsed,
        )

    # Check 6: ledger duplicate / near-duplicate check
    dup = _check_ledger_dedup(
        parsed.question,
        category,
        parsed.evidence_segment_ids,
        snapshot,
        ledger_texts=ledger_texts,
        ledger_categories=ledger_categories,
        near_dup_threshold=near_dup_threshold,
    )
    if dup is not None:
        dup_code, dup_msg = dup
        return ValidationResult(
            valid=False,
            failed_check=6,
            reason=dup_msg,
            reason_code=dup_code,
            parsed=parsed,
        )

    return ValidationResult(valid=True, parsed=parsed)


def validate_template_candidate(
    candidate: SuggestionCandidate,
    snapshot: StateSnapshot,
    brief: SessionBrief | None = None,
    *,
    max_words: int = 12,
    slot_max_words: int = 8,
    ledger_texts: Mapping[str, str] | None = None,
    ledger_categories: Mapping[str, Category] | None = None,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> ValidationResult:
    """Verify §5.6 grounding invariant, single-speaker constraint, format, DNS, & dedup."""
    active_brief = brief or snapshot.brief
    fmt_err = _check_single_question_format(candidate.text, max_words=max_words)
    if fmt_err is not None:
        return ValidationResult(
            valid=False,
            failed_check=2,
            reason=fmt_err,
            reason_code=ReasonCode.SLOT_TOO_LONG,
        )

    speakers: set[str] = set()
    for slot_name, slot in candidate.grounded_slots.items():
        if len(slot.text.split()) > slot_max_words:
            return ValidationResult(
                valid=False,
                failed_check=2,
                reason=f"Slot {slot_name!r} exceeds max_words={slot_max_words}",
                reason_code=ReasonCode.SLOT_TOO_LONG,
            )
        seg = snapshot.segments_by_id.get(slot.segment_id)
        if seg is None or not isinstance(seg, SegmentRecord):
            return ValidationResult(
                valid=False,
                failed_check=3,
                reason=f"Slot {slot_name!r} segment {slot.segment_id!r} missing",
                reason_code=ReasonCode.SLOT_UNAVAILABLE,
            )
        if not verify_grounded_slot(slot, seg):
            return ValidationResult(
                valid=False,
                failed_check=4,
                reason=f"Slot {slot_name!r} failed §5.6 contiguous-substring invariant",
                reason_code=ReasonCode.SLOT_UNAVAILABLE,
            )
        if slot.text not in candidate.text:
            return ValidationResult(
                valid=False,
                failed_check=4,
                reason=f"Slot {slot_name!r} text missing from rendered question",
                reason_code=ReasonCode.SLOT_UNAVAILABLE,
            )
        speakers.add(seg.speaker_id)

    if len(speakers) > 1:
        return ValidationResult(
            valid=False,
            failed_check=3,
            reason="Template candidate combined slots from multiple speakers",
            reason_code=ReasonCode.SLOT_UNAVAILABLE,
        )

    dns_err = _check_do_not_suggest(candidate.text, candidate.category, active_brief)
    if dns_err is not None:
        return ValidationResult(
            valid=False,
            failed_check=5,
            reason=dns_err,
            reason_code=ReasonCode.DO_NOT_SUGGEST,
        )

    dup = _check_ledger_dedup(
        candidate.text,
        candidate.category,
        [r.segment_id for r in candidate.evidence_refs],
        snapshot,
        ledger_texts=ledger_texts,
        ledger_categories=ledger_categories,
        near_dup_threshold=near_dup_threshold,
    )
    if dup is not None:
        dup_code, dup_msg = dup
        return ValidationResult(valid=False, failed_check=6, reason=dup_msg, reason_code=dup_code)

    return ValidationResult(valid=True)
