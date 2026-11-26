"""SQLite WAL storage, JSONL fixture I/O, model output cache, and §14.2 deletion."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coach.contracts import (
    CanonicalEvent,
    ClassifierResult,
    DecisionContext,
    EgressRecord,
    EventKind,
    FeedbackReason,
    LabelRecord,
    LedgerEvent,
    Mode,
    Origin,
    ReasonCode,
    RetentionConfig,
    RoutingDecision,
    SessionBrief,
    SuggestionCandidate,
    TelemetryLevel,
    TelemetryRecord,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    recipient_id TEXT NOT NULL DEFAULT 'ME',
    brief_json TEXT,
    mode TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    consent_basis TEXT NOT NULL,
    consent_note TEXT,
    effective_max_cards_per_10_min INTEGER NOT NULL,
    health_timeline_json TEXT NOT NULL DEFAULT '[]',
    reducer_counters_json TEXT NOT NULL DEFAULT '{}',
    revision_history_json TEXT NOT NULL DEFAULT '[]',
    started_ms INTEGER,
    ended_ms INTEGER,
    deleted_at_ms INTEGER,
    PRIMARY KEY (session_id, run_id)
);

CREATE TABLE IF NOT EXISTS raw_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    received_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    segment_id TEXT,
    source_revision INTEGER,
    canonical_revision INTEGER,
    channel INTEGER,
    speaker_id TEXT,
    start_ms INTEGER,
    end_ms INTEGER,
    received_ms INTEGER NOT NULL,
    source_sequence INTEGER NOT NULL,
    stability TEXT,
    text TEXT,
    payload_json TEXT,
    UNIQUE (session_id, run_id, source_id, event_id)
);
"""

_SCHEMA_SQL_2 = """
CREATE TABLE IF NOT EXISTS decisions (
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    origin TEXT NOT NULL,
    based_on_state_version INTEGER NOT NULL,
    builder_version TEXT NOT NULL,
    prefilter_version TEXT NOT NULL,
    anchor_refs_json TEXT NOT NULL,
    context_refs_json TEXT NOT NULL,
    anchor_received_ms INTEGER NOT NULL,
    display_by_ms INTEGER NOT NULL,
    token_counts_json TEXT NOT NULL,
    omitted_json TEXT NOT NULL,
    flags_json TEXT NOT NULL,
    serialized_sha256 TEXT NOT NULL,
    serialized_text TEXT,
    classifier_version TEXT,
    calibration_version TEXT,
    raw_scores_json TEXT,
    calibrated_scores_json TEXT,
    latency_ms REAL,
    input_refs_json TEXT,
    policy_version TEXT,
    policy_sha256 TEXT,
    outcome TEXT,
    route TEXT,
    category TEXT,
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    abstain_reasons_json TEXT NOT NULL DEFAULT '[]',
    thresholds_json TEXT,
    decided_ms INTEGER,
    terminal_state TEXT NOT NULL,
    terminal_reason TEXT,
    PRIMARY KEY (session_id, run_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS suggestions (
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    suggestion_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    origin TEXT NOT NULL,
    category TEXT NOT NULL,
    route TEXT NOT NULL,
    text TEXT,
    word_count INTEGER NOT NULL,
    based_on_state_version INTEGER NOT NULL,
    evidence_refs_json TEXT NOT NULL,
    template_id TEXT,
    grounded_slots_json TEXT,
    ready_ms INTEGER NOT NULL,
    display_by_ms INTEGER NOT NULL,
    display_ttl_ms INTEGER NOT NULL,
    invalidate_on_json TEXT NOT NULL,
    current_state TEXT NOT NULL,
    terminal_reason TEXT,
    transitions_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (session_id, run_id, suggestion_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    suggestion_id TEXT,
    candidate_id TEXT,
    label_id TEXT,
    feedback_reason TEXT,
    sample_source TEXT,
    worthwhile INTEGER,
    label_json TEXT,
    created_ms INTEGER NOT NULL
);
"""

_SCHEMA_SQL_3 = """
CREATE TABLE IF NOT EXISTS model_cache (
    serialized_sha256 TEXT NOT NULL,
    model_version TEXT NOT NULL,
    output_json TEXT NOT NULL,
    contains_text INTEGER NOT NULL DEFAULT 0,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY (serialized_sha256, model_version)
);

CREATE TABLE IF NOT EXISTS model_cache_sessions (
    serialized_sha256 TEXT NOT NULL,
    model_version TEXT NOT NULL,
    session_id TEXT NOT NULL,
    PRIMARY KEY (serialized_sha256, model_version, session_id)
);

CREATE TABLE IF NOT EXISTS egress_log (
    egress_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    candidate_id TEXT,
    lane TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    segment_ids_json TEXT NOT NULL,
    sent_at_ms INTEGER NOT NULL,
    first_byte_ms INTEGER,
    complete_ms INTEGER,
    finish_reason TEXT,
    usage_json TEXT NOT NULL,
    discarded INTEGER NOT NULL,
    PRIMARY KEY (session_id, run_id, egress_id)
);

CREATE TABLE IF NOT EXISTS telemetry_records (
    record_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    level TEXT NOT NULL,
    kind TEXT NOT NULL,
    candidate_id TEXT,
    suggestion_id TEXT,
    stage TEXT,
    reason_code TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (session_id, run_id, record_id)
);

CREATE TABLE IF NOT EXISTS recorder_drops (
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    level TEXT NOT NULL,
    drop_count INTEGER NOT NULL,
    PRIMARY KEY (session_id, run_id, level)
);

CREATE INDEX IF NOT EXISTS idx_events_session_run ON events (session_id, run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_session_run ON decisions (session_id, run_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_session_run ON suggestions (session_id, run_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_session_run ON telemetry_records (session_id, run_id, at_ms);
"""


@dataclass(frozen=True, slots=True)
class DeletionSummary:
    """Counts of rows purged or redacted by §14.2 session text deletion."""

    session_id: str
    raw_events_deleted: int
    events_redacted: int
    decisions_redacted: int
    suggestions_redacted: int
    feedback_redacted: int
    model_cache_purged: int
    deleted_at_ms: int
    telemetry_redacted: int = 0


def _scrub_telemetry_value(val: Any) -> Any:
    if isinstance(val, (bool, int, float)):
        return val
    if isinstance(val, Mapping):
        out: dict[str, Any] = {}
        for k, v in val.items():
            if k == "egress_id" and isinstance(v, str):
                out[k] = v
                continue
            cleaned = _scrub_telemetry_value(v)
            if cleaned is not None:
                out[k] = cleaned
        return out if out else None
    if isinstance(val, list):
        items = [c for item in val if (c := _scrub_telemetry_value(item)) is not None]
        return items if items else None
    return None


def _scrub_telemetry_payload(payload_json: str) -> str:
    loaded = json.loads(payload_json)
    if not isinstance(loaded, dict):
        return "{}"
    scrubbed = _scrub_telemetry_value(loaded)
    if not isinstance(scrubbed, dict):
        return "{}"
    return json.dumps(scrubbed, sort_keys=True)


class SQLiteStore:
    """Single-writer SQLite WAL persistence store for coach sessions and runs (§6.6)."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        if str(db_path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA secure_delete=ON;")
        self._conn.executescript(_SCHEMA_SQL + _SCHEMA_SQL_2 + _SCHEMA_SQL_3)
        cols = {
            str(r["name"]) for r in self._conn.execute("PRAGMA table_info(sessions);").fetchall()
        }
        if "recipient_id" not in cols:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN recipient_id TEXT NOT NULL DEFAULT 'ME';"
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def journal_mode(self) -> str:
        row = self._conn.execute("PRAGMA journal_mode;").fetchone()
        return str(row[0]) if row is not None else ""

    def upsert_session(
        self,
        *,
        brief: SessionBrief,
        run_id: str,
        runtime_mode: Mode = Mode.PROACTIVE,
        effective_max_cards_per_10_min: int | None = None,
        health_timeline: Sequence[Mapping[str, Any]] = (),
        reducer_counters: Mapping[str, int] | None = None,
        revision_history: Iterable[tuple[str, int]] = (),
        started_ms: int | None = None,
        ended_ms: int | None = None,
    ) -> None:
        cap = (
            brief.attention.max_cards_per_10_min
            if effective_max_cards_per_10_min is None
            else effective_max_cards_per_10_min
        )
        brief_json = (
            json.dumps(brief.to_dict(), sort_keys=True)
            if brief.retention.store_transcript
            else None
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    session_id, run_id, recipient_id, brief_json, mode, runtime_mode,
                    consent_basis, consent_note, effective_max_cards_per_10_min,
                    health_timeline_json, reducer_counters_json, revision_history_json,
                    started_ms, ended_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, run_id) DO UPDATE SET
                    recipient_id = excluded.recipient_id,
                    brief_json = excluded.brief_json,
                    mode = excluded.mode,
                    runtime_mode = excluded.runtime_mode,
                    consent_basis = excluded.consent_basis,
                    consent_note = excluded.consent_note,
                    effective_max_cards_per_10_min = excluded.effective_max_cards_per_10_min,
                    health_timeline_json = excluded.health_timeline_json,
                    reducer_counters_json = excluded.reducer_counters_json,
                    revision_history_json = excluded.revision_history_json,
                    started_ms = COALESCE(excluded.started_ms, sessions.started_ms),
                    ended_ms = COALESCE(excluded.ended_ms, sessions.ended_ms)
                """,
                (
                    brief.session_id,
                    run_id,
                    brief.recipient.participant_id,
                    brief_json,
                    str(brief.mode),
                    str(runtime_mode),
                    str(brief.consent.basis),
                    brief.consent.note,
                    cap,
                    json.dumps(list(health_timeline), sort_keys=True),
                    json.dumps(dict(reducer_counters or {}), sort_keys=True),
                    json.dumps([list(pair) for pair in sorted(revision_history)]),
                    started_ms,
                    ended_ms,
                ),
            )

    def get_session(self, session_id: str, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ? AND run_id = ?",
            (session_id, run_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def record_raw_event(
        self,
        *,
        session_id: str,
        run_id: str,
        source_id: str,
        received_ms: int,
        payload: Mapping[str, Any] | str,
        retention: RetentionConfig,
    ) -> bool:
        if not retention.store_raw_provider_events:
            return False
        payload_str = (
            payload if isinstance(payload, str) else json.dumps(dict(payload), sort_keys=True)
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO raw_events (session_id, run_id, source_id, received_ms, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, run_id, source_id, received_ms, payload_str),
            )
        return True

    def get_raw_events(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM raw_events WHERE session_id = ? AND run_id = ? ORDER BY id",
            (session_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_canonical_event(
        self,
        *,
        run_id: str,
        event: CanonicalEvent,
        retention: RetentionConfig,
        canonical_revision: int | None = None,
    ) -> None:
        keep_text = retention.store_transcript
        text_val = event.text if keep_text else None
        payload_val = json.dumps(event.to_dict(), sort_keys=True) if keep_text else None
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    session_id, run_id, event_id, source_id, kind, segment_id,
                    source_revision, canonical_revision, channel, speaker_id,
                    start_ms, end_ms, received_ms, source_sequence, stability,
                    text, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.session_id,
                    run_id,
                    event.event_id,
                    event.source_id,
                    str(event.kind),
                    event.segment_id,
                    event.source_revision,
                    canonical_revision,
                    event.channel,
                    event.speaker_id,
                    event.start_ms,
                    event.end_ms,
                    event.received_ms,
                    event.source_sequence,
                    str(event.stability) if event.stability is not None else None,
                    text_val,
                    payload_val,
                ),
            )

    def get_events(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE session_id = ? AND run_id = ? ORDER BY id",
            (session_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_decision(
        self,
        *,
        run_id: str,
        context: DecisionContext,
        terminal_state: str,
        terminal_reason: ReasonCode | None = None,
        classifier: ClassifierResult | None = None,
        routing: RoutingDecision | None = None,
        serialized_text: str | None = None,
        retention: RetentionConfig | None = None,
    ) -> None:
        keep_text = retention.store_transcript if retention is not None else True
        stored_text = serialized_text if keep_text else None
        reason_codes = [str(c) for c in routing.reason_codes] if routing is not None else []
        abstain_reasons = [str(c) for c in routing.abstain_reasons] if routing is not None else []
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO decisions (
                    session_id, run_id, candidate_id, origin, based_on_state_version,
                    builder_version, prefilter_version, anchor_refs_json, context_refs_json,
                    anchor_received_ms, display_by_ms, token_counts_json, omitted_json,
                    flags_json, serialized_sha256, serialized_text, classifier_version,
                    calibration_version, raw_scores_json, calibrated_scores_json, latency_ms,
                    input_refs_json, policy_version, policy_sha256, outcome, route, category,
                    reason_codes_json, abstain_reasons_json, thresholds_json, decided_ms,
                    terminal_state, terminal_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, run_id, candidate_id) DO UPDATE SET
                    serialized_text = COALESCE(excluded.serialized_text, decisions.serialized_text),
                    classifier_version = COALESCE(excluded.classifier_version, decisions.classifier_version),
                    calibration_version = COALESCE(excluded.calibration_version, decisions.calibration_version),
                    raw_scores_json = COALESCE(excluded.raw_scores_json, decisions.raw_scores_json),
                    calibrated_scores_json = COALESCE(excluded.calibrated_scores_json, decisions.calibrated_scores_json),
                    latency_ms = COALESCE(excluded.latency_ms, decisions.latency_ms),
                    input_refs_json = COALESCE(excluded.input_refs_json, decisions.input_refs_json),
                    policy_version = COALESCE(excluded.policy_version, decisions.policy_version),
                    policy_sha256 = COALESCE(excluded.policy_sha256, decisions.policy_sha256),
                    outcome = COALESCE(excluded.outcome, decisions.outcome),
                    route = COALESCE(excluded.route, decisions.route),
                    category = COALESCE(excluded.category, decisions.category),
                    reason_codes_json = excluded.reason_codes_json,
                    abstain_reasons_json = excluded.abstain_reasons_json,
                    thresholds_json = COALESCE(excluded.thresholds_json, decisions.thresholds_json),
                    decided_ms = COALESCE(excluded.decided_ms, decisions.decided_ms),
                    terminal_state = excluded.terminal_state,
                    terminal_reason = excluded.terminal_reason
                """,
                (
                    context.session_id,
                    run_id,
                    context.candidate_id,
                    str(context.origin),
                    context.based_on_state_version,
                    context.builder_version,
                    context.prefilter_version,
                    json.dumps([r.to_dict() for r in context.anchor_refs]),
                    json.dumps([r.to_dict() for r in context.context_refs]),
                    context.anchor_received_ms,
                    context.display_by_ms,
                    json.dumps(context.token_counts, sort_keys=True),
                    json.dumps(context.omitted, sort_keys=True),
                    json.dumps(context.flags, sort_keys=True),
                    context.serialized_sha256,
                    stored_text,
                    classifier.classifier_version if classifier is not None else None,
                    classifier.calibration_version if classifier is not None else None,
                    json.dumps(classifier.raw_scores.to_dict()) if classifier is not None else None,
                    json.dumps(classifier.calibrated_scores.to_dict())
                    if classifier is not None and classifier.calibrated_scores is not None
                    else None,
                    classifier.latency_ms if classifier is not None else None,
                    json.dumps([r.to_dict() for r in classifier.input_refs])
                    if classifier is not None
                    else None,
                    routing.policy_version if routing is not None else None,
                    routing.policy_sha256 if routing is not None else None,
                    str(routing.outcome) if routing is not None else None,
                    str(routing.route)
                    if routing is not None and routing.route is not None
                    else None,
                    str(routing.category)
                    if routing is not None and routing.category is not None
                    else None,
                    json.dumps(reason_codes),
                    json.dumps(abstain_reasons),
                    json.dumps(routing.thresholds)
                    if routing is not None and routing.thresholds is not None
                    else None,
                    routing.decided_ms if routing is not None else None,
                    terminal_state,
                    str(terminal_reason) if terminal_reason is not None else None,
                ),
            )

    def get_decisions(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM decisions WHERE session_id = ? AND run_id = ? ORDER BY anchor_received_ms, candidate_id",
            (session_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_suggestion(
        self,
        *,
        session_id: str,
        run_id: str,
        suggestion: SuggestionCandidate,
        current_state: str = "ready",
        terminal_reason: ReasonCode | None = None,
        transitions: Sequence[Mapping[str, Any]] = (),
        retention: RetentionConfig | None = None,
    ) -> None:
        keep_text = retention.store_transcript if retention is not None else True
        text_val = suggestion.text if keep_text else None
        slots_val = (
            json.dumps({k: v.to_dict() for k, v in suggestion.grounded_slots.items()})
            if keep_text
            else None
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO suggestions (
                    session_id, run_id, suggestion_id, candidate_id, origin, category, route,
                    text, word_count, based_on_state_version, evidence_refs_json, template_id,
                    grounded_slots_json, ready_ms, display_by_ms, display_ttl_ms,
                    invalidate_on_json, current_state, terminal_reason, transitions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, run_id, suggestion_id) DO UPDATE SET
                    current_state = excluded.current_state,
                    terminal_reason = excluded.terminal_reason,
                    transitions_json = excluded.transitions_json
                """,
                (
                    session_id,
                    run_id,
                    suggestion.suggestion_id,
                    suggestion.candidate_id,
                    str(suggestion.origin),
                    str(suggestion.category),
                    str(suggestion.route),
                    text_val,
                    suggestion.word_count,
                    suggestion.based_on_state_version,
                    json.dumps([r.to_dict() for r in suggestion.evidence_refs]),
                    suggestion.template_id,
                    slots_val,
                    suggestion.ready_ms,
                    suggestion.display_by_ms,
                    suggestion.display_ttl_ms,
                    json.dumps(list(suggestion.invalidate_on)),
                    current_state,
                    str(terminal_reason) if terminal_reason is not None else None,
                    json.dumps(list(transitions)),
                ),
            )

    def record_ledger_event(
        self,
        *,
        session_id: str,
        run_id: str,
        event: LedgerEvent,
    ) -> None:
        row = self._conn.execute(
            "SELECT transitions_json, candidate_id FROM suggestions WHERE session_id = ? AND run_id = ? AND suggestion_id = ?",
            (session_id, run_id, event.suggestion_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown suggestion {event.suggestion_id!r} in run {run_id!r}")
        transitions: list[dict[str, Any]] = json.loads(row["transitions_json"])
        transitions.append(event.to_dict())
        terminal_states = {"used", "dismissed", "resolved", "withdrawn", "expired", "outdated"}
        term_reason = str(event.cause) if str(event.transition) in terminal_states else None
        with self._conn:
            self._conn.execute(
                """
                UPDATE suggestions
                SET current_state = ?, terminal_reason = COALESCE(?, terminal_reason), transitions_json = ?
                WHERE session_id = ? AND run_id = ? AND suggestion_id = ?
                """,
                (
                    str(event.transition),
                    term_reason,
                    json.dumps(transitions),
                    session_id,
                    run_id,
                    event.suggestion_id,
                ),
            )
            if event.feedback is not None:
                self._conn.execute(
                    """
                    INSERT INTO feedback (
                        session_id, run_id, suggestion_id, candidate_id, feedback_reason, created_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        run_id,
                        event.suggestion_id,
                        row["candidate_id"],
                        str(event.feedback),
                        event.at_ms,
                    ),
                )

    def get_suggestions(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM suggestions WHERE session_id = ? AND run_id = ? ORDER BY ready_ms, suggestion_id",
            (session_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_feedback(
        self,
        *,
        session_id: str,
        run_id: str,
        suggestion_id: str | None = None,
        candidate_id: str | None = None,
        feedback_reason: FeedbackReason | None = None,
        created_ms: int,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO feedback (
                    session_id, run_id, suggestion_id, candidate_id, feedback_reason, created_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    run_id,
                    suggestion_id,
                    candidate_id,
                    str(feedback_reason) if feedback_reason is not None else None,
                    created_ms,
                ),
            )

    def record_label(self, *, run_id: str, label: LabelRecord, created_ms: int) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO feedback (
                    session_id, run_id, candidate_id, label_id, sample_source,
                    worthwhile, label_json, created_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label.session_id,
                    run_id,
                    label.candidate_id,
                    label.label_id,
                    str(label.sample_source),
                    1 if label.worthwhile else 0,
                    json.dumps(label.to_dict(), sort_keys=True),
                    created_ms,
                ),
            )

    def get_feedback(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE session_id = ? AND run_id = ? ORDER BY id",
            (session_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_cached_model_output(
        self,
        serialized_sha256: str,
        model_version: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT output_json FROM model_cache WHERE serialized_sha256 = ? AND model_version = ?",
            (serialized_sha256, model_version),
        ).fetchone()
        if row is None:
            return None
        if session_id is not None:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO model_cache_sessions (
                        serialized_sha256, model_version, session_id
                    ) VALUES (?, ?, ?)
                    """,
                    (serialized_sha256, model_version, session_id),
                )
        loaded = json.loads(row["output_json"])
        if not isinstance(loaded, dict):
            return {"value": loaded}
        return loaded

    def put_cached_model_output(
        self,
        *,
        serialized_sha256: str,
        model_version: str,
        output: Mapping[str, Any],
        contains_text: bool = False,
        created_ms: int | None = None,
        session_id: str | None = None,
    ) -> None:
        ts = int(time.monotonic() * 1000) if created_ms is None else created_ms
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO model_cache (
                    serialized_sha256, model_version, output_json, contains_text, created_ms
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(serialized_sha256, model_version) DO UPDATE SET
                    output_json = excluded.output_json,
                    contains_text = excluded.contains_text,
                    created_ms = excluded.created_ms
                """,
                (
                    serialized_sha256,
                    model_version,
                    json.dumps(dict(output), sort_keys=True),
                    1 if contains_text else 0,
                    ts,
                ),
            )
            if session_id is not None:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO model_cache_sessions (
                        serialized_sha256, model_version, session_id
                    ) VALUES (?, ?, ?)
                    """,
                    (serialized_sha256, model_version, session_id),
                )

    def record_egress(self, *, run_id: str, record: EgressRecord) -> None:
        usage_payload = dict(record.usage)
        if "cached_tokens" not in usage_payload and "cached_input_tokens" in usage_payload:
            usage_payload["cached_tokens"] = int(usage_payload["cached_input_tokens"])
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO egress_log (
                    egress_id, session_id, run_id, candidate_id, lane, provider, model_id,
                    segment_ids_json, sent_at_ms, first_byte_ms, complete_ms, finish_reason,
                    usage_json, discarded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, run_id, egress_id) DO UPDATE SET
                    first_byte_ms = excluded.first_byte_ms,
                    complete_ms = excluded.complete_ms,
                    finish_reason = excluded.finish_reason,
                    usage_json = excluded.usage_json,
                    discarded = excluded.discarded
                """,
                (
                    record.egress_id,
                    record.session_id,
                    run_id,
                    record.candidate_id,
                    str(record.lane),
                    record.provider,
                    record.model_id,
                    json.dumps(list(record.segment_ids)),
                    record.sent_at_ms,
                    record.first_byte_ms,
                    record.complete_ms,
                    record.finish_reason,
                    json.dumps(usage_payload, sort_keys=True),
                    1 if record.discarded else 0,
                ),
            )

    def get_egress_records(self, session_id: str, run_id: str) -> list[EgressRecord]:
        rows = self._conn.execute(
            "SELECT * FROM egress_log WHERE session_id = ? AND run_id = ? ORDER BY sent_at_ms, egress_id",
            (session_id, run_id),
        ).fetchall()
        out: list[EgressRecord] = []
        for r in rows:
            out.append(
                EgressRecord(
                    schema_version=1,
                    egress_id=r["egress_id"],
                    session_id=r["session_id"],
                    candidate_id=r["candidate_id"],
                    lane=Origin(r["lane"]),
                    provider=r["provider"],
                    model_id=r["model_id"],
                    segment_ids=tuple(json.loads(r["segment_ids_json"])),
                    sent_at_ms=r["sent_at_ms"],
                    first_byte_ms=r["first_byte_ms"],
                    complete_ms=r["complete_ms"],
                    finish_reason=r["finish_reason"],
                    usage=json.loads(r["usage_json"]),
                    discarded=bool(r["discarded"]),
                )
            )
        return out

    def record_telemetry(self, *, run_id: str, records: Iterable[TelemetryRecord]) -> None:
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO telemetry_records (
                    record_id, session_id, run_id, at_ms, level, kind,
                    candidate_id, suggestion_id, stage, reason_code, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        rec.record_id,
                        rec.session_id,
                        run_id,
                        rec.at_ms,
                        str(rec.level),
                        rec.kind,
                        rec.candidate_id,
                        rec.suggestion_id,
                        rec.stage,
                        str(rec.reason_code) if rec.reason_code is not None else None,
                        json.dumps(rec.payload, sort_keys=True),
                    )
                    for rec in records
                ],
            )

    def get_telemetry_records(self, session_id: str, run_id: str) -> list[TelemetryRecord]:
        rows = self._conn.execute(
            "SELECT * FROM telemetry_records WHERE session_id = ? AND run_id = ? ORDER BY at_ms, record_id",
            (session_id, run_id),
        ).fetchall()
        out: list[TelemetryRecord] = []
        for r in rows:
            rc_raw = r["reason_code"]
            out.append(
                TelemetryRecord(
                    schema_version=1,
                    record_id=r["record_id"],
                    session_id=r["session_id"],
                    at_ms=r["at_ms"],
                    level=TelemetryLevel(r["level"]),
                    kind=r["kind"],
                    candidate_id=r["candidate_id"],
                    suggestion_id=r["suggestion_id"],
                    stage=r["stage"],
                    reason_code=ReasonCode(rc_raw) if rc_raw is not None else None,
                    payload=json.loads(r["payload_json"]),
                )
            )
        return out

    def record_drops(
        self, *, session_id: str, run_id: str, drops_by_level: Mapping[TelemetryLevel, int]
    ) -> None:
        with self._conn:
            for level, count in drops_by_level.items():
                self._conn.execute(
                    """
                    INSERT INTO recorder_drops (session_id, run_id, level, drop_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id, run_id, level) DO UPDATE SET
                        drop_count = recorder_drops.drop_count + excluded.drop_count
                    """,
                    (session_id, run_id, str(level), int(count)),
                )

    def get_recorder_drops(self, session_id: str, run_id: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT level, drop_count FROM recorder_drops WHERE session_id = ? AND run_id = ?",
            (session_id, run_id),
        ).fetchall()
        return {str(r["level"]): int(r["drop_count"]) for r in rows}

    def delete_session_text(
        self, session_id: str, *, deleted_at_ms: int | None = None
    ) -> DeletionSummary:
        """Delete raw events, transcript text, serialized inputs, question text, and labels (§14.2)."""
        ts = int(time.time() * 1000) if deleted_at_ms is None else deleted_at_ms
        with self._conn:
            raw_cur = self._conn.execute(
                "DELETE FROM raw_events WHERE session_id = ?", (session_id,)
            )
            ev_count_row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM events WHERE session_id = ? AND (text IS NOT NULL OR (kind IN ('segment', 'brief_update') AND payload_json IS NOT NULL))",
                (session_id,),
            ).fetchone()
            events_redacted_count = int(ev_count_row["cnt"]) if ev_count_row is not None else 0
            self._conn.execute(
                "UPDATE events SET text = NULL, payload_json = NULL WHERE session_id = ? AND (text IS NOT NULL OR payload_json IS NOT NULL)",
                (session_id,),
            )
            dec_cur = self._conn.execute(
                "UPDATE decisions SET serialized_text = NULL WHERE session_id = ? AND serialized_text IS NOT NULL",
                (session_id,),
            )
            sug_cur = self._conn.execute(
                "UPDATE suggestions SET text = NULL, grounded_slots_json = NULL WHERE session_id = ? AND (text IS NOT NULL OR grounded_slots_json IS NOT NULL)",
                (session_id,),
            )
            fb_del_cur = self._conn.execute(
                "DELETE FROM feedback WHERE session_id = ? AND label_json IS NOT NULL AND feedback_reason IS NULL",
                (session_id,),
            )
            fb_upd_cur = self._conn.execute(
                "UPDATE feedback SET label_json = NULL WHERE session_id = ? AND label_json IS NOT NULL",
                (session_id,),
            )
            cache_cur = self._conn.execute(
                """
                DELETE FROM model_cache
                WHERE contains_text = 1
                  AND (
                      serialized_sha256 IN (
                          SELECT serialized_sha256 FROM decisions WHERE session_id = ?
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM model_cache_sessions mcs
                          WHERE mcs.serialized_sha256 = model_cache.serialized_sha256
                            AND mcs.model_version = model_cache.model_version
                            AND mcs.session_id = ?
                      )
                      OR (
                          NOT EXISTS (
                              SELECT 1
                              FROM decisions d
                              JOIN sessions s ON d.session_id = s.session_id AND d.run_id = s.run_id
                              WHERE d.serialized_sha256 = model_cache.serialized_sha256
                                AND s.deleted_at_ms IS NULL
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM model_cache_sessions mcs
                              WHERE mcs.serialized_sha256 = model_cache.serialized_sha256
                                AND mcs.model_version = model_cache.model_version
                                AND NOT EXISTS (
                                    SELECT 1 FROM sessions s
                                    WHERE s.session_id = mcs.session_id
                                      AND s.deleted_at_ms IS NOT NULL
                                )
                          )
                      )
                  )
                  AND serialized_sha256 NOT IN (
                      SELECT d.serialized_sha256
                      FROM decisions d
                      JOIN sessions s ON d.session_id = s.session_id AND d.run_id = s.run_id
                      WHERE d.session_id != ? AND s.deleted_at_ms IS NULL
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM model_cache_sessions mcs
                      WHERE mcs.serialized_sha256 = model_cache.serialized_sha256
                        AND mcs.model_version = model_cache.model_version
                        AND mcs.session_id != ?
                        AND NOT EXISTS (
                            SELECT 1 FROM sessions s
                            WHERE s.session_id = mcs.session_id
                              AND s.deleted_at_ms IS NOT NULL
                        )
                  )
                """,
                (session_id, session_id, session_id, session_id),
            )
            self._conn.execute(
                "DELETE FROM model_cache_sessions WHERE session_id = ?",
                (session_id,),
            )
            tel_rows = self._conn.execute(
                "SELECT run_id, record_id, payload_json FROM telemetry_records WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            telemetry_redacted = 0
            for tel_row in tel_rows:
                scrubbed_json = _scrub_telemetry_payload(str(tel_row["payload_json"]))
                if scrubbed_json != tel_row["payload_json"]:
                    self._conn.execute(
                        "UPDATE telemetry_records SET payload_json = ? WHERE session_id = ? AND run_id = ? AND record_id = ?",
                        (scrubbed_json, session_id, tel_row["run_id"], tel_row["record_id"]),
                    )
                    telemetry_redacted += 1
            self._conn.execute(
                "UPDATE sessions SET brief_json = NULL, consent_note = NULL, deleted_at_ms = ? WHERE session_id = ?",
                (ts, session_id),
            )
        self._conn.commit()
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
        self._conn.execute("VACUUM;")
        ckpt_row = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
        if ckpt_row is None:
            raise RuntimeError(
                "Cannot guarantee §14.2 WAL purge: PRAGMA wal_checkpoint(TRUNCATE) returned no status row"
            )
        busy, log_frames, checkpointed_frames = int(ckpt_row[0]), int(ckpt_row[1]), int(ckpt_row[2])
        if busy != 0 or log_frames != checkpointed_frames:
            raise RuntimeError(
                "Cannot guarantee §14.2 WAL purge: PRAGMA wal_checkpoint(TRUNCATE) blocked by active WAL reader "
                f"(busy={busy}, log={log_frames}, checkpointed={checkpointed_frames})"
            )
        return DeletionSummary(
            session_id=session_id,
            raw_events_deleted=raw_cur.rowcount,
            events_redacted=events_redacted_count,
            decisions_redacted=dec_cur.rowcount,
            suggestions_redacted=sug_cur.rowcount,
            feedback_redacted=fb_del_cur.rowcount + fb_upd_cur.rowcount,
            model_cache_purged=cache_cur.rowcount,
            deleted_at_ms=ts,
            telemetry_redacted=telemetry_redacted,
        )

    def list_runs(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self._conn.execute(
                "SELECT session_id, run_id, mode, runtime_mode, started_ms, ended_ms FROM sessions ORDER BY session_id, run_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT session_id, run_id, mode, runtime_mode, started_ms, ended_ms FROM sessions WHERE session_id = ? ORDER BY run_id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def export_events_jsonl(
    events: Iterable[CanonicalEvent],
    path: str | Path,
    *,
    brief: SessionBrief | None = None,
) -> int:
    """Write optional SessionBrief followed by CanonicalEvents as JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as fh:
        if brief is not None:
            fh.write(json.dumps(brief.to_dict(), sort_keys=True) + "\n")
        for ev in events:
            fh.write(json.dumps(ev.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def import_events_jsonl(path: str | Path) -> tuple[SessionBrief | None, list[CanonicalEvent]]:
    """Load and validate SessionBrief (if present) and CanonicalEvents from JSONL."""
    source = Path(path)
    brief: SessionBrief | None = None
    events: list[CanonicalEvent] = []
    with source.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} must be a JSON object")
            if "kind" in obj and obj["kind"] in {k.value for k in EventKind}:
                events.append(CanonicalEvent.from_dict(obj))
            elif "objective" in obj and "recipient" in obj:
                if brief is not None:
                    raise ValueError(
                        f"Multiple SessionBrief records found in {path} at line {line_no}"
                    )
                brief = SessionBrief.from_dict(obj)
            else:
                events.append(CanonicalEvent.from_dict(obj))
    return brief, events
