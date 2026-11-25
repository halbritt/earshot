"""Opaque-box SQLite WAL inspection, §14.2/§14.3 privacy, and §13.2 invariants."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.e2e.helpers.fixtures import REASON_CODES

_JSON_COLUMNS: frozenset[str] = frozenset(
    {
        "payload_json",
        "payload",
        "record_json",
        "data_json",
        "data",
        "metadata_json",
        "raw_scores_json",
        "calibrated_scores_json",
        "evidence_refs_json",
        "anchor_refs_json",
        "context_refs_json",
        "input_refs_json",
        "reason_codes_json",
        "segment_ids_json",
        "usage_json",
        "grounded_slots_json",
    }
)

_CLOSING_TRANSITIONS: frozenset[str] = frozenset(
    {"used", "dismissed", "resolved", "withdrawn", "expired", "outdated"}
)
_ROUTING_OUTCOMES: frozenset[str] = frozenset({"NONE", "ABSTAIN", "SUPPRESS", "ROUTE"})
_SCORED_OUTCOMES: frozenset[str] = _ROUTING_OUTCOMES
_PREFILTER_OUTCOMES: frozenset[str] = frozenset({"dropped", "prefilter_dropped"})
_RETIRED_OUTCOMES: frozenset[str] = frozenset({"retired", "expired_unevaluated"})


def open_db(db_path: Path, *, readonly: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection (read-only URI when readonly=True)."""
    if readonly:
        uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def assert_wal_mode(db_path: Path) -> str:
    """Assert SQLite database uses WAL journal_mode."""
    with open_db(db_path, readonly=True) as conn:
        row = conn.execute("PRAGMA journal_mode;").fetchone()
    mode = str(row[0]).lower() if row is not None else ""
    if mode != "wal":
        raise AssertionError(f"Expected SQLite journal_mode='wal', got {mode!r} for {db_path}")
    return mode


def list_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    return {str(r[0]) for r in rows}


def get_table_columns(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {
        str(r["name"]): {
            "cid": int(r["cid"]),
            "type": str(r["type"]),
            "notnull": bool(r["notnull"]),
            "pk": int(r["pk"]),
        }
        for r in rows
    }


def assert_tables_exist(
    conn: sqlite3.Connection,
    required: Sequence[str] = (
        "raw_events",
        "events",
        "decisions",
        "suggestions",
        "feedback",
        "model_cache",
        "egress_log",
    ),
) -> None:
    existing = list_tables(conn)
    missing = [t for t in required if t not in existing]
    if missing:
        raise AssertionError(f"Missing required SQLite tables: {missing}; found {sorted(existing)}")


def _unpack_row(row: sqlite3.Row) -> dict[str, Any]:
    raw_dict = dict(row)
    merged: dict[str, Any] = {}
    for col, val in raw_dict.items():
        if col in _JSON_COLUMNS and isinstance(val, str) and val.strip():
            try:
                parsed = json.loads(val)
            except json.JSONDecodeError:
                merged[col] = val
                continue
            if col in {
                "payload_json",
                "payload",
                "record_json",
                "data_json",
                "data",
            } and isinstance(parsed, dict):
                for k, v in parsed.items():
                    merged.setdefault(k, v)
            else:
                base_name = col[:-5] if col.endswith("_json") else col
                merged[base_name] = parsed
                merged[col] = parsed
        else:
            merged[col] = val
    for k, v in raw_dict.items():
        if k not in _JSON_COLUMNS:
            merged[k] = v
    return merged


def fetch_normalized_rows(
    conn: sqlite3.Connection,
    table: str,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch rows from table, unpacking JSON columns and optionally filtering by session_id."""
    if table not in list_tables(conn):
        return []
    cols = get_table_columns(conn, table)
    if session_id is not None and "session_id" in cols:
        cursor = conn.execute(f'SELECT * FROM "{table}" WHERE session_id = ?', (session_id,))
    else:
        cursor = conn.execute(f'SELECT * FROM "{table}"')
    rows = [_unpack_row(r) for r in cursor.fetchall()]
    if session_id is not None and "session_id" not in cols:
        rows = [r for r in rows if r.get("session_id") == session_id]
    return rows


def _contains_forbidden(obj: Any, forbidden: Sequence[str]) -> str | None:
    if not forbidden:
        return None
    if isinstance(obj, str):
        for needle in forbidden:
            if needle and needle in obj:
                return needle
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            found = _contains_forbidden(v, forbidden)
            if found is not None:
                return found
    elif isinstance(obj, list | tuple):
        for item in obj:
            found = _contains_forbidden(item, forbidden)
            if found is not None:
                return found
    return None


_EVENT_TEXT_KEYS = ("text", "raw_text", "transcript")
_DECISION_TEXT_KEYS = (
    "text",
    "serialized_input",
    "serialized_text",
    "prompt_text",
    "question_text",
    "anchor_text",
    "context_text",
)
_SUGGESTION_TEXT_KEYS = ("text", "question_text", "question")
_FEEDBACK_TEXT_KEYS = ("notes", "example_questions", "text")


def assert_session_text_deleted(
    db_path: Path,
    session_id: str,
    *,
    forbidden_substrings: Sequence[str] = (),
    require_decisions_preserved: bool = True,
    require_suggestions_preserved: bool = False,
) -> None:
    """Verify §14.2 session text deletion (raw/text wiped; nulled decisions preserved)."""
    with open_db(db_path, readonly=True) as conn:
        raw_rows = fetch_normalized_rows(conn, "raw_events", session_id)
        if raw_rows:
            raise AssertionError(
                f"Expected 0 raw_events for {session_id} after deletion, found {len(raw_rows)}"
            )

        ev_rows = fetch_normalized_rows(conn, "events", session_id)
        for row in ev_rows:
            for key in _EVENT_TEXT_KEYS:
                if row.get(key) not in (None, ""):
                    raise AssertionError(f"Event {row.get('event_id')} still retains {key!r}")

        dec_rows = fetch_normalized_rows(conn, "decisions", session_id)
        if require_decisions_preserved and not dec_rows:
            raise AssertionError(
                f"Expected nulled decisions preserved for {session_id}, found 0 rows"
            )
        for row in dec_rows:
            for key in _DECISION_TEXT_KEYS:
                if row.get(key) is not None:
                    raise AssertionError(
                        f"Decision {row.get('candidate_id')} still retains {key!r}"
                    )
            if not row.get("candidate_id"):
                raise AssertionError("Preserved decision row missing candidate_id")

        sug_rows = fetch_normalized_rows(conn, "suggestions", session_id)
        if require_suggestions_preserved and not sug_rows:
            raise AssertionError(
                f"Expected nulled suggestions preserved for {session_id}, found 0 rows"
            )
        for row in sug_rows:
            for key in _SUGGESTION_TEXT_KEYS:
                if row.get(key) is not None:
                    raise AssertionError(
                        f"Suggestion {row.get('suggestion_id')} still retains {key!r}"
                    )
            if isinstance(row.get("grounded_slots"), dict):
                for slot_val in row["grounded_slots"].values():
                    if isinstance(slot_val, dict) and slot_val.get("text") is not None:
                        raise AssertionError("Suggestion grounded_slots still retains slot text")

        fb_rows = fetch_normalized_rows(conn, "feedback", session_id)
        for row in fb_rows:
            for key in _FEEDBACK_TEXT_KEYS:
                val = row.get(key)
                if val not in (None, "", []):
                    raise AssertionError(f"Feedback row still retains {key!r}")

        if forbidden_substrings:
            for table in list_tables(conn):
                for row in fetch_normalized_rows(conn, table, session_id):
                    bad = _contains_forbidden(row, forbidden_substrings)
                    if bad is not None:
                        raise AssertionError(
                            f"Forbidden text {bad!r} still present in table {table!r}"
                        )


def assert_local_only_no_egress(db_path: Path, session_id: str | None = None) -> None:
    """Verify §14.3 local_only mode records zero egress_log rows."""
    with open_db(db_path, readonly=True) as conn:
        rows = fetch_normalized_rows(conn, "egress_log", session_id)
    if rows:
        raise AssertionError(f"Expected 0 egress_log rows in local_only mode, found {len(rows)}")


def assert_egress_records_valid(
    db_path: Path,
    session_id: str,
    *,
    expected_segment_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Verify §14.3 egress_log records carry segment_ids, provider, model, timestamp, lane."""
    with open_db(db_path, readonly=True) as conn:
        rows = fetch_normalized_rows(conn, "egress_log", session_id)
    if not rows:
        raise AssertionError(f"Expected at least 1 egress_log row for {session_id}")
    seen_segments: set[str] = set()
    for row in rows:
        seg_ids = row.get("segment_ids")
        if not isinstance(seg_ids, list) or not seg_ids:
            raise AssertionError(f"egress_log row missing non-empty segment_ids: {row}")
        seen_segments.update(str(s) for s in seg_ids)
        for req_key in ("provider", "model", "lane"):
            if not row.get(req_key):
                raise AssertionError(f"egress_log row missing {req_key!r}: {row}")
        if (
            row.get("timestamp_ms") is None
            and row.get("created_at") is None
            and row.get("ts") is None
        ):
            raise AssertionError(f"egress_log row missing timestamp: {row}")
    if expected_segment_ids is not None and not expected_segment_ids.issubset(seen_segments):
        missing = expected_segment_ids - seen_segments
        raise AssertionError(f"egress_log missing expected segment_ids: {sorted(missing)}")
    return rows


def assert_model_cache_has_entries(db_path: Path, *, min_count: int = 1) -> list[dict[str, Any]]:
    """Verify §13.4 model_cache entries contain serialized_sha256 and model_version."""
    with open_db(db_path, readonly=True) as conn:
        rows = fetch_normalized_rows(conn, "model_cache", None)
    if len(rows) < min_count:
        raise AssertionError(f"Expected >= {min_count} model_cache entries, found {len(rows)}")
    for row in rows:
        if not row.get("serialized_sha256") or not row.get("model_version"):
            raise AssertionError(f"model_cache entry missing key fields: {row}")
    return rows


def _extract_segment_revisions(
    events: list[dict[str, Any]],
    recipient_ids: frozenset[str],
) -> tuple[set[str], set[tuple[str, int]]]:
    """Derive committed non-recipient segment IDs and valid (segment_id, revision) pairs."""
    committed: dict[str, dict[str, Any]] = {}
    valid_revs: set[tuple[str, int]] = set()
    material_counts: dict[str, int] = {}

    def _order_key(ev: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(ev.get("received_ms") or 0),
            int(ev.get("source_sequence") or 0),
            int(ev.get("source_revision") or 0),
        )

    for ev in sorted(events, key=_order_key):
        kind = ev.get("kind")
        seg_id = ev.get("segment_id")
        if not isinstance(seg_id, str) or not seg_id:
            continue
        if kind == "segment" and ev.get("stability") == "committed":
            rev = int(ev.get("source_revision") or 1)
            prev = committed.get(seg_id)
            if prev is None or rev >= int(prev.get("source_revision") or 0):
                committed[seg_id] = dict(ev)
                material_counts[seg_id] = max(material_counts.get(seg_id, 0) + 1, rev)
                valid_revs.add((seg_id, rev))
                valid_revs.add((seg_id, material_counts[seg_id]))
        elif kind == "speaker_update" and seg_id in committed:
            rev = int(ev.get("source_revision") or (material_counts[seg_id] + 1))
            committed[seg_id]["speaker_id"] = ev.get("speaker_id")
            committed[seg_id]["source_revision"] = rev
            material_counts[seg_id] = max(material_counts[seg_id] + 1, rev)
            valid_revs.add((seg_id, rev))
            valid_revs.add((seg_id, material_counts[seg_id]))

    non_recipient = {
        sid
        for sid, seg in committed.items()
        if str(seg.get("speaker_id") or "") not in recipient_ids
    }
    return non_recipient, valid_revs


def _iter_ref_pairs(refs: Any) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    if isinstance(refs, dict):
        if "segment_id" in refs and ("revision" in refs or "source_revision" in refs):
            rev = refs.get("revision", refs.get("source_revision"))
            if rev is not None:
                pairs.append((str(refs["segment_id"]), int(rev)))
        else:
            for val in refs.values():
                pairs.extend(_iter_ref_pairs(val))
    elif isinstance(refs, list | tuple):
        if len(refs) == 2 and isinstance(refs[0], str) and isinstance(refs[1], int):
            pairs.append((refs[0], refs[1]))
        else:
            for item in refs:
                pairs.extend(_iter_ref_pairs(item))
    return pairs


def _row_reason_codes(row: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for key in ("terminal_reason", "reason_code", "reason", "cause"):
        val = row.get(key)
        if isinstance(val, str) and val:
            codes.append(val)
    multi = row.get("reason_codes")
    if isinstance(multi, list):
        codes.extend(str(c) for c in multi if c)
    return codes


def assert_six_conservation_invariants(
    db_path: Path,
    session_id: str,
    *,
    recipient_ids: frozenset[str] = frozenset({"me", "ME"}),
) -> dict[str, Any]:
    """Verify all six §13.2 end-of-session conservation invariants on SQLite WAL DB."""
    with open_db(db_path, readonly=True) as conn:
        events = fetch_normalized_rows(conn, "events", session_id)
        decisions = fetch_normalized_rows(conn, "decisions", session_id)
        suggestions = fetch_normalized_rows(conn, "suggestions", session_id)
        feedback = fetch_normalized_rows(conn, "feedback", session_id)
        egress = fetch_normalized_rows(conn, "egress_log", session_id)

    non_recip_segments, valid_revs = _extract_segment_revisions(events, recipient_ids)
    anchored_segments: set[str] = set()
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for dec in decisions:
        cid = dec.get("candidate_id")
        if cid:
            by_candidate.setdefault(str(cid), []).append(dec)
        for seg_id, _rev in _iter_ref_pairs(dec.get("anchor_refs")):
            anchored_segments.add(seg_id)

    # Inv 1: Every committed non-recipient segment in >=1 anchor; every candidate in 1 partition.
    unanchored = non_recip_segments - anchored_segments
    if unanchored:
        raise AssertionError(
            f"Inv 1 violated: unanchored non-recipient segments {sorted(unanchored)}"
        )

    scored_cids: set[str] = set()
    for cid, rows in by_candidate.items():
        partitions: set[str] = set()
        for row in rows:
            outcome = str(row.get("outcome") or row.get("partition") or row.get("stage") or "")
            has_scores = (
                row.get("raw_scores") is not None
                or row.get("calibrated_scores") is not None
                or outcome in _SCORED_OUTCOMES
            )
            if outcome in _PREFILTER_OUTCOMES:
                partitions.add("dropped")
            elif outcome in _RETIRED_OUTCOMES:
                partitions.add("retired")
            elif has_scores:
                partitions.add("scored")
                scored_cids.add(cid)
            elif outcome:
                raise AssertionError(
                    f"Inv 1 violated: unknown partition/outcome {outcome!r} for {cid}"
                )
        if len(partitions) != 1:
            raise AssertionError(
                f"Inv 1 violated: candidate {cid} has partitions {sorted(partitions)} (expected 1)"
            )

    # Inv 2: Every scored candidate has 1 RoutingDecision in {NONE, ABSTAIN, SUPPRESS, ROUTE}.
    for cid in scored_cids:
        routing_outcomes = [
            str(r.get("outcome"))
            for r in by_candidate[cid]
            if str(r.get("outcome") or "") in _ROUTING_OUTCOMES
        ]
        if len(routing_outcomes) != 1:
            raise AssertionError(
                f"Inv 2 violated: scored candidate {cid} has routing outcomes {routing_outcomes}"
            )

    # Inv 3: Every formed candidate reaches a terminal state with a valid Appendix C ReasonCode.
    sug_by_cid: dict[str, list[dict[str, Any]]] = {}
    for sug in suggestions:
        if sug.get("candidate_id"):
            sug_by_cid.setdefault(str(sug["candidate_id"]), []).append(sug)

    for cid, rows in by_candidate.items():
        codes: list[str] = []
        has_terminal = False
        for r in rows:
            codes.extend(_row_reason_codes(r))
            if r.get("terminal_state") or str(r.get("outcome") or "") in (
                _PREFILTER_OUTCOMES | _RETIRED_OUTCOMES | {"NONE", "ABSTAIN", "SUPPRESS"}
            ):
                has_terminal = True
        for s in sug_by_cid.get(cid, []):
            codes.extend(_row_reason_codes(s))
            if s.get("terminal_state") or s.get("state") in _CLOSING_TRANSITIONS | {
                "unrendered",
                "withdrawn",
                "expired",
            }:
                has_terminal = True
        if not has_terminal:
            raise AssertionError(f"Inv 3 violated: candidate {cid} has no terminal state")
        if not codes:
            raise AssertionError(f"Inv 3 violated: candidate {cid} has no ReasonCode")
        unknown = [c for c in codes if c not in REASON_CODES]
        if unknown:
            raise AssertionError(f"Inv 3 violated: candidate {cid} has unknown codes {unknown}")

    # Inv 4: Every displayed suggestion has a closing transition.
    transitions_by_sid: dict[str, list[str]] = {}
    for ev in events:
        if ev.get("kind") == "ledger" and ev.get("suggestion_id"):
            transitions_by_sid.setdefault(str(ev["suggestion_id"]), []).append(
                str(ev.get("transition") or "")
            )
    for sug in suggestions:
        sid = sug.get("suggestion_id")
        if not sid:
            continue
        sid_str = str(sid)
        if sug.get("displayed_at_ms") is not None or sug.get("state") == "displayed":
            transitions_by_sid.setdefault(sid_str, []).append("displayed")
        closing = sug.get("terminal_state") or sug.get("closing_transition")
        if isinstance(closing, str) and closing in _CLOSING_TRANSITIONS:
            transitions_by_sid.setdefault(sid_str, []).append(closing)

    for sid_str, trans in transitions_by_sid.items():
        if "displayed" in trans and not any(t in _CLOSING_TRANSITIONS for t in trans):
            raise AssertionError(
                f"Inv 4 violated: displayed suggestion {sid_str} lacks closing transition ({trans})"
            )

    # Inv 5: Every generator invocation has a recorded usage entry.
    gen_records = [
        r
        for r in [*decisions, *suggestions]
        if r.get("route") == "generator" or r.get("generator_invoked")
    ]
    for rec in gen_records:
        usage = rec.get("usage") or rec.get("generator_usage")
        has_usage = isinstance(usage, dict) and (
            usage.get("model") is not None or usage.get("latency_ms") is not None
        )
        has_flat = rec.get("model") is not None and (
            rec.get("prompt_tokens") is not None or rec.get("latency_ms") is not None
        )
        if not (has_usage or has_flat or egress):
            raise AssertionError(f"Inv 5 violated: generator invocation lacks usage record: {rec}")

    # Inv 6: Every (segment_id, revision) reference across tables exists in canonical history.
    ref_tables = [("decisions", decisions), ("suggestions", suggestions), ("feedback", feedback)]
    ref_keys = ("anchor_refs", "context_refs", "input_refs", "evidence_refs", "grounded_slots")
    for tbl_name, rows in ref_tables:
        for row in rows:
            for key in ref_keys:
                if key not in row or row[key] is None:
                    continue
                for seg_id, rev in _iter_ref_pairs(row[key]):
                    if (seg_id, rev) not in valid_revs:
                        raise AssertionError(
                            f"Inv 6 violated: {tbl_name}.{key} cites invalid {(seg_id, rev)}; "
                            f"valid={sorted(valid_revs)}"
                        )

    return {
        "session_id": session_id,
        "non_recipient_segments": len(non_recip_segments),
        "candidates": len(by_candidate),
        "suggestions": len(suggestions),
        "displayed_suggestions": sum(1 for t in transitions_by_sid.values() if "displayed" in t),
        "valid_revisions": len(valid_revs),
    }
