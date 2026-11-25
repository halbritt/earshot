"""Opaque-box SQLite WAL inspection, §14.2/§14.3 privacy, and §13.2 invariants."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.e2e.helpers.fixtures import REASON_CODES, TERMINAL_REASON_CODES

_JSON_COLUMNS: frozenset[str] = frozenset(
    {
        "payload_json",
        "payload",
        "record_json",
        "data_json",
        "data",
        "metadata_json",
        "brief_json",
        "health_timeline_json",
        "reducer_counters_json",
        "revision_history_json",
        "token_counts_json",
        "omitted_json",
        "flags_json",
        "raw_scores_json",
        "calibrated_scores_json",
        "evidence_refs_json",
        "anchor_refs_json",
        "context_refs_json",
        "input_refs_json",
        "reason_codes_json",
        "abstain_reasons_json",
        "thresholds_json",
        "segment_ids_json",
        "usage_json",
        "grounded_slots_json",
        "grounded_slots",
        "invalidate_on_json",
        "transitions_json",
        "label_json",
        "output_json",
    }
)
_RAW_JSON_STRING_COLUMNS: frozenset[str] = frozenset({"reducer_counters_json"})

_CLOSING_TRANSITIONS: frozenset[str] = frozenset(
    {"used", "dismissed", "resolved", "withdrawn", "expired", "outdated"}
)
_ROUTING_OUTCOMES: frozenset[str] = frozenset({"NONE", "ABSTAIN", "SUPPRESS", "ROUTE"})
_SCORED_OUTCOMES: frozenset[str] = _ROUTING_OUTCOMES
_PREFILTER_OUTCOMES: frozenset[str] = frozenset({"dropped", "prefilter_dropped"})
_RETIRED_OUTCOMES: frozenset[str] = frozenset({"retired", "expired_unevaluated"})
_DECISION_TERMINAL_STATES: frozenset[str] = (
    _PREFILTER_OUTCOMES
    | _RETIRED_OUTCOMES
    | {"NONE", "ABSTAIN", "SUPPRESS", "unrendered"}
    | _CLOSING_TRANSITIONS
)


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
                base_name = col[:-5] if col.endswith("_json") else col
                merged[base_name] = parsed
                merged[col] = parsed
            else:
                base_name = col[:-5] if col.endswith("_json") else col
                merged[base_name] = parsed
                merged[col] = val if col in _RAW_JSON_STRING_COLUMNS else parsed
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
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch rows from table, unpacking JSON columns and optionally filtering."""
    if table not in list_tables(conn):
        return []
    cols = get_table_columns(conn, table)
    clauses: list[str] = []
    params: list[Any] = []
    if session_id is not None and "session_id" in cols:
        clauses.append("session_id = ?")
        params.append(session_id)
    if run_id is not None and "run_id" in cols:
        clauses.append("run_id = ?")
        params.append(run_id)
    sql = f'SELECT * FROM "{table}"'
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    cursor = conn.execute(sql, params)
    rows = [_unpack_row(r) for r in cursor.fetchall()]
    if session_id is not None and "session_id" not in cols:
        rows = [r for r in rows if r.get("session_id") == session_id]
    if run_id is not None and "run_id" not in cols:
        rows = [r for r in rows if r.get("run_id") == run_id]
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
            slots_val = row.get("grounded_slots")
            if isinstance(slots_val, dict):
                for slot_val in slots_val.values():
                    if isinstance(slot_val, dict) and slot_val.get("text") is not None:
                        raise AssertionError("Suggestion grounded_slots still retains slot text")
                    if isinstance(slot_val, str) and slot_val:
                        raise AssertionError("Suggestion grounded_slots still retains slot text")
            elif slots_val not in (None, ""):
                raise AssertionError("Suggestion grounded_slots still retains unredacted text")

        fb_rows = fetch_normalized_rows(conn, "feedback", session_id)
        for row in fb_rows:
            for key in _FEEDBACK_TEXT_KEYS:
                val = row.get(key)
                if val not in (None, "", []):
                    raise AssertionError(f"Feedback row still retains {key!r}")
            lbl_val = row.get("label")
            if isinstance(lbl_val, dict):
                for key in _FEEDBACK_TEXT_KEYS:
                    val = lbl_val.get(key)
                    if val not in (None, "", []):
                        raise AssertionError(f"Feedback label_json still retains {key!r}")

        if forbidden_substrings:
            for table in list_tables(conn):
                tbl_cols = get_table_columns(conn, table)
                filter_sid = (
                    None if table == "model_cache" and "session_id" not in tbl_cols else session_id
                )
                for row in fetch_normalized_rows(conn, table, filter_sid):
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
    run_id: str | None = None,
    expected_segment_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Verify §14.3 egress_log records carry segment_ids, provider, model, timestamp, lane."""
    with open_db(db_path, readonly=True) as conn:
        rows = fetch_normalized_rows(conn, "egress_log", session_id, run_id=run_id)
    if not rows:
        raise AssertionError(f"Expected at least 1 egress_log row for {session_id}")
    seen_segments: set[str] = set()
    for row in rows:
        seg_ids = row.get("segment_ids")
        if not isinstance(seg_ids, list) or not seg_ids:
            raise AssertionError(f"egress_log row missing non-empty segment_ids: {row}")
        seen_segments.update(str(s) for s in seg_ids)
        for req_key in ("provider", "lane"):
            if not row.get(req_key):
                raise AssertionError(f"egress_log row missing {req_key!r}: {row}")
        if not (row.get("model") or row.get("model_id")):
            raise AssertionError(f"egress_log row missing 'model'/'model_id': {row}")
        if all(
            row.get(ts_key) is None for ts_key in ("timestamp_ms", "created_at", "ts", "sent_at_ms")
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
        canon_rev = ev.get("canonical_revision")
        has_canon_rev = (
            isinstance(canon_rev, int) and not isinstance(canon_rev, bool) and canon_rev >= 1
        )
        if kind == "segment" and ev.get("stability") == "committed":
            src_rev = int(ev.get("source_revision") or 1)
            prev = committed.get(seg_id)
            if prev is None:
                committed[seg_id] = dict(ev)
                if has_canon_rev:
                    assert isinstance(canon_rev, int)
                    material_counts[seg_id] = max(material_counts.get(seg_id, 0), canon_rev)
                    valid_revs.add((seg_id, canon_rev))
                else:
                    material_counts[seg_id] = 1
                    valid_revs.add((seg_id, 1))
            elif has_canon_rev or src_rev > int(prev.get("source_revision") or 0):
                committed[seg_id] = dict(ev)
                if has_canon_rev:
                    assert isinstance(canon_rev, int)
                    material_counts[seg_id] = max(material_counts.get(seg_id, 0), canon_rev)
                    valid_revs.add((seg_id, canon_rev))
                else:
                    next_rev = material_counts[seg_id] + 1
                    material_counts[seg_id] = next_rev
                    valid_revs.add((seg_id, next_rev))
        elif kind == "speaker_update" and seg_id in committed:
            src_rev = int(ev.get("source_revision") or (material_counts[seg_id] + 1))
            committed[seg_id]["speaker_id"] = ev.get("speaker_id")
            committed[seg_id]["source_revision"] = src_rev
            if has_canon_rev:
                assert isinstance(canon_rev, int)
                material_counts[seg_id] = max(material_counts.get(seg_id, 0), canon_rev)
                valid_revs.add((seg_id, canon_rev))
            else:
                next_rev = material_counts[seg_id] + 1
                material_counts[seg_id] = next_rev
                valid_revs.add((seg_id, next_rev))

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
    run_id: str | None = None,
    recipient_ids: frozenset[str] = frozenset({"me", "ME"}),
) -> dict[str, Any]:
    """Verify all six §13.2 end-of-session conservation invariants on SQLite WAL DB."""
    with open_db(db_path, readonly=True) as conn:
        assert_tables_exist(conn)
        events = fetch_normalized_rows(conn, "events", session_id, run_id=run_id)
        decisions = fetch_normalized_rows(conn, "decisions", session_id, run_id=run_id)
        suggestions = fetch_normalized_rows(conn, "suggestions", session_id, run_id=run_id)
        feedback = fetch_normalized_rows(conn, "feedback", session_id, run_id=run_id)
        egress = fetch_normalized_rows(conn, "egress_log", session_id, run_id=run_id)

    non_recip_segments, valid_revs = _extract_segment_revisions(events, recipient_ids)
    anchor_cids_by_seg_rev: dict[tuple[str, int], set[str]] = {}
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for dec in decisions:
        cid = dec.get("candidate_id")
        if not isinstance(cid, str) or not cid.strip():
            raise AssertionError(f"Inv 1 violated: decision row missing candidate_id: {dec}")
        by_candidate.setdefault(cid, []).append(dec)
        origin = str(dec.get("origin") or "proactive")
        seen_in_row: set[str] = set()
        for seg_id, rev in _iter_ref_pairs(dec.get("anchor_refs")):
            if seg_id in seen_in_row:
                raise AssertionError(
                    f"Inv 1 violated: segment {seg_id} repeated in anchor_refs for {cid}"
                )
            seen_in_row.add(seg_id)
            if origin != "manual":
                anchor_cids_by_seg_rev.setdefault((seg_id, rev), set()).add(cid)

    # Inv 1: Every committed non-recipient segment (at its latest canonical revision)
    # belongs to == 1 proactive anchor (and <= 1 per historical revision);
    # every candidate belongs to 1 partition.
    for (seg_id, rev), owners in sorted(anchor_cids_by_seg_rev.items()):
        if len(owners) > 1:
            raise AssertionError(
                f"Inv 1 violated: non-recipient segment {seg_id} rev {rev} in {len(owners)} "
                f"anchors ({sorted(owners)}), expected <= 1"
            )

    latest_rev_by_seg: dict[str, int] = {}
    for seg_id, rev in valid_revs:
        if rev > latest_rev_by_seg.get(seg_id, 0):
            latest_rev_by_seg[seg_id] = rev

    for seg_id in sorted(non_recip_segments):
        latest_rev = latest_rev_by_seg.get(seg_id, 1)
        owners = anchor_cids_by_seg_rev.get((seg_id, latest_rev), set())
        if len(owners) != 1:
            raise AssertionError(
                f"Inv 1 violated: non-recipient segment {seg_id} (rev={latest_rev}) in "
                f"{len(owners)} anchors ({sorted(owners)}), expected 1"
            )

    scored_cids: set[str] = set()
    for cid, rows in by_candidate.items():
        partitions: set[str] = set()
        for row in rows:
            outcome = str(row.get("outcome") or row.get("partition") or row.get("stage") or "")
            term_state = str(row.get("terminal_state") or "")
            has_raw_or_cal = (
                row.get("raw_scores") is not None or row.get("calibrated_scores") is not None
            )
            has_scores = has_raw_or_cal or outcome in _SCORED_OUTCOMES
            is_dropped = outcome in _PREFILTER_OUTCOMES or term_state in _PREFILTER_OUTCOMES
            is_retired = outcome in _RETIRED_OUTCOMES or term_state in _RETIRED_OUTCOMES
            if is_dropped:
                if has_raw_or_cal or outcome in _SCORED_OUTCOMES:
                    raise AssertionError(
                        f"Inv 1 violated: dropped candidate {cid} carries scores/routing: {row}"
                    )
                partitions.add("dropped")
            elif is_retired:
                if has_raw_or_cal or outcome in _SCORED_OUTCOMES:
                    raise AssertionError(
                        f"Inv 1 violated: retired candidate {cid} carries scores/routing: {row}"
                    )
                partitions.add("retired")
            elif has_scores:
                partitions.add("scored")
                scored_cids.add(cid)
            elif outcome or term_state:
                raise AssertionError(
                    f"Inv 1 violated: unknown partition {(outcome, term_state)!r} for {cid}"
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

    # Inv 3: Every formed candidate reaches 1 terminal state with terminal Appendix C ReasonCode.
    sug_by_cid: dict[str, list[dict[str, Any]]] = {}
    for sug in suggestions:
        scid = sug.get("candidate_id")
        if not isinstance(scid, str) or not scid.strip():
            raise AssertionError(f"Inv 3 violated: suggestion row missing candidate_id: {sug}")
        sug_by_cid.setdefault(scid, []).append(sug)

    all_cids = set(by_candidate.keys()) | set(sug_by_cid.keys())
    for cid in sorted(all_cids):
        rows = by_candidate.get(cid, [])
        sugs = sug_by_cid.get(cid, [])
        codes: list[str] = []
        per_sug_codes: list[tuple[str, list[str]]] = []
        terminal_states: set[str] = set()
        has_non_terminal_sug = False
        for r in rows:
            codes.extend(_row_reason_codes(r))
            tr = r.get("terminal_reason")
            if isinstance(tr, str) and tr and tr not in TERMINAL_REASON_CODES:
                raise AssertionError(
                    f"Inv 3 violated: candidate {cid} non-terminal decision terminal_reason {tr!r}"
                )
            d_term = r.get("terminal_state")
            d_out = str(r.get("outcome") or "")
            if isinstance(d_term, str) and d_term in _DECISION_TERMINAL_STATES:
                terminal_states.add(d_term)
            elif d_out in (
                _PREFILTER_OUTCOMES | _RETIRED_OUTCOMES | {"NONE", "ABSTAIN", "SUPPRESS"}
            ):
                terminal_states.add(d_out)
        for s in sugs:
            sc = _row_reason_codes(s)
            codes.extend(sc)
            per_sug_codes.append((str(s.get("suggestion_id") or "<unknown>"), sc))
            tr = s.get("terminal_reason")
            if isinstance(tr, str) and tr and tr not in TERMINAL_REASON_CODES:
                raise AssertionError(
                    f"Inv 3 violated: candidate {cid} non-terminal suggestion reason {tr!r}"
                )
            s_state = s.get("terminal_state") or s.get("current_state") or s.get("state")
            if isinstance(s_state, str) and s_state in (_CLOSING_TRANSITIONS | {"unrendered"}):
                terminal_states.add(s_state)
            elif s_state is not None:
                has_non_terminal_sug = True
        if has_non_terminal_sug or len(terminal_states) != 1:
            raise AssertionError(
                f"Inv 3 violated: candidate {cid} terminal_states={sorted(terminal_states)} "
                f"(non_terminal_sug={has_non_terminal_sug}, expected exactly 1)"
            )
        if not codes:
            raise AssertionError(f"Inv 3 violated: candidate {cid} has no ReasonCode")
        unknown = [c for c in codes if c not in REASON_CODES]
        if unknown:
            raise AssertionError(f"Inv 3 violated: candidate {cid} has unknown codes {unknown}")
        if not any(c in TERMINAL_REASON_CODES for c in codes):
            raise AssertionError(
                f"Inv 3 violated: candidate {cid} lacks terminal ReasonCode ({codes})"
            )
        for sid_label, sc in per_sug_codes:
            if not any(c in TERMINAL_REASON_CODES for c in sc):
                raise AssertionError(
                    f"Inv 3 violated: suggestion {sid_label} for {cid} lacks terminal "
                    f"ReasonCode ({sc})"
                )

    # Inv 4: Every displayed suggestion has a closing transition (allowing used -> resolved).
    trans_events_by_sid: dict[str, list[tuple[str, str | None]]] = {}
    for ev in events:
        if ev.get("kind") == "ledger" and ev.get("suggestion_id"):
            sid_str = str(ev["suggestion_id"])
            eid = str(ev["event_id"]) if ev.get("event_id") else None
            trans_events_by_sid.setdefault(sid_str, []).append(
                (str(ev.get("transition") or ""), eid)
            )
    for sug in suggestions:
        sid = sug.get("suggestion_id")
        if not sid:
            continue
        sid_str = str(sid)
        entries = trans_events_by_sid.setdefault(sid_str, [])
        seen_eids = {eid for _, eid in entries if eid is not None}
        sug_trans = sug.get("transitions")
        if isinstance(sug_trans, list):
            for item in sug_trans:
                if isinstance(item, dict):
                    t_name = str(item.get("transition") or "")
                    eid = str(item["event_id"]) if item.get("event_id") else None
                    if eid is not None and eid in seen_eids:
                        continue
                    if eid is not None:
                        seen_eids.add(eid)
                    if t_name:
                        entries.append((t_name, eid))
                elif isinstance(item, str) and item:
                    entries.append((item, None))
        names_so_far = [t for t, _ in entries]
        cur_state = sug.get("current_state") or sug.get("state")
        if "displayed" not in names_so_far and (
            sug.get("displayed_at_ms") is not None or cur_state == "displayed"
        ):
            entries.append(("displayed", None))
        closing = sug.get("terminal_state") or sug.get("closing_transition") or cur_state
        if (
            isinstance(closing, str)
            and closing in _CLOSING_TRANSITIONS
            and not any(t in _CLOSING_TRANSITIONS for t, _ in entries)
        ):
            entries.append((closing, None))

    displayed_count = 0
    for sid_str, entries in trans_events_by_sid.items():
        trans = [t for t, _ in entries]
        if "displayed" in trans:
            displayed_count += 1
            closing_list = [t for t in trans if t in _CLOSING_TRANSITIONS]
            if not (len(closing_list) == 1 or closing_list == ["used", "resolved"]):
                raise AssertionError(
                    f"Inv 4 violated: displayed suggestion {sid_str} has closing transitions "
                    f"{closing_list} (all={trans})"
                )

    # Inv 5: Every generator invocation has a recorded usage entry (no session-wide fallback).
    usage_metric_keys = (
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "latency_ms",
    )
    sid_to_cid = {
        str(s["suggestion_id"]): str(s["candidate_id"])
        for s in suggestions
        if s.get("suggestion_id") and s.get("candidate_id")
    }
    gen_records = [
        r
        for r in [*decisions, *suggestions]
        if r.get("route") == "generator" or r.get("generator_invoked")
    ]
    for rec in gen_records:
        usage = rec.get("usage") or rec.get("generator_usage")
        has_usage = isinstance(usage, dict) and any(
            usage.get(k) is not None
            for k in ("model", "model_id", "latency_ms", "prompt_tokens", "completion_tokens")
        )
        has_flat = (rec.get("model") is not None or rec.get("model_id") is not None) and (
            rec.get("prompt_tokens") is not None or rec.get("latency_ms") is not None
        )
        rec_cid = rec.get("candidate_id")
        rec_sid = rec.get("suggestion_id")
        rec_segs = {
            s
            for s, _ in _iter_ref_pairs(
                rec.get("anchor_refs") or rec.get("evidence_refs") or rec.get("grounded_slots")
            )
        }
        has_matched_egress = False
        for eg in egress:
            eg_usage = eg.get("usage")
            eg_usage_metrics = isinstance(eg_usage, dict) and any(
                eg_usage.get(k) is not None for k in usage_metric_keys
            )
            is_discarded = bool(eg.get("discarded"))
            eg_has_metrics = bool(eg.get("model") or eg.get("model_id")) and (
                eg_usage_metrics
                or eg.get("complete_ms") is not None
                or (
                    is_discarded
                    and (eg.get("sent_at_ms") is not None or eg.get("timestamp_ms") is not None)
                )
            )
            if not eg_has_metrics:
                continue
            eg_cid = eg.get("candidate_id")
            eg_sid = eg.get("suggestion_id")
            if eg_cid not in (None, "", rec_cid):
                continue
            if rec_sid and eg_sid not in (None, "", rec_sid):
                continue
            if rec_cid and eg_sid and sid_to_cid.get(str(eg_sid)) not in (None, rec_cid):
                continue
            eg_segs = {str(s) for s in (eg.get("segment_ids") or [])}
            if (
                (rec_cid and eg_cid == rec_cid)
                or (rec_sid and eg_sid == rec_sid)
                or (rec_segs and eg_segs and bool(rec_segs & eg_segs))
            ):
                has_matched_egress = True
                break
        if not (has_usage or has_flat or has_matched_egress):
            raise AssertionError(f"Inv 5 violated: generator invocation lacks usage record: {rec}")

    # Inv 6: Every (segment_id, revision) reference across tables exists in canonical history.
    ref_tables: list[tuple[str, list[dict[str, Any]], tuple[str, ...]]] = [
        (
            "decisions",
            decisions,
            ("anchor_refs", "context_refs", "input_refs", "evidence_refs", "grounded_slots"),
        ),
        (
            "suggestions",
            suggestions,
            ("anchor_refs", "context_refs", "input_refs", "evidence_refs", "grounded_slots"),
        ),
        ("feedback", feedback, ("evidence_refs",)),
        ("events", events, ("evidence_refs",)),
    ]
    for tbl_name, rows, keys in ref_tables:
        for row in rows:
            for key in keys:
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
        "displayed_suggestions": displayed_count,
        "valid_revisions": len(valid_revs),
    }
