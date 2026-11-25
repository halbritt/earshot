"""CLI entry point for the Live Conversation Coach (`coach`)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from coach.storage import SQLiteStore
from coach.telemetry import build_session_report, diff_runs, explain_silence


def _resolve_run_id(store: SQLiteStore, session_id: str, run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    runs = store.list_runs(session_id)
    if not runs:
        raise SystemExit(f"No runs found for session {session_id!r}")
    return str(runs[-1]["run_id"])


def _emit_output(payload: dict[str, Any], *, as_json: bool, summary_lines: Sequence[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in summary_lines:
            print(line)


def _cmd_report(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        run_id = _resolve_run_id(store, args.session, args.run_id)
        report = build_session_report(store, args.session, run_id)
    payload = report.to_dict()
    lines = [
        f"Session: {report.session_id} (run={report.run_id})",
        f"Mode: {report.mode} | Runtime: {report.runtime_mode} | Consent: {report.consent_basis}",
        f"Duration: {report.duration_ms} ms | Cards: {report.displayed_card_count} "
        f"(budget={report.effective_max_cards_per_10_min}/10m, compliant={report.budget_compliant})",
        f"Invariants passed: {report.invariants.passed} "
        f"({len(report.invariants.violations)} violations)",
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0 if report.invariants.passed else 2


def _cmd_explain(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        run_id = _resolve_run_id(store, args.session, args.run_id)
        explanation = explain_silence(
            store, args.session, run_id, at_ms=args.at_ms, window_ms=args.window_ms
        )
    payload = explanation.to_dict()
    lines = [
        f"Silence explanation for {explanation.session_id} (run={explanation.run_id}) "
        f"at {explanation.at_ms} ms (window={explanation.window_ms} ms): "
        f"{len(explanation.anchors)} anchors"
    ]
    for a in explanation.anchors:
        lines.append(
            f"  - {a.candidate_id} @ {a.anchor_received_ms}ms: "
            f"outcome={a.outcome} state={a.terminal_state} reason={a.terminal_reason}"
        )
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        diff = diff_runs(store, args.run_a, args.run_b, session_id=args.session)
    payload = diff.to_dict()
    lines = [
        f"Diff for {diff.session_id}: {diff.run_a} -> {diff.run_b}",
        f"Gained: {len(diff.gained)} | Lost: {len(diff.lost)} | Changed: {len(diff.changed)}",
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def _cmd_delete_session(args: argparse.Namespace) -> int:
    with SQLiteStore(args.db) as store:
        summary = store.delete_session_text(args.session)
    payload = asdict(summary)
    lines = [
        f"Deleted session text for {summary.session_id}: "
        f"raw_events={summary.raw_events_deleted}, events_redacted={summary.events_redacted}, "
        f"decisions_redacted={summary.decisions_redacted}, suggestions_redacted={summary.suggestions_redacted}"
    ]
    _emit_output(payload, as_json=args.json, summary_lines=lines)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coach", description="Live Conversation Coach CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="Generate per-session report (§13.5)")
    p_report.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_report.add_argument("--session", required=True, help="Session ID")
    p_report.add_argument("--run-id", default=None, help="Run ID (defaults to latest run)")
    p_report.add_argument("--json", action="store_true", help="Emit JSON output")
    p_report.set_defaults(handler=_cmd_report)

    p_explain = sub.add_parser("explain", help="Explain silence in a time window (§13.3)")
    p_explain.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_explain.add_argument("--session", required=True, help="Session ID")
    p_explain.add_argument("--at-ms", required=True, type=int, help="Window end timestamp (ms)")
    p_explain.add_argument("--window-ms", type=int, default=60_000, help="Window length (ms)")
    p_explain.add_argument("--run-id", default=None, help="Run ID (defaults to latest run)")
    p_explain.add_argument("--json", action="store_true", help="Emit JSON output")
    p_explain.set_defaults(handler=_cmd_explain)

    p_diff = sub.add_parser("diff", help="Diff two runs of a session (§13.4)")
    p_diff.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_diff.add_argument("--run-a", required=True, help="Baseline run ID")
    p_diff.add_argument("--run-b", required=True, help="Candidate run ID")
    p_diff.add_argument("--session", default=None, help="Session ID (auto-detected if unique)")
    p_diff.add_argument("--json", action="store_true", help="Emit JSON output")
    p_diff.set_defaults(handler=_cmd_diff)

    p_del = sub.add_parser("delete-session", help="Delete session text per §14.2")
    p_del.add_argument("--db", required=True, help="Path to SQLite WAL database")
    p_del.add_argument("--session", required=True, help="Session ID")
    p_del.add_argument("--json", action="store_true", help="Emit JSON output")
    p_del.set_defaults(handler=_cmd_delete_session)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
