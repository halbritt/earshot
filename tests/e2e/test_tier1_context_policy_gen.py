"""Tier 1 opaque-box E2E tests for Area C: Context, Policy, Templates, Generator & Provider."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from tests.e2e.helpers.cli_runner import build_coach_test_env, run_coach
from tests.e2e.helpers.db_assertions import (
    assert_egress_records_valid,
    assert_local_only_no_egress,
    assert_six_conservation_invariants,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
    REASON_CODES,
    build_boundary_event,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    deep_merge_dict,
    write_brief_json,
    write_events_jsonl,
    write_policy_dict,
)
from tests.e2e.helpers.mock_openai import MockCompletionSpec, MockOpenAIServer

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _extract_reason_codes(row: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for key in ("terminal_reason", "reason_code", "reason"):
        val = row.get(key)
        if isinstance(val, str) and val:
            codes.add(val)
    for list_key in ("reason_codes", "abstain_reasons"):
        vals = row.get(list_key)
        if isinstance(vals, list):
            codes.update(str(v) for v in vals if v)
    return codes


def _replay_session(
    ws: dict[str, Path],
    events: list[dict[str, Any]],
    *,
    brief: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    db_name: str = "coach.db",
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    brief_path = write_brief_json(
        ws["root"] / f"{db_name}.brief.json", brief or build_session_brief()
    )
    events_path = write_events_jsonl(ws["sessions_dir"] / f"{db_name}.jsonl", events)
    policy_path = (
        write_policy_dict(ws["policies_dir"] / f"{db_name}.yaml", policy)
        if policy is not None
        else ws["policy_path"]
    )
    db_path = ws["root"] / db_name
    args = [
        "--session",
        str(events_path),
        "--brief",
        str(brief_path),
        "--policy",
        str(policy_path),
        "--db",
        str(db_path),
        "--json",
        *(extra_args or []),
    ]
    res = run_coach("replay", args, env=env)
    assert res.returncode == 0, (
        f"coach replay failed ({res.returncode}): {res.stderr}\n{res.stdout}"
    )
    payload = res.json()
    assert isinstance(payload, dict)
    return db_path, payload


def test_anchor_coalescing_audited_prefilter_and_deterministic_candidate_id(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-c-coalesce"
    brief = build_session_brief(session_id=sid)
    long_part_a = " ".join(["replication"] * 38)
    long_part_b = " ".join(["verification"] * 32)
    events: list[dict[str, Any]] = [
        # Recipient speech must never become an anchor
        build_segment_event(
            session_id=sid,
            event_id="ev-01",
            segment_id="seg-me-1",
            text="We should verify the cutover schedule today.",
            speaker_id="me",
            start_ms=500,
            end_ms=1500,
            received_ms=1600,
            source_sequence=1,
        ),
        # Coalescing trigger 1: punctuation closure across two adjacent segments
        build_segment_event(
            session_id=sid,
            event_id="ev-02",
            segment_id="seg-punct-1",
            text="First we drain traffic",
            speaker_id="participant-b",
            start_ms=2000,
            end_ms=3000,
            received_ms=3100,
            source_sequence=2,
            boundary=None,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-03",
            segment_id="seg-punct-2",
            text="from the primary pool.",
            speaker_id="participant-b",
            start_ms=3050,
            end_ms=4200,
            received_ms=4300,
            source_sequence=3,
        ),
        # Coalescing trigger 2: explicit boundary event without punctuation
        build_segment_event(
            session_id=sid,
            event_id="ev-04",
            segment_id="seg-bnd-1",
            text="Check the replica lag",
            speaker_id="participant-b",
            start_ms=5000,
            end_ms=5900,
            received_ms=6000,
            source_sequence=4,
            boundary=None,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-05",
            segment_id="seg-bnd-2",
            text="before flipping DNS",
            speaker_id="participant-b",
            start_ms=6050,
            end_ms=7000,
            received_ms=7100,
            source_sequence=5,
            boundary=None,
        ),
        build_boundary_event(
            session_id=sid,
            event_id="ev-06",
            channel=1,
            speaker_id="participant-b",
            end_ms=7000,
            received_ms=7200,
            source_sequence=6,
            boundary="utterance_end",
        ),
        # Coalescing trigger 3: speaker change (participant-b -> participant-c)
        build_segment_event(
            session_id=sid,
            event_id="ev-07",
            segment_id="seg-spk-1",
            text="Let us verify the staging cluster",
            speaker_id="participant-b",
            start_ms=8000,
            end_ms=9100,
            received_ms=9200,
            source_sequence=7,
            boundary=None,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-08",
            segment_id="seg-spk-2",
            text="Someone should own the rollback plan.",
            speaker_id="participant-c",
            start_ms=9300,
            end_ms=10800,
            received_ms=10900,
            source_sequence=8,
        ),
        # Prefilter trigger: repeat of preceding anchor
        build_segment_event(
            session_id=sid,
            event_id="ev-09",
            segment_id="seg-pre-repeat",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=11000,
            end_ms=12200,
            received_ms=12300,
            source_sequence=9,
        ),
        # Prefilter trigger: backchannel, too_short, stock_phrase
        build_segment_event(
            session_id=sid,
            event_id="ev-10",
            segment_id="seg-pre-backchannel",
            text="Yeah.",
            speaker_id="participant-b",
            start_ms=12500,
            end_ms=12800,
            received_ms=12900,
            source_sequence=10,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-11",
            segment_id="seg-pre-short",
            text="Okay sure.",
            speaker_id="participant-c",
            start_ms=13000,
            end_ms=13400,
            received_ms=13500,
            source_sequence=11,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-12",
            segment_id="seg-pre-stock",
            text="Thank you for watching.",
            speaker_id="participant-b",
            start_ms=13600,
            end_ms=14400,
            received_ms=14500,
            source_sequence=12,
        ),
        # Coalescing trigger 4: ~60-token cap forces split across 70 punctuation-free tokens
        build_segment_event(
            session_id=sid,
            event_id="ev-13",
            segment_id="seg-cap-1",
            text=long_part_a,
            speaker_id="participant-b",
            start_ms=15000,
            end_ms=18000,
            received_ms=18100,
            source_sequence=13,
            boundary=None,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-14",
            segment_id="seg-cap-2",
            text=long_part_b,
            speaker_id="participant-b",
            start_ms=18050,
            end_ms=21000,
            received_ms=21100,
            source_sequence=14,
            boundary=None,
        ),
    ]
    db1, _ = _replay_session(e2e_workspace, events, brief=brief, db_name="coalesce_1.db")
    db2, _ = _replay_session(e2e_workspace, events, brief=brief, db_name="coalesce_2.db")
    assert_six_conservation_invariants(db1, sid)
    assert_six_conservation_invariants(db2, sid)

    with open_db(db1) as conn1, open_db(db2) as conn2:
        decs1 = fetch_normalized_rows(conn1, "decisions", sid)
        decs2 = fetch_normalized_rows(conn2, "decisions", sid)

    cids1 = sorted(str(d["candidate_id"]) for d in decs1)
    cids2 = sorted(str(d["candidate_id"]) for d in decs2)
    assert cids1 == cids2, "Replaying identical stream must emit identical candidate_ids"
    assert all(_SHA256_RE.match(cid) for cid in cids1), f"Non-SHA256 candidate_id: {cids1}"

    seg_groups: list[set[str]] = []
    dropped_reasons: set[str] = set()
    for d in decs1:
        refs = {str(r["segment_id"]) for r in (d.get("anchor_refs") or []) if isinstance(r, dict)}
        assert "seg-me-1" not in refs, f"Recipient segment formed anchor: {d}"
        seg_groups.append(refs)
        if str(d.get("outcome") or "") in {"dropped", "prefilter_dropped"} or str(
            d.get("terminal_state") or ""
        ) in {"dropped", "prefilter_dropped"}:
            assert d.get("raw_scores") is None, f"Prefilter drop must skip classifier: {d}"
            dropped_reasons.update(_extract_reason_codes(d))

    assert {"seg-punct-1", "seg-punct-2"} in seg_groups
    assert {"seg-bnd-1", "seg-bnd-2"} in seg_groups
    assert {"seg-spk-1"} in seg_groups and {"seg-spk-2"} in seg_groups
    assert {"seg-cap-1", "seg-cap-2"} not in seg_groups, "60-token cap must split long anchor"
    assert {"backchannel", "too_short", "repeat", "stock_phrase"}.issubset(dropped_reasons)

    explain = run_coach(
        "explain",
        [
            "--db",
            str(db1),
            "--session",
            sid,
            "--at-ms",
            "25000",
            "--window-ms",
            "30000",
            "--json",
        ],
    )
    assert explain.returncode == 0
    explain_data = explain.json()
    assert len(explain_data["anchors"]) == len(decs1)


def test_context_serialization_layout_role_anon_ages_and_oldest_first_truncation(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-c-serialize"
    brief = build_session_brief(
        session_id=sid,
        participants=[
            {"participant_id": "me", "label": "ME", "role": "reviewer"},
            {"participant_id": "participant-b", "label": "S1", "role": "engineer"},
            {"participant_id": "participant-c", "label": "S2", "role": "operator"},
        ],
    )
    # Build turns across age buckets plus enough history to force oldest-first truncation at 160t
    events: list[dict[str, Any]] = [
        build_segment_event(
            session_id=sid,
            event_id="ev-s-1",
            segment_id="seg-s-1",
            text="Older turn alpha about cluster topology and database primary failover steps.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3000,
            received_ms=3100,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-s-2",
            segment_id="seg-s-2",
            text="Unmapped speaker turn beta discussing canary metrics and error budgets.",
            speaker_id="participant-d",
            start_ms=500000,
            end_ms=502000,
            received_ms=502100,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-s-3",
            segment_id="seg-s-3",
            text="Recipient follow-up checking how quickly the read replicas catch up.",
            speaker_id="me",
            start_ms=700000,
            end_ms=702000,
            received_ms=702100,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-s-4",
            segment_id="seg-s-4",
            text="Recent context gamma explaining connection draining during blue green cutover.",
            speaker_id="participant-c",
            start_ms=790000,
            end_ms=792000,
            received_ms=792100,
            source_sequence=4,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-s-5",
            segment_id="seg-s-5",
            text="Newest context delta confirming DNS TTL was lowered yesterday.",
            speaker_id="participant-b",
            start_ms=805000,
            end_ms=807000,
            received_ms=807100,
            source_sequence=5,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-s-anchor",
            segment_id="seg-s-anchor",
            text="Someone should own the rollback plan.",
            speaker_id="participant-c",
            start_ms=810000,
            end_ms=812000,
            received_ms=812100,
            source_sequence=6,
        ),
    ]
    # 1. Full serialization check at 512 tokens
    pol_full = build_policy_dict(input_limit_tokens=512, settle_ms=200, cooldown_ms=1000)
    db_full, _ = _replay_session(
        e2e_workspace, events, brief=brief, policy=pol_full, db_name="ser_full.db"
    )
    assert_six_conservation_invariants(db_full, sid)
    with open_db(db_full) as conn:
        full_decs = fetch_normalized_rows(conn, "decisions", sid)
    anchor_dec = next(
        d
        for d in full_decs
        if any(r.get("segment_id") == "seg-s-anchor" for r in (d.get("anchor_refs") or []))
    )
    ser_text = str(anchor_dec.get("serialized_text") or "")
    markers = ("[OBJECTIVE]", "[ROLE]", "[KNOWN]", "[SUGGESTED]", "[CONTEXT]", "[ANCHOR]")
    positions = [ser_text.index(m) for m in markers]
    assert positions == sorted(positions), f"Out-of-order §5.3 blocks:\n{ser_text}"
    assert "ME=" in ser_text and "S1=" in ser_text and "S2=" in ser_text
    assert "S?:" in ser_text or "S?=" in ser_text
    assert "participant-b" not in ser_text and "participant-d" not in ser_text
    assert any(b in ser_text for b in ("<30s", "<2m", "<10m", "older"))

    # 2. Oldest-first whole-turn truncation + context_truncated_material
    pol_trunc = build_policy_dict(input_limit_tokens=96, settle_ms=200, cooldown_ms=1000)
    db_trunc, _ = _replay_session(
        e2e_workspace, events, brief=brief, policy=pol_trunc, db_name="ser_trunc.db"
    )
    assert_six_conservation_invariants(db_trunc, sid)
    with open_db(db_trunc) as conn:
        trunc_decs = fetch_normalized_rows(conn, "decisions", sid)
    trunc_anchor = next(
        d
        for d in trunc_decs
        if any(r.get("segment_id") == "seg-s-anchor" for r in (d.get("anchor_refs") or []))
    )
    trunc_text = str(trunc_anchor.get("serialized_text") or "")
    assert "Older turn alpha" not in trunc_text
    assert "Someone should own the rollback plan." in trunc_text
    assert "[OBJECTIVE]" in trunc_text and "[ROLE]" in trunc_text and "[ANCHOR]" in trunc_text
    omitted = trunc_anchor.get("omitted") or {}
    if isinstance(omitted, str):
        omitted = json.loads(omitted)
    assert int(omitted.get("turns", 0)) > 0
    assert trunc_anchor.get("outcome") == "ABSTAIN"
    assert "context_truncated_material" in _extract_reason_codes(trunc_anchor)

    # 3. Mandatory blocks exceeding tiny limit -> ABSTAIN with input_overflow
    pol_over = build_policy_dict(input_limit_tokens=12, settle_ms=200)
    db_over, _ = _replay_session(
        e2e_workspace,
        [events[-1]],
        brief=brief,
        policy=pol_over,
        db_name="ser_over.db",
    )
    assert_six_conservation_invariants(db_over, sid)
    with open_db(db_over) as conn:
        over_decs = fetch_normalized_rows(conn, "decisions", sid)
    assert over_decs[0].get("outcome") == "ABSTAIN"
    assert "input_overflow" in _extract_reason_codes(over_decs[0])


def test_span_rules_v0_grounding_and_generic_referent_gating(
    e2e_workspace: dict[str, Path],
) -> None:
    pol = build_policy_dict(settle_ms=200, cooldown_ms=1000)

    # (a) Strict contiguous-substring grounding (char_start..char_end)
    sid_a = "sess-c-span-grounded"
    anchor_text = "Someone should own the rollback plan."
    ev_a = [
        build_segment_event(
            session_id=sid_a,
            event_id="ev-span-1",
            segment_id="seg-span-1",
            text=anchor_text,
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=4200,
            received_ms=4450,
            source_sequence=1,
        )
    ]
    db_a, _ = _replay_session(
        e2e_workspace,
        ev_a,
        brief=build_session_brief(session_id=sid_a),
        policy=pol,
        db_name="span_a.db",
    )
    assert_six_conservation_invariants(db_a, sid_a)
    with open_db(db_a) as conn:
        sugs_a = fetch_normalized_rows(conn, "suggestions", sid_a)
    assert len(sugs_a) == 1
    assert sugs_a[0].get("template_id") == "confirm_owner_v1"
    slots_a = sugs_a[0].get("grounded_slots") or {}
    assert "action" in slots_a
    action_slot = slots_a["action"]
    assert action_slot["segment_id"] == "seg-span-1"
    c_start = int(action_slot["char_start"])
    c_end = int(action_slot["char_end"])
    assert anchor_text[c_start:c_end].strip().rstrip(".,?!").lower() == (
        str(action_slot["text"]).strip().rstrip(".,?!").lower()
    )
    assert str(action_slot["text"]) in str(sugs_a[0]["text"])

    # (b) Unextractable (>8 word) slot falls back to generic when latest other speech & <=6000 ms
    sid_b = "sess-c-span-generic-ok"
    overlong_anchor = (
        "Someone should own the extremely complex multi region active active "
        "database failover orchestration verification procedure."
    )
    ev_b = [
        build_segment_event(
            session_id=sid_b,
            event_id="ev-gen-1",
            segment_id="seg-gen-ok",
            text=overlong_anchor,
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=4000,
            received_ms=4200,
            source_sequence=1,
        )
    ]
    db_b, _ = _replay_session(
        e2e_workspace,
        ev_b,
        brief=build_session_brief(session_id=sid_b),
        policy=pol,
        db_name="span_b.db",
    )
    assert_six_conservation_invariants(db_b, sid_b)
    with open_db(db_b) as conn:
        sugs_b = fetch_normalized_rows(conn, "suggestions", sid_b)
    assert len(sugs_b) == 1
    assert sugs_b[0].get("text") == "Who owns that next step?"

    # (c) Newer non-ME speech arriving before render/display blocks generic referent
    sid_c = "sess-c-span-generic-blocked"
    pol_slow_settle = build_policy_dict(settle_ms=7000, cooldown_ms=1000)
    ev_c = [
        build_segment_event(
            session_id=sid_c,
            event_id="ev-blk-1",
            segment_id="seg-blk-1",
            text=overlong_anchor,
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=4000,
            received_ms=4200,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid_c,
            event_id="ev-blk-2",
            segment_id="seg-blk-2",
            text="Let us also check the DNS zone TTL values.",
            speaker_id="participant-c",
            start_ms=4300,
            end_ms=5200,
            received_ms=5300,
            source_sequence=2,
        ),
    ]
    db_c, _ = _replay_session(
        e2e_workspace,
        ev_c,
        brief=build_session_brief(session_id=sid_c),
        policy=pol_slow_settle,
        db_name="span_c.db",
    )
    assert_six_conservation_invariants(db_c, sid_c)
    with open_db(db_c) as conn:
        decs_c = fetch_normalized_rows(conn, "decisions", sid_c)
        sugs_c = fetch_normalized_rows(conn, "suggestions", sid_c)
    first_reasons = _extract_reason_codes(decs_c[0])
    for s in sugs_c:
        if s.get("candidate_id") == decs_c[0].get("candidate_id"):
            first_reasons.update(_extract_reason_codes(s))
    assert "generic_referent_unclear" in first_reasons


def test_policy_immutability_hash_startup_calibration_check_stricter_cap_and_5_templates(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-c-policy-tpls"
    extra_cats: dict[str, Any] = {
        "CLARIFY_SCOPE": {
            "t_low": 0.20,
            "t_compute": 0.50,
            "t_display": 0.75,
            "route": "template",
            "template": "clarify_scope_v1",
            "required_lookback_s": 120,
        },
        "REQUEST_EVIDENCE": {
            "t_low": 0.20,
            "t_compute": 0.50,
            "t_display": 0.75,
            "route": "template",
            "template": "request_evidence_v1",
            "required_lookback_s": 120,
        },
    }
    pol_a = build_policy_dict(
        settle_ms=100,
        cooldown_ms=100,
        max_cards_per_10_min=5,
        extra_categories=extra_cats,
    )
    pol_b = deep_merge_dict(pol_a, {"display": {"settle_ms": 150}})
    brief = build_session_brief(
        session_id=sid,
        attention={"max_cards_per_10_min": 5, "proactive": True},
    )
    events_5 = [
        build_segment_event(
            session_id=sid,
            event_id="ev-t-1",
            segment_id="seg-t-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=2500,
            received_ms=2600,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t-2",
            segment_id="seg-t-2",
            text="The schema migration needs to finish soon.",
            speaker_id="participant-c",
            start_ms=5000,
            end_ms=6500,
            received_ms=6600,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t-3",
            segment_id="seg-t-3",
            text="We should verify the canary rollout works.",
            speaker_id="participant-b",
            start_ms=9000,
            end_ms=10500,
            received_ms=10600,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t-4",
            segment_id="seg-t-4",
            text="This phase covers the billing service migration including legacy webhooks.",
            speaker_id="participant-c",
            start_ms=13000,
            end_ms=14500,
            received_ms=14600,
            source_sequence=4,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-t-5",
            segment_id="seg-t-5",
            text="The cache layer never drops writes under peak traffic.",
            speaker_id="participant-b",
            start_ms=17000,
            end_ms=18500,
            received_ms=18600,
            source_sequence=5,
        ),
    ]
    db_a, _ = _replay_session(
        e2e_workspace, events_5, brief=brief, policy=pol_a, db_name="pol_a.db"
    )
    db_b, _ = _replay_session(
        e2e_workspace, events_5, brief=brief, policy=pol_b, db_name="pol_b.db"
    )
    assert_six_conservation_invariants(db_a, sid)
    with open_db(db_a) as ca, open_db(db_b) as cb:
        decs_a = fetch_normalized_rows(ca, "decisions", sid)
        decs_b = fetch_normalized_rows(cb, "decisions", sid)
        sugs_a = fetch_normalized_rows(ca, "suggestions", sid)
    hashes_a = {str(d.get("policy_sha256") or "") for d in decs_a}
    hashes_b = {str(d.get("policy_sha256") or "") for d in decs_b}
    assert len(hashes_a) == 1 and len(hashes_b) == 1
    hash_a = next(iter(hashes_a))
    hash_b = next(iter(hashes_b))
    assert _SHA256_RE.match(hash_a) and _SHA256_RE.match(hash_b)
    assert hash_a != hash_b
    assert {str(d.get("policy_version") or "") for d in decs_a} == {"policy-v0-a"}

    seen_templates = {str(s.get("template_id") or "") for s in sugs_a}
    expected_tpls = {
        "confirm_owner_v1",
        "confirm_time_v1",
        "success_criterion_v1",
        "clarify_scope_v1",
        "request_evidence_v1",
    }
    assert expected_tpls.issubset(seen_templates), f"Missing templates: {seen_templates}"
    for s in sugs_a:
        q = str(s.get("text") or "").strip()
        assert q.endswith("?") and len(q.split()) <= 12, f"Invalid template question: {q!r}"

    # Startup mismatch rejection: bad calibration name and bad mode
    ev_path = e2e_workspace["sessions_dir"] / "pol_a.db.jsonl"
    brief_path = e2e_workspace["root"] / "pol_a.db.brief.json"
    bad_cal_path = write_policy_dict(
        e2e_workspace["policies_dir"] / "bad_cal.yaml",
        build_policy_dict(calibration="mismatched-cal-v99"),
    )
    res_bad_cal = run_coach(
        "replay",
        [
            "--session",
            str(ev_path),
            "--brief",
            str(brief_path),
            "--policy",
            str(bad_cal_path),
            "--db",
            str(e2e_workspace["root"] / "bad_cal.db"),
        ],
    )
    assert res_bad_cal.returncode != 0

    bad_mode_pol = deep_merge_dict(build_policy_dict(), {"mode": "personal_logistics"})
    bad_mode_path = write_policy_dict(e2e_workspace["policies_dir"] / "bad_mode.yaml", bad_mode_pol)
    res_bad_mode = run_coach(
        "replay",
        [
            "--session",
            str(ev_path),
            "--brief",
            str(brief_path),
            "--policy",
            str(bad_mode_path),
            "--db",
            str(e2e_workspace["root"] / "bad_mode.db"),
        ],
    )
    assert res_bad_mode.returncode != 0

    # Stricter-wins attention cap: brief=1 vs policy=3 -> effective=1
    brief_cap1 = build_session_brief(
        session_id="sess-c-cap1",
        attention={"max_cards_per_10_min": 1, "proactive": True},
    )
    pol_cap3 = build_policy_dict(
        settle_ms=100,
        cooldown_ms=100,
        max_cards_per_10_min=3,
        extra_categories=extra_cats,
    )
    ev_cap = [dict(e, session_id="sess-c-cap1") for e in events_5]
    db_cap, _ = _replay_session(
        e2e_workspace, ev_cap, brief=brief_cap1, policy=pol_cap3, db_name="cap1.db"
    )
    rep = run_coach("report", ["--db", str(db_cap), "--session", "sess-c-cap1", "--json"])
    assert rep.returncode == 0
    rep_json = rep.json()
    assert rep_json["effective_max_cards_per_10_min"] == 1
    assert rep_json["displayed_card_count"] <= 1
    assert rep_json["budget_compliant"] is True


def test_question_generator_4block_cache_prompt_decline_and_egress_usage(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    sid = "sess-c-gen-cache"
    mock_openai.enqueue(
        MockCompletionSpec.valid_question(
            "Has failover been tested under realistic load?",
            ["seg-gen-1"],
            cached_tokens=96,
        ),
        MockCompletionSpec.decline("already answered", cached_tokens=96),
    )
    pol = deep_merge_dict(
        build_policy_dict(generator_enabled=True, settle_ms=100, cooldown_ms=100),
        {
            "categories": {
                "CONFIRM_OWNER": {"route": "generator", "t_display": 0.50},
                "PROBE_TRADEOFF": {
                    "t_low": 0.20,
                    "t_compute": 0.40,
                    "t_display": 0.50,
                    "route": "generator",
                    "required_lookback_s": 120,
                },
            }
        },
    )
    brief = build_session_brief(session_id=sid)
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-g-1",
            segment_id="seg-gen-1",
            text="Someone should own the failover plan before cutover.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3000,
            received_ms=3100,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-g-2",
            segment_id="seg-gen-2",
            text="On the other hand, active-active replication adds cross-region latency.",
            speaker_id="participant-c",
            start_ms=10000,
            end_ms=12500,
            received_ms=12600,
            source_sequence=2,
        ),
    ]
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    db_path, _ = _replay_session(
        e2e_workspace, events, brief=brief, policy=pol, db_name="gen_cache.db", env=env
    )
    assert_six_conservation_invariants(db_path, sid)
    assert len(mock_openai.requests) == 2, f"Decline must not retry: {len(mock_openai.requests)}"

    req0 = mock_openai.requests[0]
    joined = req0.joined_prompt()
    low = joined.lower()
    # §9.2 4-block cache order: Policy -> Brief -> Quoted Transcript -> Opportunity
    p_idx = low.index("policy")
    b_idx = low.index("objective")
    t_idx = low.index("seg-gen-1")
    o_idx = low.index("opportunity")
    assert p_idx < b_idx < t_idx < o_idx
    assert "not an instruction" in low or "untrusted" in low or "quoted" in low

    egress_rows = assert_egress_records_valid(
        db_path, sid, expected_segment_ids={"seg-gen-1", "seg-gen-2"}
    )
    assert any(int((r.get("usage") or {}).get("cached_tokens", 0)) == 96 for r in egress_rows), (
        f"Expected cached_tokens=96 in egress_log: {egress_rows}"
    )

    with open_db(db_path) as conn:
        sugs = fetch_normalized_rows(conn, "suggestions", sid)
        decs = fetch_normalized_rows(conn, "decisions", sid)
    assert any(
        s.get("route") == "generator"
        and s.get("text") == "Has failover been tested under realistic load?"
        for s in sugs
    )
    decline_reasons = set()
    for row in [*decs, *sugs]:
        decline_reasons.update(_extract_reason_codes(row))
    assert "generator_declined" in decline_reasons


def test_candidate_validator_rejects_all_6_deterministic_failure_modes(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    sid = "sess-c-validator-6"
    # Anchor 0 produces a valid displayed card; anchors 1..6 trigger each of the 6 §9.2 rejections
    two_sent_over20 = (
        "First sentence adds extra context. Second sentence asks about "
        "every detail across more than twenty words without ending cleanly"
    )
    mock_openai.enqueue(
        MockCompletionSpec.valid_question("Who owns the cutover runbook?", ["seg-v-0"]),
        MockCompletionSpec.malformed_truncated('{"question": "Is this'),
        MockCompletionSpec(
            content=json.dumps(
                {
                    "question": two_sent_over20,
                    "evidence_segment_ids": ["seg-v-2"],
                    "decline_reason": None,
                }
            )
        ),
        MockCompletionSpec.valid_question("Who owns the rollback plan?", ["segment-9999"]),
        MockCompletionSpec.ungrounded("Did Alice approve $42,000 on May 19?", ["seg-v-4"]),
        MockCompletionSpec.valid_question("Should we increase the budget now?", ["seg-v-5"]),
        MockCompletionSpec.valid_question("Who owns the cutover runbook?", ["seg-v-6"]),
    )
    pol = deep_merge_dict(
        build_policy_dict(
            generator_enabled=True,
            settle_ms=50,
            cooldown_ms=50,
            max_cards_per_10_min=10,
        ),
        {
            "categories": {"CONFIRM_OWNER": {"route": "generator", "t_display": 0.50}},
            "display": {"display_ttl_ms": 30000},
        },
    )
    brief = build_session_brief(session_id=sid, do_not_suggest=["budget"])
    prompts = [
        "Someone should own the cutover runbook.",
        "Someone should own the staging verification.",
        "Someone should own the canary promotion.",
        "Someone should own the DNS failover.",
        "Someone should own the database cutover.",
        "Someone should own the infrastructure allocation.",
        "Someone should own the backup restore test.",
    ]
    events = [
        build_segment_event(
            session_id=sid,
            event_id=f"ev-v-{idx}",
            segment_id=f"seg-v-{idx}",
            text=text,
            speaker_id="participant-b" if idx % 2 == 0 else "participant-c",
            start_ms=1000 + idx * 4000,
            end_ms=2500 + idx * 4000,
            received_ms=2600 + idx * 4000,
            source_sequence=idx + 1,
        )
        for idx, text in enumerate(prompts)
    ]
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    db_path, _ = _replay_session(
        e2e_workspace, events, brief=brief, policy=pol, db_name="val_6.db", env=env
    )
    assert_six_conservation_invariants(db_path, sid)
    egress = assert_egress_records_valid(db_path, sid)
    assert len(egress) >= 7

    with open_db(db_path) as conn:
        sugs = fetch_normalized_rows(conn, "suggestions", sid)
        decs = fetch_normalized_rows(conn, "decisions", sid)

    by_seg: dict[str, list[dict[str, Any]]] = {}
    for row in [*decs, *sugs]:
        for ref in row.get("anchor_refs") or row.get("evidence_refs") or []:
            if isinstance(ref, dict) and ref.get("segment_id"):
                by_seg.setdefault(str(ref["segment_id"]), []).append(row)

    for idx in range(1, 7):
        seg_id = f"seg-v-{idx}"
        rows = by_seg.get(seg_id, [])
        assert rows, f"Missing decision/suggestion records for {seg_id}"
        for s in [r for r in rows if r.get("suggestion_id")]:
            assert s.get("current_state") != "displayed", f"Invalid output displayed for {seg_id}"
        codes = set().union(*(_extract_reason_codes(r) for r in rows))
        assert codes & REASON_CODES, f"No valid ReasonCode recorded for {seg_id}: {rows}"
        assert codes & {"generator_invalid_output", "do_not_suggest", "duplicate", "near_duplicate"}


def test_provider_adapter_startup_capability_validation_and_local_only_mode(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    sid = "sess-c-provider"
    brief = build_session_brief(session_id=sid)
    brief_path = write_brief_json(e2e_workspace["root"] / "prov_brief.json", brief)
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-p-1",
            segment_id="seg-p-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3000,
            received_ms=3100,
            source_sequence=1,
        )
    ]
    events_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "prov.jsonl", events)

    # (a) Startup capability validation rejects unsupported (model_id, reasoning_setting)
    bad_cap_pol = deep_merge_dict(
        build_policy_dict(generator_enabled=True),
        {
            "generator": {
                "provider": "hosted",
                "model_id": "gpt-6.2-sol",
                "reasoning": "none",
            }
        },
    )
    bad_cap_path = write_policy_dict(e2e_workspace["policies_dir"] / "bad_cap.yaml", bad_cap_pol)
    bad_db = e2e_workspace["root"] / "bad_cap.db"
    env_with_openai = build_coach_test_env(openai_base_url=mock_openai.base_url)
    res_bad_cap = run_coach(
        "replay",
        [
            "--session",
            str(events_path),
            "--brief",
            str(brief_path),
            "--policy",
            str(bad_cap_path),
            "--db",
            str(bad_db),
        ],
        env=env_with_openai,
    )
    assert res_bad_cap.returncode != 0
    assert len(mock_openai.requests) == 0, "Startup rejection must occur before any model call"

    # (b) §14.6 Secret hygiene: zero API keys inside policy/template YAML files
    for yaml_file in e2e_workspace["policies_dir"].rglob("*.yaml"):
        loaded = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        dumped = json.dumps(loaded).lower()
        assert "sk-" not in dumped and "api_key" not in dumped

    # (c) --local-only mode: zero hosted egress and health=template_only
    local_pol = deep_merge_dict(
        build_policy_dict(generator_enabled=True, settle_ms=100, cooldown_ms=100),
        {
            "generator": {"provider": "hosted", "model_id": "gpt-6-luna", "reasoning": "low"},
            "categories": {"CONFIRM_OWNER": {"route": "generator", "t_display": 0.50}},
        },
    )
    mock_openai.clear_requests()
    mock_openai.enqueue(MockCompletionSpec.valid_question("Who owns rollback?", ["seg-p-1"]))
    db_local, _ = _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=local_pol,
        db_name="local_only.db",
        extra_args=["--local-only"],
        env=env_with_openai,
    )
    assert len(mock_openai.requests) == 0, "--local-only must make zero hosted HTTP requests"
    assert_local_only_no_egress(db_local, sid)
    assert_six_conservation_invariants(db_local, sid)

    rep = run_coach("report", ["--db", str(db_local), "--session", sid, "--json"])
    assert rep.returncode == 0
    health_states = [
        str(h.get("health") or h.get("state") or "") for h in rep.json().get("health_timeline", [])
    ]
    assert "template_only" in health_states


def test_provider_adapter_circuit_breaker_3_failures_and_half_open_recovery(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    sid = "sess-c-breaker"
    mock_openai.enqueue(
        MockCompletionSpec.http_error(500, "err-1"),
        MockCompletionSpec.http_error(500, "err-2"),
        MockCompletionSpec.http_error(500, "err-3"),
        MockCompletionSpec.valid_question("What rollback risk remains untested?", ["seg-cb-5"]),
    )
    pol = deep_merge_dict(
        build_policy_dict(
            generator_enabled=True,
            settle_ms=50,
            cooldown_ms=50,
            max_cards_per_10_min=10,
        ),
        {
            "generator": {"breaker": {"consecutive_failures": 3, "open_ms": 800}},
            "categories": {"CONFIRM_OWNER": {"route": "generator", "t_display": 0.50}},
        },
    )
    brief = build_session_brief(session_id=sid)
    # Anchors 1-3 fail (500) -> open breaker; anchor 4 inside 800ms skipped; anchor 5 recovers
    timings_ms = (100, 300, 500, 700, 1700)
    events = [
        build_segment_event(
            session_id=sid,
            event_id=f"ev-cb-{idx}",
            segment_id=f"seg-cb-{idx}",
            text=f"Someone should own cutover task {idx}.",
            speaker_id="participant-b" if idx % 2 == 1 else "participant-c",
            start_ms=t_ms,
            end_ms=t_ms + 80,
            received_ms=t_ms + 100,
            source_sequence=idx,
        )
        for idx, t_ms in enumerate(timings_ms, start=1)
    ]
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    db_path, _ = _replay_session(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol,
        db_name="breaker.db",
        extra_args=["--speed", "1.0"],
        env=env,
    )
    assert_six_conservation_invariants(db_path, sid)
    # Only 4 HTTP calls: 3 consecutive failures + 1 half-open probe after open_ms
    assert len(mock_openai.requests) == 4, (
        f"Expected 4 OpenAI calls, got {len(mock_openai.requests)}"
    )

    rep = run_coach("report", ["--db", str(db_path), "--session", sid, "--json"])
    assert rep.returncode == 0
    timeline = [
        str(h.get("health") or h.get("state") or "") for h in rep.json().get("health_timeline", [])
    ]
    assert "template_only" in timeline
    assert timeline[-1] == "ok"

    with open_db(db_path) as conn:
        sugs = fetch_normalized_rows(conn, "suggestions", sid)
    assert any(
        s.get("text") == "What rollback risk remains untested?"
        and any(r.get("segment_id") == "seg-cb-5" for r in (s.get("evidence_refs") or []))
        for s in sugs
    )
