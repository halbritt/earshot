"""Tier 1 opaque-box E2E tests for Areas D (Classifier/Routing), E (Eval/Bench), G (Annotator)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.e2e.helpers.cli_runner import (
    build_coach_test_env,
    extract_ui_url_and_token,
    run_coach,
    start_coach_process,
)
from tests.e2e.helpers.db_assertions import (
    assert_six_conservation_invariants,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
    CATEGORIES,
    NONE_REASONS,
    REASON_CODES,
    SAMPLE_SOURCES,
    SCHEMA_VERSION,
    build_decision_context_row,
    build_label_record,
    build_policy_dict,
    build_segment_event,
    build_session_brief,
    deep_merge_dict,
    read_jsonl,
    write_brief_json,
    write_events_jsonl,
    write_jsonl,
    write_policy_dict,
)
from tests.e2e.helpers.mock_openai import MockCompletionSpec, MockOpenAIServer
from tests.e2e.helpers.ui_client import AnnotationClient


def _reason_set(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("terminal_reason", "reason_code", "reason"):
        val = row.get(key)
        if isinstance(val, str) and val:
            out.add(val)
    for key in ("reason_codes", "abstain_reasons"):
        vals = row.get(key)
        if isinstance(vals, list):
            out.update(str(v) for v in vals if v)
    return out


def _replay(
    ws: dict[str, Path],
    events: list[dict[str, Any]],
    *,
    brief: dict[str, Any],
    policy: dict[str, Any] | None = None,
    name: str = "run",
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    brief_path = write_brief_json(ws["root"] / f"{name}.brief.json", brief)
    events_path = write_events_jsonl(ws["sessions_dir"] / f"{name}.jsonl", events)
    policy_path = (
        write_policy_dict(ws["policies_dir"] / f"{name}.yaml", policy)
        if policy is not None
        else ws["policy_path"]
    )
    db_path = ws["root"] / f"{name}.db"
    res = run_coach(
        "replay",
        [
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
        ],
        env=env,
    )
    assert res.returncode == 0, f"coach replay failed ({res.returncode}): {res.stderr}"
    payload = res.json()
    assert isinstance(payload, dict)
    return db_path, payload


def test_area_d_rules_v0_classifier_and_deadline_propagation(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-d-rules"
    brief = build_session_brief(session_id=sid)
    pol = build_policy_dict(settle_ms=100, cooldown_ms=100, max_cards_per_10_min=6)
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-d1-1",
            segment_id="seg-d1-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=2500,
            received_ms=2600,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-d1-2",
            segment_id="seg-d1-2",
            text="The schema migration needs to finish at some point.",
            speaker_id="participant-c",
            start_ms=4000,
            end_ms=5500,
            received_ms=5600,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-d1-3",
            segment_id="seg-d1-3",
            text="We should check how the canary rollout succeeded.",
            speaker_id="participant-b",
            start_ms=7000,
            end_ms=8500,
            received_ms=8600,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-d1-4",
            segment_id="seg-d1-4",
            text="The server rack uses twenty four volt power supplies.",
            speaker_id="participant-c",
            start_ms=10000,
            end_ms=11500,
            received_ms=11600,
            source_sequence=4,
        ),
    ]
    db_path, _ = _replay(e2e_workspace, events, brief=brief, policy=pol, name="d_rules")
    assert_six_conservation_invariants(db_path, sid)
    with open_db(db_path) as conn:
        decs = fetch_normalized_rows(conn, "decisions", sid)
    assert len(decs) == 4
    for dec in decs:
        assert dec.get("classifier_version") == "rules_v0"
        assert isinstance(dec.get("raw_scores"), dict)
        assert isinstance(dec.get("calibrated_scores"), dict)
        assert float(dec.get("latency_ms", -1.0)) >= 0.0
        assert dec.get("input_refs") == dec.get("anchor_refs")
        assert int(dec["display_by_ms"]) == int(dec["anchor_received_ms"]) + 8000

    # Tight deadline propagation (max_candidate_age_ms=10ms) skips/expires without overrun
    tight_pol = deep_merge_dict(pol, {"display": {"max_candidate_age_ms": 10}})
    db_tight, _ = _replay(
        e2e_workspace, [events[0]], brief=brief, policy=tight_pol, name="d_rules_tight"
    )
    assert_six_conservation_invariants(db_tight, sid)
    with open_db(db_tight) as conn:
        tight_decs = fetch_normalized_rows(conn, "decisions", sid)
    assert int(tight_decs[0]["display_by_ms"]) == int(tight_decs[0]["anchor_received_ms"]) + 10
    assert any(
        c in {"expired_unevaluated", "display_by_passed", "insufficient_budget"}
        for c in _reason_set(tight_decs[0])
    )


def test_area_d_readout_v0_and_llm_baseline_v0_via_mock_openai(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    sid = "sess-d-readout-llm"
    brief = build_session_brief(session_id=sid)
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-d2-1",
            segment_id="seg-d2-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3000,
            received_ms=3100,
            source_sequence=1,
        )
    ]
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)

    # 1. readout_v0 via /v1/completions logprobs over closed label set
    mock_openai.enqueue(
        MockCompletionSpec(
            content="CONFIRM_OWNER",
            top_logprobs=[
                {
                    "CONFIRM_OWNER": -0.10,
                    "CONFIRM_TIME": -2.40,
                    "SUCCESS_CRITERION": -2.80,
                    "NONE": -3.10,
                }
            ],
        )
    )
    pol_readout = deep_merge_dict(
        build_policy_dict(calibration="readout_v0-platt-v0-a", settle_ms=100),
        {"classifier": {"version": "readout_v0"}},
    )
    db_ro, _ = _replay(
        e2e_workspace, events, brief=brief, policy=pol_readout, name="d_readout", env=env
    )
    assert_six_conservation_invariants(db_ro, sid)
    with open_db(db_ro) as conn:
        ro_decs = fetch_normalized_rows(conn, "decisions", sid)
    assert ro_decs[0].get("classifier_version") == "readout_v0"
    ro_scores = ro_decs[0].get("raw_scores") or {}
    assert float((ro_scores.get("categories") or {}).get("CONFIRM_OWNER", 0.0)) > 0.5
    assert any(r.path.endswith("/completions") for r in mock_openai.requests)

    # 2. llm_baseline_v0 via /v1/chat/completions
    mock_openai.clear_requests()
    mock_openai.enqueue(
        MockCompletionSpec(
            content=json.dumps(
                {
                    "opportunity": 0.91,
                    "categories": {
                        "CONFIRM_OWNER": 0.89,
                        "CONFIRM_TIME": 0.08,
                        "SUCCESS_CRITERION": 0.12,
                    },
                }
            )
        )
    )
    pol_llm = deep_merge_dict(
        build_policy_dict(calibration="llm_baseline_v0-platt-v0-a", settle_ms=100),
        {"classifier": {"version": "llm_baseline_v0"}},
    )
    db_llm, _ = _replay(
        e2e_workspace, events, brief=brief, policy=pol_llm, name="d_llm_base", env=env
    )
    assert_six_conservation_invariants(db_llm, sid)
    with open_db(db_llm) as conn:
        llm_decs = fetch_normalized_rows(conn, "decisions", sid)
    assert llm_decs[0].get("classifier_version") == "llm_baseline_v0"
    llm_scores = llm_decs[0].get("raw_scores") or {}
    assert float((llm_scores.get("categories") or {}).get("CONFIRM_OWNER", 0.0)) > 0.7
    assert any(r.path.endswith("/chat/completions") for r in mock_openai.requests)
    assert "[ANCHOR]" in mock_openai.requests[-1].joined_prompt()


def test_area_d_train_embed_logreg_masked_bce_and_platt_calibration(
    e2e_workspace: dict[str, Path],
) -> None:
    train_rows = (
        [
            build_decision_context_row(
                candidate_id=f"tr-pos-{i}",
                session_id=f"tr-sess-{i % 3}",
                anchor_text="Someone should own the rollback plan before Friday.",
                worthwhile=True,
                categories=("CONFIRM_OWNER",),
                sample_source="random",
            )
            for i in range(6)
        ]
        + [
            build_decision_context_row(
                candidate_id=f"tr-time-{i}",
                session_id=f"tr-sess-{i % 3}",
                anchor_text="When does the database migration cutover finish?",
                worthwhile=True,
                categories=("CONFIRM_TIME",),
                sample_source="random",
            )
            for i in range(4)
        ]
        + [
            build_decision_context_row(
                candidate_id=f"tr-neg-{i}",
                session_id=f"tr-sess-{i % 3}",
                anchor_text="The rack draws twenty four volts on primary power.",
                worthwhile=False,
                categories=(),
                sample_source="random",
            )
            for i in range(6)
        ]
        + [
            # Missing/contested category (worthwhile=True, categories=[]): masked BCE (§5.5)
            build_decision_context_row(
                candidate_id=f"tr-mask-{i}",
                session_id=f"tr-sess-{i % 2}",
                anchor_text="What happens if both regions fail simultaneously?",
                worthwhile=True,
                categories=(),
                sample_source="pass1_mark",
            )
            for i in range(3)
        ]
    )
    train_path = write_jsonl(e2e_workspace["root"] / "train.jsonl", train_rows)
    model_dir = e2e_workspace["root"] / "embed_logreg_model"

    tr_res = run_coach(
        "train",
        ["--data", str(train_path), "--out", str(model_dir), "--backend", "embed_logreg"],
    )
    assert tr_res.returncode == 0, f"coach train failed: {tr_res.stderr}\n{tr_res.stdout}"
    assert model_dir.exists() and any(model_dir.iterdir())

    cal_rows = [
        build_decision_context_row(
            candidate_id=f"cal-pos-{i}",
            session_id=f"cal-sess-{i % 4}",
            anchor_text="Someone should own the cutover checklist.",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            sample_source="random",
        )
        for i in range(6)
    ] + [
        build_decision_context_row(
            candidate_id=f"cal-neg-{i}",
            session_id=f"cal-sess-{i % 4}",
            anchor_text="Everything is already documented in the ticket.",
            worthwhile=False,
            categories=(),
            sample_source="random",
        )
        for i in range(6)
    ]
    cal_path = write_jsonl(e2e_workspace["root"] / "cal_random.jsonl", cal_rows)
    cal_out = e2e_workspace["root"] / "cal_platt.json"

    cal_res = run_coach(
        "calibrate",
        [
            "--data",
            str(cal_path),
            "--classifier",
            str(model_dir),
            "--mode",
            "technical_working_discussion",
            "--out",
            str(cal_out),
        ],
    )
    assert cal_res.returncode == 0, f"coach calibrate failed: {cal_res.stderr}\n{cal_res.stdout}"
    cal_data = json.loads(cal_out.read_text(encoding="utf-8"))
    assert cal_data["mode"] == "technical_working_discussion"
    assert cal_data["calibration_version"]
    assert cal_data["classifier_version"]
    assert cal_data["builder_version"] == "ctx-v0"
    assert cal_data["prefilter_version"] == "pre-v0"
    assert "brier_score" in cal_data or "brier" in json.dumps(cal_data)
    assert "reliability_bins" in cal_data or "bins" in json.dumps(cal_data)
    assert "bootstrap" in json.dumps(cal_data) or "ci" in json.dumps(cal_data)

    pol_trained = deep_merge_dict(
        build_policy_dict(calibration=str(cal_data["calibration_version"]), settle_ms=100),
        {
            "classifier": {
                "version": str(cal_data["classifier_version"]),
                "model_path": str(model_dir),
                "calibration_path": str(cal_out),
            }
        },
    )
    ev = [
        build_segment_event(
            session_id="sess-d-trained",
            event_id="ev-dt-1",
            segment_id="seg-dt-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3000,
            received_ms=3100,
            source_sequence=1,
        )
    ]
    db_tr, _ = _replay(
        e2e_workspace,
        ev,
        brief=build_session_brief(session_id="sess-d-trained"),
        policy=pol_trained,
        name="d_trained",
    )
    assert_six_conservation_invariants(db_tr, "sess-d-trained")
    with open_db(db_tr) as conn:
        decs = fetch_normalized_rows(conn, "decisions", "sess-d-trained")
    assert decs[0].get("calibration_version") == cal_data["calibration_version"]


def test_area_d_calibration_refuses_non_random_sample_source(
    e2e_workspace: dict[str, Path],
) -> None:
    bad_rows = [
        build_label_record(
            label_id="lbl-bad-1",
            candidate_id="cand-bad-1",
            sample_source="model_positive",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
        ),
        build_label_record(
            label_id="lbl-bad-2",
            candidate_id="cand-bad-2",
            sample_source="shadow_review",
            worthwhile=False,
            categories=(),
            example_questions=(),
            none_reason="no_gap",
        ),
    ]
    bad_cal_path = write_jsonl(e2e_workspace["root"] / "cal_nonrandom.jsonl", bad_rows)
    cal_out = e2e_workspace["root"] / "should_not_exist.json"
    res = run_coach(
        "calibrate",
        [
            "--data",
            str(bad_cal_path),
            "--classifier",
            "rules_v0",
            "--mode",
            "technical_working_discussion",
            "--out",
            str(cal_out),
        ],
    )
    assert res.returncode != 0
    combined = (res.stdout + "\n" + res.stderr).lower()
    assert "random" in combined or "sample_source" in combined
    assert not cal_out.exists()


def test_area_d_routing_policy_abstain_suppress_none_route_matrix(
    e2e_workspace: dict[str, Path],
) -> None:
    # 1. Proactive run exercising ROUTE, NONE, SUPPRESS (do_not_suggest, known_fact), ABSTAIN
    sid = "sess-d-matrix"
    brief = build_session_brief(
        session_id=sid,
        do_not_suggest=["budget"],
        known_facts=[{"key": "launch_date", "value": "March 3", "source": "brief"}],
    )
    pol = build_policy_dict(
        settle_ms=100,
        cooldown_ms=100,
        max_cards_per_10_min=10,
        asr_confidence_floor=0.75,
    )
    events = [
        # ROUTE -> above_t_display
        build_segment_event(
            session_id=sid,
            event_id="ev-m-1",
            segment_id="seg-m-route",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=2500,
            received_ms=2600,
            source_sequence=1,
        ),
        # NONE -> below_t_low
        build_segment_event(
            session_id=sid,
            event_id="ev-m-2",
            segment_id="seg-m-none",
            text="The chassis fans spin at two thousand rpm.",
            speaker_id="participant-c",
            start_ms=4000,
            end_ms=5500,
            received_ms=5600,
            source_sequence=2,
        ),
        # SUPPRESS -> do_not_suggest ("budget")
        build_segment_event(
            session_id=sid,
            event_id="ev-m-3",
            segment_id="seg-m-dns",
            text="Someone should own the budget approval spreadsheet.",
            speaker_id="participant-b",
            start_ms=7000,
            end_ms=8500,
            received_ms=8600,
            source_sequence=3,
        ),
        # SUPPRESS -> known_fact ("March 3" launch_date)
        build_segment_event(
            session_id=sid,
            event_id="ev-m-4",
            segment_id="seg-m-known",
            text="When is the launch_date set for March 3?",
            speaker_id="participant-c",
            start_ms=10000,
            end_ms=11500,
            received_ms=11600,
            source_sequence=4,
        ),
        # ABSTAIN -> speaker_uncertain
        build_segment_event(
            session_id=sid,
            event_id="ev-m-5",
            segment_id="seg-m-spk",
            text="Someone should own the DNS migration runbook.",
            speaker_id="participant-d",
            speaker_confidence=0.25,
            start_ms=13000,
            end_ms=14500,
            received_ms=14600,
            source_sequence=5,
            extra={"speaker_uncertain": True},
        ),
        # ABSTAIN -> asr_low_confidence
        build_segment_event(
            session_id=sid,
            event_id="ev-m-6",
            segment_id="seg-m-asr",
            text="Someone should own the canary alert threshold.",
            speaker_id="participant-b",
            start_ms=16000,
            end_ms=17500,
            received_ms=17600,
            source_sequence=6,
            extra={"asr_confidence": 0.40},
        ),
        # ABSTAIN -> score_in_band (t_low=0.20..t_display=0.75)
        build_segment_event(
            session_id=sid,
            event_id="ev-m-7",
            segment_id="seg-m-band",
            text="Maybe someone could possibly check staging eventually.",
            speaker_id="participant-c",
            start_ms=19000,
            end_ms=20500,
            received_ms=20600,
            source_sequence=7,
        ),
    ]
    db_path, _ = _replay(e2e_workspace, events, brief=brief, policy=pol, name="d_matrix")
    assert_six_conservation_invariants(db_path, sid)
    with open_db(db_path) as conn:
        decs = fetch_normalized_rows(conn, "decisions", sid)
    by_seg = {str(d["anchor_refs"][0]["segment_id"]): d for d in decs if d.get("anchor_refs")}
    assert by_seg["seg-m-route"]["outcome"] == "ROUTE"
    assert "above_t_display" in _reason_set(by_seg["seg-m-route"])
    assert by_seg["seg-m-none"]["outcome"] == "NONE"
    assert "below_t_low" in _reason_set(by_seg["seg-m-none"])
    assert by_seg["seg-m-dns"]["outcome"] == "SUPPRESS"
    assert "do_not_suggest" in _reason_set(by_seg["seg-m-dns"])
    assert by_seg["seg-m-known"]["outcome"] == "SUPPRESS"
    assert "known_fact" in _reason_set(by_seg["seg-m-known"])
    assert by_seg["seg-m-spk"]["outcome"] == "ABSTAIN"
    assert "speaker_uncertain" in _reason_set(by_seg["seg-m-spk"])
    assert by_seg["seg-m-asr"]["outcome"] == "ABSTAIN"
    assert "asr_low_confidence" in _reason_set(by_seg["seg-m-asr"])
    assert by_seg["seg-m-band"]["outcome"] == "ABSTAIN"
    assert "score_in_band" in _reason_set(by_seg["seg-m-band"])
    for dec in decs:
        assert _reason_set(dec).issubset(REASON_CODES)

    # 2. Quiet mode -> SUPPRESS (proactive_disabled)
    quiet_brief = build_session_brief(
        session_id="sess-d-quiet", attention={"max_cards_per_10_min": 3, "proactive": False}
    )
    db_quiet, _ = _replay(
        e2e_workspace,
        [dict(events[0], session_id="sess-d-quiet")],
        brief=quiet_brief,
        policy=pol,
        name="d_quiet",
    )
    with open_db(db_quiet) as conn:
        q_decs = fetch_normalized_rows(conn, "decisions", "sess-d-quiet")
    assert q_decs[0]["outcome"] == "SUPPRESS"
    assert "proactive_disabled" in _reason_set(q_decs[0])

    # 3. Missing calibration for mode -> ABSTAIN (mode_mismatch)
    pol_uncal = deep_merge_dict(pol, {"classifier": {"calibration": None}})
    db_uncal, _ = _replay(
        e2e_workspace,
        [dict(events[0], session_id="sess-d-uncal")],
        brief=build_session_brief(session_id="sess-d-uncal"),
        policy=pol_uncal,
        name="d_uncal",
    )
    with open_db(db_uncal) as conn:
        u_decs = fetch_normalized_rows(conn, "decisions", "sess-d-uncal")
    assert u_decs[0]["outcome"] == "ABSTAIN"
    assert "mode_mismatch" in _reason_set(u_decs[0])


def test_area_d_routing_rejects_invalid_threshold_ordering(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-d-bad-thresh"
    brief_path = write_brief_json(
        e2e_workspace["root"] / "bad_thresh_brief.json", build_session_brief(session_id=sid)
    )
    ev_path = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "bad_thresh.jsonl",
        [build_segment_event(session_id=sid)],
    )
    bad_cases = (
        ("t_compute_below_t_low", {"t_low": 0.60, "t_compute": 0.40, "t_display": 0.80}),
        ("t_display_below_t_compute", {"t_low": 0.20, "t_compute": 0.75, "t_display": 0.50}),
    )
    for label, thresholds in bad_cases:
        pol = deep_merge_dict(build_policy_dict(), {"categories": {"CONFIRM_OWNER": thresholds}})
        pol_path = write_policy_dict(e2e_workspace["policies_dir"] / f"{label}.yaml", pol)
        res = run_coach(
            "replay",
            [
                "--session",
                str(ev_path),
                "--brief",
                str(brief_path),
                "--policy",
                str(pol_path),
                "--db",
                str(e2e_workspace["root"] / f"{label}.db"),
            ],
        )
        assert res.returncode != 0, f"Expected threshold validation error for {label}"


def test_area_e_conversation_and_series_splitters_and_eval_look_counter(
    e2e_workspace: dict[str, Path],
) -> None:
    # Build 6 sessions across 3 recurring series (2 sessions each)
    dataset_rows: list[dict[str, Any]] = []
    for series_id in ("series-alpha", "series-beta", "series-gamma"):
        for conv_idx in range(2):
            sid = f"{series_id}-conv-{conv_idx}"
            for turn_idx in range(2):
                row = build_decision_context_row(
                    candidate_id=f"cand-{sid}-{turn_idx}",
                    session_id=sid,
                    anchor_text="Someone should own the rollback plan.",
                    worthwhile=(turn_idx == 0),
                    categories=("CONFIRM_OWNER",) if turn_idx == 0 else (),
                    sample_source="random",
                )
                row["series_id"] = series_id
                row["conversation_id"] = sid
                dataset_rows.append(row)

    manifest_path = write_jsonl(e2e_workspace["root"] / "series_dataset.jsonl", dataset_rows)
    eval_log = e2e_workspace["root"] / "eval_looks.json"

    res1 = run_coach(
        "replay",
        [
            "--session",
            str(manifest_path),
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--split-by",
            "series",
            "--eval-log",
            str(eval_log),
            "--json",
        ],
    )
    assert res1.returncode == 0, f"Eval split run 1 failed: {res1.stderr}"
    out1 = res1.json()
    splits = out1["splits"]
    train_series = set(splits["train_series"])
    cal_series = set(splits["cal_series"])
    eval_series = set(splits["eval_series"])
    assert train_series.isdisjoint(cal_series)
    assert train_series.isdisjoint(eval_series)
    assert cal_series.isdisjoint(eval_series)
    train_convs = set(splits["train_conversations"])
    cal_convs = set(splits["cal_conversations"])
    eval_convs = set(splits["eval_conversations"])
    assert train_convs.isdisjoint(cal_convs) and train_convs.isdisjoint(eval_convs)
    assert cal_convs.isdisjoint(eval_convs)
    assert int(out1["look_count"]) == 1

    res2 = run_coach(
        "replay",
        [
            "--session",
            str(manifest_path),
            "--brief",
            str(e2e_workspace["brief_path"]),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--split-by",
            "series",
            "--eval-log",
            str(eval_log),
            "--json",
        ],
    )
    assert res2.returncode == 0
    assert int(res2.json()["look_count"]) == 2


def test_area_e_validity_interval_matcher_and_unlabeled_review_queue(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-e-matcher"
    brief = build_session_brief(
        session_id=sid, attention={"max_cards_per_10_min": 20, "proactive": True}
    )
    pol = deep_merge_dict(
        build_policy_dict(settle_ms=100, cooldown_ms=50, max_cards_per_10_min=20),
        {"display": {"display_ttl_ms": 1500}},
    )
    # 6 anchors: (1) useful, (2) repetition, (3) late, (4) premature, (5) distracting, (6) unlabeled
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-em-1",
            segment_id="seg-em-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=2000,
            received_ms=2100,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-em-2",
            segment_id="seg-em-2",
            text="We still need someone to own the backup runbook.",
            speaker_id="participant-c",
            start_ms=3000,
            end_ms=4000,
            received_ms=4100,
            source_sequence=2,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-em-3",
            segment_id="seg-em-3",
            text="Someone should own the cutover checklist.",
            speaker_id="participant-b",
            start_ms=6000,
            end_ms=7000,
            received_ms=7100,
            source_sequence=3,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-em-4",
            segment_id="seg-em-4",
            text="Someone should own the canary alerts.",
            speaker_id="participant-c",
            start_ms=9000,
            end_ms=10000,
            received_ms=10100,
            source_sequence=4,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-em-5",
            segment_id="seg-em-5",
            text="Someone should own the joke slide deck.",
            speaker_id="participant-b",
            start_ms=12000,
            end_ms=13000,
            received_ms=13100,
            source_sequence=5,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-em-6",
            segment_id="seg-em-6",
            text="Someone should own the postmortem follow-up.",
            speaker_id="participant-c",
            start_ms=15000,
            end_ms=16000,
            received_ms=16100,
            source_sequence=6,
        ),
    ]
    labels = [
        # (1)+(2) same opportunity validity interval -> seg-em-1 useful, seg-em-2 repetition
        build_label_record(
            label_id="lbl-em-1",
            session_id=sid,
            candidate_id="cand-em-1",
            sample_source="random",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            evidence_refs=(
                {"segment_id": "seg-em-1", "revision": 1},
                {"segment_id": "seg-em-2", "revision": 1},
            ),
            open_ms=2000,
            ideal_ms=2500,
            close_ms=8000,
            closed_by="answered",
        ),
        # (3) close_ms=7120 already closed before display -> late (closed_by=answered)
        build_label_record(
            label_id="lbl-em-3",
            session_id=sid,
            candidate_id="cand-em-3",
            sample_source="random",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            evidence_refs=({"segment_id": "seg-em-3", "revision": 1},),
            open_ms=7000,
            ideal_ms=7050,
            close_ms=7120,
            closed_by="answered",
        ),
        # (4) premature: resolved by same speaker inside settle window
        build_label_record(
            label_id="lbl-em-4",
            session_id=sid,
            candidate_id="cand-em-4",
            sample_source="random",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            evidence_refs=({"segment_id": "seg-em-4", "revision": 1},),
            open_ms=10100,
            ideal_ms=10120,
            close_ms=10150,
            closed_by="answered",
        ),
        # (5) distracting: worthwhile=False with none_reason="rhetorical"
        build_label_record(
            label_id="lbl-em-5",
            session_id=sid,
            candidate_id="cand-em-5",
            sample_source="random",
            worthwhile=False,
            categories=(),
            example_questions=(),
            evidence_refs=({"segment_id": "seg-em-5", "revision": 1},),
            open_ms=13100,
            ideal_ms=None,
            close_ms=None,
            closed_by=None,
            none_reason="rhetorical",
        ),
    ]
    labels_path = write_jsonl(e2e_workspace["root"] / "matcher_labels.jsonl", labels)
    db_path, out = _replay(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol,
        name="e_matcher",
        extra_args=["--labels", str(labels_path)],
    )
    assert_six_conservation_invariants(db_path, sid)
    eval_metrics = out.get("evaluation") or out
    assert int(eval_metrics["useful_count"]) >= 1
    assert int(eval_metrics["repetition_count"]) >= 1
    assert int(eval_metrics["late_count"]) >= 1
    assert "answered" in (eval_metrics.get("late_by_closed_by") or {})
    assert int(eval_metrics["premature_count"]) >= 1
    assert int(eval_metrics["distracting_count"]) >= 1
    assert "rhetorical" in (eval_metrics.get("distracting_by_none_reason") or {})
    assert int(eval_metrics["unscored_pending_review_count"]) >= 1
    assert len(eval_metrics.get("review_queue") or []) >= 1


def test_area_e_threshold_sweeper_under_attention_budget(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-e-sweep"
    brief = build_session_brief(session_id=sid)
    pol = build_policy_dict(settle_ms=100, cooldown_ms=100, max_cards_per_10_min=3)
    events = [
        build_segment_event(
            session_id=sid,
            event_id=f"ev-sw-{idx}",
            segment_id=f"seg-sw-{idx}",
            text=text,
            speaker_id="participant-b" if idx % 2 == 0 else "participant-c",
            start_ms=1000 + idx * 5000,
            end_ms=2500 + idx * 5000,
            received_ms=2600 + idx * 5000,
            source_sequence=idx + 1,
        )
        for idx, text in enumerate(
            [
                "Someone should own the rollback plan.",
                "Maybe someone could check the staging logs later.",
                "Someone should own the cutover checklist.",
                "The joke deployment script runs on coffee.",
                "Someone should own the database failover runbook.",
            ]
        )
    ]
    labels = [
        build_label_record(
            label_id="lbl-sw-1",
            session_id=sid,
            candidate_id="cand-sw-1",
            sample_source="random",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            evidence_refs=({"segment_id": "seg-sw-0", "revision": 1},),
            open_ms=2600,
            close_ms=10000,
            closed_by="answered",
        ),
        build_label_record(
            label_id="lbl-sw-2",
            session_id=sid,
            candidate_id="cand-sw-2",
            sample_source="random",
            worthwhile=False,
            categories=(),
            example_questions=(),
            evidence_refs=({"segment_id": "seg-sw-3", "revision": 1},),
            open_ms=17600,
            ideal_ms=None,
            close_ms=None,
            closed_by=None,
            none_reason="rhetorical",
        ),
    ]
    labels_path = write_jsonl(e2e_workspace["root"] / "sweep_labels.jsonl", labels)
    _, out = _replay(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol,
        name="e_sweep",
        extra_args=[
            "--labels",
            str(labels_path),
            "--sweep-t-display",
            "0.40,0.60,0.75,0.90",
        ],
    )
    curve = out.get("sweep_curve") or out.get("threshold_sweep") or []
    assert len(curve) == 4
    for pt in curve:
        assert "t_display" in pt
        assert "useful_per_10_min" in pt
        assert "distracting_per_10_min" in pt
        assert float(pt["cards_per_10_min"]) <= 3.0


def test_area_e_paired_conversation_bootstrap_and_low_n_flagging(
    e2e_workspace: dict[str, Path],
) -> None:
    # Compare two runs across multiple conversations where SUCCESS_CRITERION has N=2
    events: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for c_idx in range(4):
        sid = f"sess-e-boot-{c_idx}"
        events.append(
            build_segment_event(
                session_id=sid,
                event_id=f"ev-b-{c_idx}-1",
                segment_id=f"seg-b-{c_idx}-1",
                text="Someone should own the rollback plan.",
                speaker_id="participant-b",
                start_ms=1000 + c_idx * 20000,
                end_ms=2500 + c_idx * 20000,
                received_ms=2600 + c_idx * 20000,
                source_sequence=len(events) + 1,
            )
        )
        labels.append(
            build_label_record(
                label_id=f"lbl-b-own-{c_idx}",
                session_id=sid,
                candidate_id=f"cand-b-own-{c_idx}",
                sample_source="random",
                worthwhile=True,
                categories=("CONFIRM_OWNER",),
                evidence_refs=({"segment_id": f"seg-b-{c_idx}-1", "revision": 1},),
                open_ms=2600 + c_idx * 20000,
                ideal_ms=None,
                close_ms=12000 + c_idx * 20000,
                closed_by="answered",
            )
        )
        if c_idx < 2:
            events.append(
                build_segment_event(
                    session_id=sid,
                    event_id=f"ev-b-{c_idx}-2",
                    segment_id=f"seg-b-{c_idx}-2",
                    text="We need to verify the canary rollout works.",
                    speaker_id="participant-c",
                    start_ms=6000 + c_idx * 20000,
                    end_ms=7500 + c_idx * 20000,
                    received_ms=7600 + c_idx * 20000,
                    source_sequence=len(events) + 1,
                )
            )
            labels.append(
                build_label_record(
                    label_id=f"lbl-b-suc-{c_idx}",
                    session_id=sid,
                    candidate_id=f"cand-b-suc-{c_idx}",
                    sample_source="random",
                    worthwhile=True,
                    categories=("SUCCESS_CRITERION",),
                    evidence_refs=({"segment_id": f"seg-b-{c_idx}-2", "revision": 1},),
                    open_ms=7600 + c_idx * 20000,
                    ideal_ms=None,
                    close_ms=17000 + c_idx * 20000,
                    closed_by="answered",
                )
            )

    brief = build_session_brief(session_id="sess-e-boot-0")
    pol_a = deep_merge_dict(
        build_policy_dict(settle_ms=100, cooldown_ms=100),
        {"categories": {"CONFIRM_OWNER": {"t_display": 0.90}}},
    )
    pol_b = deep_merge_dict(
        build_policy_dict(settle_ms=100, cooldown_ms=100),
        {"categories": {"CONFIRM_OWNER": {"t_display": 0.50}}},
    )
    labels_path = write_jsonl(e2e_workspace["root"] / "boot_labels.jsonl", labels)
    db_a, _ = _replay(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol_a,
        name="boot_a",
        extra_args=["--labels", str(labels_path), "--run-id", "run-boot-a"],
    )
    _replay(
        e2e_workspace,
        events,
        brief=brief,
        policy=pol_b,
        name="boot_a",
        extra_args=["--labels", str(labels_path), "--run-id", "run-boot-b"],
    )
    diff_res = run_coach(
        "diff",
        [
            "--db",
            str(db_a),
            "--run-a",
            "run-boot-a",
            "--run-b",
            "run-boot-b",
            "--labels",
            str(labels_path),
            "--bootstrap",
            "--json",
        ],
    )
    assert diff_res.returncode == 0, f"coach diff --bootstrap failed: {diff_res.stderr}"
    diff_json = diff_res.json()
    paired = diff_json["paired_bootstrap"]
    assert "ci_low" in paired and "ci_high" in paired
    per_cat = paired["per_category"]
    assert per_cat["SUCCESS_CRITERION"]["low_n"] is True
    assert int(per_cat["SUCCESS_CRITERION"]["n_positives"]) == 2


def test_area_e_coach_bench_latency_percentiles_and_memory(
    e2e_workspace: dict[str, Path],
) -> None:
    turn_text = (
        "S1: We need to verify the database failover runbook and canary alerts before cutover."
    )
    contexts = [
        build_decision_context_row(
            candidate_id=f"bench-128-{i}",
            input_limit_tokens=128,
            context_turns=tuple([turn_text] * 5),
        )
        for i in range(6)
    ] + [
        build_decision_context_row(
            candidate_id=f"bench-512-{i}",
            input_limit_tokens=512,
            context_turns=tuple([turn_text] * 24),
        )
        for i in range(6)
    ]
    ctx_path = write_jsonl(e2e_workspace["root"] / "bench_contexts.jsonl", contexts)
    res = run_coach(
        "bench",
        [
            "--contexts",
            str(ctx_path),
            "--classifier",
            "rules_v0",
            "--batch-sizes",
            "1,4",
            "--tokens",
            "128,512",
            "--json",
        ],
    )
    assert res.returncode == 0, f"coach bench failed: {res.stderr}\n{res.stdout}"
    data = res.json()
    cells = data.get("cells") or data.get("results") or []
    seen_pairs = {(int(c["tokens"]), int(c["batch_size"])) for c in cells}
    assert {(128, 1), (128, 4), (512, 1), (512, 4)}.issubset(seen_pairs)
    for cell in cells:
        assert float(cell["p50_ms"]) <= float(cell["p95_ms"]) <= float(cell["p99_ms"])
        assert float(cell["p99_ms"]) <= float(cell["max_ms"])
        assert float(cell["peak_rss_mb"]) > 0.0
        assert int(cell["warmup_discarded"]) >= 1
        assert int(cell["timed_calls"]) >= 1


def test_area_g_pass1_online_prefix_only_and_no_future_scroll(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-g-pass1"
    events = [
        build_segment_event(
            session_id=sid,
            event_id=f"ev-g1-{idx}",
            segment_id=f"seg-g1-{idx}",
            text=f"Turn {idx} someone should own step {idx}.",
            speaker_id="participant-b",
            start_ms=idx * 4000,
            end_ms=idx * 4000 + 2000,
            received_ms=idx * 4000 + 2100,
            source_sequence=idx,
        )
        for idx in range(1, 6)
    ]
    events_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "g_pass1.jsonl", events)
    labels_out = e2e_workspace["root"] / "pass1_marks.jsonl"
    port = free_port_factory()

    with start_coach_process(
        "annotate",
        [
            "--session",
            str(events_path),
            "--out",
            str(labels_out),
            "--pass",
            "1",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        url, token = extract_ui_url_and_token(proc, default_port=port, timeout=5.0)
        client = AnnotationClient(url, token=token)
        prefix = client.get_pass1_prefix(sid, at_ms=7000)
        assert prefix.get("events"), "Expected prefix events up to at_ms=7000"
        client.assert_pass1_no_future_scroll(sid, at_ms=7000, requested_ms=22000)
        client.post_pass1_mark(sid, at_ms=7000, state_version=1, question="Who owns step 1?")

    saved = read_jsonl(labels_out)
    assert len(saved) >= 1
    assert saved[0]["sample_source"] == "pass1_mark"
    assert int(saved[0]["state_version_at_decision"]) == 1


def test_area_g_pass2_step1_lock_invariant_before_and_after_step2_unlock(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-g-pass2"
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-g2-1",
            segment_id="seg-g2-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=4200,
            received_ms=4450,
            source_sequence=1,
        ),
        build_segment_event(
            session_id=sid,
            event_id="ev-g2-2",
            segment_id="seg-g2-2",
            text="Priya confirmed she will own the rollback plan.",
            speaker_id="participant-c",
            start_ms=12000,
            end_ms=14500,
            received_ms=14900,
            source_sequence=2,
        ),
    ]
    db_path, _ = _replay(
        e2e_workspace,
        events,
        brief=build_session_brief(session_id=sid),
        policy=build_policy_dict(settle_ms=100, cooldown_ms=100),
        name="g_pass2_seed",
    )
    with open_db(db_path) as conn:
        decs = fetch_normalized_rows(conn, "decisions", sid)
    cand_id = str(decs[0]["candidate_id"])

    events_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "g_pass2.jsonl", events)
    labels_out = e2e_workspace["root"] / "pass2_labels.jsonl"
    port = free_port_factory()

    with start_coach_process(
        "annotate",
        [
            "--session",
            str(events_path),
            "--db",
            str(db_path),
            "--out",
            str(labels_out),
            "--pass",
            "2",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        url, token = extract_ui_url_and_token(proc, default_port=port, timeout=5.0)
        client = AnnotationClient(url, token=token)
        step1_view = client.get_step1_view(cand_id)
        assert step1_view.get("candidate_id") == cand_id
        client.submit_step1(
            cand_id,
            worthwhile=True,
            categories=["CONFIRM_OWNER"],
            example_questions=["Who will own the rollback plan?"],
            sample_source="random",
        )
        unlocked = client.unlock_step2(cand_id)
        assert unlocked.get("continuation_events"), "Step 2 unlock must reveal continuation"
        client.assert_step1_locked(
            cand_id,
            {
                "worthwhile": False,
                "categories": ["CONFIRM_TIME"],
                "example_questions": [],
                "none_reason": "no_gap",
            },
        )
        client.submit_step2(cand_id, close_ms=14900, closed_by="answered", ideal_ms=5200)

    saved = read_jsonl(labels_out)
    assert len(saved) == 1
    rec = saved[0]
    assert rec["candidate_id"] == cand_id
    assert rec["worthwhile"] is True
    assert rec["categories"] == ["CONFIRM_OWNER"]
    assert rec["validity"]["close_ms"] == 14900
    assert rec["validity"]["closed_by"] == "answered"


def test_area_g_other_category_none_reason_and_all_sample_sources(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-g-schema"
    events = [
        build_segment_event(
            session_id=sid,
            event_id=f"ev-g3-{idx}",
            segment_id=f"seg-g3-{idx}",
            text=text,
            speaker_id="participant-b",
            start_ms=idx * 5000,
            end_ms=idx * 5000 + 2500,
            received_ms=idx * 5000 + 2600,
            source_sequence=idx,
        )
        for idx, text in enumerate(
            [
                "What SLA applies if the rollback fails halfway?",
                "Well obviously nobody likes outages.",
            ],
            start=1,
        )
    ]
    db_path, _ = _replay(
        e2e_workspace,
        events,
        brief=build_session_brief(session_id=sid),
        policy=build_policy_dict(settle_ms=100, cooldown_ms=100),
        name="g_schema_seed",
    )
    with open_db(db_path) as conn:
        decs = fetch_normalized_rows(conn, "decisions", sid)
    cids = [str(d["candidate_id"]) for d in decs]
    assert len(cids) == 2

    events_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "g_schema.jsonl", events)
    labels_out = e2e_workspace["root"] / "schema_labels.jsonl"
    port = free_port_factory()

    with start_coach_process(
        "annotate",
        [
            "--session",
            str(events_path),
            "--db",
            str(db_path),
            "--out",
            str(labels_out),
            "--pass",
            "2",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        url, token = extract_ui_url_and_token(proc, default_port=port, timeout=5.0)
        client = AnnotationClient(url, token=token)
        # Item 1: OTHER category + custom question + sample_source="random"
        client.get_step1_view(cids[0])
        client.submit_step1(
            cids[0],
            worthwhile=True,
            categories=["OTHER"],
            example_questions=["What is the rollback SLA?"],
            sample_source="random",
        )
        client.unlock_step2(cids[0])
        client.submit_step2(cids[0], close_ms=12000, closed_by="topic_moved")

        # Item 2: Negative with none_reason="rhetorical" + sample_source="shadow_review"
        client.get_step1_view(cids[1])
        client.submit_step1(
            cids[1],
            worthwhile=False,
            categories=[],
            example_questions=[],
            none_reason="rhetorical",
            sample_source="shadow_review",
        )
        client.unlock_step2(cids[1])
        client.submit_step2(cids[1], close_ms=None, closed_by="session_end")

    saved = read_jsonl(labels_out)
    assert len(saved) == 2
    for rec in saved:
        assert rec["schema_version"] == SCHEMA_VERSION
        assert rec["sample_source"] in SAMPLE_SOURCES
        assert set(rec["categories"]).issubset(set(CATEGORIES))
        assert rec["none_reason"] is None or rec["none_reason"] in NONE_REASONS
    assert saved[0]["categories"] == ["OTHER"]
    assert saved[0]["example_questions"] == ["What is the rollback SLA?"]
    assert saved[1]["worthwhile"] is False
    assert saved[1]["none_reason"] == "rhetorical"


def test_area_g_review_queue_unlabeled_cards_workflow(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-g-review"
    events = [
        build_segment_event(
            session_id=sid,
            event_id="ev-g4-1",
            segment_id="seg-g4-1",
            text="Someone should own the rollback plan.",
            speaker_id="participant-b",
            start_ms=1000,
            end_ms=3000,
            received_ms=3100,
            source_sequence=1,
        )
    ]
    empty_labels = write_jsonl(e2e_workspace["root"] / "empty_labels.jsonl", [])
    db_path, out1 = _replay(
        e2e_workspace,
        events,
        brief=build_session_brief(session_id=sid),
        policy=build_policy_dict(settle_ms=100, cooldown_ms=100),
        name="g_review_1",
        extra_args=["--labels", str(empty_labels)],
    )
    ev1 = out1.get("evaluation") or out1
    assert int(ev1["unscored_pending_review_count"]) >= 1

    events_path = e2e_workspace["sessions_dir"] / "g_review_1.jsonl"
    labels_out = e2e_workspace["root"] / "reviewed_labels.jsonl"
    port = free_port_factory()

    with start_coach_process(
        "annotate",
        [
            "--session",
            str(events_path),
            "--db",
            str(db_path),
            "--out",
            str(labels_out),
            "--pass",
            "2",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        url, token = extract_ui_url_and_token(proc, default_port=port, timeout=5.0)
        client = AnnotationClient(url, token=token)
        queue = client.get_review_queue()
        assert len(queue) >= 1
        cand_id = str(queue[0]["candidate_id"])
        client.get_step1_view(cand_id)
        client.submit_step1(
            cand_id,
            worthwhile=True,
            categories=["CONFIRM_OWNER"],
            example_questions=["Who will own the rollback plan?"],
            sample_source="shadow_review",
        )
        client.unlock_step2(cand_id)
        client.submit_step2(cand_id, close_ms=15000, closed_by="answered")

    _, out2 = _replay(
        e2e_workspace,
        events,
        brief=build_session_brief(session_id=sid),
        policy=build_policy_dict(settle_ms=100, cooldown_ms=100),
        name="g_review_2",
        extra_args=["--labels", str(labels_out)],
    )
    ev2 = out2.get("evaluation") or out2
    assert int(ev2["unscored_pending_review_count"]) == 0
    assert int(ev2["useful_count"]) >= 1


def test_area_g_blind_test_retest_and_cohens_kappa_report(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-g-kappa"
    events = [
        build_segment_event(
            session_id=sid,
            event_id=f"ev-gk-{idx}",
            segment_id=f"seg-gk-{idx}",
            text=f"Someone should handle task {idx}.",
            speaker_id="participant-b",
            start_ms=idx * 3000,
            end_ms=idx * 3000 + 1500,
            received_ms=idx * 3000 + 1600,
            source_sequence=idx,
        )
        for idx in range(1, 7)
    ]
    events_path = write_events_jsonl(e2e_workspace["sessions_dir"] / "g_kappa.jsonl", events)

    first_pass = [
        build_label_record(
            label_id="lbl-k-1",
            session_id=sid,
            candidate_id="cand-k-1",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            evidence_refs=({"segment_id": "seg-gk-1", "revision": 1},),
        ),
        build_label_record(
            label_id="lbl-k-2",
            session_id=sid,
            candidate_id="cand-k-2",
            worthwhile=True,
            categories=("CONFIRM_OWNER",),
            evidence_refs=({"segment_id": "seg-gk-2", "revision": 1},),
        ),
        build_label_record(
            label_id="lbl-k-3",
            session_id=sid,
            candidate_id="cand-k-3",
            worthwhile=False,
            categories=(),
            example_questions=(),
            none_reason="no_gap",
            evidence_refs=({"segment_id": "seg-gk-3", "revision": 1},),
        ),
        build_label_record(
            label_id="lbl-k-4",
            session_id=sid,
            candidate_id="cand-k-4",
            worthwhile=False,
            categories=(),
            example_questions=(),
            none_reason="not_material",
            evidence_refs=({"segment_id": "seg-gk-4", "revision": 1},),
        ),
        build_label_record(
            label_id="lbl-k-5",
            session_id=sid,
            candidate_id="cand-k-5",
            worthwhile=True,
            categories=("CONFIRM_TIME",),
            evidence_refs=({"segment_id": "seg-gk-5", "revision": 1},),
        ),
        build_label_record(
            label_id="lbl-k-6",
            session_id=sid,
            candidate_id="cand-k-6",
            worthwhile=True,
            categories=("CONFIRM_TIME",),
            evidence_refs=({"segment_id": "seg-gk-6", "revision": 1},),
        ),
    ]
    first_pass_path = write_jsonl(e2e_workspace["root"] / "first_pass.jsonl", first_pass)
    retest_out = e2e_workspace["root"] / "retest_labels.jsonl"
    port = free_port_factory()

    with start_coach_process(
        "annotate",
        [
            "--session",
            str(events_path),
            "--out",
            str(retest_out),
            "--kappa",
            str(first_pass_path),
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        url, token = extract_ui_url_and_token(proc, default_port=port, timeout=5.0)
        client = AnnotationClient(url, token=token)
        retest_inputs = [
            ("cand-k-1", True, ["CONFIRM_OWNER"], None),
            ("cand-k-2", True, ["CONFIRM_OWNER"], None),
            ("cand-k-3", False, [], "no_gap"),
            ("cand-k-4", False, [], "not_material"),
            ("cand-k-5", True, ["CONFIRM_TIME"], None),
            ("cand-k-6", False, [], "no_gap"),
        ]
        for item_id, ww, cats, nr in retest_inputs:
            blind_view = client.get_retest_item(item_id)
            assert blind_view.get("candidate_id") == item_id or blind_view.get("item_id") == item_id
            client.submit_retest(item_id, worthwhile=ww, categories=cats, none_reason=nr)

        kappa = client.get_kappa_report()
        wk = float(kappa["worthwhile_kappa"])
        assert -1.0 <= wk <= 1.0
        per_cat = kappa["per_category"]
        assert "CONFIRM_OWNER" in per_cat and "CONFIRM_TIME" in per_cat
        assert per_cat["CONFIRM_TIME"].get("low_n") is True
