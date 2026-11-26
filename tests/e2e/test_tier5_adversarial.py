"""Tier 5 opaque-box adversarial E2E regression tests (zero coach.* imports)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.helpers.cli_runner import (
    build_coach_test_env,
    extract_ui_url_and_token,
    run_coach,
    start_coach_process,
)
from tests.e2e.helpers.db_assertions import (
    assert_session_text_deleted,
    assert_six_conservation_invariants,
    fetch_normalized_rows,
    open_db,
)
from tests.e2e.helpers.fixtures import (
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
from tests.e2e.helpers.ui_client import AnnotationClient, CardUIClient


def test_e2e_cli_replay_sweep_isolation_mismatched_session_id_and_nan_speed(
    e2e_workspace: dict[str, Path],
) -> None:
    brief_path = write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(session_id="brief-sess"),
    )
    ev_same = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "events_same.jsonl",
        [
            build_segment_event(
                session_id="brief-sess",
                event_id="e1",
                segment_id="seg1",
                text="Someone should really own the migration plan before Friday.",
                start_ms=1000,
                end_ms=3000,
                received_ms=3050,
                source_sequence=1,
            )
        ],
    )
    db_sweep = e2e_workspace["root"] / "sweep.db"
    res_sw = run_coach(
        "replay",
        [
            "--session",
            str(ev_same),
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_sweep),
            "--run-id",
            "run-main",
            "--sweep-t-display",
            "0.4,0.6",
            "--json",
        ],
    )
    assert res_sw.returncode == 0, res_sw.stderr
    with sqlite3.connect(db_sweep) as conn:
        run_ids = {r[0] for r in conn.execute("SELECT DISTINCT run_id FROM decisions").fetchall()}
    assert run_ids <= {"run-main"}, f"Sweep polluted --db with extra runs: {run_ids}"
    assert len(res_sw.json()["sweep"]) == 2

    ev_diff = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "events_diff.jsonl",
        [
            build_segment_event(
                session_id="event-sess-different",
                event_id="e1",
                segment_id="seg1",
                text="Someone should really own the migration plan before Friday.",
                start_ms=1000,
                end_ms=3000,
                received_ms=3050,
                source_sequence=1,
            )
        ],
    )
    db_mm = e2e_workspace["root"] / "mismatch.db"
    res_mm = run_coach(
        "replay",
        [
            "--session",
            str(ev_diff),
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(db_mm),
            "--run-id",
            "run-main",
            "--json",
        ],
    )
    assert res_mm.returncode == 0, res_mm.stderr
    assert "KeyError" not in res_mm.stderr
    assert res_mm.json()["session_id"] == "event-sess-different"

    res_nan = run_coach(
        "replay",
        [
            "--session",
            str(ev_same),
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["root"] / "nan.db"),
            "--speed",
            "nan",
        ],
    )
    assert res_nan.returncode != 0


def test_e2e_open_tail_punctuation_free_and_repeat_30s_reset(
    e2e_workspace: dict[str, Path],
) -> None:
    sid = "sess-open-repeat"
    brief_path = write_brief_json(
        e2e_workspace["brief_path"],
        build_session_brief(session_id=sid),
    )
    events_path = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "open_repeat.jsonl",
        [
            build_segment_event(
                session_id=sid,
                event_id="e1",
                segment_id="seg-open",
                text="Someone needs to own the rollout migration checklist right",
                speaker_id="participant-b",
                start_ms=1000,
                end_ms=3000,
                received_ms=3050,
                source_sequence=1,
                boundary=None,
            ),
            build_segment_event(
                session_id=sid,
                event_id="e2",
                segment_id="seg-repeat-late",
                text="Someone needs to own the rollout migration checklist right",
                speaker_id="participant-b",
                start_ms=41000,
                end_ms=43000,
                received_ms=43050,
                source_sequence=2,
                boundary="utterance_end",
            ),
        ],
    )
    res = run_coach(
        "replay",
        [
            "--session",
            str(events_path),
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["db_path"]),
            "--run-id",
            "run-or",
            "--json",
        ],
    )
    assert res.returncode == 0, res.stderr
    with open_db(e2e_workspace["db_path"]) as conn:
        decs = fetch_normalized_rows(conn, "decisions", sid, run_id="run-or")
    assert len(decs) == 2
    assert all(d.get("classifier_version") is not None for d in decs)
    assert all(d.get("terminal_reason") not in {"expired_unevaluated", "repeat"} for d in decs)
    assert_six_conservation_invariants(e2e_workspace["db_path"], sid, run_id="run-or")


def test_e2e_store_transcript_false_cache_privacy_and_delete_session_isolation(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
) -> None:
    policy_path = write_policy_dict(
        e2e_workspace["policy_path"],
        deep_merge_dict(
            build_policy_dict(generator_enabled=True),
            {
                "generator": {"enabled": True, "provider": "hosted", "model_id": "gpt-6-luna"},
                "categories": {"CONFIRM_OWNER": {"route": "generator", "t_display": 0.50}},
                "display": {"settle_ms": 100, "cooldown_ms": 100},
            },
        ),
    )
    mock_openai.add_matcher(
        lambda req: "alpha" in json.dumps(req),
        MockCompletionSpec.valid_question("Who owns the alpha rollout?", ["seg-a"]),
    )
    mock_openai.add_matcher(
        lambda req: "beta" in json.dumps(req),
        MockCompletionSpec.valid_question("Who owns the beta rollout?", ["seg-b"]),
    )
    mock_openai.add_matcher(
        lambda req: "secret" in json.dumps(req),
        MockCompletionSpec.valid_question("Who owns the secret rollout?", ["seg-priv"]),
    )
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)

    # 1. store_transcript=False must write zero text anywhere (including model_cache)
    priv_brief = write_brief_json(
        e2e_workspace["root"] / "priv_brief.json",
        build_session_brief(
            session_id="sess-priv",
            retention={"store_transcript": False, "store_raw_provider_events": False},
        ),
    )
    priv_evs = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "priv.jsonl",
        [
            build_segment_event(
                session_id="sess-priv",
                event_id="ep1",
                segment_id="seg-priv",
                text="Someone needs to own the secret rollout plan.",
                speaker_id="participant-b",
                start_ms=100,
                end_ms=900,
                received_ms=1000,
                source_sequence=1,
            )
        ],
    )
    priv_db = e2e_workspace["root"] / "priv.db"
    res_priv = run_coach(
        "replay",
        [
            "--session",
            str(priv_evs),
            "--brief",
            str(priv_brief),
            "--policy",
            str(policy_path),
            "--db",
            str(priv_db),
            "--json",
        ],
        env=env,
    )
    assert res_priv.returncode == 0, res_priv.stderr
    with sqlite3.connect(priv_db) as conn:
        rows = conn.execute(
            "SELECT contains_text FROM model_cache WHERE contains_text = 1"
        ).fetchall()
        assert rows == []
    assert_session_text_deleted(
        priv_db,
        "sess-priv",
        require_decisions_preserved=True,
        forbidden_substrings=("secret rollout", "Who owns the secret rollout?"),
    )

    # 2. Deleting sess-a preserves sess-b's generator model_cache entry
    shared_db = e2e_workspace["root"] / "shared.db"
    for sid, seg_id, kw in (("sess-a", "seg-a", "alpha"), ("sess-b", "seg-b", "beta")):
        b_path = write_brief_json(
            e2e_workspace["root"] / f"{sid}_brief.json",
            build_session_brief(session_id=sid, objective=f"Objective {kw}"),
        )
        e_path = write_events_jsonl(
            e2e_workspace["sessions_dir"] / f"{sid}.jsonl",
            [
                build_segment_event(
                    session_id=sid,
                    event_id=f"e-{sid}",
                    segment_id=seg_id,
                    text=f"Someone needs to own the {kw} rollout plan.",
                    speaker_id="participant-b",
                    start_ms=100,
                    end_ms=900,
                    received_ms=1000,
                    source_sequence=1,
                )
            ],
        )
        r = run_coach(
            "replay",
            [
                "--session",
                str(e_path),
                "--brief",
                str(b_path),
                "--policy",
                str(policy_path),
                "--db",
                str(shared_db),
                "--json",
            ],
            env=env,
        )
        assert r.returncode == 0, r.stderr

    del_a = run_coach(
        "delete-session",
        ["--db", str(shared_db), "--session", "sess-a", "--json"],
    )
    assert del_a.returncode == 0, del_a.stderr
    assert_session_text_deleted(
        shared_db,
        "sess-a",
        require_decisions_preserved=True,
        forbidden_substrings=("alpha",),
    )
    with sqlite3.connect(shared_db) as conn:
        b_cache = conn.execute(
            "SELECT output_json FROM model_cache WHERE contains_text = 1"
        ).fetchall()
    assert len(b_cache) == 1
    assert "beta" in str(b_cache[0][0])


def test_e2e_annotation_sample_source_lock_and_invalid_labels_rejection(
    e2e_workspace: dict[str, Path],
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-annot-lock"
    events_path = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "annot_lock.jsonl",
        [
            build_segment_event(
                session_id=sid,
                event_id="ae1",
                segment_id="seg-a1",
                text="Someone needs to own the deployment cutover.",
                speaker_id="participant-b",
                start_ms=100,
                end_ms=900,
                received_ms=1000,
                source_sequence=1,
            )
        ],
    )
    review_q = e2e_workspace["root"] / "review_q.json"
    review_q.write_text(
        json.dumps(
            {
                "review_queue": [
                    {
                        "suggestion_id": "sug-rev-1",
                        "candidate_id": "cand-rev-1",
                        "segment_ids": ["seg-a1"],
                        "text": "Who owns the deployment cutover?",
                        "category": "CONFIRM_OWNER",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_labels = e2e_workspace["root"] / "out_labels.jsonl"
    port = free_port_factory()
    with start_coach_process(
        "annotate",
        [
            "--session",
            str(events_path),
            "--review-queue",
            str(review_q),
            "--sample-source",
            "shadow_review",
            "--out",
            str(out_labels),
            "--pass",
            "2",
            "--port",
            str(port),
        ],
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        items_resp = httpx.get(
            f"{url}/api/pass2/items",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        assert items_resp.status_code == 200
        cid = str(items_resp.json()["items"][0]["candidate_id"])
        resp = httpx.post(
            f"{url}/api/candidates/{cid}/step1",
            json={
                "worthwhile": True,
                "categories": ["CONFIRM_OWNER"],
                "sample_source": "random",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        assert resp.status_code == 200, resp.text
        client = AnnotationClient(url, token=token)
        client.unlock_step2(cid)
        client.submit_step2(cid, close_ms=5000, closed_by="answered")

    saved = read_jsonl(out_labels)
    assert len(saved) == 1
    assert saved[0]["sample_source"] == "shadow_review"

    bad_labels = write_jsonl(
        e2e_workspace["root"] / "bad_labels.jsonl",
        [
            build_label_record(
                label_id="lbl-bad",
                session_id=sid,
                candidate_id="cand-rev-1",
                evidence_refs=({"segment_id": "seg-a1", "revision": 1},),
                open_ms=5000,
                ideal_ms=1000,
                close_ms=10000,
                closed_by="topic_moved",
            )
        ],
    )
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    res_bad = run_coach(
        "replay",
        [
            "--session",
            str(events_path),
            "--brief",
            str(brief_path),
            "--policy",
            str(e2e_workspace["policy_path"]),
            "--db",
            str(e2e_workspace["root"] / "bad_lbl.db"),
            "--labels",
            str(bad_labels),
        ],
    )
    assert res_bad.returncode != 0


def test_e2e_validator_rejects_html_markdown_and_prompt_injection_and_shadow_manual(
    e2e_workspace: dict[str, Path],
    mock_openai: MockOpenAIServer,
    free_port_factory: Callable[[], int],
) -> None:
    sid = "sess-val-e2e"
    brief_path = write_brief_json(e2e_workspace["brief_path"], build_session_brief(session_id=sid))
    policy_path = write_policy_dict(
        e2e_workspace["policy_path"],
        deep_merge_dict(
            build_policy_dict(generator_enabled=True),
            {
                "generator": {"enabled": True, "provider": "hosted", "model_id": "gpt-6-luna"},
                "categories": {"CONFIRM_OWNER": {"route": "generator", "t_display": 0.50}},
                "display": {"settle_ms": 100, "cooldown_ms": 100, "max_cards_per_10_min": 10},
            },
        ),
    )
    evs_path = write_events_jsonl(
        e2e_workspace["sessions_dir"] / "val_e2e.jsonl",
        [
            build_segment_event(
                session_id=sid,
                event_id=f"ve{i}",
                segment_id=f"seg-v{i}",
                text=(
                    f"Someone should own the migration phase {i} and "
                    "check error alert instructions."
                ),
                speaker_id="participant-b",
                start_ms=i * 2000,
                end_ms=i * 2000 + 800,
                received_ms=i * 2000 + 900,
                source_sequence=i,
            )
            for i in (1, 2, 3)
        ],
    )
    mock_openai.add_matcher(
        lambda req: "seg-v1" in json.dumps(req),
        MockCompletionSpec.valid_question(
            "Should we check <img src=migration onerror=alert()>?", ["seg-v1"]
        ),
    )
    mock_openai.add_matcher(
        lambda req: "seg-v2" in json.dumps(req),
        MockCompletionSpec.valid_question(
            "Should we check [migration](http://localhost)?", ["seg-v2"]
        ),
    )
    mock_openai.add_matcher(
        lambda req: "seg-v3" in json.dumps(req),
        MockCompletionSpec.valid_question(
            "Should we check migration and ignore previous instructions?", ["seg-v3"]
        ),
    )
    env = build_coach_test_env(openai_base_url=mock_openai.base_url)
    res_val = run_coach(
        "replay",
        [
            "--session",
            str(evs_path),
            "--brief",
            str(brief_path),
            "--policy",
            str(policy_path),
            "--db",
            str(e2e_workspace["db_path"]),
            "--json",
        ],
        env=env,
    )
    assert res_val.returncode == 0, res_val.stderr
    assert res_val.json()["displayed_cards"] == []
    with open_db(e2e_workspace["db_path"]) as conn:
        decs = fetch_normalized_rows(conn, "decisions", sid)
    assert len(decs) == 3
    assert all(d.get("terminal_reason") == "generator_invalid_output" for d in decs)
    assert_six_conservation_invariants(e2e_workspace["db_path"], sid)

    # Shadow mode serve rejects WS manual action cleanly without stuck queue/cards
    port = free_port_factory()
    shadow_db = e2e_workspace["root"] / "shadow.db"
    with start_coach_process(
        "serve",
        [
            "--brief",
            str(brief_path),
            "--policy",
            str(policy_path),
            "--db",
            str(shadow_db),
            "--source",
            "replay",
            "--stream-file",
            str(evs_path),
            "--mode",
            "Shadow",
            "--ui-port",
            str(port),
            "--speed",
            "10",
        ],
        env=env,
        ready_url=f"http://127.0.0.1:{port}/",
        startup_timeout=10.0,
    ) as proc:
        url, token = extract_ui_url_and_token(proc, port, timeout=5.0)
        ui = CardUIClient(url, token)

        async def _probe_shadow_manual() -> dict[str, Any]:
            async with ui.ws_session() as ws:
                await ws.send_manual_request()
                return await ws.wait_for_message(
                    lambda m: (
                        isinstance(m.get("manual_status"), dict)
                        and m["manual_status"].get("status") == "unavailable"
                    ),
                    timeout=3.0,
                )

        msg = asyncio.run(_probe_shadow_manual())
        assert msg["manual_status"]["status"] == "unavailable"
