"""Unit tests for private localhost Card UI security/actions and 2-pass Annotation Tool (§3.4, §12.2–§12.4, §14.4)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import InvalidStatus

from coach.annotation_tool import (
    AnnotationLockError,
    AnnotationServer,
    Pass1Session,
    Pass2Session,
    build_blind_retest_queue,
    build_pass2_items,
    compute_cohens_kappa,
    load_labels_jsonl,
    load_review_queue_items,
)
from coach.contracts import (
    AnnotationPass,
    CanonicalEvent,
    Category,
    ClosedBy,
    EventKind,
    HealthState,
    LabelRecord,
    Mode,
    NoneReason,
    RecipientConfig,
    SampleSource,
    SegmentRef,
    SessionBrief,
    SessionMode,
    Stability,
    ValidityInterval,
)
from coach.ui import CardUIServer, UIStateSnapshot


def _brief(sid: str = "sess-ui") -> SessionBrief:
    return SessionBrief(
        schema_version=1,
        session_id=sid,
        mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
        recipient=RecipientConfig(participant_id="me", role="reviewer"),
        objective="Confirm rollback owner and cutover schedule.",
    )


def _events(sid: str = "sess-ui") -> list[CanonicalEvent]:
    return [
        CanonicalEvent(
            schema_version=1,
            session_id=sid,
            event_id="e1",
            kind=EventKind.SEGMENT,
            received_ms=1000,
            source_sequence=1,
            segment_id="seg-1",
            source_revision=1,
            channel=0,
            speaker_id="S1",
            start_ms=500,
            end_ms=950,
            text="Someone needs to own the database rollback tonight.",
            stability=Stability.COMMITTED,
        ),
        CanonicalEvent(
            schema_version=1,
            session_id=sid,
            event_id="e2",
            kind=EventKind.SEGMENT,
            received_ms=3000,
            source_sequence=2,
            segment_id="seg-2",
            source_revision=1,
            channel=0,
            speaker_id="S2",
            start_ms=2200,
            end_ms=2900,
            text="I will own the database rollback tonight.",
            stability=Stability.COMMITTED,
        ),
    ]


def test_card_ui_rejects_non_loopback_and_static_has_no_innerhtml() -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        CardUIServer(session_id="s1", host="0.0.0.0", port=0)

    app_js = (Path(__file__).resolve().parents[2] / "coach" / "ui" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "textContent" in app_js
    for forbidden in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "Notification",
    ):
        assert forbidden not in app_js


async def test_card_ui_auth_origin_plain_text_and_ws_actions() -> None:
    received_actions: list[dict[str, Any]] = []

    async def _on_action(msg: dict[str, Any]) -> dict[str, Any]:
        received_actions.append(msg)
        act = msg.get("action")
        if act == "off_record":
            return UIStateSnapshot(
                session_id="sess-ui",
                mode=Mode.OFF_THE_RECORD,
                health=HealthState.OK,
                off_record=True,
                active_card=None,
            ).to_dict()
        return UIStateSnapshot(
            session_id="sess-ui",
            mode=Mode.PROACTIVE,
            health=HealthState.OK,
            active_card=None,
        ).to_dict()

    xss_text = "<script>alert('xss')</script> Who owns rollback?"
    init_state = UIStateSnapshot(
        session_id="sess-ui",
        mode=Mode.PROACTIVE,
        health=HealthState.OK,
        active_card={
            "suggestion_id": "sug-1",
            "candidate_id": "cand-1",
            "category": "CONFIRM_OWNER",
            "category_chip": "owner",
            "text": xss_text,
            "word_count": 5,
            "pinned": False,
            "origin": "proactive",
            "route": "template",
            "evidence": [{"speaker": "S1", "age": "<30s", "text": "<img src=x onerror=alert(1)>"}],
        },
    )
    srv = CardUIServer(
        session_id="sess-ui",
        port=0,
        on_action=_on_action,
        initial_state=init_state,
    )
    await srv.start()
    try:
        base_http = f"http://127.0.0.1:{srv.port}"
        async with httpx.AsyncClient() as client:
            for route in ("/", "/app.js", "/styles.css", "/state", "/health"):
                r_no_tok = await client.get(base_http + route)
                assert r_no_tok.status_code == 401
                r_bad_tok = await client.get(base_http + route + "?token=wrong")
                assert r_bad_tok.status_code == 401

            r_ok = await client.get(
                base_http + "/", headers={"Authorization": f"Bearer {srv.token}"}
            )
            assert r_ok.status_code == 200
            assert "Content-Security-Policy" in r_ok.headers
            assert r_ok.headers.get("X-Content-Type-Options") == "nosniff"

            r_state = await client.get(base_http + f"/state?token={srv.token}")
            assert r_state.status_code == 200
            assert r_state.json()["active_card"]["text"] == xss_text

        # WS missing/wrong token -> 401; non-loopback origin -> 403
        with pytest.raises(InvalidStatus) as exc_unauth:
            async with ws_connect(f"ws://127.0.0.1:{srv.port}/ws"):
                pass
        assert exc_unauth.value.response.status_code == 401

        with pytest.raises(InvalidStatus) as exc_origin:
            async with ws_connect(
                srv.ws_url,
                origin="https://evil.example",  # type: ignore[arg-type]
            ):
                pass
        assert exc_origin.value.response.status_code == 403

        # Valid WS connection receives state and handles actions
        async with ws_connect(srv.ws_url) as ws:
            first = json.loads(await ws.recv())
            assert first["type"] == "state"
            assert first["active_card"]["text"] == xss_text

            await ws.send(
                json.dumps(
                    {
                        "type": "action",
                        "action": "dismiss",
                        "suggestion_id": "sug-1",
                        "feedback": "not_relevant",
                    }
                )
            )
            after_dismiss = json.loads(await ws.recv())
            assert after_dismiss["active_card"] is None

            await ws.send(json.dumps({"type": "action", "action": "off_record"}))
            after_otr = json.loads(await ws.recv())
            assert after_otr["off_record"] is True
            assert len(received_actions) == 2
    finally:
        await srv.stop()


def test_annotation_pass1_prefix_only_and_http_409_scroll_ahead(tmp_path: Path) -> None:
    out_p = tmp_path / "pass1.jsonl"
    p1 = Pass1Session(
        brief=_brief(),
        events=_events(),
        initial_ms=0,
        out_path=out_p,
    )
    assert p1.visible_events() == []
    with pytest.raises(AnnotationLockError):
        p1.visible_events(at_ms=2000)
    with pytest.raises(AnnotationLockError):
        p1.seek(1500)

    srv = AnnotationServer(port=0, pass1=p1)
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv.port}"
        with httpx.Client() as client:
            assert client.get(base + "/api/pass1/state").status_code == 401
            headers = {"Authorization": f"Bearer {srv.token}"}
            r_seek = client.post(
                base + "/api/pass1/seek", json={"target_ms": 2000}, headers=headers
            )
            assert r_seek.status_code == 409
            r_fut = client.get(base + "/api/pass1/events?at_ms=2000", headers=headers)
            assert r_fut.status_code == 409

            r_step = client.post(base + "/api/pass1/step", json={}, headers=headers)
            assert r_step.status_code == 200
            st = r_step.json()
            assert st["now_ms"] == 1000
            assert len(st["visible_events"]) == 1

            r_mark = client.post(
                base + "/api/pass1/mark",
                json={"categories": ["CONFIRM_OWNER"], "example_questions": ["Who owns rollback?"]},
                headers=headers,
            )
            assert r_mark.status_code == 200
            lbl = LabelRecord.from_dict(r_mark.json()["label"])
            assert lbl.sample_source is SampleSource.PASS1_MARK
            assert lbl.annotation_pass is AnnotationPass.ONLINE
            assert lbl.worthwhile is True
            assert len(load_labels_jsonl(out_p)) == 1
    finally:
        srv.stop()


def test_annotation_pass2_step1_lock_review_queue_and_kappa(tmp_path: Path) -> None:
    brief = _brief()
    events = _events()
    items = build_pass2_items(brief, events, sample_source=SampleSource.RANDOM)
    assert len(items) == 2
    out_p = tmp_path / "pass2.jsonl"
    p2 = Pass2Session(items=items, out_path=out_p)

    # Continuation hidden before lock
    iid = items[0].item_id
    view0 = p2.get_item_view(iid)
    assert view0["continuation"] is None
    assert view0["continuation_available"] is False
    with pytest.raises(AnnotationLockError):
        p2.get_continuation(iid)
    with pytest.raises(AnnotationLockError):
        p2.submit_step2(iid, close_ms=3000, closed_by=ClosedBy.ANSWERED)

    srv = AnnotationServer(port=0, pass2=p2)
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv.port}"
        headers = {"Authorization": f"Bearer {srv.token}"}
        with httpx.Client() as client:
            assert (
                client.get(
                    f"{base}/api/pass2/items/{iid}/continuation", headers=headers
                ).status_code
                == 409
            )
            r_s1 = client.post(
                f"{base}/api/pass2/items/{iid}/step1",
                json={
                    "worthwhile": True,
                    "categories": ["CONFIRM_OWNER", "OTHER"],
                    "example_questions": ["Who owns rollback?"],
                },
                headers=headers,
            )
            assert r_s1.status_code == 200
            r_rev = client.post(f"{base}/api/pass2/items/{iid}/reveal", json={}, headers=headers)
            assert r_rev.status_code == 200
            rev_body = r_rev.json()
            assert rev_body["step1_locked"] is True
            assert rev_body["continuation_available"] is True
            assert len(rev_body["continuation"]) == 1

            # Mutating Step 1 after lock rejected with HTTP 409 (both /step1 and /step2)
            assert (
                client.post(
                    f"{base}/api/pass2/items/{iid}/step1",
                    json={"worthwhile": False, "none_reason": "no_gap"},
                    headers=headers,
                ).status_code
                == 409
            )
            assert (
                client.post(
                    f"{base}/api/pass2/items/{iid}/step2",
                    json={
                        "worthwhile": False,
                        "close_ms": 3000,
                        "closed_by": "answered",
                    },
                    headers=headers,
                ).status_code
                == 409
            )
            r_s2 = client.post(
                f"{base}/api/pass2/items/{iid}/step2",
                json={"ideal_ms": 1500, "close_ms": 3000, "closed_by": "answered"},
                headers=headers,
            )
            assert r_s2.status_code == 200
            saved = load_labels_jsonl(out_p)
            assert len(saved) == 1
            assert saved[0].sample_source is SampleSource.RANDOM
            assert Category.OTHER in saved[0].categories
    finally:
        srv.stop()

    # Review queue item builder + blind retest + kappa (low-N and sufficient-N)
    rq_items = load_review_queue_items(
        brief,
        events,
        [{"suggestion_id": "sug-rq", "segment_ids": ["seg-1"], "category": "CONFIRM_OWNER"}],
    )
    assert len(rq_items) == 1
    assert rq_items[0].sample_source is SampleSource.SHADOW_REVIEW

    def _mk_lbl(
        idx: int, worthwhile: bool, cat: Category | None = Category.CONFIRM_OWNER
    ) -> LabelRecord:
        return LabelRecord(
            schema_version=1,
            label_id=f"l-{idx}",
            session_id="s-k",
            annotator="recipient",
            sample_source=SampleSource.RANDOM,
            annotation_pass=AnnotationPass.OFFLINE,
            candidate_id=f"c-{idx}",
            state_version_at_decision=idx,
            worthwhile=worthwhile,
            categories=(cat,) if worthwhile and cat is not None else (),
            example_questions=(),
            evidence_refs=(SegmentRef("seg-1", 1),),
            validity=ValidityInterval(open_ms=1000, close_ms=3000, closed_by=ClosedBy.ANSWERED),
            none_reason=None if worthwhile else NoneReason.NO_GAP,
        )

    small_a = [_mk_lbl(i, i % 2 == 0) for i in range(6)]
    retest_q = build_blind_retest_queue(small_a, fraction=0.5, seed=0)
    assert len(retest_q) == 3
    assert "worthwhile" not in retest_q[0]
    k_small = compute_cohens_kappa(small_a, small_a)
    assert k_small["worthwhile"]["low_n"] is True
    assert k_small["worthwhile"]["kappa"] == pytest.approx(1.0)

    big_a = [_mk_lbl(i, i < 10) for i in range(24)]
    k_big = compute_cohens_kappa(big_a, big_a)
    assert k_big["worthwhile"]["low_n"] is False
    assert k_big["worthwhile"]["kappa"] == pytest.approx(1.0)
    assert k_big["per_category"]["CONFIRM_OWNER"]["low_n"] is False


async def test_card_ui_server_rejects_stale_seq_and_serializes_broadcasts() -> None:
    srv = CardUIServer(session_id="s-seq", port=0, emit_card_events=True)
    await srv.start()
    try:
        card2 = {
            "suggestion_id": "c2",
            "candidate_id": "cand-2",
            "category": "CONFIRM_OWNER",
            "category_chip": "owner",
            "text": "Who owns rollback?",
            "word_count": 3,
            "pinned": False,
            "origin": "proactive",
            "route": "template",
        }
        async with ws_connect(srv.ws_url) as ws:
            init_msg = json.loads(await ws.recv())
            assert init_msg["type"] == "state"
            assert init_msg["active_card"] is None

            await srv.broadcast_state(UIStateSnapshot(session_id="s-seq", active_card=card2, seq=1))
            st1 = json.loads(await ws.recv())
            cf1 = json.loads(await ws.recv())
            assert st1["active_card"]["suggestion_id"] == "c2"
            assert cf1["type"] == "card"
            assert cf1["suggestion_id"] == "c2"

            await srv.broadcast_state(UIStateSnapshot(session_id="s-seq", active_card=None, seq=3))
            st3 = json.loads(await ws.recv())
            assert st3["active_card"] is None

            # Out-of-order stale seq=2 must be dropped by both broadcast_state and update_state
            await srv.broadcast_state(
                UIStateSnapshot(
                    session_id="s-seq",
                    active_card={**card2, "pinned": True},
                    seq=2,
                )
            )
            srv.update_state(UIStateSnapshot(session_id="s-seq", active_card=card2, seq=2))
            assert srv.current_state["active_card"] is None
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=0.1)

            # Concurrent broadcasts serialize cleanly under _broadcast_lock
            card4 = {**card2, "suggestion_id": "c4", "candidate_id": "cand-4"}
            t4 = asyncio.create_task(
                srv.broadcast_state(UIStateSnapshot(session_id="s-seq", active_card=card4, seq=4))
            )
            t5 = asyncio.create_task(
                srv.broadcast_state(UIStateSnapshot(session_id="s-seq", active_card=None, seq=5))
            )
            await asyncio.gather(t4, t5)
            m4_state = json.loads(await ws.recv())
            m4_card = json.loads(await ws.recv())
            m5_state = json.loads(await ws.recv())
            assert m4_state["active_card"]["suggestion_id"] == "c4"
            assert m4_card["type"] == "card" and m4_card["suggestion_id"] == "c4"
            assert m5_state["active_card"] is None
    finally:
        await srv.stop()


async def test_card_ui_allowed_origins_binary_audio_feed_and_test_audio(tmp_path: Path) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"0123456789ABCDEF")
    received_audio: list[bytes] = []

    async def _on_audio(chunk: bytes) -> None:
        received_audio.append(chunk)

    srv = CardUIServer(
        session_id="s-live",
        port=0,
        allowed_origins=("https://proximal.tail0ecc2e.ts.net:8798",),
        on_audio_chunk=_on_audio,
        feed_provider=lambda: {
            "segments": [{"segment_id": "seg-1", "speaker_id": "S1", "text": "hello"}],
            "decisions": [],
            "audio_rx_bytes": sum(len(c) for c in received_audio),
        },
        test_audio_path=audio_file,
    )
    await srv.start()
    try:
        base_http = f"http://127.0.0.1:{srv.port}"
        async with httpx.AsyncClient() as client:
            assert (await client.get(f"{base_http}/feed")).status_code == 401
            r_feed = await client.get(f"{base_http}/feed?token={srv.token}")
            assert r_feed.status_code == 200
            assert r_feed.json()["has_test_audio"] is True
            assert r_feed.json()["segments"][0]["text"] == "hello"

            assert (await client.get(f"{base_http}/test-audio")).status_code == 401
            r_full = await client.get(f"{base_http}/test-audio?token={srv.token}")
            assert r_full.status_code == 200
            assert r_full.content == b"0123456789ABCDEF"

            r_range = await client.get(
                f"{base_http}/test-audio?token={srv.token}",
                headers={"Range": "bytes=4-7"},
            )
            assert r_range.status_code == 206
            assert r_range.content == b"4567"
            assert r_range.headers["Content-Range"] == "bytes 4-7/16"

        with pytest.raises(InvalidStatus) as exc_bad_origin:
            async with ws_connect(
                srv.ws_url,
                origin="https://evil.example",  # type: ignore[arg-type]
            ):
                pass
        assert exc_bad_origin.value.response.status_code == 403

        async with ws_connect(
            srv.ws_url,
            origin="https://proximal.tail0ecc2e.ts.net:8798",  # type: ignore[arg-type]
        ) as ws:
            _ = json.loads(await ws.recv())
            await ws.send(b"\x01\x02\x03\x04")
            await asyncio.sleep(0.05)
        assert received_audio == [b"\x01\x02\x03\x04"]
    finally:
        await srv.stop()
