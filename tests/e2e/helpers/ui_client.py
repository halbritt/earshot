"""Opaque-box HTTP/WebSocket clients for private Card UI and Annotation Tool."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidHandshake, WebSocketException

from tests.e2e.helpers.fixtures import FEEDBACK_REASONS

_DANGEROUS_DOM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("innerHTML", re.compile(r"\.innerHTML\b")),
    ("outerHTML", re.compile(r"\.outerHTML\b")),
    ("insertAdjacentHTML", re.compile(r"\binsertAdjacentHTML\b")),
    ("document.write", re.compile(r"\bdocument\.write\s*\(")),
    ("Notification(", re.compile(r"\bNotification\s*\(")),
)
_HTML_TAG_RE = re.compile(r"<\s*/?\s*(script|img|iframe|svg|body|on\w+)\b", re.IGNORECASE)


def assert_loopback_url(url: str, *, allow_localhost: bool = False) -> None:
    """Assert URL binds strictly to loopback (127.0.0.1 by default)."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    allowed = {"127.0.0.1", "localhost"} if allow_localhost else {"127.0.0.1"}
    if host not in allowed:
        raise AssertionError(f"Expected loopback host in {sorted(allowed)}, got {host!r} ({url})")


class CardUISession:
    """Active authenticated WebSocket session with Card UI."""

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws

    async def recv_json(self, timeout: float = 2.0) -> dict[str, Any]:
        raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise AssertionError(f"Expected JSON object from UI WS, got {type(data).__name__}")
        return data

    async def wait_for_card(self, timeout: float = 5.0) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            rem = deadline - asyncio.get_running_loop().time()
            if rem <= 0:
                raise TimeoutError("Timed out waiting for card message on UI WS")
            msg = await self.recv_json(timeout=rem)
            msg_type = str(msg.get("type") or "")
            if msg_type in {"card", "card_update", "suggestion"} or (
                "question" in msg and msg.get("question")
            ):
                return msg

    async def wait_for_health(self, expected_health: str, timeout: float = 5.0) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            rem = deadline - asyncio.get_running_loop().time()
            if rem <= 0:
                raise TimeoutError(f"Timed out waiting for health={expected_health!r}")
            msg = await self.recv_json(timeout=rem)
            health = msg.get("health") or msg.get("state")
            if msg.get("type") == "health" and health == expected_health:
                return msg
            if health == expected_health:
                return msg

    async def send_action(self, action: str, **payload: Any) -> None:
        body: dict[str, Any] = {"type": "action", "action": action}
        body.update(payload)
        await self._ws.send(json.dumps(body))

    async def send_use(self, suggestion_id: str) -> None:
        await self.send_action("use", suggestion_id=suggestion_id, key="u")

    async def send_dismiss(self, suggestion_id: str, key: str = "d") -> None:
        feedback = FEEDBACK_REASONS.get(key)
        extra: dict[str, Any] = {"suggestion_id": suggestion_id, "key": key}
        if feedback is not None:
            extra["feedback"] = feedback
        await self.send_action("dismiss", **extra)

    async def send_pin(self, suggestion_id: str) -> None:
        await self.send_action("pin", suggestion_id=suggestion_id, key="p")

    async def send_manual_request(self) -> None:
        await self.send_action("manual")

    async def send_mode(self, mode: str) -> None:
        await self.send_action("mode", mode=mode)

    async def send_off_record(self, active: bool = True) -> None:
        await self.send_action("off_record", active=active)


class CardUIClient:
    """Opaque-box client verifying §3.4 / §14.4-14.5 private localhost Card UI."""

    def __init__(self, base_url: str, token: str, *, allow_localhost: bool = False) -> None:
        assert_loopback_url(base_url, allow_localhost=allow_localhost)
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _ws_url(self, ws_path: str = "/ws", token: str | None = None) -> str:
        parsed = urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = ws_path if ws_path.startswith("/") else f"/{ws_path}"
        base_ws = f"{scheme}://{parsed.netloc}{path}"
        if token is not None:
            sep = "&" if "?" in base_ws else "?"
            return f"{base_ws}{sep}token={token}"
        return base_ws

    def assert_auth_enforced(
        self,
        *,
        ws_path: str = "/ws",
        state_path: str = "/",
    ) -> dict[str, int]:
        """Verify missing/wrong token is rejected on HTTP and WS; valid token succeeds."""
        path = state_path if state_path.startswith("/") else f"/{state_path}"
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=5.0) as client:
            no_auth = client.get(url)
            bad_bearer = client.get(url, headers={"Authorization": "Bearer wrong-token"})
            bad_query = client.get(f"{url}?token=wrong-token")
            ok_bearer = client.get(url, headers={"Authorization": f"Bearer {self.token}"})
            ok_query = client.get(f"{url}?token={self.token}")

        for label, resp in (
            ("unauthenticated", no_auth),
            ("bad_bearer", bad_bearer),
            ("bad_query", bad_query),
        ):
            if resp.status_code not in (401, 403):
                raise AssertionError(
                    f"Expected 401/403 for {label} HTTP GET {url}, got {resp.status_code}"
                )
        for label, resp in (("ok_bearer", ok_bearer), ("ok_query", ok_query)):
            if resp.status_code != 200:
                raise AssertionError(
                    f"Expected 200 for {label} HTTP GET {url}, got {resp.status_code}"
                )

        async def _check_ws_auth() -> None:
            for label, target_url in (
                ("ws_no_token", self._ws_url(ws_path, token=None)),
                ("ws_bad_token", self._ws_url(ws_path, token="wrong-token")),
            ):
                rejected = False
                try:
                    async with connect(target_url, open_timeout=3.0) as ws:
                        await asyncio.wait_for(ws.recv(), timeout=1.0)
                except InvalidHandshake, WebSocketException, OSError, TimeoutError:
                    rejected = True
                if not rejected:
                    raise AssertionError(f"Expected WS rejection for {label} ({target_url})")

            async with connect(self._ws_url(ws_path, token=self.token), open_timeout=3.0):
                pass

        asyncio.run(_check_ws_auth())
        return {
            "unauthenticated": no_auth.status_code,
            "bad_bearer": bad_bearer.status_code,
            "bad_query": bad_query.status_code,
            "ok_bearer": ok_bearer.status_code,
            "ok_query": ok_query.status_code,
        }

    @staticmethod
    def inspect_html_xss_safety(
        html: str,
        *,
        forbidden_substrings: Sequence[str] = (),
    ) -> None:
        """Verify UI HTML/JS uses plain-text DOM APIs without Notification or XSS sinks."""
        for label, pattern in _DANGEROUS_DOM_PATTERNS:
            if pattern.search(html):
                raise AssertionError(f"Unsafe DOM/API pattern {label!r} found in Card UI HTML")
        for needle in forbidden_substrings:
            if needle and needle in html:
                raise AssertionError(f"Unescaped forbidden substring {needle!r} in Card UI HTML")

    @staticmethod
    def assert_card_plain_text(
        card_payload: Mapping[str, Any],
        *,
        raw_adversarial_text: str = "",
        max_words: int = 20,
    ) -> None:
        """Verify displayed card text obeys word cap, plain text, and adversarial isolation."""
        question = str(card_payload.get("question") or card_payload.get("text") or "").strip()
        if not question:
            raise AssertionError(f"Card payload missing question/text: {card_payload}")
        words = question.split()
        if len(words) > max_words:
            raise AssertionError(f"Card exceeds {max_words} words ({len(words)}): {question!r}")
        if _HTML_TAG_RE.search(question):
            raise AssertionError(f"Unescaped HTML markup in card question: {question!r}")
        if (
            raw_adversarial_text
            and "<script" in raw_adversarial_text.lower()
            and ("<script" in question.lower())
        ):
            raise AssertionError(f"Adversarial script tag leaked into card: {question!r}")

    @asynccontextmanager
    async def ws_session(self, ws_path: str = "/ws") -> AsyncIterator[CardUISession]:
        async with connect(self._ws_url(ws_path, token=self.token), open_timeout=5.0) as ws:
            yield CardUISession(ws)


class AnnotationClient:
    """Opaque-box HTTP client for Pass 1 prefix-only & Pass 2 locked Annotation Tool (§12.4)."""

    def __init__(self, base_url: str, token: str | None = None) -> None:
        assert_loopback_url(base_url, allow_localhost=True)
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_pass1_prefix(
        self,
        session_id: str,
        at_ms: int,
        *,
        path_template: str = "/api/sessions/{session_id}/prefix",
        extra_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = path_template.format(session_id=session_id)
        params: dict[str, Any] = {"at_ms": at_ms}
        if extra_params:
            params.update(extra_params)
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url}{path}", params=params, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise AssertionError(f"Expected dict from {path}, got {type(data).__name__}")
        for ev in data.get("events", []):
            if isinstance(ev, dict) and int(ev.get("received_ms", 0)) > at_ms:
                raise AssertionError(f"Future leakage in Pass 1 prefix at {at_ms}: event={ev}")
        return data

    def assert_pass1_no_future_scroll(
        self,
        session_id: str,
        *,
        at_ms: int,
        requested_ms: int,
        path_template: str = "/api/sessions/{session_id}/prefix",
    ) -> None:
        """Verify requesting beyond playhead at_ms either rejects or clamps to at_ms."""
        path = path_template.format(session_id=session_id)
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                f"{self.base_url}{path}",
                params={"at_ms": at_ms, "scroll_to_ms": requested_ms, "include_future": "true"},
                headers=self._headers(),
            )
        if resp.status_code in (400, 403, 409, 422):
            return
        resp.raise_for_status()
        data = resp.json()
        for ev in data.get("events", []):
            if isinstance(ev, dict) and int(ev.get("received_ms", 0)) > at_ms:
                raise AssertionError(
                    f"Pass 1 allowed scrolling past at_ms={at_ms} to {requested_ms}: {ev}"
                )

    def post_pass1_mark(
        self,
        session_id: str,
        *,
        at_ms: int,
        state_version: int,
        question: str | None = None,
        path_template: str = "/api/sessions/{session_id}/marks",
    ) -> dict[str, Any]:
        path = path_template.format(session_id=session_id)
        payload = {"at_ms": at_ms, "state_version": state_version, "question": question}
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{self.base_url}{path}", json=payload, headers=self._headers())
        resp.raise_for_status()
        res = resp.json()
        return res if isinstance(res, dict) else {"result": res}

    def get_step1_view(
        self,
        candidate_id: str,
        *,
        path_template: str = "/api/candidates/{candidate_id}/step1",
    ) -> dict[str, Any]:
        path = path_template.format(candidate_id=candidate_id)
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url}{path}", headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise AssertionError(f"Expected dict from {path}")
        if data.get("continuation_events"):
            raise AssertionError("Step 1 view leaked continuation_events before Step 2 unlock")
        return data

    def submit_step1(
        self,
        candidate_id: str,
        *,
        worthwhile: bool,
        categories: Sequence[str],
        example_questions: Sequence[str],
        none_reason: str | None = None,
        sample_source: str = "random",
        path_template: str = "/api/candidates/{candidate_id}/step1",
    ) -> dict[str, Any]:
        path = path_template.format(candidate_id=candidate_id)
        payload = {
            "worthwhile": worthwhile,
            "categories": list(categories),
            "example_questions": list(example_questions),
            "none_reason": none_reason,
            "sample_source": sample_source,
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{self.base_url}{path}", json=payload, headers=self._headers())
        resp.raise_for_status()
        res = resp.json()
        return res if isinstance(res, dict) else {"result": res}

    def unlock_step2(
        self,
        candidate_id: str,
        *,
        path_template: str = "/api/candidates/{candidate_id}/step2/unlock",
    ) -> dict[str, Any]:
        path = path_template.format(candidate_id=candidate_id)
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{self.base_url}{path}", headers=self._headers())
        resp.raise_for_status()
        res = resp.json()
        if not isinstance(res, dict):
            raise AssertionError(f"Expected dict from {path}")
        return res

    def assert_step1_locked(
        self,
        candidate_id: str,
        mutated_payload: Mapping[str, Any],
        *,
        path_template: str = "/api/candidates/{candidate_id}/step1",
    ) -> int:
        """Assert Step 1 fields cannot be mutated once Step 2 is unlocked (§12.4)."""
        path = path_template.format(candidate_id=candidate_id)
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{self.base_url}{path}", json=dict(mutated_payload), headers=self._headers()
            )
        if resp.status_code not in (400, 403, 409, 422):
            raise AssertionError(
                f"Expected Step 1 lock rejection (400/403/409/422), got {resp.status_code}"
            )
        return resp.status_code

    def submit_step2(
        self,
        candidate_id: str,
        *,
        close_ms: int | None,
        closed_by: str,
        ideal_ms: int | None = None,
        path_template: str = "/api/candidates/{candidate_id}/step2",
    ) -> dict[str, Any]:
        path = path_template.format(candidate_id=candidate_id)
        payload = {"close_ms": close_ms, "closed_by": closed_by, "ideal_ms": ideal_ms}
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{self.base_url}{path}", json=payload, headers=self._headers())
        resp.raise_for_status()
        res = resp.json()
        return res if isinstance(res, dict) else {"result": res}

    def get_retest_item(
        self,
        item_id: str,
        *,
        path_template: str = "/api/retest/{item_id}",
    ) -> dict[str, Any]:
        path = path_template.format(item_id=item_id)
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url}{path}", headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise AssertionError(f"Expected dict from {path}")
        for secret_field in ("worthwhile", "categories", "none_reason", "annotator"):
            if data.get(secret_field) not in (None, [], ""):
                raise AssertionError(
                    f"Blind retest item leaked prior label field {secret_field!r}: {data}"
                )
        return data

    def get_kappa_report(self, *, path: str = "/api/retest/kappa") -> dict[str, Any]:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url}{path}", headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise AssertionError(f"Expected dict from {path}")
        if "worthwhile_kappa" not in data or "per_category" not in data:
            raise AssertionError(f"Kappa report missing expected metrics: {data}")
        return data
