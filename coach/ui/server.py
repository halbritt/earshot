"""Private 127.0.0.1-only token-gated HTTP/WebSocket Card UI server (§3.4, §14.4)."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import Server, ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from coach.contracts import Category, HealthState, Mode

STATIC_DIR = Path(__file__).resolve().parent / "static"

CATEGORY_CHIPS: dict[str, str] = {
    Category.CONFIRM_OWNER.value: "owner",
    Category.CONFIRM_TIME.value: "deadline",
    Category.SUCCESS_CRITERION.value: "success criterion",
    Category.CLARIFY_SCOPE.value: "scope",
    Category.REQUEST_EVIDENCE.value: "evidence",
    Category.CLARIFY_REFERENCE.value: "reference",
    Category.PROBE_TRADEOFF.value: "tradeoff",
    Category.FOLLOW_UP.value: "follow-up",
    Category.OTHER.value: "other",
}

_SECURITY_HEADERS = [
    ("X-Content-Type-Options", "nosniff"),
    ("Cache-Control", "no-store"),
    (
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self' ws://127.0.0.1:*; object-src 'none'; base-uri 'none'",
    ),
]
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True, slots=True)
class UIStateSnapshot:
    """Serializable Card UI state pushed over WebSocket and GET /state."""

    session_id: str
    mode: Mode | str = Mode.PROACTIVE
    health: HealthState | str = HealthState.OK
    health_reason: str | None = None
    off_record: bool = False
    active_card: dict[str, Any] | None = None
    manual_status: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    seq: int | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "state",
            "session_id": self.session_id,
            "mode": str(self.mode),
            "health": str(self.health),
            "health_reason": self.health_reason,
            "off_record": self.off_record,
            "active_card": self.active_card,
            "manual_status": self.manual_status,
        }
        if self.extra:
            out.update(self.extra)
        return out


ActionHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None] | dict[str, Any] | None]


class CardUIServer:
    """Loopback-only token-gated HTTP + WebSocket UI server (§3.4, §14.4)."""

    def __init__(
        self,
        *,
        session_id: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        session_token: str | None = None,
        on_action: ActionHandler | None = None,
        initial_state: UIStateSnapshot | None = None,
        emit_card_events: bool = False,
        send_timeout_s: float = 1.0,
        client_queue_maxsize: int = 64,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError(f"CardUIServer must bind strictly to 127.0.0.1, got {host!r}")
        if port < 0 or port > 65535:
            raise ValueError(f"Invalid UI port {port}")
        self._host = host
        self._requested_port = port
        self._bound_port = port
        self._token = session_token or secrets.token_urlsafe(32)
        self._on_action = on_action
        self._emit_card_events = bool(emit_card_events)
        self._send_timeout_s = max(0.05, float(send_timeout_s))
        self._client_queue_maxsize = max(1, int(client_queue_maxsize))
        self._last_emitted_card_id: str | None = None
        self._broadcast_lock = asyncio.Lock()
        self._last_broadcast_seq: int = (
            initial_state.seq if initial_state is not None and initial_state.seq is not None else -1
        )
        self._state: dict[str, Any] = (
            initial_state.to_dict()
            if initial_state is not None
            else UIStateSnapshot(session_id=session_id).to_dict()
        )
        self._server: Server | None = None
        self._clients: set[ServerConnection] = set()
        self._client_queues: dict[ServerConnection, asyncio.Queue[tuple[str, ...]]] = {}
        self._client_tasks: dict[ServerConnection, asyncio.Task[None]] = {}
        self._close_tasks: set[asyncio.Task[None]] = set()
        self._static_cache: dict[str, bytes] = {
            "index.html": (STATIC_DIR / "index.html").read_bytes(),
            "app.js": (STATIC_DIR / "app.js").read_bytes(),
            "styles.css": (STATIC_DIR / "styles.css").read_bytes(),
        }

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._bound_port

    @property
    def token(self) -> str:
        return self._token

    @property
    def http_url(self) -> str:
        return f"http://{self._host}:{self._bound_port}/?token={self._token}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self._host}:{self._bound_port}/ws?token={self._token}"

    @property
    def current_state(self) -> dict[str, Any]:
        return dict(self._state)

    def set_action_handler(self, handler: ActionHandler | None) -> None:
        self._on_action = handler

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await ws_serve(
            self._ws_handler,
            self._host,
            self._requested_port,
            process_request=self._process_request,
        )
        if self._server.sockets:
            self._bound_port = int(self._server.sockets[0].getsockname()[1])

    @staticmethod
    async def _close_ws_quiet(ws: ServerConnection) -> None:
        with contextlib.suppress(OSError, ConnectionClosed):
            await ws.close()

    def _drop_client_sync(self, ws: ServerConnection, *, close: bool = False) -> None:
        self._clients.discard(ws)
        self._client_queues.pop(ws, None)
        task = self._client_tasks.pop(ws, None)
        cur = asyncio.current_task()
        if task is not None and task is not cur and not task.done():
            task.cancel()
        if close:
            close_task = asyncio.create_task(self._close_ws_quiet(ws))
            self._close_tasks.add(close_task)
            close_task.add_done_callback(self._close_tasks.discard)

    async def _remove_client(self, ws: ServerConnection) -> None:
        self._clients.discard(ws)
        self._client_queues.pop(ws, None)
        task = self._client_tasks.pop(ws, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _ensure_client_queue(self, ws: ServerConnection) -> asyncio.Queue[tuple[str, ...]]:
        q = self._client_queues.get(ws)
        if q is None:
            q = asyncio.Queue(maxsize=self._client_queue_maxsize)
            self._client_queues[ws] = q
            self._client_tasks[ws] = asyncio.create_task(self._client_writer(ws, q))
        return q

    async def _client_writer(
        self, ws: ServerConnection, queue: asyncio.Queue[tuple[str, ...]]
    ) -> None:
        try:
            while True:
                frames = await queue.get()
                try:
                    for frame in frames:
                        await asyncio.wait_for(ws.send(frame), timeout=self._send_timeout_s)
                finally:
                    queue.task_done()
        except OSError, ConnectionClosed, TimeoutError:
            self._drop_client_sync(ws, close=True)
        except asyncio.CancelledError:
            raise

    async def stop(self) -> None:
        clients = list(self._clients)
        tasks = list(self._client_tasks.values()) + list(self._close_tasks)
        self._clients.clear()
        self._client_queues.clear()
        self._client_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for ws in clients:
            with contextlib.suppress(OSError, ConnectionClosed):
                await ws.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _extract_token(self, request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        qs = parse_qs(urlsplit(request.path).query)
        vals = qs.get("token")
        return vals[0] if vals else ""

    def _text_response(self, status: int, reason: str, content_type: str, body: bytes) -> Response:
        headers = Headers([("Content-Type", content_type), *_SECURITY_HEADERS])
        return Response(status, reason, headers, body)

    def _process_request(self, _connection: ServerConnection, request: Request) -> Response | None:
        supplied = self._extract_token(request)
        if not supplied or not hmac.compare_digest(supplied, self._token):
            return Response(
                401,
                "Unauthorized",
                Headers(
                    [
                        ("Content-Type", "text/plain; charset=utf-8"),
                        ("Cache-Control", "no-store"),
                    ]
                ),
                b"Unauthorized\n",
            )
        parsed = urlsplit(request.path)
        path = parsed.path
        is_ws_upgrade = request.headers.get("Upgrade", "").lower() == "websocket"
        if path == "/ws" or is_ws_upgrade:
            origin = request.headers.get("Origin")
            if origin is not None:
                origin_host = (urlsplit(origin).hostname or "").lower()
                if origin_host not in _LOOPBACK_HOSTS:
                    return self._text_response(
                        403, "Forbidden", "text/plain; charset=utf-8", b"Forbidden Origin\n"
                    )
            return None
        if path in {"/", "/index.html"}:
            return self._text_response(
                200, "OK", "text/html; charset=utf-8", self._static_cache["index.html"]
            )
        if path == "/app.js":
            return self._text_response(
                200, "OK", "application/javascript; charset=utf-8", self._static_cache["app.js"]
            )
        if path == "/styles.css":
            return self._text_response(
                200, "OK", "text/css; charset=utf-8", self._static_cache["styles.css"]
            )
        if path in {"/state", "/api/state"}:
            body = json.dumps(self._state, sort_keys=True).encode("utf-8")
            return self._text_response(200, "OK", "application/json; charset=utf-8", body)
        if path == "/health":
            body = json.dumps(
                {
                    "status": "ok",
                    "health": self._state.get("health", "ok"),
                    "mode": self._state.get("mode", "proactive"),
                },
                sort_keys=True,
            ).encode("utf-8")
            return self._text_response(200, "OK", "application/json; charset=utf-8", body)
        return self._text_response(404, "Not Found", "text/plain; charset=utf-8", b"Not Found\n")

    @staticmethod
    def _build_card_frame(state: Mapping[str, Any]) -> dict[str, Any] | None:
        card = state.get("active_card")
        if not isinstance(card, Mapping):
            return None
        frame = dict(card)
        frame["type"] = "card"
        frame["session_id"] = state.get("session_id")
        frame["mode"] = state.get("mode")
        frame["health"] = state.get("health")
        frame["health_reason"] = state.get("health_reason")
        if "question" not in frame:
            frame["question"] = card.get("question") or card.get("text")
        return frame

    async def broadcast_state(
        self,
        state: UIStateSnapshot | Mapping[str, Any],
        *,
        seq: int | None = None,
    ) -> None:
        async with self._broadcast_lock:
            eff_seq = (
                seq
                if seq is not None
                else (state.seq if isinstance(state, UIStateSnapshot) else None)
            )
            if eff_seq is not None and eff_seq <= self._last_broadcast_seq:
                return
            if eff_seq is not None:
                self._last_broadcast_seq = eff_seq
            if isinstance(state, UIStateSnapshot):
                payload = state.to_dict()
            else:
                payload = {"type": "state", **dict(state)}
            self._state = payload
            card = payload.get("active_card")
            card_sid = (
                str(card["suggestion_id"])
                if isinstance(card, Mapping) and card.get("suggestion_id")
                else None
            )
            card_frame: dict[str, Any] | None = None
            if self._emit_card_events:
                if card_sid is not None and card_sid != self._last_emitted_card_id:
                    self._last_emitted_card_id = card_sid
                    card_frame = self._build_card_frame(payload)
                elif card_sid is None:
                    self._last_emitted_card_id = None
            if not self._clients:
                return
            encoded = json.dumps(payload, sort_keys=True)
            frames: tuple[str, ...] = (
                (encoded, json.dumps(card_frame, sort_keys=True))
                if card_frame is not None
                else (encoded,)
            )
            stalled: list[ServerConnection] = []
            for ws in list(self._clients):
                q = self._ensure_client_queue(ws)
                try:
                    q.put_nowait(frames)
                except asyncio.QueueFull:
                    stalled.append(ws)
            for ws in stalled:
                self._drop_client_sync(ws, close=True)
        await asyncio.sleep(0)

    def update_state(
        self,
        state: UIStateSnapshot | Mapping[str, Any],
        *,
        seq: int | None = None,
    ) -> None:
        eff_seq = (
            seq if seq is not None else (state.seq if isinstance(state, UIStateSnapshot) else None)
        )
        if eff_seq is not None and eff_seq <= self._last_broadcast_seq:
            return
        if eff_seq is not None:
            self._last_broadcast_seq = eff_seq
        if isinstance(state, UIStateSnapshot):
            self._state = state.to_dict()
        else:
            self._state = {"type": "state", **dict(state)}

    async def _send_ws_safe(self, websocket: ServerConnection, payload: Mapping[str, Any]) -> bool:
        try:
            await asyncio.wait_for(
                websocket.send(json.dumps(payload)),
                timeout=self._send_timeout_s,
            )
            return True
        except OSError, ConnectionClosed, TimeoutError:
            return False

    async def _ws_handler(self, websocket: ServerConnection) -> None:
        async with self._broadcast_lock:
            self._clients.add(websocket)
            q = self._ensure_client_queue(websocket)
            init_frames: list[str] = [json.dumps(self._state, sort_keys=True)]
            if self._emit_card_events:
                card_frame = self._build_card_frame(self._state)
                if card_frame is not None:
                    init_frames.append(json.dumps(card_frame, sort_keys=True))
            try:
                q.put_nowait(tuple(init_frames))
            except asyncio.QueueFull:
                self._drop_client_sync(websocket, close=True)
                return
        try:
            async for raw in websocket:
                if not isinstance(raw, str):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    if not await self._send_ws_safe(
                        websocket, {"type": "error", "error": "invalid_json"}
                    ):
                        break
                    continue
                if not isinstance(msg, dict):
                    if not await self._send_ws_safe(
                        websocket, {"type": "error", "error": "invalid_message"}
                    ):
                        break
                    continue
                if msg.get("type") == "ping":
                    if not await self._send_ws_safe(websocket, {"type": "pong"}):
                        break
                    continue
                if msg.get("type") == "action" and self._on_action is not None:
                    try:
                        res = self._on_action(msg)
                        if asyncio.iscoroutine(res):
                            res = await res
                        if isinstance(res, Mapping):
                            await self.broadcast_state(res)
                    except Exception as exc:
                        if not await self._send_ws_safe(
                            websocket, {"type": "error", "error": str(exc)}
                        ):
                            break
        finally:
            await self._remove_client(websocket)
