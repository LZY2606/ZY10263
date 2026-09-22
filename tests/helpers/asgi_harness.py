"""ASGI event harness for driving the middleware without a real server.

The harness plays the role of both the ASGI server (it feeds `receive`
messages and records `send` messages) and the clock-free scheduler (all
interleaving is fixed with anyio events, never with sleeps). Scripted apps
are plain async callables that use the provided `receive`/`send` callables,
so every test exercises the real
`prometheus_fastapi_instrumentator.middleware.PrometheusInstrumentatorMiddleware`.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Union

from starlette.types import Message, Receive, Scope, Send

from prometheus_fastapi_instrumentator import metrics
from prometheus_fastapi_instrumentator.middleware import (
    PrometheusInstrumentatorMiddleware,
)


class InfoRecorder:
    """Instrumentation closure that records every `metrics.Info` it gets."""

    def __init__(self) -> None:
        self.infos: List[metrics.Info] = []

    def __call__(self, info: metrics.Info) -> None:
        self.infos.append(info)


class ASGIEventHarness:
    """Drives one middleware instance with a scripted ASGI event stream."""

    def __init__(self, middleware: PrometheusInstrumentatorMiddleware) -> None:
        self.middleware = middleware
        self.incoming: "asyncio.Queue[Union[Message, BaseException]]" = asyncio.Queue()
        self.sent: List[Message] = []
        self.gauge_samples: List[Optional[float]] = []
        self.send_failures: Dict[int, BaseException] = {}

    # -- server-side scripting -------------------------------------------------

    def push(self, message: Message) -> None:
        """Queues a message the scripted app will get from `receive`."""
        self.incoming.put_nowait(message)

    def push_request_body(self, body: bytes, more_body: bool = False) -> None:
        self.push(
            {"type": "http.request", "body": body, "more_body": more_body}
        )

    def push_disconnect(self) -> None:
        self.push({"type": "http.disconnect"})

    def push_receive_error(self, exc: BaseException) -> None:
        """Makes the next `receive` call raise instead of returning a message."""
        self.incoming.put_nowait(exc)

    def fail_send_on(self, index: int, exc: BaseException) -> None:
        """Makes the n-th `send` call (0-based) raise after recording."""
        self.send_failures[index] = exc

    # -- middleware-facing callables -------------------------------------------

    def _gauge_value(self) -> Optional[float]:
        gauge = self.middleware.inprogress
        if gauge is None:
            return None
        return float(gauge._value.get())

    async def receive(self) -> Message:
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, message: Message) -> None:
        index = len(self.sent)
        self.sent.append(message)
        self.gauge_samples.append(self._gauge_value())
        if index in self.send_failures:
            raise self.send_failures[index]

    # -- driving ---------------------------------------------------------------

    async def run(self, app, scope: Scope) -> None:
        """Runs the middleware once; exceptions propagate to the caller."""
        await self.middleware(scope, self.receive, self.send)

    def sent_types(self) -> List[str]:
        return [message["type"] for message in self.sent]

    def sent_body(self) -> bytes:
        return b"".join(
            message.get("body", b"")
            for message in self.sent
            if message["type"] == "http.response.body"
        )


def http_scope(
    path: str = "/items/42",
    method: str = "GET",
    app=None,
    extensions: Optional[dict] = None,
) -> Scope:
    """Builds a minimal HTTP scope, optionally bound to an app for routing."""
    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "root_path": "",
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    if app is not None:
        scope["app"] = app
    if extensions:
        scope["extensions"] = extensions
    return scope
