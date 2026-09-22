"""Event-scripted ASGI harness tests for the Prometheus middleware.

These tests drive `PrometheusInstrumentatorMiddleware` directly with scripted
ASGI event streams (chunked request bodies, disconnects, multi-chunk
responses, send failures, cancellation) without any real server, network, or
clock-based waiting. Interleaving is fixed with anyio events. Every script
asserts that metrics are recorded exactly once and that the in-progress
gauge returns to zero so per-request state stays recyclable.

Where behavior is ambiguous (e.g. whether duration is still observed after a
client disconnect), the tests lock in the current implementation's behavior.
"""

import asyncio

import anyio
import pytest
from fastapi import FastAPI
from helpers import utils
from helpers.asgi_harness import ASGIEventHarness, InfoRecorder, http_scope

from prometheus_fastapi_instrumentator.middleware import (
    PrometheusInstrumentatorMiddleware,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    utils.reset_collectors()


def _routing_app() -> FastAPI:
    """App whose routes only serve templated handler resolution."""

    app = FastAPI()

    @app.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item_id": item_id}

    @app.get("/ignore")
    def read_ignore():
        return {"message": "ignored"}

    return app


def _build(app, recorder: InfoRecorder, extra_instrumentations=(), **kwargs):
    kwargs.setdefault("should_instrument_requests_inprogress", True)
    middleware = PrometheusInstrumentatorMiddleware(
        app,
        instrumentations=[recorder, *extra_instrumentations],
        **kwargs,
    )
    return ASGIEventHarness(middleware)


def _start(status: int = 200) -> dict:
    return {"type": "http.response.start", "status": status, "headers": []}


def _body(body: bytes, more_body: bool = False) -> dict:
    return {"type": "http.response.body", "body": body, "more_body": more_body}


async def _ok_app(scope, receive, send):
    await receive()
    await send(_start(200))
    await send(_body(b"ok"))


# ------------------------------------------------------------------------------
# Streaming happy paths


async def test_chunked_request_body_and_streamed_response():
    recorder = InfoRecorder()

    received = []

    async def app(scope, receive, send):
        received.append(await receive())
        received.append(await receive())
        await send(_start(200))
        await send(_body(b"he", more_body=True))
        await send(_body(b"ll", more_body=True))
        await send(_body(b"o"))

    harness = _build(app, recorder)
    harness.push_request_body(b"re", more_body=True)
    harness.push_request_body(b"quest")

    await harness.run(app, http_scope(app=_routing_app()))

    assert [m["type"] for m in received] == ["http.request", "http.request"]
    assert received[0]["more_body"] is True
    assert harness.sent_types() == ["http.response.start"] + [
        "http.response.body"
    ] * 3
    assert harness.sent_body() == b"hello"

    assert len(recorder.infos) == 1
    info = recorder.infos[0]
    assert info.modified_handler == "/items/{item_id}"
    assert info.modified_status == "2xx"
    assert info.modified_duration >= 0.0

    assert harness.gauge_samples == [1.0] * 4
    assert harness._gauge_value() == 0.0


async def test_empty_chunk_and_duplicate_final_body_aggregated_as_sent():
    recorder = InfoRecorder()

    async def app(scope, receive, send):
        await receive()
        await send(_start(200))
        await send(_body(b"ab", more_body=True))
        await send(_body(b"", more_body=True))  # empty intermediate chunk
        await send(_body(b"cd"))
        await send(_body(b"cd"))  # misbehaving duplicate final body

    harness = _build(app, recorder, body_handlers=["/items.*"])
    harness.push_request_body(b"")

    await harness.run(app, http_scope(app=_routing_app()))

    assert len(recorder.infos) == 1
    # Aggregation is exactly what was sent: empty chunks add nothing and the
    # duplicated final chunk is counted as sent (current behavior locked).
    assert recorder.infos[0].response.body == b"abcdcd"
    assert harness.sent_body() == b"abcdcd"
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Client disconnects (duration still observed: current behavior locked)


async def test_disconnect_before_response_start_still_observed():
    recorder = InfoRecorder()

    async def app(scope, receive, send):
        await receive()
        assert await receive() == {"type": "http.disconnect"}
        await send(_start(200))
        await send(_body(b"late"))

    harness = _build(app, recorder)
    harness.push_request_body(b"")
    harness.push_disconnect()

    await harness.run(app, http_scope(app=_routing_app()))

    assert len(recorder.infos) == 1
    assert recorder.infos[0].modified_status == "2xx"
    assert recorder.infos[0].modified_duration >= 0.0
    assert harness._gauge_value() == 0.0


async def test_disconnect_during_response_stream_still_observed():
    recorder = InfoRecorder()

    async def app(scope, receive, send):
        await receive()
        await send(_start(200))
        await send(_body(b"part", more_body=True))
        assert await receive() == {"type": "http.disconnect"}
        await send(_body(b"rest"))

    harness = _build(app, recorder)
    harness.push_request_body(b"")
    harness.push_disconnect()

    await harness.run(app, http_scope(app=_routing_app()))

    assert len(recorder.infos) == 1
    assert recorder.infos[0].modified_status == "2xx"
    assert harness.sent_body() == b"partrest"
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Handler failures before vs after response start


async def test_handler_raises_before_response_start():
    recorder = InfoRecorder()
    app_error = RuntimeError("boom before start")

    async def app(scope, receive, send):
        await receive()
        raise app_error

    harness = _build(app, recorder)
    harness.push_request_body(b"")

    with pytest.raises(RuntimeError) as exc_info:
        await harness.run(app, http_scope(app=_routing_app()))

    assert exc_info.value is app_error
    assert harness.sent == []
    assert len(recorder.infos) == 1
    assert recorder.infos[0].modified_status == "5xx"
    assert harness._gauge_value() == 0.0


async def test_handler_raises_after_response_start():
    recorder = InfoRecorder()
    app_error = RuntimeError("boom after start")

    async def app(scope, receive, send):
        await receive()
        await send(_start(201))
        raise app_error

    harness = _build(app, recorder)
    harness.push_request_body(b"")

    with pytest.raises(RuntimeError) as exc_info:
        await harness.run(app, http_scope(app=_routing_app()))

    assert exc_info.value is app_error
    assert harness.sent_types() == ["http.response.start"]
    assert len(recorder.infos) == 1
    # Status was already captured from the response start message.
    assert recorder.infos[0].modified_status == "2xx"
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Send failures


async def test_send_failure_on_response_start():
    recorder = InfoRecorder()
    send_error = ConnectionError("client gone on start")

    async def app(scope, receive, send):
        await receive()
        await send(_start(200))

    harness = _build(app, recorder)
    harness.push_request_body(b"")
    harness.fail_send_on(0, send_error)

    with pytest.raises(ConnectionError) as exc_info:
        await harness.run(app, http_scope(app=_routing_app()))

    assert exc_info.value is send_error
    assert len(recorder.infos) == 1
    # send_wrapper captured the status before the failed send (locked).
    assert recorder.infos[0].modified_status == "2xx"
    assert harness._gauge_value() == 0.0


async def test_send_failure_mid_stream_keeps_aggregated_body():
    recorder = InfoRecorder()
    send_error = ConnectionError("client gone mid stream")

    async def app(scope, receive, send):
        await receive()
        await send(_start(200))
        await send(_body(b"aaa", more_body=True))
        await send(_body(b"bbb"))

    harness = _build(app, recorder, body_handlers=["/items.*"])
    harness.push_request_body(b"")
    harness.fail_send_on(2, send_error)

    with pytest.raises(ConnectionError) as exc_info:
        await harness.run(app, http_scope(app=_routing_app()))

    assert exc_info.value is send_error
    assert len(recorder.infos) == 1
    # The failed chunk was aggregated before send raised (locked).
    assert recorder.infos[0].response.body == b"aaabbb"
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Receive failures


async def test_receive_raises_mid_request():
    recorder = InfoRecorder()
    receive_error = OSError("socket blew up")

    async def app(scope, receive, send):
        await receive()
        await receive()  # raises

    harness = _build(app, recorder)
    harness.push_request_body(b"first", more_body=True)
    harness.push_receive_error(receive_error)

    with pytest.raises(OSError) as exc_info:
        await harness.run(app, http_scope(app=_routing_app()))

    assert exc_info.value is receive_error
    assert harness.sent == []
    assert len(recorder.infos) == 1
    assert recorder.infos[0].modified_status == "5xx"
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Cancellation (no sleeps; interleaving fixed with anyio events)


async def test_cancellation_before_response_start():
    recorder = InfoRecorder()
    app_blocked = anyio.Event()
    never = anyio.Event()

    async def app(scope, receive, send):
        await receive()
        app_blocked.set()
        await never.wait()

    harness = _build(app, recorder)
    harness.push_request_body(b"")

    task = asyncio.create_task(harness.run(app, http_scope(app=_routing_app())))
    await app_blocked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(recorder.infos) == 1
    assert recorder.infos[0].modified_status == "5xx"
    assert harness._gauge_value() == 0.0


async def test_cancellation_during_response_stream():
    recorder = InfoRecorder()
    streaming = anyio.Event()
    never = anyio.Event()

    async def app(scope, receive, send):
        await receive()
        await send(_start(200))
        await send(_body(b"chunk", more_body=True))
        streaming.set()
        await never.wait()

    harness = _build(app, recorder)
    harness.push_request_body(b"")

    task = asyncio.create_task(harness.run(app, http_scope(app=_routing_app())))
    await streaming.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(recorder.infos) == 1
    assert recorder.infos[0].modified_status == "2xx"
    assert harness.sent_body() == b"chunk"
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Metric callback failures


async def test_metric_callback_raises_preserves_app_exception_context():
    recorder = InfoRecorder()
    app_error = RuntimeError("app exploded")
    metric_error = ValueError("metric exploded")

    def failing_instrumentation(info):
        raise metric_error

    async def app(scope, receive, send):
        await receive()
        raise app_error

    harness = _build(app, recorder, extra_instrumentations=[failing_instrumentation])
    harness.push_request_body(b"")

    with pytest.raises(ValueError) as exc_info:
        await harness.run(app, http_scope(app=_routing_app()))

    assert exc_info.value is metric_error
    # The original handler exception survives as the cause/context chain.
    assert exc_info.value.__context__ is app_error
    assert len(recorder.infos) == 1
    assert harness._gauge_value() == 0.0


async def test_metric_callback_raises_after_clean_response():
    recorder = InfoRecorder()
    metric_error = ValueError("metric exploded")

    def failing_instrumentation(info):
        raise metric_error

    harness = _build(
        _ok_app, recorder, extra_instrumentations=[failing_instrumentation]
    )
    harness.push_request_body(b"")

    with pytest.raises(ValueError) as exc_info:
        await harness.run(_ok_app, http_scope(app=_routing_app()))

    assert exc_info.value is metric_error
    assert exc_info.value.__context__ is None
    assert len(recorder.infos) == 1
    assert recorder.infos[0].modified_status == "2xx"
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Handler resolution: templated / untemplated / excluded


async def test_templated_vs_grouped_untemplated_handler():
    templated_recorder = InfoRecorder()
    harness = _build(_ok_app, templated_recorder)
    harness.push_request_body(b"")
    await harness.run(_ok_app, http_scope(app=_routing_app()))
    assert [i.modified_handler for i in templated_recorder.infos] == [
        "/items/{item_id}"
    ]

    untemplated_recorder = InfoRecorder()
    utils.reset_collectors()
    harness = _build(_ok_app, untemplated_recorder, should_group_untemplated=True)
    harness.push_request_body(b"")
    await harness.run(_ok_app, http_scope())  # no app in scope: untemplated
    assert [i.modified_handler for i in untemplated_recorder.infos] == ["none"]
    assert harness._gauge_value() == 0.0


async def test_ignored_untemplated_records_nothing():
    recorder = InfoRecorder()
    harness = _build(_ok_app, recorder, should_ignore_untemplated=True)
    harness.push_request_body(b"")

    await harness.run(_ok_app, http_scope())

    assert recorder.infos == []
    assert harness.sent_types() == ["http.response.start", "http.response.body"]
    assert harness.gauge_samples == [0.0, 0.0]
    assert harness._gauge_value() == 0.0


async def test_excluded_handler_records_nothing():
    recorder = InfoRecorder()
    harness = _build(_ok_app, recorder, excluded_handlers=["/ignore"])
    harness.push_request_body(b"")

    await harness.run(_ok_app, http_scope(path="/ignore", app=_routing_app()))

    assert recorder.infos == []
    assert harness.sent_types() == ["http.response.start", "http.response.body"]
    assert harness.gauge_samples == [0.0, 0.0]
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Status grouping and latency rounding options


async def test_grouped_vs_ungrouped_status():
    grouped = InfoRecorder()
    harness = _build(_ok_app, grouped, should_group_status_codes=True)
    harness.push_request_body(b"")
    await harness.run(_ok_app, http_scope(app=_routing_app()))
    assert [i.modified_status for i in grouped.infos] == ["2xx"]

    ungrouped = InfoRecorder()
    utils.reset_collectors()
    harness = _build(_ok_app, ungrouped, should_group_status_codes=False)
    harness.push_request_body(b"")
    await harness.run(_ok_app, http_scope(app=_routing_app()))
    assert [i.modified_status for i in ungrouped.infos] == ["200"]


async def test_round_latency_decimals():
    recorder = InfoRecorder()
    harness = _build(
        _ok_app,
        recorder,
        should_round_latency_decimals=True,
        round_latency_decimals=3,
    )
    harness.push_request_body(b"")

    await harness.run(_ok_app, http_scope(app=_routing_app()))

    assert len(recorder.infos) == 1
    info = recorder.infos[0]
    assert info.modified_duration == round(info.modified_duration, 3)
    assert info.modified_duration_without_streaming == round(
        info.modified_duration_without_streaming, 3
    )
    assert 0.0 <= info.modified_duration_without_streaming
    assert info.modified_duration_without_streaming <= info.modified_duration
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# Background work is not part of the response body


async def test_background_task_not_part_of_response_body():
    recorder = InfoRecorder()
    events = []

    def metric_recorded(info):
        events.append("metric-recorded")

    async def app(scope, receive, send):
        await receive()
        await send(_start(200))
        await send(_body(b"payload"))
        # Background work runs after the final body, inside the app call.
        events.append("background-done")

    harness = _build(
        app,
        recorder,
        extra_instrumentations=[metric_recorded],
        body_handlers=["/items.*"],
    )
    harness.push_request_body(b"")

    await harness.run(app, http_scope(app=_routing_app()))

    assert harness.sent_body() == b"payload"
    assert len(recorder.infos) == 1
    assert recorder.infos[0].response.body == b"payload"
    # Metrics are recorded only after the background work finished.
    assert events == ["background-done", "metric-recorded"]
    assert harness._gauge_value() == 0.0


# ------------------------------------------------------------------------------
# State recyclability across sequential requests


async def test_state_recoverable_across_sequential_requests():
    recorder = InfoRecorder()
    harness = _build(_ok_app, recorder)

    for _ in range(2):
        harness.push_request_body(b"")
        await harness.run(_ok_app, http_scope(app=_routing_app()))
        assert harness._gauge_value() == 0.0

    assert len(recorder.infos) == 2
    assert all(info.modified_status == "2xx" for info in recorder.infos)
    assert harness.gauge_samples == [1.0] * 4
