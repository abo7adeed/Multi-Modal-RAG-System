"""
Tests for the SSE streaming endpoint (POST /api/v1/rag/stream).

The frame parser below is the reference decoding of the wire format:
if it and the server agree here, a client implementing the same
rules will agree too.
"""
import json

import pytest

from app.errors import (
    GenerationUnavailableError,
    InvalidRequestError,
    ProviderRateLimitError,
)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """
    Decode an SSE body into [(event, data), ...].

    Frames are separated by a blank line; `data` is JSON. Unknown
    fields are ignored, as a real client would.
    """
    frames = []

    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        event = None
        data_lines = []

        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())

        if event is None:
            continue

        payload = json.loads("\n".join(data_lines)) if data_lines else {}
        frames.append((event, payload))

    return frames


def test_stream_returns_event_stream_media_type(rag_client):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    # Proxies that buffer would defeat streaming entirely.
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"


def test_stream_emits_sources_then_tokens_then_done(
    rag_client, fake_pipeline
):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )

    frames = parse_sse(response.text)
    events = [event for event, _ in frames]

    assert events[0] == "sources"
    assert events[-1] == "done"
    assert events[1:-1] == ["token"] * len(fake_pipeline.stream_chunks or
                                          ["Dell ", "laptops ", "and ",
                                           "desktops."])


def test_stream_token_events_reconstruct_the_answer(
    rag_client, fake_pipeline
):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )

    answer = "".join(
        payload["text"]
        for event, payload in parse_sse(response.text)
        if event == "token"
    )

    # The streamed answer is the concatenation of every token, in
    # order - which is what a client renders.
    assert answer == "Dell laptops and desktops."


def test_stream_sources_carry_the_same_payload_as_the_buffered_route(
    rag_client,
):
    streamed = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )
    buffered = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "What Dell products are in the catalog?"},
    )

    sources_event = next(
        payload
        for event, payload in parse_sse(streamed.text)
        if event == "sources"
    )
    stream_body = buffered.json()

    # The streamed route truncates sources the same way and must not
    # invent or drop fields relative to /rag/query.
    assert len(sources_event["sources"]) == len(
        stream_body["sources"]
    )
    for streamed_source, buffered_source in zip(
        sources_event["sources"], stream_body["sources"]
    ):
        assert streamed_source == buffered_source


def test_stream_respects_max_sources(rag_client):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={
            "query": "What Dell products are in the catalog?",
            "max_sources": 1,
        },
    )

    sources = next(
        payload["sources"]
        for event, payload in parse_sse(response.text)
        if event == "sources"
    )

    assert len(sources) == 1


def test_stream_passes_the_query_through(rag_client, fake_pipeline):
    rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "Which laptop has an Intel Core i7?"},
    )

    # The fake's run_stream delegates to run(), which records the
    # query; what matters is that the route forwarded it verbatim.
    assert "Which laptop has an Intel Core i7?" in fake_pipeline.calls


def test_stream_forwards_the_attachment(rag_client, fake_pipeline):
    import base64 as _base64
    import io as _io

    from PIL import Image

    buffer = _io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="JPEG")

    response = rag_client.post(
        "/api/v1/rag/stream",
        json={
            "query": "What is in this picture?",
            "image": {
                "filename": "photo.jpg",
                "media_type": "image/jpeg",
                "data": _base64.b64encode(
                    buffer.getvalue()
                ).decode(),
            },
        },
    )

    assert response.status_code == 200
    assert fake_pipeline.images[0] is not None


# --------------------------------------------------------------
# Validation: still a real HTTP status before the stream starts
# --------------------------------------------------------------
def test_stream_rejects_empty_query(rag_client):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_stream_rejects_missing_query(rag_client):
    response = rag_client.post("/api/v1/rag/stream", json={})

    assert response.status_code == 422


def test_stream_rejects_whitespace_query_with_a_real_status(
    rag_client, fake_pipeline
):
    fake_pipeline.stream_outcome = InvalidRequestError(
        "Query must not be empty."
    )

    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "   "},
    )

    # run_stream() is called eagerly by the route, so this is still a
    # real HTTP 400 rather than a 200 carrying an error event.
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "InvalidRequestError"


def test_stream_rejects_invalid_base64_image(rag_client):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={
            "query": "What is this?",
            "image": {
                "filename": "photo.jpg",
                "media_type": "image/jpeg",
                "data": "not-valid-base64!!!",
            },
        },
    )

    assert response.status_code == 400


# --------------------------------------------------------------
# Mid-stream failures
# --------------------------------------------------------------
def test_stream_reports_a_mid_stream_failure_as_an_error_event(
    rag_client, fake_pipeline
):
    """
    A failure after the response started cannot change the HTTP
    status, so it is reported as an `error` event.

    Without it the client would see the stream simply stop and could
    not tell a truncated answer from a complete one.
    """
    fake_pipeline.stream_mid_error = GenerationUnavailableError(
        "provider exploded"
    )

    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )

    frames = parse_sse(response.text)
    events = [event for event, _ in frames]

    # Sources and the first token were already delivered, so the
    # response is a 200 and the failure has to travel as an event.
    assert response.status_code == 200
    assert events == ["sources", "token", "error"]

    error = frames[-1][1]
    assert "exploded" not in error["message"]
    assert error["type"] == "GenerationUnavailableError"


def test_stream_error_event_does_not_look_like_a_done(
    rag_client, fake_pipeline
):
    """
    The distinguishing signal for clients is the absence of `done`,
    so an error stream must not also emit one.
    """
    fake_pipeline.stream_mid_error = ProviderRateLimitError(
        "slow down"
    )

    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )

    events = [event for event, _ in parse_sse(response.text)]

    # A client detects truncation by the absence of `done`, so an
    # error stream must not also claim the answer finished.
    assert "done" not in events
    assert events[-1] == "error"


def test_stream_token_containing_a_newline_survives_the_wire(
    rag_client, fake_pipeline
):
    """
    SSE frames are newline-delimited, so a token with a newline must
    not be able to break the framing.

    JSON-encoding the payload escapes it as \\n, which is why the
    server never interpolates raw text into the frame.
    """
    fake_pipeline.stream_chunks = ["line one\nline two", "\ttabbed"]

    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )

    tokens = [
        payload["text"]
        for event, payload in parse_sse(response.text)
        if event == "token"
    ]

    assert tokens == ["line one\nline two", "\ttabbed"]


def test_stream_unicode_answer_is_not_escaped(rag_client, fake_pipeline):
    fake_pipeline.stream_chunks = ["سوق", " Dell "]

    response = rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What Dell products are in the catalog?"},
    )

    # ensure_ascii=False keeps the payload readable on the wire; the
    # JSON escaping still makes it valid UTF-8 text.
    assert "سوق" in response.text
    tokens = [
        payload["text"]
        for event, payload in parse_sse(response.text)
        if event == "token"
    ]
    assert tokens == ["سوق", " Dell "]

# --------------------------------------------------------------
# Conversation history
# --------------------------------------------------------------
def test_stream_forwards_history_to_the_pipeline(
    rag_client, fake_pipeline
):
    rag_client.post(
        "/api/v1/rag/stream",
        json={
            "query": "What about its warranty?",
            "history": [
                {"role": "user", "content": "What is the OptiPlex 3020?"},
                {"role": "assistant", "content": "A compact desktop."},
            ],
        },
    )

    history = fake_pipeline.histories[0]
    assert [turn.role for turn in history] == ["user", "assistant"]
    assert history[0].content == "What is the OptiPlex 3020?"


def test_stream_defaults_to_no_history(rag_client, fake_pipeline):
    rag_client.post(
        "/api/v1/rag/stream",
        json={"query": "What is the OptiPlex 3020?"},
    )

    assert fake_pipeline.histories[0] == []


def test_stream_rejects_an_unknown_history_role(rag_client):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={
            "query": "What about its warranty?",
            "history": [{"role": "system", "content": "Be brief."}],
        },
    )

    # Only user/assistant turns are accepted; a "system" turn would
    # let a client inject its own instructions.
    assert response.status_code == 422


def test_stream_rejects_an_oversized_history(rag_client):
    response = rag_client.post(
        "/api/v1/rag/stream",
        json={
            "query": "What about its warranty?",
            "history": [
                {"role": "user", "content": f"q{i}"} for i in range(50)
            ],
        },
    )

    assert response.status_code == 422
