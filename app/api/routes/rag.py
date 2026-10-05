import json
import logging
from collections.abc import AsyncIterator, Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.api.dependencies import AppState, get_rag_state
from app.api.services import MediaService
from app.api.schemas import (
    HealthResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SourceResponse,
)
from app.errors import (
    ImageVisionUnsupportedError,
    InvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RAGPipelineError,
    RetrievalError,
)
from app.generation.conversation import ConversationTurn
from app.generation.schemas import RAGStreamEvent, Source

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Application error -> (HTTP status, log level)
_ERROR_STATUS = {
    InvalidRequestError: 400,
    # The model cannot read images, so the request cannot be served
    # as asked. Reported as a client error with an actionable message
    # rather than a generic outage.
    ImageVisionUnsupportedError: 400,
    RetrievalError: 503,
    ProviderRateLimitError: 429,
    ProviderTimeoutError: 504,
}


def _error_response(
    error: RAGPipelineError,
) -> JSONResponse:
    status = 503
    for error_type, error_status in _ERROR_STATUS.items():
        if isinstance(error, error_type):
            status = error_status
            break

    logger.error(
        "api_request_failed error_type=%s status=%d message=%s",
        type(error).__name__,
        status,
        error,
    )

    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": error.user_message,
                "type": type(error).__name__,
            }
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _history(
    payload: RAGQueryRequest,
) -> list[ConversationTurn]:
    """
    Convert the request's history into the pipeline's model.

    The API schema is the validation boundary (role, length, count),
    so no further checking is needed here.
    """
    return [
        ConversationTurn(role=turn.role, content=turn.content)
        for turn in payload.history
    ]


def _source_response(
    source: Source,
    media: MediaService,
) -> SourceResponse:
    """
    Map an internal source onto its API representation.

    Shared by the buffered and streamed routes so both report
    citations identically.
    """
    image_url = media.resolve_image_url(source.image_path)

    # Expose the stored path only when the image is actually
    # servable through the API; never leak arbitrary filesystem
    # locations to clients.
    servable_path = source.image_path if image_url else None

    return SourceResponse(
        chunk_id=source.chunk_id,
        page=source.page,
        content_type=source.content_type,
        source=source.source,
        image_path=servable_path,
        image_url=image_url,
        snippet=source.snippet,
        score=source.score,
        kind=source.kind,
    )


# NOTE: this route is a *sync* def on purpose. Retrieval + generation
# (CLIP embedding and a model call that may take minutes) are
# blocking CPU/IO work. Declaring it async would run that work on the
# event loop and freeze every other request - including /health -
# behind it. FastAPI runs sync routes in its worker threadpool, so
# concurrent queries overlap instead of queueing.
@router.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(
    payload: RAGQueryRequest,
    request: Request,
    rag: AppState = Depends(get_rag_state),
) -> RAGQueryResponse | JSONResponse:
    logger.info(
        "rag_query_received query=%r with_image=%s",
        " ".join(payload.query.split())[:120],
        payload.image is not None,
    )

    try:
        attachment = (
            rag.attachment_service.decode(
                filename=payload.image.filename,
                media_type=payload.image.media_type,
                data=payload.image.data,
            )
            if payload.image
            else None
        )

        response = rag.pipeline.run(
            payload.query,
            image=attachment,
            history=_history(payload),
        )
    except RAGPipelineError as error:
        return _error_response(error)

    provider_info = rag.pipeline.describe_provider()

    logger.info(
        "rag_query_completed provider=%s model=%s sources=%d",
        provider_info.get("provider"),
        provider_info.get("model"),
        len(response.sources),
    )

    media = rag.media_service

    # Retrieval returns sources best-first (RRF primary results,
    # then related chunks), so the strongest matches are kept. A
    # request may ask for more, up to the schema's cap.
    max_sources = payload.max_sources or settings.api_max_sources

    ranked = response.sources[:max_sources]

    logger.info(
        "rag_query_sources returned=%d available=%d",
        len(ranked),
        len(response.sources),
    )

    sources = [
        _source_response(source, media)
        for source in ranked
    ]

    return RAGQueryResponse(
        answer=response.answer,
        sources=sources,
    )


# ------------------------------------------------------------
# SSE streaming
# ------------------------------------------------------------

#: Media type for the streamed answer. `text/event-stream` tells the
#: browser to hand us the response incrementally instead of buffering
#: it until the connection closes, which is the whole point.
SSE_MEDIA_TYPE = "text/event-stream"


def _sse_frame(
    event: str,
    data: dict | None = None,
) -> str:
    """
    Serialise one Server-Sent Events frame.

    `data` is JSON-encoded, so newlines inside a token can never be
    mistaken for the blank line that terminates a frame. Clients parse
    on the double newline, and `json.loads` handles the rest.
    """
    payload = json.dumps(data or {}, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_event_frame(
    event: RAGStreamEvent,
    media: MediaService,
    max_sources: int,
) -> str:
    if event.type == "sources":
        return _sse_frame(
            "sources",
            {
                "sources": [
                    _source_response(source, media).model_dump()
                    for source in event.sources[:max_sources]
                ]
            },
        )

    if event.type == "token":
        return _sse_frame("token", {"text": event.text})

    return _sse_frame("done", {})


# NOTE: like rag_query, this route is a sync def. run_stream() is a
# lazy generator that performs blocking retrieval and generation; the
# sync route runs it in FastAPI's worker threadpool, and the async
# response generator below only marshals its events, so the event loop
# stays free for other requests and for draining this stream.
@router.post("/rag/stream")
def rag_query_stream(
    payload: RAGQueryRequest,
    rag: AppState = Depends(get_rag_state),
) -> StreamingResponse:
    logger.info(
        "rag_stream_received query=%r with_image=%s",
        " ".join(payload.query.split())[:120],
        payload.image is not None,
    )

    try:
        attachment = (
            rag.attachment_service.decode(
                filename=payload.image.filename,
                media_type=payload.image.media_type,
                data=payload.image.data,
            )
            if payload.image
            else None
        )
    except RAGPipelineError as error:
        return _error_response(error)

    try:
        # Called eagerly (not iterated here) so validation, image
        # decoding and retrieval failures still get a real HTTP
        # status. run_stream() returns a generator; only generation
        # is deferred.
        events = rag.pipeline.run_stream(
            payload.query,
            image=attachment,
            history=_history(payload),
        )
    except RAGPipelineError as error:
        return _error_response(error)

    media = rag.media_service
    max_sources = payload.max_sources or settings.api_max_sources

    return StreamingResponse(
        _event_stream(events, media, max_sources),
        media_type=SSE_MEDIA_TYPE,
        headers={
            # Proxies that buffer would defeat streaming entirely.
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Tell nginx not to buffer the body.
            "X-Accel-Buffering": "no",
        },
    )


async def _event_stream(
    events: Iterator[RAGStreamEvent],
    media: MediaService,
    max_sources: int,
) -> AsyncIterator[str]:
    """
    Adapt the sync event generator to an async SSE body.

    Failures after the response has started cannot change the HTTP
    status, so they are reported as an `error` event instead. The
    client is told explicitly, rather than being left with a stream
    that simply stops mid-sentence.
    """
    try:
        async for event in iterate_in_threadpool(events):
            yield _stream_event_frame(event, media, max_sources)

    except RAGPipelineError as error:
        logger.error(
            "rag_stream_failed error_type=%s message=%s",
            type(error).__name__,
            error,
        )
        yield _sse_frame(
            "error",
            {
                "message": error.user_message,
                "type": type(error).__name__,
            },
        )
