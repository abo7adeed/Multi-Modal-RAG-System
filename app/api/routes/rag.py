import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import AppState, get_rag_state
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


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(
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
            payload.query, image=attachment
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
    # then related chunks), so the strongest matches are kept.
    ranked = response.sources[: settings.api_max_sources]

    sources = []
    for source in ranked:
        image_url = media.resolve_image_url(source.image_path)

        # Expose the stored path only when the image is actually
        # servable through the API; never leak arbitrary filesystem
        # locations to clients.
        servable_path = source.image_path if image_url else None

        sources.append(
            SourceResponse(
                chunk_id=source.chunk_id,
                page=source.page,
                content_type=source.content_type,
                source=source.source,
                image_path=servable_path,
                image_url=image_url,
                snippet=source.snippet,
            )
        )

    return RAGQueryResponse(
        answer=response.answer,
        sources=sources,
    )
