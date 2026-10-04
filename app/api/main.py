"""
FastAPI application factory for the multimodal RAG system.

The API layer contains no RAG business logic: routes validate input,
delegate to the pipeline/services, and translate application errors
into HTTP responses with user-facing messages.
"""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.dependencies import lifespan
from app.api.routes import documents_router, rag_router
from app.api.services import MediaService
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Modal RAG System",
        version="0.1.0",
        description=(
            "Multimodal retrieval-augmented generation over "
            "text and images extracted from documents."
        ),
        lifespan=lifespan,
    )

    # ----------------------------------------------------------
    # CORS (environment-configured origins)
    # ----------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------
    # Routers
    # ----------------------------------------------------------
    app.include_router(rag_router, prefix=API_PREFIX)
    app.include_router(documents_router, prefix=API_PREFIX)

    # ----------------------------------------------------------
    # Media serving (retrieved images)
    #
    # Files are resolved through MediaService, which rejects any
    # path escaping the processed-images directory. MediaService is
    # read-only, so it is safe to construct per request.
    # ----------------------------------------------------------
    media_prefix = settings.api_media_url_prefix

    @app.get(
        media_prefix + "/{file_path:path}",
        include_in_schema=False,
        response_model=None,
    )
    async def serve_media(
        request: Request, file_path: str
    ) -> FileResponse | JSONResponse:
        rag_state = getattr(request.app.state, "rag", None)
        media = (
            rag_state.media_service
            if rag_state is not None
            else MediaService()
        )

        safe_path = media.resolve_safe_path(file_path)

        if safe_path is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Media not found."},
            )

        return FileResponse(safe_path)

    # ----------------------------------------------------------
    # Fallback exception handlers
    # ----------------------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "unhandled_exception path=%s error_type=%s",
            request.url.path,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": (
                        "An unexpected error occurred. Please try again."
                    ),
                    "type": "InternalServerError",
                }
            },
        )

    return app


app = create_app()


def main() -> None:
    """Entrypoint: python -m app.api.main"""
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
