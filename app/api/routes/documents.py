import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from app.api.dependencies import AppState, get_rag_state
from app.api.schemas import (
    DocumentDeleteResponse,
    DocumentUploadResponse,
)
from app.errors import (
    DocumentIngestionError,
    DocumentNotFoundError,
    InvalidRequestError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF or image document."),
    rag: AppState = Depends(get_rag_state),
) -> DocumentUploadResponse:
    try:
        result = await rag.ingestion_service.ingest_upload(file)
    except InvalidRequestError as error:
        raise HTTPException(
            status_code=400,
            detail=error.user_message,
        ) from error
    except DocumentIngestionError as error:
        raise HTTPException(
            status_code=422,
            detail=error.user_message,
        ) from error

    logger.info(
        "document_uploaded filename=%s chunks=%d",
        result["filename"],
        result["chunks_indexed"],
    )

    return DocumentUploadResponse(**result)


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=200,
)
async def delete_document(
    document_id: str,
    rag: AppState = Depends(get_rag_state),
) -> DocumentDeleteResponse:
    """
    Remove a document and every chunk indexed from it.

    404 rather than 200-with-zero when the id is unknown: a client
    retrying a delete must be able to tell "already gone" from
    "never existed".
    """
    try:
        result = await rag.ingestion_service.delete_document(document_id)
    except InvalidRequestError as error:
        raise HTTPException(
            status_code=400,
            detail=error.user_message,
        ) from error
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=error.user_message,
        ) from error

    return DocumentDeleteResponse(**result)
