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
from app.api.schemas import DocumentUploadResponse
from app.errors import (
    DocumentIngestionError,
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
