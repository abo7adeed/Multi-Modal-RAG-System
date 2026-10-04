from pydantic import BaseModel, Field

from app.pipeline import MAX_QUERY_LENGTH


class ImageAttachmentRequest(BaseModel):
    """
    An image the user attached to their question, carried inline so
    the chat stays a single JSON request.

    Validation (real image, size limits) happens in
    AttachmentService: a Pydantic model cannot tell a PNG from a PDF
    renamed to .png.
    """

    filename: str = Field(
        default="attachment",
        max_length=255,
        description="Original filename, used for display only.",
    )
    media_type: str = Field(
        default="image/jpeg",
        max_length=100,
        description="Declared content type of the image.",
    )
    data: str = Field(
        ...,
        min_length=1,
        description=(
            "Base64-encoded image bytes, without a data: URL prefix."
        ),
    )


class RAGQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="User question about the indexed documents.",
        examples=["What Dell products are shown in the catalog?"],
    )
    image: ImageAttachmentRequest | None = Field(
        default=None,
        description=(
            "Optional image the question is about. It is sent to the "
            "vision model directly, and similar document images are "
            "retrieved as supporting context."
        ),
    )


class SourceResponse(BaseModel):
    chunk_id: str | None = None
    page: int | None = None
    content_type: str | None = None
    source: str | None = None
    image_path: str | None = None
    # URL of the image served by this API (null for text sources).
    image_url: str | None = None
    # Matched text preview (null for image sources).
    snippet: str | None = None


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
