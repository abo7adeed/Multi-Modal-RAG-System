from typing import Any

from pydantic import BaseModel, Field


class ImageAttachment(BaseModel):
    """
    An image the user attached to their own question.

    Unlike a retrieved page image, this one is the *subject* of the
    question: the model answers about it directly, and the index is
    only used to add supporting context.

    Encoding is a transport concern (base64 in JSON). By the time an
    attachment reaches the pipeline it holds raw bytes.
    """

    filename: str = "attachment"
    media_type: str = "image/jpeg"
    data: bytes = b""


class Source(BaseModel):
    chunk_id: str | None = None
    page: int | None = None
    content_type: str | None = None
    source: str | None = None
    image_path: str | None = None
    # Short preview of the matched text, so clients can show *why*
    # a source was retrieved and distinguish chunks from the same
    # page. None for image sources.
    snippet: str | None = None


class RAGResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)


class GenerationContext(BaseModel):
    """
    Structured context handed to the generation layer.

    Modelled as a type rather than a bare dict because its shape is a
    contract shared by the context builder, the generator and every
    provider.
    """

    query: str
    text_context: list[dict[str, Any]] = Field(default_factory=list)
    image_context: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)