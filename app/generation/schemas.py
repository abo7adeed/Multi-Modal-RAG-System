from typing import Any, Literal

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
    # Retrieval rank score (RRF for fused results; None for chunks
    # pulled in purely as related context).
    score: float | None = None
    # How this source relates to the answer:
    #   "document" - an indexed chunk that grounds the answer
    #   "attachment" - the user's own image (never emitted as a source)
    #   "visual_match" - a lookalike page image used only as context
    kind: str = "document"


class RAGResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)


class RAGStreamEvent(BaseModel):
    """
    One event in a streamed RAG answer.

    The streaming API emits these instead of a RAGResponse:

      - "sources": the citations, sent before any token so the client
        can render them while the answer is still being written.
      - "token": a fragment of the answer. Clients append these.
      - "done": the answer finished; no more tokens are coming.

    Failures are not an event: they raise, and the API layer converts
    the exception into an SSE `error` event. Keeping errors out of the
    event union means a client can treat "stream ended without `done`"
    as a failure without inspecting message text.
    """

    type: Literal["sources", "token", "done"]
    text: str | None = None
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
    # Recent turns rendered as text, for resolving follow-up questions.
    # None for a single-turn request, so providers omit the section
    # entirely rather than sending an empty header.
    conversation: str | None = None