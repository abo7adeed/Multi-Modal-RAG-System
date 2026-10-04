from typing import Any, Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str

    document_id: str

    content_type: Literal["text", "image", "multimodal"]

    text: str | None = None

    image_path: str | None = None

    page_number: int | None = None

    parent_chunk_id: str | None = None

    related_chunk_ids: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)