from typing import Any

from pydantic import BaseModel, Field


class ImageAsset(BaseModel):
    image_id: str
    path: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageContent(BaseModel):
    page_id: str
    page_number: int
    text: str = ""
    images: list[ImageAsset] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    document_id: str
    source: str
    document_type: str
    pages: list[PageContent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)