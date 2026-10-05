from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str
    chunks_indexed: int = Field(ge=0)


class DocumentDeleteResponse(BaseModel):
    """
    What a delete removed.

    `chunks_deleted` is reported rather than assumed: a client that
    re-indexes needs to know how much of the index actually changed.
    """

    document_id: str
    chunks_deleted: int = Field(ge=1)
