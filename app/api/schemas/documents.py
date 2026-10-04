from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str
    chunks_indexed: int = Field(ge=0)
