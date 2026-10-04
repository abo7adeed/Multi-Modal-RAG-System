from uuid import uuid4

from .chunk_schemas import Chunk
from .schemas import Document


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        chunks = []

        for page in document.pages:

            # --------------------------------------------------
            # 1. Create text chunks
            # --------------------------------------------------
            text_chunks = []

            if page.text.strip():
                text_parts = self._split_text(page.text)

                for text in text_parts:
                    text_chunk = Chunk(
                        chunk_id=str(uuid4()),
                        document_id=document.document_id,
                        content_type="text",
                        text=text,
                        page_number=page.page_number,
                        metadata={
                            "source": document.source,
                            "document_type": document.document_type,
                            "page_id": page.page_id,
                        },
                    )

                    text_chunks.append(text_chunk)
                    chunks.append(text_chunk)

            # --------------------------------------------------
            # 2. Create image chunks
            # --------------------------------------------------
            image_chunks = []

            for image in page.images:
                image_chunk = Chunk(
                    chunk_id=str(uuid4()),
                    document_id=document.document_id,
                    content_type="image",
                    image_path=image.path,
                    page_number=page.page_number,
                    metadata={
                        "source": document.source,
                        "document_type": document.document_type,
                        "page_id": page.page_id,
                        **image.metadata,
                    },
                )

                image_chunks.append(image_chunk)
                chunks.append(image_chunk)

            # --------------------------------------------------
            # 3. Create multimodal relationships
            # --------------------------------------------------

            text_chunk_ids = [
                chunk.chunk_id
                for chunk in text_chunks
            ]

            image_chunk_ids = [
                chunk.chunk_id
                for chunk in image_chunks
            ]

            # Text → Images
            for text_chunk in text_chunks:
                text_chunk.related_chunk_ids.extend(
                    image_chunk_ids
                )

            # Images → Text
            for image_chunk in image_chunks:
                image_chunk.related_chunk_ids.extend(
                    text_chunk_ids
                )

        return chunks

    # ----------------------------------------------------------
    # Text splitting
    # ----------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n")
            if paragraph.strip()
        ]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:

            # Start a new chunk
            if not current_chunk:
                current_chunk = paragraph
                continue

            candidate = (
                current_chunk
                + "\n"
                + paragraph
            )

            # Paragraph fits inside current chunk
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate

            # Current chunk is full
            else:
                chunks.append(current_chunk)

                # Keep overlap from previous chunk
                overlap_text = current_chunk[
                    -self.chunk_overlap:
                ]

                current_chunk = (
                    overlap_text
                    + "\n"
                    + paragraph
                )

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks