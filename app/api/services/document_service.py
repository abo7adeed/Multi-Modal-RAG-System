"""
Document upload/ingestion service.

Reuses the existing ingestion stack (loaders, chunker) and the
existing retrieval stack (embedder, vector store) without modifying
them. Called only by the API layer.
"""
import logging
from pathlib import Path, PureWindowsPath

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.errors import DocumentIngestionError, InvalidRequestError
from app.ingestion.chunker import DocumentChunker
from app.ingestion.image_loader import ImageLoader
from app.ingestion.pdf_loader import PDFLoader
from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.retriever import MultimodalRetriever
from app.retrieval.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class DocumentIngestionService:
    def __init__(
        self,
        retriever: MultimodalRetriever,
        uploads_dir: Path | str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.retriever = retriever
        self.uploads_dir = Path(uploads_dir or settings.uploads_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest_upload(self, file: UploadFile) -> dict:
        try:
            return await self._ingest_upload(file)
        finally:
            # Always release the upload, including on validation
            # failures that happen before the file is read.
            await file.close()

    async def _ingest_upload(self, file: UploadFile) -> dict:
        # --------------------------------------------------------
        # Sanitise the client-supplied filename.
        #
        # UploadFile.filename is attacker controlled: "../../x.pdf"
        # would otherwise escape the uploads directory and write
        # anywhere the server can reach. Strip any path components
        # and keep only the base name.
        # --------------------------------------------------------
        filename = self._safe_filename(file.filename)
        suffix = Path(filename).suffix.lower()

        self._validate(filename, suffix, file.size)

        # --------------------------------------------------------
        # Persist the upload to disk (loaders read it back)
        # --------------------------------------------------------
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        upload_path = self._resolve_upload_path(filename)

        try:
            size = 0
            with upload_path.open("wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_size_bytes:
                        raise InvalidRequestError(
                            "File exceeds the maximum allowed size of "
                            f"{settings.api_max_upload_size_mb} MB."
                        )
                    buffer.write(chunk)
        except InvalidRequestError:
            upload_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            logger.error(
                "upload_write_failed filename=%s error=%s", filename, exc
            )
            raise DocumentIngestionError(
                "The file could not be stored."
            ) from exc

        # --------------------------------------------------------
        # Ingest through the existing pipeline
        #
        # Parsing, chunking and CLIP embedding are blocking work, so
        # they run in a worker thread. Doing them inline in this
        # async method would stall the event loop (and every other
        # in-flight request) for the whole ingestion.
        # --------------------------------------------------------
        try:
            return await run_in_threadpool(self._ingest, upload_path)
        except (InvalidRequestError, DocumentIngestionError):
            raise
        except Exception as exc:
            logger.error(
                "ingest_failed filename=%s error_type=%s error=%s",
                filename,
                type(exc).__name__,
                exc,
            )
            raise DocumentIngestionError(
                "The document could not be processed."
            ) from exc

    # ------------------------------------------------------------
    # Filename safety
    # ------------------------------------------------------------

    @staticmethod
    def _safe_filename(raw: str | None) -> str:
        """
        Reduce a client-supplied filename to a safe base name.

        PureWindowsPath is used because uploads may originate from
        Windows clients that send backslash separators.
        """
        base = PureWindowsPath(raw or "").name.strip()

        # Drop any remaining separators or traversal remnants.
        base = base.replace("/", "").replace("\\", "").strip()

        if not base or base in {".", ".."}:
            raise InvalidRequestError("Invalid filename.")

        return base

    def _resolve_upload_path(self, filename: str) -> Path:
        """
        Resolve the final destination, guaranteeing it stays inside
        the uploads directory.
        """
        base_dir = self.uploads_dir.resolve()
        candidate = (base_dir / filename).resolve()

        try:
            candidate.relative_to(base_dir)
        except ValueError as exc:
            logger.error(
                "upload_path_rejected filename=%s", filename
            )
            raise InvalidRequestError(
                "Invalid filename."
            ) from exc

        return candidate

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def _validate(
        self,
        filename: str,
        suffix: str,
        declared_size: int | None,
    ) -> None:
        if suffix not in PDF_SUFFIXES | IMAGE_SUFFIXES:
            raise InvalidRequestError(
                "Unsupported file type. Supported: PDF, JPG, PNG, WEBP."
            )

        if declared_size is not None and declared_size > (
            settings.max_upload_size_bytes
        ):
            raise InvalidRequestError(
                "File exceeds the maximum allowed size of "
                f"{settings.api_max_upload_size_mb} MB."
            )

    # ------------------------------------------------------------
    # Ingestion (existing stack)
    # ------------------------------------------------------------

    def _ingest(self, upload_path: Path) -> dict:
        suffix = upload_path.suffix.lower()

        if suffix in PDF_SUFFIXES:
            document = PDFLoader().load(upload_path)
        else:
            document = ImageLoader().load(upload_path)

        chunker = DocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = chunker.chunk(document)

        # --------------------------------------------------------
        # Normalize image chunks so CLIP can read them
        # --------------------------------------------------------
        for chunk in chunks:
            if chunk.content_type == "image" and chunk.image_path:
                chunk.image_path = str(
                    self._normalize_image(
                        Path(chunk.image_path),
                        document_id=document.document_id,
                    )
                )

        self.retriever.index_chunks(chunks)

        logger.info(
            "document_ingested filename=%s document_id=%s chunks=%d",
            upload_path.name,
            document.document_id,
            len(chunks),
        )

        return {
            "document_id": document.document_id,
            "filename": upload_path.name,
            "document_type": document.document_type,
            "chunks_indexed": len(chunks),
        }

    def _normalize_image(
        self,
        image_path: Path,
        document_id: str,
    ) -> Path:
        """
        Convert an image chunk to JPEG inside the processed-media
        tree.

        The normalized copy must live under the processed images
        directory: that tree is what the API serves as media, so
        writing next to the upload would make the image unviewable
        in the UI.
        """
        from app.ingestion.image_normalizer import ImageNormalizer

        output_dir = (
            Path(settings.processed_images_directory) / document_id
        )
        output_path = output_dir / f"{image_path.stem}.jpg"

        normalizer = ImageNormalizer()
        return normalizer.normalize(image_path, output_path)

    async def delete_document(self, document_id: str) -> dict:
        """
        Remove a document from the index.

        Returns the number of chunks deleted. Raises
        DocumentNotFoundError when the id matches nothing, so the API
        can answer 404 rather than a misleading 200 with a zero
        count.
        """
        document_id = (document_id or "").strip()

        if not document_id:
            raise InvalidRequestError(
                "A document id is required."
            )

        # Chroma is blocking I/O; a delete can touch hundreds of
        # chunks, so it must not run on the event loop.
        removed = await run_in_threadpool(
            self.retriever.delete_document, document_id
        )

        if not removed:
            raise DocumentNotFoundError(document_id)

        logger.info(
            "document_deleted document_id=%s chunks=%d",
            document_id,
            removed,
        )

        return {"document_id": document_id, "chunks_deleted": removed}
