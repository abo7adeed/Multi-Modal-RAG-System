"""
Upload security and validation tests.

UploadFile.filename is attacker controlled: a filename containing path
separators must never be able to write outside the uploads directory.
"""
import asyncio
import io
from pathlib import Path

import pytest

from app.api.services.document_service import DocumentIngestionService
from app.errors import InvalidRequestError


def make_pdf_bytes(text: str = "Security test document") -> bytes:
    """Build a real, minimal, valid PDF."""
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, text)
    pdf.save()
    return buffer.getvalue()


class FakeUpload:
    """Minimal UploadFile stand-in."""

    def __init__(self, filename, content=None, size=None):
        self.filename = filename
        self._content = (
            make_pdf_bytes() if content is None else content
        )
        self.size = len(self._content) if size is None else size
        self.closed = False

    async def read(self, size=-1):
        if not self._content:
            return b""
        chunk, self._content = self._content, b""
        return chunk

    async def close(self):
        self.closed = True


class RecordingRetriever:
    def __init__(self):
        self.indexed = []

    def index_chunks(self, chunks):
        self.indexed.extend(chunks)


@pytest.fixture
def service(tmp_path):
    return DocumentIngestionService(
        retriever=RecordingRetriever(),
        uploads_dir=tmp_path / "uploads",
    )


# --------------------------------------------------------------
# Filename sanitisation (no I/O)
# --------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../pwned.pdf", "pwned.pdf"),
        ("..\\..\\pwned.pdf", "pwned.pdf"),
        ("/etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Windows\\evil.pdf", "evil.pdf"),
        ("sub/dir/nested.pdf", "nested.pdf"),
        ("  spaced.pdf  ", "spaced.pdf"),
        ("My Report.pdf", "My Report.pdf"),
        ("أحمد.pdf", "أحمد.pdf"),
    ],
)
def test_safe_filename_strips_path_components(raw, expected):
    assert DocumentIngestionService._safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "..", ".", "/", "\\", None, "   "])
def test_safe_filename_rejects_dangerous_values(raw):
    with pytest.raises(InvalidRequestError):
        DocumentIngestionService._safe_filename(raw)


def test_resolve_upload_path_stays_inside_uploads_dir(service, tmp_path):
    base = (tmp_path / "uploads").resolve()
    base.mkdir(parents=True, exist_ok=True)

    resolved = service._resolve_upload_path("report.pdf")

    assert resolved.parent == base


def test_resolve_upload_path_rejects_escape(service, tmp_path):
    base = (tmp_path / "uploads").resolve()
    base.mkdir(parents=True, exist_ok=True)

    # Defence in depth: even if sanitisation were bypassed, the
    # resolved path must not escape.
    with pytest.raises(InvalidRequestError):
        service._resolve_upload_path("../escaped.pdf")


# --------------------------------------------------------------
# End-to-end upload
# --------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    ["../../pwned.pdf", "..\\..\\pwned.pdf", "/etc/passwd.pdf"],
)
def test_traversal_upload_writes_inside_uploads_dir(
    service, tmp_path, raw
):
    result = asyncio.run(service.ingest_upload(FakeUpload(raw)))

    uploads = tmp_path / "uploads"
    written = list(uploads.glob("*"))

    assert len(written) == 1
    assert written[0].parent == uploads.resolve()
    assert result["filename"] == written[0].name
    assert ".." not in result["filename"]

    # Nothing escaped to the project root or the tmp dir.
    assert not (tmp_path / "pwned.pdf").exists()


def test_valid_upload_is_indexed(service, tmp_path):
    result = asyncio.run(
        service.ingest_upload(FakeUpload("cv.pdf"))
    )

    assert result["filename"] == "cv.pdf"
    assert result["chunks_indexed"] >= 1
    assert (tmp_path / "uploads" / "cv.pdf").exists()


def test_unicode_filename_survives_ingestion(service):
    result = asyncio.run(
        service.ingest_upload(FakeUpload("أحمد-sectional.pdf"))
    )

    assert result["filename"] == "أحمد-sectional.pdf"


def test_unsupported_extension_is_rejected(service):
    with pytest.raises(InvalidRequestError):
        asyncio.run(
            service.ingest_upload(
                FakeUpload("malware.exe", content=b"MZ", size=2)
            )
        )


def test_oversized_file_is_rejected_and_cleaned_up(
    service, monkeypatch, tmp_path
):
    from app.config import settings

    monkeypatch.setattr(settings, "api_max_upload_size_mb", 1)

    big = FakeUpload(
        "big.pdf",
        content=b"x" * (2 * 1024 * 1024),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(service.ingest_upload(big))

    assert not (tmp_path / "uploads" / "big.pdf").exists()


def test_upload_is_closed_even_on_success(service):
    upload = FakeUpload("close-me.pdf")
    asyncio.run(service.ingest_upload(upload))

    assert upload.closed is True


def test_upload_is_closed_even_on_failure(service):
    upload = FakeUpload("bad.exe", content=b"MZ", size=2)

    with pytest.raises(InvalidRequestError):
        asyncio.run(service.ingest_upload(upload))

    assert upload.closed is True


def test_corrupt_pdf_is_reported_as_ingestion_error(service):
    from app.errors import DocumentIngestionError

    upload = FakeUpload(
        "broken.pdf", content=b"%PDF-1.4 truncated garbage"
    )

    with pytest.raises(DocumentIngestionError):
        asyncio.run(service.ingest_upload(upload))


# --------------------------------------------------------------
# Normalized images must land in the servable media tree
# --------------------------------------------------------------
def test_uploaded_image_is_normalized_into_processed_media(service):
    """
    An uploaded image must be normalized under the processed
    directory, otherwise it is indexed but cannot be displayed:
    the API only serves files from the processed media tree.
    """
    import io as _io

    from PIL import Image
    from app.config import settings

    buffer = _io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(
        buffer, format="PNG"
    )

    upload = FakeUpload("portrait.png", content=buffer.getvalue())

    result = asyncio.run(service.ingest_upload(upload))

    assert result["document_type"] == "image"
    assert result["chunks_indexed"] == 1

    indexed = service.retriever.indexed[0]
    assert indexed.image_path is not None
    assert indexed.image_path.endswith(".jpg")

    processed_root = Path(settings.processed_images_directory)
    normalized = Path(indexed.image_path)

    # The normalized file must exist and be inside the media tree.
    assert normalized.exists()
    assert normalized.suffix == ".jpg"
    assert processed_root.resolve() in normalized.resolve().parents