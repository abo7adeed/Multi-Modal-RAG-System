"""
Image ingestion regression tests.

ImageLoader previously built PageContent without the required page_id
field, so every image upload failed with a Pydantic ValidationError.
"""
from pathlib import Path

import pytest
from PIL import Image

from app.ingestion.chunker import DocumentChunker
from app.ingestion.image_loader import ImageLoader


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "photo.png"
    Image.new("RGB", (64, 48), color=(120, 30, 200)).save(path)
    return path


def test_image_loader_returns_valid_document(sample_image: Path):
    document = ImageLoader().load(sample_image)

    assert document.document_type == "image"
    assert len(document.pages) == 1
    assert document.metadata["filename"] == "photo.png"
    assert document.metadata["width"] == 64
    assert document.metadata["height"] == 48


def test_page_has_required_page_id(sample_image: Path):
    document = ImageLoader().load(sample_image)
    page = document.pages[0]

    # page_id is required by the schema and must match the document.
    assert page.page_id == f"{document.document_id}:page:1"
    assert page.page_number == 1


def test_page_contains_the_image_asset(sample_image: Path):
    document = ImageLoader().load(sample_image)
    page = document.pages[0]

    assert len(page.images) == 1
    assert page.images[0].path == str(sample_image)
    assert page.images[0].metadata["format"] == "PNG"


def test_image_document_chunks_without_error(sample_image: Path):
    document = ImageLoader().load(sample_image)

    chunks = DocumentChunker().chunk(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content_type == "image"
    assert chunk.page_number == 1
    assert chunk.image_path == str(sample_image)
    assert chunk.metadata["document_type"] == "image"


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ImageLoader().load(tmp_path / "nope.png")


def test_unsupported_format_raises(tmp_path: Path):
    bad = tmp_path / "file.txt"
    bad.write_text("not an image")

    with pytest.raises(ValueError):
        ImageLoader().load(bad)