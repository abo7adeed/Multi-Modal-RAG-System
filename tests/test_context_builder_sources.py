"""
Context builder tests: source attribution must use the real document
filename, not the RRF provenance tag.
"""
from app.generation.context_builder import ContextBuilder, document_name


def build(results):
    return ContextBuilder().build(query="q", results=results)


# --------------------------------------------------------------
# document_name
# --------------------------------------------------------------
def test_document_name_handles_windows_paths():
    assert (
        document_name("data\\raw\\documents\\dell_catalog.pdf")
        == "dell_catalog.pdf"
    )


def test_document_name_handles_posix_paths():
    assert (
        document_name("data/raw/documents/dell_catalog.pdf")
        == "dell_catalog.pdf"
    )


def test_document_name_handles_unicode_filenames():
    assert (
        document_name("data/uploads\\أحmingdocuments\\cv.pdf") == "cv.pdf"
    )


def test_document_name_returns_none_for_empty():
    assert document_name(None) is None
    assert document_name("") is None


# --------------------------------------------------------------
# Source attribution
# --------------------------------------------------------------
def test_sources_use_document_filename_not_provenance_tag():
    context = build([
        {
            "chunk_id": "text-001",
            "document": "Dell OptiPlex 3020 Micro desktop.",
            "metadata": {
                "content_type": "text",
                "page_number": 2,
                "source": "data\\raw\\documents\\dell_catalog.pdf",
            },
            "source": "retrieved",
        }
    ])

    source = context.sources[0]
    assert source.source == "dell_catalog.pdf"
    assert context.text_context[0]["source"] == "dell_catalog.pdf"


def test_uploaded_document_filename_is_reported():
    context = build([
        {
            "chunk_id": "text-001",
            "document": "Ahmed Abohadeed is an AI Engineer.",
            "metadata": {
                "content_type": "text",
                "page_number": 1,
                "source": "data\\uploads\\My First Board.pdf",
            },
            "source": "retrieved",
        }
    ])

    assert context.sources[0].source == "My First Board.pdf"


def test_falls_back_to_provenance_when_metadata_has_no_document():
    context = build([
        {
            "chunk_id": "text-001",
            "document": "Some content",
            "metadata": {"content_type": "text", "page_number": 1},
            "source": "retrieved",
        }
    ])

    assert context.sources[0].source == "retrieved"


def test_related_chunks_report_their_document_too():
    context = build([
        {
            "chunk_id": "image-001",
            "document": "Image from page 2",
            "metadata": {
                "content_type": "image",
                "page_number": 2,
                "source": "data\\raw\\documents\\dell_catalog.pdf",
                "image_path": "img.jpg",
            },
            "source": "related",
        }
    ])

    source = context.sources[0]
    assert source.source == "dell_catalog.pdf"
    assert source.content_type == "image"


# --------------------------------------------------------------
# Attached image
# --------------------------------------------------------------
def test_attached_image_leads_the_visual_context():
    context = ContextBuilder().build(
        query="What is this?",
        results=[],
        attachments=[
            {
                "image_path": "tmp/attachment.jpg",
                "filename": "photo.png",
            }
        ],
    )

    assert len(context.image_context) == 1
    assert context.image_context[0]["kind"] == "attachment"
    assert context.image_context[0]["source"] == "photo.png"


def test_attached_image_is_not_reported_as_a_source():
    # The client already shows its own image, and it is not part of
    # the indexed corpus, so it must not appear as a citation.
    context = ContextBuilder().build(
        query="What is this?",
        results=[],
        attachments=[
            {
                "image_path": "tmp/attachment.jpg",
                "filename": "photo.png",
            }
        ],
    )

    assert context.sources == []