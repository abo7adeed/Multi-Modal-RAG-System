"""
API tests: health, RAG query (validation, success, error mapping),
document upload, and media security.
"""
import io

from app.errors import (
    DocumentIngestionError,
    InvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RetrievalError,
)


# --------------------------------------------------------------
# Health
# --------------------------------------------------------------
def test_health_returns_ok(rag_client):
    response = rag_client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------
# RAG query: validation
# --------------------------------------------------------------
def test_rag_query_rejects_empty_query(rag_client):
    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_rag_query_rejects_whitespace_query(rag_client, fake_pipeline):
    fake_pipeline.outcome = InvalidRequestError(
        "Query must not be empty."
    )

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "   "},
    )

    assert response.status_code == 400


def test_rag_query_rejects_oversized_query(rag_client):
    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "x" * 2001},
    )

    assert response.status_code == 422


def test_rag_query_rejects_missing_query(rag_client):
    response = rag_client.post(
        "/api/v1/rag/query",
        json={},
    )

    assert response.status_code == 422


# --------------------------------------------------------------
# RAG query: success
# --------------------------------------------------------------
def test_rag_query_returns_only_best_source(rag_client, fake_pipeline):
    # Retrieval returns sources best-first; the API keeps the
    # strongest match only.
    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "What Dell products are shown?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert len(body["sources"]) == 1
    # The first ranked source is the text chunk on page 4.
    assert body["sources"][0]["chunk_id"] == "text-001"
    assert body["sources"][0]["page"] == 4


def test_rag_query_source_limit_is_configurable(rag_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_max_sources", 2)

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "What Dell products are shown?"},
    )

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 2


def test_rag_query_includes_text_snippet(rag_client, fake_pipeline):
    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "Dell laptop"},
    )

    source = response.json()["sources"][0]

    # Text sources must expose why they matched.
    assert source["snippet"]
    assert "Dell Precision" in source["snippet"]


def test_rag_query_success(rag_client, fake_pipeline):
    from app.config import settings
    import pytest_asyncio

    # Restore multi-source output for the shape assertions below.
    original = settings.api_max_sources
    settings.api_max_sources = 2
    try:
        response = rag_client.post(
            "/api/v1/rag/query",
            json={"query": "What Dell products are shown in the catalog?"},
        )

        assert response.status_code == 200
        body = response.json()

        assert body["answer"] == (
            "The catalog contains Dell laptops and desktops."
        )
        assert len(body["sources"]) == 2

        text_source, image_source = body["sources"]

        assert text_source["content_type"] == "text"
        assert text_source["page"] == 4
        assert text_source["chunk_id"] == "text-001"
        assert text_source["image_url"] is None

        assert image_source["content_type"] == "image"
        assert image_source["page"] == 3
        # Image lives inside the media root -> URL is exposed
        assert image_source["image_url"] is not None
        assert image_source["image_url"].startswith("/media/")

        # The servable image must be fetchable through the media route
        media_response = rag_client.get(image_source["image_url"])
        assert media_response.status_code == 200
    finally:
        settings.api_max_sources = original

    assert fake_pipeline.calls == [
        "What Dell products are shown in the catalog?"
    ]


def test_rag_query_passes_query_through(rag_client, fake_pipeline):
    # Whitespace normalization is the pipeline's responsibility
    # (covered by tests/test_pipeline.py); the route passes the
    # payload through unchanged.
    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "  Dell laptops  "},
    )

    assert response.status_code == 200
    assert fake_pipeline.calls == ["  Dell laptops  "]


# --------------------------------------------------------------
# RAG query: attached image
# --------------------------------------------------------------
def test_rag_query_forwards_attachment_to_the_pipeline(
    rag_client, fake_pipeline
):
    import base64
    from PIL import Image

    payload = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(payload, format="PNG")
    raw = payload.getvalue()

    response = rag_client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is in this picture?",
            "image": {
                "filename": "photo.png",
                "media_type": "image/png",
                "data": base64.b64encode(raw).decode("utf-8"),
            },
        },
    )

    assert response.status_code == 200

    attachment = fake_pipeline.images[0]
    assert attachment.filename == "photo.png"
    assert attachment.media_type == "image/png"
    assert attachment.data == raw


def test_rag_query_without_image_sends_none(rag_client, fake_pipeline):
    rag_client.post(
        "/api/v1/rag/query",
        json={"query": "Dell laptop"},
    )

    assert fake_pipeline.images == [None]


def test_rag_query_rejects_non_image_payload(rag_client, fake_pipeline):
    import base64

    response = rag_client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is this?",
            "image": {
                "filename": "invoice.png",
                "media_type": "image/png",
                "data": base64.b64encode(
                    b"%PDF-1.4 not an image"
                ).decode("utf-8"),
            },
        },
    )

    assert response.status_code == 400
    # The user must be told *why* the image was refused, not just that
    # the request was invalid.
    assert "not a valid image" in (
        response.json()["error"]["message"]
    )
    # The pipeline must never see an unvalidated attachment.
    assert fake_pipeline.calls == []


def test_rag_query_rejects_invalid_base64_image(rag_client):
    response = rag_client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is this?",
            "image": {
                "filename": "photo.png",
                "media_type": "image/png",
                "data": "not base64 !!!",
            },
        },
    )

    assert response.status_code == 400


def test_rag_query_rejects_empty_image_data(rag_client):
    response = rag_client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is this?",
            "image": {
                "filename": "photo.png",
                "media_type": "image/png",
                "data": "",
            },
        },
    )

    assert response.status_code == 422


def test_rag_query_maps_unsupported_vision_to_400(
    rag_client, fake_pipeline
):
    from app.errors import ImageVisionUnsupportedError

    fake_pipeline.outcome = ImageVisionUnsupportedError()

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "What is this?"},
    )

    assert response.status_code == 400
    assert "cannot read images" in (
        response.json()["error"]["message"]
    )


# --------------------------------------------------------------
# RAG query: provider failure mapping
# --------------------------------------------------------------
def test_rag_query_rate_limit_returns_429(rag_client, fake_pipeline):
    fake_pipeline.outcome = ProviderRateLimitError(
        "provider=OpenRouter model=test-model"
    )

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "test"},
    )

    assert response.status_code == 429
    body = response.json()
    assert "rate limited" in body["error"]["message"]
    assert "test-model" not in body["error"]["message"]
    assert body["error"]["type"] == "ProviderRateLimitError"


def test_rag_query_timeout_returns_504(rag_client, fake_pipeline):
    fake_pipeline.outcome = ProviderTimeoutError(
        "provider=OpenRouter model=test-model"
    )

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "test"},
    )

    assert response.status_code == 504
    body = response.json()
    assert "took too long" in body["error"]["message"]
    assert "test-model" not in body["error"]["message"]
    assert body["error"]["type"] == "ProviderTimeoutError"


def test_rag_query_retrieval_failure_returns_503(rag_client, fake_pipeline):
    fake_pipeline.outcome = RetrievalError("chroma down")

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "test"},
    )

    assert response.status_code == 503


def test_rag_query_generic_failure_returns_503(rag_client, fake_pipeline):
    from app.errors import RAGPipelineError

    fake_pipeline.outcome = RAGPipelineError("boom")

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "test"},
    )

    assert response.status_code == 503
    assert "try again" in response.json()["error"]["message"]


def test_error_responses_never_leak_technical_details(
    rag_client, fake_pipeline
):
    fake_pipeline.outcome = ProviderRateLimitError(
        "secret-api-key-details-here"
    )

    response = rag_client.post(
        "/api/v1/rag/query",
        json={"query": "test"},
    )

    assert "secret-api-key-details-here" not in response.text


# --------------------------------------------------------------
# Document upload
# --------------------------------------------------------------
def test_upload_success(rag_client, fake_ingestion):
    response = rag_client.post(
        "/api/v1/documents",
        files={"file": ("uploaded.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "doc-123"
    assert body["chunks_indexed"] == 7
    assert fake_ingestion.calls == ["uploaded.pdf"]


def test_upload_unsupported_type(rag_client):
    response = rag_client.post(
        "/api/v1/documents",
        files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/x-msdownload")},
    )

    assert response.status_code == 400


def test_upload_ingestion_error(rag_client, fake_ingestion):
    fake_ingestion.outcome = DocumentIngestionError("bad pdf")

    response = rag_client.post(
        "/api/v1/documents",
        files={"file": ("uploaded.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 422


# --------------------------------------------------------------
# Media security
# --------------------------------------------------------------
def test_media_route_rejects_traversal(rag_client):
    response = rag_client.get("/media/../.env")

    assert response.status_code in (400, 404)


def test_media_route_returns_404_for_missing_file(rag_client):
    response = rag_client.get("/media/does-not-exist.jpg")

    assert response.status_code == 404
