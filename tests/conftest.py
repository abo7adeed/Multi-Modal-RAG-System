"""
Shared fixtures for API tests.

Heavy components (CLIP, Chroma, providers) are faked/stubbed so the
API layer is tested in isolation, without network or model loading.
"""
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app.api.dependencies import AppState
from app.api.main import create_app
from app.api.services import DocumentIngestionService, MediaService
from app.errors import InvalidRequestError
from app.generation.schemas import RAGResponse
from app.pipeline import MultimodalRAGPipeline

# --------------------------------------------------------------
# Environment isolation: never read the developer's real .env
# --------------------------------------------------------------
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test-model")
    monkeypatch.setenv(
        "API_CORS_ORIGINS", "http://localhost:5173"
    )
    monkeypatch.setenv(
        "PROCESSED_IMAGES_DIRECTORY", str(tmp_path / "processed/images")
    )
    monkeypatch.setenv("UPLOADS_DIRECTORY", str(tmp_path / "uploads"))
    yield


class FakePipeline:
    """Stand-in for MultimodalRAGPipeline with a configurable outcome."""

    def __init__(self, media_service: MediaService | None = None):
        self.outcome: Exception | None = None
        self.stream_outcome: Exception | None = None
        self.stream_mid_error: Exception | None = None
        self.calls: list[str] = []
        self.images: list = []
        self.media_service = media_service
        self.stream_chunks: list[str] | None = None
        self.stream_calls: list[str] = []
        self.last_response: RAGResponse | None = None

    def run(self, query: str, image=None, history=None):
        self.calls.append(query)
        self.images.append(image)
        if self.outcome is not None:
            raise self.outcome
        from app.generation.schemas import Source

        image_path = None
        if self.media_service is not None:
            image_path = (
                self.media_service.base_path
                / "dell_catalog/page_3_image_0.jpg"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                image_path.write_bytes(b"fake-jpeg-bytes")

        from app.generation.schemas import RAGResponse

        self.last_response = RAGResponse(
            answer="The catalog contains Dell laptops and desktops.",
            sources=[
                Source(
                    chunk_id="text-001",
                    page=4,
                    content_type="text",
                    source="retrieved",
                    image_path=None,
                    snippet="Dell Precision laptop with Intel Core i7.",
                ),
                Source(
                    chunk_id="image-001",
                    page=3,
                    content_type="image",
                    source="retrieved",
                    image_path=str(image_path) if image_path else None,
                ),
            ],
        )

        return self.last_response

    def run_stream(self, query: str, image=None, history=None):
        """
        Mirror run() for the SSE route: sources, tokens, done.

        Not a generator function, matching the real pipeline: setup
        failures are raised when the route calls this, so they can
        still be answered with a real HTTP status.
        """
        from app.generation.schemas import RAGStreamEvent

        self.stream_calls = getattr(self, "stream_calls", [])
        self.stream_calls.append(query)
        self.histories = getattr(self, "histories", [])
        self.histories.append(list(history or []))

        if self.stream_outcome is not None:
            raise self.stream_outcome

        self.run(query, image=image)

        def generate():
            yield RAGStreamEvent(
                type="sources", sources=self.last_response.sources
            )

            chunks = self.stream_chunks or [
                "Dell ", "laptops ", "and ", "desktops."
            ]
            for index, chunk in enumerate(chunks):
                # Mirrors the real pipeline: a generation failure
                # surfaces mid-iteration, once the response has
                # already started.
                if self.stream_mid_error and index == 1:
                    raise self.stream_mid_error
                yield RAGStreamEvent(type="token", text=chunk)

            yield RAGStreamEvent(type="done")

        return generate()

    def describe_provider(self):
        return {"provider": "FakePipeline", "model": "test-model"}


class FakeIngestion:
    """Mirrors DocumentIngestionService validation, minus heavy work."""

    def __init__(self, media_service: MediaService | None = None):
        self.calls: list = []
        self.outcome: Exception | None = None
        self.media_service = media_service
        self.result = {
            "document_id": "doc-123",
            "filename": "uploaded.pdf",
            "document_type": "pdf",
            "chunks_indexed": 7,
        }
        self.delete_calls: list[str] = []
        self.delete_chunks = 7
        self.delete_error: Exception | None = None

    async def delete_document(self, document_id: str):
        self.delete_calls.append(document_id)
        if self.delete_error is not None:
            raise self.delete_error
        if self.delete_chunks == 0:
            from app.errors import DocumentNotFoundError

            raise DocumentNotFoundError(document_id)
        return {
            "document_id": document_id,
            "chunks_deleted": self.delete_chunks,
        }

    async def ingest_upload(self, file):
        from pathlib import Path

        from app.config import settings

        self.calls.append(file.filename)

        suffix = Path(file.filename or "").suffix.lower()
        allowed = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}
        if suffix not in allowed:
            raise InvalidRequestError(
                "Unsupported file type. Supported: PDF, JPG, PNG, WEBP."
            )

        if (
            file.size is not None
            and file.size > settings.max_upload_size_bytes
        ):
            raise InvalidRequestError("File exceeds the maximum size.")

        if self.outcome is not None:
            raise self.outcome
        return dict(self.result)


@pytest_asyncio.fixture
async def client(
    monkeypatch,
    tmp_path: Path,
):
    """TestClient with fakes wired into app.state.rag."""
    from contextlib import asynccontextmanager

    import pytest_asyncio

    media_service = MediaService(
        base_path=tmp_path / "processed/images"
    )

    fake_pipeline = FakePipeline(media_service=media_service)
    fake_ingestion = FakeIngestion(media_service=media_service)

    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.rag = AppState(
            pipeline=fake_pipeline,
            ingestion_service=fake_ingestion,
            media_service=media_service,
        )
        yield

    test_app = create_app()
    test_app.router.lifespan_context = fake_lifespan

    with TestClient(test_app) as test_client:
        yield test_client, fake_pipeline, fake_ingestion


@pytest.fixture
def rag_client(client):
    test_client, fake_pipeline, fake_ingestion = client
    return test_client


@pytest.fixture
def fake_pipeline(client):
    _, fake_pipeline, _ = client
    return fake_pipeline


@pytest.fixture
def fake_ingestion(client):
    _, _, fake_ingestion = client
    return fake_ingestion
