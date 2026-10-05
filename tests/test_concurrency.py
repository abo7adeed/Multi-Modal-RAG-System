"""
Regression tests for event-loop blocking.

Both the query route and the upload route do blocking work (CLIP
embedding, PDF parsing, a model call that may take minutes). If that
work runs on the event loop, concurrent requests - and /health -
queue behind it. These tests pin the fix: overlapping wall time and
observed peak concurrency.

They are timing-based, so the margins are deliberately loose.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from io import BytesIO

import httpx
import pytest
from starlette.datastructures import UploadFile

from app.api.dependencies import AppState
from app.api.main import create_app
from app.api.services import DocumentIngestionService, MediaService
from app.generation.schemas import RAGResponse

# One unit of fake "slow" work, and how many run at once.
SLEEP = 0.4
CONCURRENCY = 3
# Serialized execution would take SLEEP * CONCURRENCY; overlapped
# execution should land near SLEEP. 0.7 leaves room for CI jitter
# while still failing loudly if the requests serialize.
MAX_OVERLAPPED = SLEEP * CONCURRENCY * 0.7


class SlowPipeline:
    """Fake pipeline that records peak concurrent execution."""

    def __init__(self):
        self.active = 0
        self.peak = 0

    def run(self, query, image=None, history=None):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            time.sleep(SLEEP)
        finally:
            self.active -= 1
        return RAGResponse(answer="ok", sources=[])

    def describe_provider(self):
        return {"provider": "SlowPipeline", "model": "test"}


class StubIngestion:
    async def ingest_upload(self, file):
        return {}


@pytest.mark.asyncio
async def test_concurrent_queries_overlap(tmp_path):
    """Three simultaneous queries must not serialize on the loop."""
    media = MediaService(base_path=tmp_path / "images")
    slow = SlowPipeline()

    app = create_app()

    @asynccontextmanager
    async def fake_lifespan(app_):
        app_.state.rag = AppState(
            pipeline=slow,
            ingestion_service=StubIngestion(),
            media_service=media,
        )
        yield

    app.router.lifespan_context = fake_lifespan

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            start = time.perf_counter()
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/api/v1/rag/query", json={"query": f"q{i}"}
                    )
                    for i in range(CONCURRENCY)
                ]
            )
            elapsed = time.perf_counter() - start

    assert [r.status_code for r in responses] == [200] * CONCURRENCY
    assert slow.peak == CONCURRENCY, (
        f"only {slow.peak} of {CONCURRENCY} queries ran concurrently"
    )
    assert elapsed < MAX_OVERLAPPED, (
        f"queries serialized: {elapsed:.2f}s for "
        f"{CONCURRENCY} x {SLEEP}s"
    )


@pytest.mark.asyncio
async def test_concurrent_ingestions_overlap(tmp_path):
    """Ingestion's blocking core must run off the event loop too."""
    service = DocumentIngestionService(
        retriever=object(),
        uploads_dir=tmp_path / "uploads",
    )

    def slow_ingest(upload_path):
        time.sleep(SLEEP)
        return {
            "document_id": "doc",
            "filename": upload_path.name,
            "document_type": "pdf",
            "chunks_indexed": 1,
        }

    service._ingest = slow_ingest

    files = [
        UploadFile(
            BytesIO(b"%PDF-1.4\n"),
            size=9,
            filename=f"doc{i}.pdf",
        )
        for i in range(CONCURRENCY)
    ]

    start = time.perf_counter()
    results = await asyncio.gather(
        *[service.ingest_upload(f) for f in files]
    )
    elapsed = time.perf_counter() - start

    assert len(results) == CONCURRENCY
    assert elapsed < MAX_OVERLAPPED, (
        f"ingestions serialized: {elapsed:.2f}s for "
        f"{CONCURRENCY} x {SLEEP}s"
    )
