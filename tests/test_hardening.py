"""
Tests for the hardening layer: request identity, API-key auth,
rate limiting, fail-fast provider configuration, and document
deletion.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import validate_provider_settings
from app.api.main import create_app
from app.api.middleware import (
    REQUEST_ID_HEADER,
    SlidingWindowRateLimiter,
)
from app.config import settings
from app.errors import DocumentNotFoundError


# --------------------------------------------------------------
# Request identity
# --------------------------------------------------------------
def test_every_response_carries_a_request_id(rag_client):
    response = rag_client.get("/api/v1/health")

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id


def test_a_client_supplied_request_id_is_preserved(rag_client):
    # Honouring it lets a proxy, the API and the frontend share one
    # identifier for the same request.
    supplied = "trace-abc-123"

    response = rag_client.get(
        "/api/v1/health",
        headers={REQUEST_ID_HEADER: supplied},
    )

    assert response.headers[REQUEST_ID_HEADER] == supplied


def test_request_ids_are_unique_per_request(rag_client):
    first = rag_client.get("/api/v1/health").headers[REQUEST_ID_HEADER]
    second = rag_client.get("/api/v1/health").headers[REQUEST_ID_HEADER]

    assert first != second


# --------------------------------------------------------------
# API key authentication
# --------------------------------------------------------------
@pytest.fixture
def authed_client(monkeypatch):
    """A client for an app that requires an API key."""
    from contextlib import asynccontextmanager

    monkeypatch.setattr(settings, "api_key", "secret-key")

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app = create_app()
    app.router.lifespan_context = noop_lifespan

    with TestClient(app) as client:
        yield client


def test_auth_is_disabled_when_no_key_is_configured(rag_client):
    # Local development should not require configuration.
    assert settings.api_key == ""
    assert rag_client.get("/api/v1/health").status_code == 200


def test_a_missing_key_is_rejected(authed_client):
    response = authed_client.get("/api/v1/health")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "AuthenticationError"
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_a_wrong_key_is_rejected(authed_client):
    response = authed_client.get(
        "/api/v1/health",
        headers={"X-API-Key": "wrong"},
    )

    assert response.status_code == 401


def test_the_correct_key_is_accepted(authed_client):
    response = authed_client.get(
        "/api/v1/health",
        headers={"X-API-Key": "secret-key"},
    )

    assert response.status_code == 200


def test_auth_errors_do_not_echo_the_expected_key(authed_client):
    response = authed_client.get("/api/v1/health")

    assert "secret-key" not in response.text


# --------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------
def test_limiter_allows_up_to_the_limit():
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60)

    assert [limiter.check("ip")[0] for _ in range(3)] == [True] * 3
    allowed, retry_after = limiter.check("ip")

    assert allowed is False
    assert retry_after > 0


def test_limiter_is_per_key():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)

    assert limiter.check("a")[0] is True
    # One client exhausting the limit must not throttle another.
    assert limiter.check("b")[0] is True


def test_limiter_forgets_hits_outside_the_window():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)

    limiter.check("ip", now=0.0)
    limiter.check("ip", now=1.0)
    assert limiter.check("ip", now=2.0)[0] is False

    # The window has moved on, so the old hits no longer count.
    assert limiter.check("ip", now=20.0)[0] is True


def test_limiter_slides_rather_than_resetting_at_a_boundary():
    """
    A fixed window would allow 2x the limit across the boundary:
    the full quota just before it resets, then the full quota again.
    """
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)

    limiter.check("ip", now=9.0)
    limiter.check("ip", now=9.5)
    assert limiter.check("ip", now=9.9)[0] is False

    # 10.0s later the first hits have expired but not the second.
    assert limiter.check("ip", now=10.5)[0] is False
    assert limiter.check("ip", now=19.5)[0] is True


def test_rate_limited_requests_return_429(monkeypatch):
    """
    The limit comes from settings at app construction, so it has to
    be patched before create_app() - which is also how a real
    deployment configures it.
    """
    from contextlib import asynccontextmanager

    monkeypatch.setattr(settings, "api_rate_limit_requests", 3)

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app = create_app()
    app.router.lifespan_context = noop_lifespan

    with TestClient(app) as client:
        statuses = [
            client.get("/api/v1/health").status_code for _ in range(4)
        ]

    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429


def test_rate_limited_response_explains_how_to_recover(monkeypatch):
    from contextlib import asynccontextmanager

    monkeypatch.setattr(settings, "api_rate_limit_requests", 1)

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app = create_app()
    app.router.lifespan_context = noop_lifespan

    with TestClient(app) as client:
        client.get("/api/v1/health")
        response = client.get("/api/v1/health")

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
    assert response.json()["error"]["type"] == "RateLimitError"


def test_rate_limit_response_says_when_to_retry(rag_client):
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)

    allowed, retry_after = limiter.check("ip")
    assert allowed is True
    assert retry_after == 0

    allowed, retry_after = limiter.check("ip")
    assert allowed is False
    assert 0 < retry_after <= 60


# --------------------------------------------------------------
# Fail-fast provider configuration
# --------------------------------------------------------------
def test_missing_openrouter_key_fails_at_startup(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "")

    with pytest.raises(ValueError, match="openrouter"):
        validate_provider_settings()


def test_missing_opencode_key_fails_at_startup(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "opencode")
    monkeypatch.setattr(settings, "opencode_api_key", "")

    with pytest.raises(ValueError, match="opencode"):
        validate_provider_settings()


def test_local_ollama_needs_no_key(monkeypatch):
    # A local server has no auth, so demanding a key would make the
    # default local setup impossible to start.
    monkeypatch.setattr(settings, "generation_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")
    monkeypatch.setattr(settings, "ollama_api_key", "")

    validate_provider_settings()


def test_ollama_cloud_requires_a_key(monkeypatch):
    # https:// is Ollama Cloud, which always needs a token.
    monkeypatch.setattr(settings, "generation_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "https://ollama.com")
    monkeypatch.setattr(settings, "ollama_api_key", "")

    with pytest.raises(ValueError, match="ollama"):
        validate_provider_settings()


def test_a_configured_provider_passes(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "key")

    validate_provider_settings()


# --------------------------------------------------------------
# Document deletion
# --------------------------------------------------------------
def test_delete_document_removes_it_from_the_index(
    rag_client, fake_ingestion
):
    response = rag_client.delete("/api/v1/documents/doc-123")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-123",
        "chunks_deleted": 7,
    }
    assert fake_ingestion.delete_calls == ["doc-123"]


def test_delete_unknown_document_returns_404(
    rag_client, fake_ingestion
):
    fake_ingestion.delete_chunks = 0

    response = rag_client.delete("/api/v1/documents/missing")

    # 200-with-zero would tell a retrying client the delete succeeded
    # when nothing was ever there.
    assert response.status_code == 404
    assert response.json()["detail"] == (
        "No document with that id is indexed."
    )


async def test_delete_rejects_a_blank_document_id():
    """
    An empty id must not be treated as "delete everything".

    Exercised against the real service (not the fake), because this
    check lives there and a fake would only test itself.
    """
    from app.api.services import DocumentIngestionService
    from app.errors import InvalidRequestError

    class _Retriever:
        def __init__(self):
            self.calls: list[str] = []

        def delete_document(self, document_id: str) -> int:
            self.calls.append(document_id)
            return 3

    retriever = _Retriever()
    service = DocumentIngestionService(retriever=retriever)

    for blank in ("", "   "):
        with pytest.raises(InvalidRequestError):
            await service.delete_document(blank)

    # Nothing may reach the store, or a blank id would delete the
    # whole collection.
    assert retriever.calls == []