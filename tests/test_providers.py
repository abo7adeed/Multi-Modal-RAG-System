"""
Provider tests: factory selection, configuration wiring, and SDK
error translation. No network calls are made.
"""
import base64
from typing import Any

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

from app.config import settings
from app.generation.providers import (
    OllamaMultimodalProvider,
    OpenCodeMultimodalProvider,
    OpenRouterMultimodalProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    build_provider,
)
from app.generation.providers.base import (
    ProviderRateLimitError as BaseProviderRateLimitError,
    ProviderTimeoutError as BaseProviderTimeoutError,
)


# --------------------------------------------------------------
# Factory
# --------------------------------------------------------------
@pytest.mark.parametrize(
    ("provider_name", "expected"),
    [
        ("openrouter", OpenRouterMultimodalProvider),
        ("ollama", OllamaMultimodalProvider),
        ("opencode", OpenCodeMultimodalProvider),
    ],
)
def test_build_provider_selects_configured_provider(
    monkeypatch, provider_name, expected
):
    monkeypatch.setattr(settings, "generation_provider", provider_name)

    provider = build_provider()

    assert isinstance(provider, expected)


def test_build_provider_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "generation_provider", "does-not-exist")

    with pytest.raises(ProviderError):
        build_provider()


# --------------------------------------------------------------
# Configuration wiring
# --------------------------------------------------------------
def test_openrouter_uses_settings_timeout_and_retries(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_model", "my-model")
    monkeypatch.setattr(settings, "openrouter_timeout", 42.0)
    monkeypatch.setattr(settings, "openrouter_max_retries", 5)
    monkeypatch.setattr(settings, "openrouter_api_key", "key")

    provider = OpenRouterMultimodalProvider()

    assert provider.model == "my-model"
    assert provider.client.timeout == 42.0
    assert provider.client.max_retries == 5
    assert str(provider.client.base_url).startswith(
        "https://openrouter.ai"
    )


def test_opencode_uses_settings(monkeypatch):
    monkeypatch.setattr(settings, "opencode_model", "zen-model")
    monkeypatch.setattr(settings, "opencode_api_key", "zen-key")
    monkeypatch.setattr(
        settings, "opencode_base_url", "https://example.test/v1"
    )

    provider = OpenCodeMultimodalProvider()

    assert provider.model == "zen-model"
    assert provider.client.api_key == "zen-key"
    assert str(provider.client.base_url).rstrip("/") == (
        "https://example.test/v1"
    )


def test_ollama_uses_settings(monkeypatch):
    monkeypatch.setattr(settings, "ollama_model", "vision:7b")
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://ollama.test:11434"
    )

    provider = OllamaMultimodalProvider()

    assert provider.model == "vision:7b"
    assert provider.base_url == "http://ollama.test:11434"


# --------------------------------------------------------------
# Ollama Cloud (hosted models)
# --------------------------------------------------------------
def test_ollama_cloud_sends_bearer_token(monkeypatch):
    monkeypatch.setattr(settings, "ollama_base_url", "https://ollama.com")
    monkeypatch.setattr(settings, "ollama_api_key", "cloud-key")
    monkeypatch.setattr(settings, "ollama_model", "gpt-oss:120b-cloud")

    provider = OllamaMultimodalProvider()

    assert provider.base_url == "https://ollama.com"
    assert provider.model == "gpt-oss:120b-cloud"
    assert provider.api_key == "cloud-key"
    assert provider.client._client.headers.get(
        "authorization"
    ) == "Bearer cloud-key"


def test_ollama_local_sends_no_auth(monkeypatch):
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )
    monkeypatch.setattr(settings, "ollama_api_key", "")

    provider = OllamaMultimodalProvider()

    assert "authorization" not in provider.client._client.headers


def test_ollama_encodes_images_as_base64(monkeypatch, tmp_path):
    # Cloud models cannot read local file paths, so images must be
    # inlined before being sent.
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    image = tmp_path / "page.jpg"
    image.write_bytes(b"fake-jpeg-bytes")

    provider = OllamaMultimodalProvider()

    captured: dict[str, Any] = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {"message": {"content": "ok"}}

    provider.client.chat = fake_chat

    provider.generate(
        query="What is shown?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[{"image_path": str(image)}],
    )

    messages = captured["messages"]
    # Grounding rules live in a system message; images ride on the
    # user message alongside the question.
    assert messages[0]["role"] == "system"
    assert "Never greet, chat" in messages[0]["content"]
    assert messages[1]["role"] == "user"

    images = messages[1]["images"]
    assert images == [
        base64.b64encode(b"fake-jpeg-bytes").decode("utf-8")
    ]


# --------------------------------------------------------------
# Multimodal capability
# --------------------------------------------------------------
def test_openrouter_supports_images():
    assert OpenRouterMultimodalProvider.supports_images is True


def test_opencode_is_text_only():
    # Zen's catalogue has no vision models; images must be dropped
    # rather than crashing the request.
    assert OpenCodeMultimodalProvider.supports_images is False


def test_text_only_provider_drops_image_parts(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "opencode_api_key", "zen-key")

    image = tmp_path / "page.jpg"
    image.write_bytes(b"fake-jpeg")

    provider = OpenCodeMultimodalProvider()

    content = provider._build_content(
        query="What is shown?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[{"image_path": str(image)}],
    )

    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_multimodal_provider_keeps_image_parts(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "openrouter_api_key", "key")

    image = tmp_path / "page.jpg"
    image.write_bytes(b"fake-jpeg")

    provider = OpenRouterMultimodalProvider()

    content = provider._build_content(
        query="What is shown?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[{"image_path": str(image)}],
    )

    assert len(content) == 2
    assert content[1]["type"] == "image_url"


# --------------------------------------------------------------
# Attached image (the subject of the question)
# --------------------------------------------------------------
def test_attached_image_is_sent_first_and_labelled(tmp_path):
    attachment = tmp_path / "attachment.jpg"
    attachment.write_bytes(b"user-photo")
    page = tmp_path / "page.jpg"
    page.write_bytes(b"catalog-page")

    provider = OpenRouterMultimodalProvider()

    # Deliberately out of order: the attachment must lead, and must
    # be called out so the model treats it as the question's subject
    # rather than just another context image.
    content = provider._build_content(
        query="What is this?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[
            {"image_path": str(page), "kind": "retrieved"},
            {"image_path": str(attachment), "kind": "attachment"},
        ],
    )

    labels = [
        part["text"] for part in content if part["type"] == "text"
    ]
    assert any("FIRST image" in text for text in labels)
    # The attachment prompt is the one that treats the picture itself
    # as evidence, instead of only the retrieved context.
    assert any("primary source of truth" in text for text in labels)

    images = [
        part["image_url"]["url"]
        for part in content
        if part["type"] == "image_url"
    ]
    assert base64.b64encode(b"user-photo").decode(
        "utf-8"
    ) in images[0]
    assert base64.b64encode(b"catalog-page").decode(
        "utf-8"
    ) in images[1]


def test_ollama_labels_the_attached_image(monkeypatch, tmp_path):
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    attachment = tmp_path / "attachment.jpg"
    attachment.write_bytes(b"user-photo")

    provider = OllamaMultimodalProvider()

    captured: dict[str, Any] = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {"message": {"content": "ok"}}

    provider.client.chat = fake_chat

    provider.generate(
        query="What is this?",
        text_context=[],
        image_context=[
            {"image_path": str(attachment), "kind": "attachment"}
        ],
    )

    user_message = captured["messages"][1]
    assert "FIRST image" in user_message["content"]
    assert user_message["images"] == [
        base64.b64encode(b"user-photo").decode("utf-8")
    ]
    # The system prompt must make the attachment itself admissible as
    # evidence, otherwise a question about the picture is refused.
    assert "primary source of truth" in captured["messages"][0]["content"]


def test_document_grounded_prompt_is_kept_without_an_attachment(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    page = tmp_path / "page.jpg"
    page.write_bytes(b"catalog-page")

    provider = OllamaMultimodalProvider()

    captured: dict[str, Any] = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {"message": {"content": "ok"}}

    provider.client.chat = fake_chat

    provider.generate(
        query="What is shown?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[{"image_path": str(page), "kind": "retrieved"}],
    )

    system_message = captured["messages"][0]["content"]
    assert "I don't have enough information" in system_message
    assert "primary source of truth" not in system_message


def test_ollama_omits_the_attachment_note_for_retrieved_images(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    page = tmp_path / "page.jpg"
    page.write_bytes(b"catalog-page")

    provider = OllamaMultimodalProvider()

    captured: dict[str, Any] = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return {"message": {"content": "ok"}}

    provider.client.chat = fake_chat

    provider.generate(
        query="What is shown?",
        text_context=[{"page": 1, "content": "text"}],
        image_context=[{"image_path": str(page), "kind": "retrieved"}],
    )

    assert "FIRST image" not in captured["messages"][1]["content"]


# --------------------------------------------------------------
# Error translation (no network)
# --------------------------------------------------------------
class _FakeCompletions:
    def __init__(self, error):
        self.error = error

    def create(self, **kwargs):
        raise self.error


class _FakeChat:
    def __init__(self, error):
        self.completions = _FakeCompletions(error)


class _FakeClient:
    def __init__(self, error):
        self.chat = _FakeChat(error)
        self.api_key = "key"


def test_rate_limit_is_translated(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "key")

    provider = OpenRouterMultimodalProvider()
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    provider.client = _FakeClient(
        RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=request),
            body=None,
        )
    )

    with pytest.raises(BaseProviderRateLimitError):
        provider.generate(
            query="q",
            text_context=[],
            image_context=[],
        )


def test_timeout_is_translated(monkeypatch):
    monkeypatch.setattr(settings, "opencode_api_key", "key")

    provider = OpenCodeMultimodalProvider()
    request = httpx.Request("POST", "https://opencode.ai/zen/v1/chat/completions")
    provider.client = _FakeClient(
        APITimeoutError(request=request)
    )

    with pytest.raises(BaseProviderTimeoutError):
        provider.generate(
            query="q",
            text_context=[],
            image_context=[],
        )


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.setattr(settings, "opencode_api_key", "")

    provider = OpenCodeMultimodalProvider()

    with pytest.raises(ProviderError):
        provider.generate(
            query="q",
            text_context=[],
            image_context=[],
        )


def test_ollama_connection_error_is_translated(monkeypatch):
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    provider = OllamaMultimodalProvider()

    def raise_connect_error(**kwargs):
        raise httpx.ConnectError("refused")

    provider.client.chat = raise_connect_error

    with pytest.raises(ProviderError):
        provider.generate(
            query="q",
            text_context=[],
            image_context=[],
        )


def test_ollama_timeout_is_translated(monkeypatch):
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    provider = OllamaMultimodalProvider()

    def raise_timeout(**kwargs):
        raise httpx.ReadTimeout("too slow")

    provider.client.chat = raise_timeout

    with pytest.raises(BaseProviderTimeoutError):
        provider.generate(
            query="q",
            text_context=[],
            image_context=[],
        )

# --------------------------------------------------------------
# Streaming
# --------------------------------------------------------------
def test_base_stream_defaults_to_one_chunk():
    """
    A provider with no streaming support must still stream.

    The API always speaks SSE, so the default implementation emits the
    buffered answer as a single chunk instead of failing - a
    text-only gateway degrades to "all at once", not to an error.
    """
    from app.generation.providers.base import BaseMultimodalProvider

    class BufferedOnly(BaseMultimodalProvider):
        def generate(
            self, query, text_context, image_context, conversation=None
        ):
            return "complete answer"

    chunks = list(
        BufferedOnly().stream(
            query="q",
            text_context=[],
            image_context=[],
        )
    )

    assert chunks == ["complete answer"]


def test_ollama_stream_yields_deltas(monkeypatch):
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    provider = OllamaMultimodalProvider()

    captured: dict[str, Any] = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return iter(
            [
                {"message": {"content": "Dell "}},
                {"message": {"content": "Precision"}},
                {"message": {"content": ""}},
            ]
        )

    provider.client.chat = fake_chat

    chunks = list(
        provider.stream(
            query="What is shown?",
            text_context=[{"page": 1, "content": "text"}],
            image_context=[],
        )
    )

    assert captured["stream"] is True
    # Empty deltas are dropped: forwarding them would make the client
    # render stray separators.
    assert chunks == ["Dell ", "Precision"]


def test_ollama_stream_prompts_identically_to_generate(
    monkeypatch, tmp_path
):
    """Both paths must send the same messages, or answers diverge."""
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    attachment = tmp_path / "attachment.jpg"
    attachment.write_bytes(b"user-photo")

    provider = OllamaMultimodalProvider()

    buffered: dict[str, Any] = {}
    streamed: dict[str, Any] = {}

    provider.client.chat = lambda **kw: buffered.update(kw) or {
        "message": {"content": "ok"}
    }
    provider.generate(
        query="What is this?",
        text_context=[],
        image_context=[
            {"image_path": str(attachment), "kind": "attachment"}
        ],
    )

    provider.client.chat = lambda **kw: streamed.update(kw) or iter([])
    list(
        provider.stream(
            query="What is this?",
            text_context=[],
            image_context=[
                {"image_path": str(attachment), "kind": "attachment"}
            ],
        )
    )

    assert buffered["messages"] == streamed["messages"]


def test_ollama_stream_translates_mid_stream_error(monkeypatch):
    """
    A failure after some text has been emitted must still raise.

    Ending the generator quietly would leave the client showing a
    half-finished answer that looks complete.
    """
    monkeypatch.setattr(
        settings, "ollama_base_url", "http://localhost:11434"
    )

    provider = OllamaMultimodalProvider()

    def failing_chat(**kwargs):
        def generate():
            yield {"message": {"content": "Dell "}}
            raise httpx.ReadTimeout("too slow")

        return generate()

    provider.client.chat = failing_chat

    stream = provider.stream(
        query="q", text_context=[], image_context=[]
    )

    assert next(stream) == "Dell "

    with pytest.raises(BaseProviderTimeoutError):
        next(stream)


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeStreamCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return iter(
                [
                    _FakeChunk("Dell "),
                    _FakeChunk(None),
                    _FakeChunk("Precision"),
                ]
            )
        return None


class _FakeStreamClient:
    def __init__(self):
        self.completions = _FakeStreamCompletions()

        class Chat:
            completions = self.completions

        self.chat = Chat()
        self.api_key = "key"


def test_openai_compatible_stream_yields_deltas(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "key")

    provider = OpenRouterMultimodalProvider()
    fake_client = _FakeStreamClient()
    provider.client = fake_client

    chunks = list(
        provider.stream(
            query="q",
            text_context=[],
            image_context=[],
        )
    )

    assert fake_client.completions.kwargs["stream"] is True
    assert chunks == ["Dell ", "Precision"]


def test_openai_compatible_stream_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "opencode_api_key", "")

    provider = OpenCodeMultimodalProvider()

    with pytest.raises(ProviderError):
        list(
            provider.stream(
                query="q",
                text_context=[],
                image_context=[],
            )
        )
