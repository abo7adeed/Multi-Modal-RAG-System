"""
Unit tests for the hardened pipeline: input validation,
provider-error translation, and application error mapping.

Uses fakes for retriever/context builder/generator - no models,
no network, no vector store.
"""
import pytest

from app.errors import (
    GenerationUnavailableError,
    InvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RetrievalError,
)
from app.generation.context_builder import ContextBuilder
from app.generation.schemas import RAGResponse, Source
from app.generation.providers.base import (
    ProviderError,
    ProviderRateLimitError as ProviderRateLimitErrorBase,
    ProviderTimeoutError as ProviderTimeoutErrorBase,
)
from app.pipeline import MultimodalRAGPipeline


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error

    def search_multimodal(self, query, **kwargs):
        if self.error:
            raise self.error
        return self.results


class FakeGenerator:
    def __init__(self, answer="ok", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def generate(self, query, context):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.answer


SAMPLE_RESULTS = [
    {
        "chunk_id": "text-001",
        "document": "Dell Precision laptop with Intel Core i7.",
        "metadata": {
            "content_type": "text",
            "page_number": 10,
            "image_path": "",
        },
        "source": "retrieved",
    }
]


def build_pipeline(retriever, generator):
    return MultimodalRAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generator=generator,
    )


# --------------------------------------------------------------
# Input validation
# --------------------------------------------------------------
def test_run_rejects_empty_query():
    pipeline = build_pipeline(FakeRetriever(), FakeGenerator())

    with pytest.raises(InvalidRequestError):
        pipeline.run("   ")


def test_run_rejects_oversized_query():
    pipeline = build_pipeline(FakeRetriever(), FakeGenerator())

    with pytest.raises(InvalidRequestError):
        pipeline.run("x" * 2001)


def test_run_strips_query():
    retriever = FakeRetriever(results=SAMPLE_RESULTS)
    pipeline = build_pipeline(retriever, FakeGenerator())

    pipeline.run("  Dell laptops  ")

    assert retriever.results == SAMPLE_RESULTS


# --------------------------------------------------------------
# Happy path
# --------------------------------------------------------------
def test_run_returns_response_with_sources():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeGenerator(answer="Dell Precision is in the catalog."),
    )

    response = pipeline.run("Dell laptop")

    assert isinstance(response, RAGResponse)
    assert response.answer == "Dell Precision is in the catalog."
    assert response.sources[0].page == 10
    assert response.sources[0].content_type == "text"


# --------------------------------------------------------------
# Off-topic short circuit
# --------------------------------------------------------------
def test_run_short_circuits_when_nothing_is_relevant():
    generator = FakeGenerator(answer="Hello! How can I help?")

    pipeline = build_pipeline(
        FakeRetriever(results=[]),  # retrieval found no match
        generator,
    )

    response = pipeline.run("hello")

    # The model must not be consulted for an unmatched query, so a
    # conversational reply like "How can I help?" cannot be returned.
    assert response.answer == (
        "I don't have enough information in the provided documents."
    )
    assert response.sources == []
    assert generator.calls == []


def test_run_still_generates_when_results_exist():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeGenerator(answer="Grounded answer."),
    )

    response = pipeline.run("Dell laptop")

    assert response.answer == "Grounded answer."
    assert len(response.sources) == 1


def test_run_rejects_empty_answer():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeGenerator(answer="   "),
    )

    with pytest.raises(GenerationUnavailableError):
        pipeline.run("Dell laptop")


# --------------------------------------------------------------
# Retrieval failures
# --------------------------------------------------------------
def test_retrieval_error_wraps_unexpected_exception():
    pipeline = build_pipeline(
        FakeRetriever(error=RuntimeError("chroma exploded")),
        FakeGenerator(),
    )

    with pytest.raises(RetrievalError):
        pipeline.run("Dell laptop")


# --------------------------------------------------------------
# Provider failure translation
# --------------------------------------------------------------
def test_provider_rate_limit_maps_to_app_error():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeGenerator(error=ProviderRateLimitErrorBase("429")),
    )

    with pytest.raises(ProviderRateLimitError):
        pipeline.run("Dell laptop")


def test_provider_timeout_maps_to_app_error():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeGenerator(error=ProviderTimeoutErrorBase("timeout")),
    )

    with pytest.raises(ProviderTimeoutError):
        pipeline.run("Dell laptop")


def test_provider_generic_error_maps_to_unavailable():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeGenerator(error=ProviderError("boom")),
    )

    with pytest.raises(GenerationUnavailableError):
        pipeline.run("Dell laptop")


def test_provider_errors_never_propagate_raw():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeGenerator(error=ProviderRateLimitErrorBase("raw sdk error")),
    )

    with pytest.raises(ProviderRateLimitError):
        pipeline.run("Dell laptop")


# --------------------------------------------------------------
# describe_provider introspection
# --------------------------------------------------------------
def test_describe_provider_reports_model():
    class FakeProvider:
        model = "test-model"

    class GeneratorWithProvider(FakeGenerator):
        provider = FakeProvider()

    pipeline = build_pipeline(FakeRetriever(), GeneratorWithProvider())

    info = pipeline.describe_provider()

    assert info["model"] == "test-model"
    assert info["provider"] == "FakeProvider"
