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
        self.queries: list[str] = []

    def search_multimodal(self, query, **kwargs):
        self.queries.append(query)
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
        self.contexts = getattr(self, "contexts", [])
        self.contexts.append(context)
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


# --------------------------------------------------------------
# Streaming
# --------------------------------------------------------------
class FakeStreamingGenerator(FakeGenerator):
    """Generator that yields the answer in fragments."""

    def __init__(self, chunks=None, error=None):
        super().__init__(error=error)
        self.chunks = chunks if chunks is not None else ["Dell ", "Precision."]
        self.stream_calls = []

    def stream(self, query, context):
        self.stream_calls.append(query)

        def generate():
            for index, chunk in enumerate(self.chunks):
                # A failure part-way through must propagate, not end
                # the generator quietly.
                if self.error and index == 1:
                    raise self.error
                yield chunk

        return generate()


def test_run_stream_emits_sources_then_tokens_then_done():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeStreamingGenerator(),
    )

    events = list(
        pipeline.run_stream("What is the Dell Precision?")
    )

    types = [event.type for event in events]
    # Order is part of the contract: clients render sources as soon
    # as they arrive, and treat a missing `done` as a failure.
    assert types == ["sources", "token", "token", "done"]

    assert [event.text for event in events if event.type == "token"] == [
        "Dell ",
        "Precision.",
    ]
    assert events[0].sources[0].chunk_id == "text-001"


def test_run_stream_does_not_call_the_buffered_path():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeStreamingGenerator(),
    )

    list(pipeline.run_stream("What is the Dell Precision?"))

    assert pipeline.generator.stream_calls == [
        "What is the Dell Precision?"
    ]
    assert pipeline.generator.calls == []


def test_run_stream_short_circuits_off_topic_without_the_model():
    generator = FakeStreamingGenerator()

    pipeline = build_pipeline(FakeRetriever(results=[]), generator)

    events = list(pipeline.run_stream("What is the capital of France?"))

    assert [event.type for event in events] == ["token", "done"]
    assert events[0].text == (
        "I don't have enough information in the provided documents."
    )
    assert generator.stream_calls == []


def test_run_stream_validates_before_returning_the_generator():
    """
    Input validation runs eagerly, when run_stream() is called.

    The API layer calls it while it can still send a real HTTP status;
    deferring validation into the generator would turn a 400 into a
    200 carrying an error event.
    """
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeStreamingGenerator(),
    )

    with pytest.raises(InvalidRequestError):
        pipeline.run_stream("   ")

    assert pipeline.generator.stream_calls == []


def test_run_stream_translates_provider_errors():
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeStreamingGenerator(error=ProviderError("provider down")),
    )

    stream = pipeline.run_stream("What is the Dell Precision?")

    assert next(stream).type == "sources"
    assert next(stream).text == "Dell "

    with pytest.raises(GenerationUnavailableError):
        next(stream)


def test_run_stream_translates_rate_limit_and_timeout():
    for provider_error, expected in [
        (ProviderRateLimitErrorBase("429"), ProviderRateLimitError),
        (ProviderTimeoutErrorBase("504"), ProviderTimeoutError),
    ]:
        pipeline = build_pipeline(
            FakeRetriever(results=SAMPLE_RESULTS),
            FakeStreamingGenerator(error=provider_error),
        )

        stream = pipeline.run_stream("What is the Dell Precision?")

        # `sources`, then the first token; the fake fails on the
        # second, which is where the provider would really break.
        next(stream)
        next(stream)

        with pytest.raises(expected):
            next(stream)


def test_run_stream_rejects_an_empty_answer():
    """
    An empty stream is a failure, not an answer.

    Without this, a provider that returns nothing would produce a
    clean `done` event and the user would see a blank reply with no
    indication that anything went wrong.
    """
    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        FakeStreamingGenerator(chunks=["", ""]),
    )

    stream = pipeline.run_stream("What is the Dell Precision?")

    # `sources` is emitted, but the blank tokens are dropped and the
    # stream then fails instead of ending in a clean `done`.
    next(stream)

    with pytest.raises(GenerationUnavailableError):
        next(stream)


def test_run_stream_rejects_an_image_for_a_text_only_model():
    """
    A model that cannot read images must refuse before generating.

    Answering anyway would mean inventing the answer, so this is an
    error rather than a degraded reply.
    """
    import io as _io

    from PIL import Image

    from app.errors import ImageVisionUnsupportedError
    from app.generation.schemas import ImageAttachment

    class TextOnlyGenerator(FakeGenerator):
        supports_images = False

    buffer = _io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="JPEG")

    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS),
        TextOnlyGenerator(),
    )

    with pytest.raises(ImageVisionUnsupportedError):
        pipeline.run_stream(
            "What colour is this?",
            image=ImageAttachment(data=buffer.getvalue()),
        )
