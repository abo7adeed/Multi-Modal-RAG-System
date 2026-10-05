"""
Questions about an image the user attaches to the chat message.

The attached image is the *subject* of the question, so it must reach
the model even when the index has nothing textually related to it.
"""
import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from app.errors import (
    ImageVisionUnsupportedError,
    InvalidRequestError,
)
from app.generation.context_builder import ContextBuilder
from app.generation.schemas import ImageAttachment
from app.pipeline import MultimodalRAGPipeline


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def png_bytes(size=(8, 8), color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def attachment(data: bytes | None = None, filename="photo.png") -> ImageAttachment:
    return ImageAttachment(
        filename=filename,
        media_type="image/png",
        data=data if data is not None else png_bytes(),
    )


SAMPLE_RESULTS = [
    {
        "chunk_id": "text-001",
        "document": "Dell OptiPlex 3020 Micro desktop.",
        "metadata": {
            "content_type": "text",
            "page_number": 2,
            "image_path": "",
            "source": "data\\raw\\documents\\dell_catalog.pdf",
        },
        "source": "retrieved",
    }
]


class FakeRetriever:
    def __init__(self, results=None, visual=None, visual_error=None):
        self.results = results or []
        self.visual = visual or []
        self.visual_error = visual_error
        self.visual_queries = []

    def search_multimodal(self, query, **kwargs):
        return self.results

    def search_by_image(self, image_path, top_k=2):
        self.visual_queries.append((image_path, top_k))
        if self.visual_error:
            raise self.visual_error
        return self.visual


class FakeGenerator:
    supports_images = True

    def __init__(self, answer="ok", error=None):
        self.answer = answer
        self.error = error
        self.contexts = []
        # Snapshot of the staged attachment *while the provider still
        # has it*: the pipeline deletes it once the request is done.
        self.staged = None

    def _snapshot(self, context):
        for item in context.image_context:
            if item.get("kind") == "attachment":
                path = Path(item["image_path"])
                if path.exists():
                    with Image.open(path) as image:
                        self.staged = {
                            "path": str(path),
                            "format": image.format,
                            "size": image.size,
                        }
                        return

    def generate(self, query, context):
        self.contexts.append(context)
        self._snapshot(context)
        if self.error:
            raise self.error
        return self.answer


class TextOnlyGenerator(FakeGenerator):
    supports_images = False


def build_pipeline(retriever, generator):
    return MultimodalRAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generator=generator,
    )


# --------------------------------------------------------------
# The gate must not swallow an attached image
# --------------------------------------------------------------
def test_attachment_answers_even_with_no_relevant_documents():
    # The lexical gate returns nothing here (as it does for any
    # question phrased in words absent from the index), but the user
    # is asking about a picture, not about the documents.
    generator = FakeGenerator(answer="A red square.")

    pipeline = build_pipeline(
        FakeRetriever(results=[]), generator
    )

    response = pipeline.run(
        "What is in this picture?", image=attachment()
    )

    assert response.answer == "A red square."
    assert generator.contexts, "the model must be consulted"


def test_attachment_is_passed_to_the_model_as_image_context():
    generator = FakeGenerator(answer="A red square.")

    pipeline = build_pipeline(FakeRetriever(), generator)
    pipeline.run("What is this?", image=attachment())

    image_context = generator.contexts[0].image_context
    assert len(image_context) == 1
    assert image_context[0]["kind"] == "attachment"
    assert image_context[0]["source"] == "photo.png"


def test_attachment_is_staged_as_a_readable_jpeg():
    generator = FakeGenerator(answer="ok")

    pipeline = build_pipeline(FakeRetriever(), generator)
    pipeline.run("What is this?", image=attachment())

    # The staged file must exist while the provider reads it, and be
    # a real JPEG whatever the client uploaded.
    assert generator.staged["format"] == "JPEG"
    assert generator.staged["size"] == (8, 8)


def test_attachment_is_never_reported_as_a_source():
    generator = FakeGenerator(answer="A red square.")

    pipeline = build_pipeline(FakeRetriever(), generator)
    response = pipeline.run("What is this?", image=attachment())

    assert response.sources == []


def test_temporary_attachment_file_is_deleted_afterwards():
    generator = FakeGenerator(answer="ok")

    pipeline = build_pipeline(FakeRetriever(), generator)
    pipeline.run("What is this?", image=attachment())

    assert not Path(generator.staged["path"]).exists()


def test_temporary_attachment_file_is_deleted_when_generation_fails():
    generator = FakeGenerator(error=RuntimeError("provider down"))

    pipeline = build_pipeline(FakeRetriever(), generator)

    with pytest.raises(Exception):
        pipeline.run("What is this?", image=attachment())

    assert not Path(generator.staged["path"]).exists()


# --------------------------------------------------------------
# Visual enrichment
# --------------------------------------------------------------
def test_visual_matches_support_the_answer_but_are_not_cited():
    retriever = FakeRetriever(
        visual=[
            {
                "chunk_id": "image-001",
                "document": "Image from page 4",
                "metadata": {
                    "content_type": "image",
                    "page_number": 4,
                    "image_path": "page_4.jpg",
                },
                "source": "retrieved",
            }
        ]
    )
    generator = FakeGenerator(answer="Same product.")

    pipeline = build_pipeline(retriever, generator)
    response = pipeline.run("Is this the same?", image=attachment())

    # The lookalike page reaches the model as supporting context...
    image_context = generator.contexts[0].image_context
    assert any(
        item.get("kind") == "visual_match"
        and item.get("chunk_id") == "image-001"
        for item in image_context
    )

    # ...but it is never reported as a source: merely resembling the
    # user's photo does not mean it grounded the answer, and citing it
    # would point the user at an unrelated catalog page.
    assert response.sources == []


def test_visual_search_uses_the_configured_limit():
    retriever = FakeRetriever()

    pipeline = build_pipeline(retriever, FakeGenerator())
    pipeline.run("What is this?", image=attachment())

    from app.config import settings

    _path, top_k = retriever.visual_queries[0]
    assert top_k == settings.retrieval_visual_top_k


def test_no_visual_search_for_text_only_questions():
    retriever = FakeRetriever(results=SAMPLE_RESULTS)

    pipeline = build_pipeline(retriever, FakeGenerator())
    pipeline.run("What is the OptiPlex?")

    assert retriever.visual_queries == []


def test_visual_search_failure_does_not_fail_the_request():
    retriever = FakeRetriever(visual_error=RuntimeError("CLIP down"))
    generator = FakeGenerator(answer="A red square.")

    pipeline = build_pipeline(retriever, generator)
    response = pipeline.run("What is this?", image=attachment())

    assert response.answer == "A red square."


def test_no_visual_search_without_an_attachment():
    retriever = FakeRetriever(results=SAMPLE_RESULTS)

    pipeline = build_pipeline(retriever, FakeGenerator())
    pipeline.run("What is the OptiPlex?")

    assert retriever.visual_queries == []


# --------------------------------------------------------------
# Text-only models
# --------------------------------------------------------------
def test_text_only_model_refuses_image_questions():
    generator = TextOnlyGenerator()

    pipeline = build_pipeline(FakeRetriever(), generator)

    with pytest.raises(ImageVisionUnsupportedError):
        pipeline.run("What is this?", image=attachment())


def test_text_only_model_still_answers_text_questions():
    generator = TextOnlyGenerator(answer="Grounded answer.")

    pipeline = build_pipeline(
        FakeRetriever(results=SAMPLE_RESULTS), generator
    )
    response = pipeline.run("What is the OptiPlex?")

    assert response.answer == "Grounded answer."


# --------------------------------------------------------------
# Staging rejects payloads that are not images
# --------------------------------------------------------------
def test_non_image_bytes_are_rejected():
    pipeline = build_pipeline(FakeRetriever(), FakeGenerator())

    with pytest.raises(InvalidRequestError):
        pipeline.run(
            "What is this?",
            image=attachment(b"%PDF-1.4 not an image at all"),
        )


def test_filename_cannot_escape_the_temporary_directory():
    generator = FakeGenerator(answer="ok")

    pipeline = build_pipeline(FakeRetriever(), generator)
    pipeline.run(
        "What is this?",
        image=attachment(filename="../../../../etc/passwd"),
    )

    # The client filename is display-only: it must never reach a path.
    assert Path(generator.staged["path"]).name == "attachment.jpg"
    assert generator.contexts[0].image_context[0]["source"] == (
        "../../../../etc/passwd"
    )
