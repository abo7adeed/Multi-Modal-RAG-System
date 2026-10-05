import logging
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from app.config import settings
from app.errors import (
    GenerationUnavailableError,
    ImageVisionUnsupportedError,
    InvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RAGPipelineError,
    RetrievalError,
    log_pipeline_error,
)
from app.generation.context_builder import ContextBuilder
from app.generation.conversation import (
    ConversationMemory,
    ConversationTurn,
    rewrite_query,
)
from app.generation.generator import MultimodalGenerator
from app.generation.providers.base import ProviderError
from app.generation.providers.base import (
    ProviderRateLimitError as ProviderRateLimitErrorBase,
)
from app.generation.providers.base import (
    ProviderTimeoutError as ProviderTimeoutErrorBase,
)
from app.generation.schemas import (
    GenerationContext,
    ImageAttachment,
    RAGResponse,
    RAGStreamEvent,
)
from app.ingestion.image_normalizer import ImageNormalizer
from app.retrieval.retriever import MultimodalRetriever

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 2000

NO_INFORMATION_ANSWER = (
    "I don't have enough information in the provided documents."
)


class MultimodalRAGPipeline:
    """
    Orchestrates retrieval -> context -> generation.

    Raises only application-level errors (app.errors). Provider and
    SDK exceptions are translated here; the API layer never sees raw
    SDK exceptions. Technical details go to structured logs, while
    user-facing messages travel on the exception itself.
    """

    def __init__(
        self,
        retriever: MultimodalRetriever,
        context_builder: ContextBuilder,
        generator: MultimodalGenerator,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

    def run(
        self,
        query: str,
        image: ImageAttachment | None = None,
        history: list[ConversationTurn] | None = None,
    ) -> RAGResponse:
        query = self._validate_query(query)

        # --------------------------------------------------------
        # Attached image
        #
        # The temporary file lives only for this request: providers
        # read images from disk, so the upload has to be written out
        # somewhere, and it must never outlive the answer.
        # --------------------------------------------------------
        with self._staged_image(image) as attachments:
            # Answering about an image the model cannot see would mean
            # inventing the answer, so refuse before generating.
            vision_capable = getattr(
                self.generator, "supports_images", True
            )
            if attachments and not vision_capable:
                raise ImageVisionUnsupportedError()

            return self._answer(
                query=query,
                attachments=attachments,
                memory=ConversationMemory(history),
            )

    def run_stream(
        self,
        query: str,
        image: ImageAttachment | None = None,
        history: list[ConversationTurn] | None = None,
    ) -> Iterator[RAGStreamEvent]:
        """
        Answer a query, yielding events as the answer arrives.

        Deliberately NOT a generator function: everything that can
        fail cheaply - input validation, attachment decoding, the
        vision-capability check and retrieval - runs eagerly, before
        this returns. The caller therefore still has no response
        bytes on the wire and can report those failures with a real
        HTTP status, exactly like the buffered route.

        Only generation is deferred, because that is the slow part
        and the entire reason for streaming. A failure *there* has to
        be reported as an SSE `error` event, since the response has
        already started by then.

        Event order is guaranteed: `sources` (when the model is
        called at all) precedes every `token`, and `done` is last.
        """
        query = self._validate_query(query)

        # The staging context manager spans the whole iteration: the
        # attachment must still exist while the provider streams, so
        # it is entered here and closed when the token generator ends
        # (including on early exit).
        with ExitStack() as staging:
            attachments = staging.enter_context(self._staged_image(image))

            # Answering about an image the model cannot see would mean
            # inventing the answer, so refuse before generating.
            vision_capable = getattr(
                self.generator, "supports_images", True
            )
            if attachments and not vision_capable:
                raise ImageVisionUnsupportedError()

            context = self._prepare(
                query=query,
                attachments=attachments,
                memory=ConversationMemory(history),
            )

            return self._stream_tokens(
                query=query,
                context=context,
                staging=staging,
            )

    def _stream_tokens(
        self,
        query: str,
        context: GenerationContext | None,
        staging: ExitStack,
    ) -> Iterator[RAGStreamEvent]:
        """
        Yield the answer as it is generated.

        `staging` is closed when iteration ends, so the temporary
        attachment file lives exactly as long as the stream.
        """
        try:
            if context is None:
                # Off-topic: answered without calling the model.
                yield RAGStreamEvent(
                    type="token",
                    text=NO_INFORMATION_ANSWER,
                )
                yield RAGStreamEvent(type="done")
                return

            yield RAGStreamEvent(
                type="sources",
                sources=context.sources,
            )

            emitted = False

            try:
                for chunk in self.generator.stream(
                    query=query,
                    context=context,
                ):
                    # Blank deltas are noise from the provider; they
                    # would render as stray whitespace in the UI.
                    if not chunk:
                        continue
                    emitted = True
                    yield RAGStreamEvent(type="token", text=chunk)

            except ProviderRateLimitErrorBase as exc:
                log_pipeline_error(
                    exc, stage="generation", query=query
                )
                raise ProviderRateLimitError(str(exc)) from exc
            except ProviderTimeoutErrorBase as exc:
                log_pipeline_error(
                    exc, stage="generation", query=query
                )
                raise ProviderTimeoutError(str(exc)) from exc
            except ProviderError as exc:
                log_pipeline_error(
                    exc, stage="generation", query=query
                )
                raise GenerationUnavailableError(str(exc)) from exc
            except RAGPipelineError as exc:
                log_pipeline_error(
                    exc, stage="generation", query=query
                )
                raise
            except Exception as exc:
                log_pipeline_error(
                    exc, stage="generation", query=query
                )
                raise GenerationUnavailableError(
                    "Generation failed unexpectedly."
                ) from exc

            # An empty answer is an error, not an answer - the same
            # rule as run(). Raised after the tokens so a client that
            # already rendered text can tell it is incomplete, and so
            # a blank stream never ends in a clean `done`.
            if not emitted:
                raise GenerationUnavailableError(
                    "The model returned an empty answer."
                )

            yield RAGStreamEvent(type="done")

        finally:
            staging.close()

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Normalise and bounds-check the query.

        Raises InvalidRequestError, which the API maps to a 400 -
        for the streaming route this happens before the response
        body starts, so it can still use a real HTTP status.
        """
        if not isinstance(query, str):
            raise InvalidRequestError("Query must be a string.")

        query = query.strip()

        if not query:
            raise InvalidRequestError("Query must not be empty.")

        if len(query) > MAX_QUERY_LENGTH:
            raise InvalidRequestError(
                "Query exceeds the maximum length of "
                f"{MAX_QUERY_LENGTH} characters."
            )

        return query

    def _answer(
        self,
        query: str,
        attachments: list[dict[str, str]],
        memory: ConversationMemory | None = None,
    ) -> RAGResponse:
        """
        Answer a prepared query and return the whole response.

        Shares retrieval and context building with run_stream() via
        _prepare(), so the streamed and non-streamed paths cannot
        drift apart in what they retrieve or cite.
        """
        context = self._prepare(
            query=query,
            attachments=attachments,
            memory=memory,
        )

        if context is None:
            # Off-topic: answered without calling the model.
            return RAGResponse(
                answer=NO_INFORMATION_ANSWER,
                sources=[],
            )

        # --------------------------------------------------------
        # Generation
        # --------------------------------------------------------
        try:
            answer = self.generator.generate(
                query=query,
                context=context,
            )
        except ProviderRateLimitErrorBase as exc:
            log_pipeline_error(exc, stage="generation", query=query)
            raise ProviderRateLimitError(str(exc)) from exc
        except ProviderTimeoutErrorBase as exc:
            log_pipeline_error(exc, stage="generation", query=query)
            raise ProviderTimeoutError(str(exc)) from exc
        except ProviderError as exc:
            log_pipeline_error(exc, stage="generation", query=query)
            raise GenerationUnavailableError(str(exc)) from exc
        except RAGPipelineError as exc:
            log_pipeline_error(exc, stage="generation", query=query)
            raise
        except Exception as exc:
            log_pipeline_error(exc, stage="generation", query=query)
            raise GenerationUnavailableError(
                "Generation failed unexpectedly."
            ) from exc

        if not answer or not answer.strip():
            raise GenerationUnavailableError(
                "The model returned an empty answer."
            )

        return RAGResponse(
            answer=answer,
            sources=context.sources,
        )

    def _prepare(
        self,
        query: str,
        attachments: list[dict[str, str]],
        memory: ConversationMemory | None = None,
    ) -> GenerationContext | None:
        """
        Retrieve and build the generation context.

        Returns None when the query is off-topic and the model
        should not be called at all.
        """
        # --------------------------------------------------------
        # Follow-up resolution
        #
        # A question like "what about its warranty?" has no subject
        # to search for, so the referent is resolved against the
        # conversation *before* retrieval. The model still receives
        # the user's own wording plus the earlier turns.
        # --------------------------------------------------------
        rewritten = rewrite_query(query, memory)

        if rewritten.was_rewritten:
            logger.info(
                "pipeline_follow_up_rewrite reason=%s "
                "search_query=%r",
                rewritten.reason,
                rewritten.search_query,
            )

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------
        try:
            results = self.retriever.search_multimodal(
                rewritten.search_query
            )
        except RAGPipelineError as exc:
            log_pipeline_error(exc, stage="retrieval", query=query)
            raise
        except Exception as exc:
            log_pipeline_error(exc, stage="retrieval", query=query)
            raise RetrievalError("Failed to retrieve documents.") from exc

        # --------------------------------------------------------
        # Visual enrichment
        #
        # An attached image makes a text question that shares no
        # words with the index answerable, so the lexical gate below
        # must not end the request. Instead, page images that look
        # like the attachment are added as supporting context.
        # --------------------------------------------------------
        if attachments:
            results = results + self._visual_matches(
                attachments[0]["image_path"]
            )

        # --------------------------------------------------------
        # Off-topic short circuit
        #
        # If retrieval found nothing genuinely relevant (no lexical
        # overlap with the index), answer without calling the model.
        # This keeps the system grounded and avoids both wasted
        # generation and conversational drift on unrelated questions.
        #
        # It does not apply when an image is attached: the user is
        # asking about that image, which is evidence in itself.
        # --------------------------------------------------------
        if not results and not attachments:
            logger.info(
                "pipeline_no_match query=%r", " ".join(query.split())[:120]
            )
            return None

        # --------------------------------------------------------
        # Context building
        # --------------------------------------------------------
        try:
            context = self.context_builder.build(
                query=query,
                results=results,
                attachments=attachments,
                conversation=(
                    memory.as_prompt_context() if memory else None
                ),
            )
        except RAGPipelineError as exc:
            log_pipeline_error(exc, stage="context", query=query)
            raise
        except Exception as exc:
            log_pipeline_error(exc, stage="context", query=query)
            raise RAGPipelineError(
                "Failed to build generation context."
            ) from exc

        return context

    # ------------------------------------------------------------
    # Attached image staging
    # ------------------------------------------------------------

    @contextmanager
    def _staged_image(
        self,
        image: ImageAttachment | None,
    ) -> Iterator[list[dict[str, str]]]:
        """
        Materialise an attached image as a temporary JPEG.

        The name is fixed on purpose: the client-supplied filename is
        kept for display only, never used to build a path.
        """
        if image is None:
            yield []
            return

        with tempfile.TemporaryDirectory(
            prefix="rag-attachment-"
        ) as temp_dir:
            upload = Path(temp_dir) / "upload.bin"
            upload.write_bytes(image.data)

            # Re-encoding through the ingestion normalizer guarantees
            # the model receives a format it can decode (JPEG/RGB),
            # whatever the client uploaded. It doubles as a backstop
            # for payloads that are not decodable images at all.
            try:
                normalized = ImageNormalizer().normalize(
                    upload, Path(temp_dir) / "attachment.jpg"
                )
            except Exception as exc:
                log_pipeline_error(
                    exc, stage="attachment", query=None
                )
                reason = "The attached file is not a valid image."
                raise InvalidRequestError(
                    reason, user_message=reason
                ) from exc

            yield [
                {
                    "image_path": str(normalized),
                    "filename": image.filename or "attachment",
                }
            ]

    def _visual_matches(
        self,
        image_path: str,
    ) -> list[dict]:
        """
        Page images that resemble the attachment.

        Best effort: the attachment alone is enough to answer, so a
        CLIP failure must never fail the request.
        """
        try:
            matches = self.retriever.search_by_image(
                image_path,
                top_k=settings.retrieval_visual_top_k,
            )
        except Exception as exc:
            logger.warning(
                "visual_match_unavailable error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return []

        # Tag them so the context builder can feed them to the model as
        # supporting context without reporting them as sources: they
        # are lookalikes, not evidence for the answer.
        for match in matches:
            match["source"] = "visual_match"

        return matches

    # ------------------------------------------------------------
    # Introspection helper (used by the API layer)
    # ------------------------------------------------------------

    def describe_provider(self) -> dict[str, Any]:
        provider = getattr(self.generator, "provider", None)
        return {
            "provider": type(provider).__name__ if provider else None,
            "model": getattr(provider, "model", None),
        }