"""
Application-level error taxonomy.

Provider/SDK exceptions never cross module boundaries: the pipeline
translates every failure into one of these application errors.

The API layer maps these to HTTP status codes and user-facing
messages; technical details stay in the logs.
"""
import logging

logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """
    Base class for recoverable RAG pipeline failures.

    Carries a user-facing message that is safe to return to clients.
    Technical details (provider, model, error type) must go to logs,
    never to the user.
    """

    #: Safe, user-facing message. Subclasses override this.
    user_message: str = (
        "The request could not be completed. Please try again."
    )

    def __init__(
        self,
        *args: object,
        user_message: str | None = None,
    ):
        super().__init__(*args)

        # Per-instance override, for errors whose reason is already
        # written for end users (e.g. a rejected image upload) and
        # would only be replaced by a generic message.
        if user_message:
            self.user_message = user_message


class RetrievalError(RAGPipelineError):
    """Raised when document retrieval fails (store/embedder issues)."""

    user_message = (
        "The document index is temporarily unavailable. "
        "Please try again."
    )


class GenerationUnavailableError(RAGPipelineError):
    """Raised when the generation provider cannot serve the request."""

    user_message = (
        "Generation service is temporarily unavailable. "
        "Please try again."
    )


class ProviderRateLimitError(GenerationUnavailableError):
    """Raised when the upstream provider rate limit is exceeded."""

    user_message = (
        "Generation service is temporarily rate limited. "
        "Please try again in a moment."
    )


class ProviderTimeoutError(GenerationUnavailableError):
    """Raised when the upstream provider request times out."""

    user_message = (
        "Generation took too long to respond. Please try again."
    )


class InvalidRequestError(RAGPipelineError, ValueError):
    """
    Raised when the caller-supplied input is invalid
    (empty query, oversized input, unsupported file, ...).
    """

    user_message = "The request is invalid. Please check your input."


class ImageVisionUnsupportedError(RAGPipelineError):
    """
    Raised when a question about an attached image reaches a provider
    whose model cannot read images.

    Answering anyway would mean inventing an answer, so the request is
    refused with an actionable message instead.
    """

    user_message = (
        "The current model cannot read images. "
        "Switch to a vision-capable model to ask questions about "
        "an attached image."
    )


class DocumentIngestionError(RAGPipelineError):
    """Raised when an uploaded document cannot be ingested."""

    user_message = (
        "The document could not be processed. "
        "Please verify the file and try again."
    )


def log_pipeline_error(
    error: Exception,
    *,
    stage: str,
    query: str | None = None,
) -> None:
    """
    Log a structured technical error record.

    The query is truncated and stripped of newlines to keep log lines
    safe and single-line. Never log API keys or stack internals here.
    """
    safe_query = ""
    if query:
        safe_query = " ".join(query.split())[:120]

    logger.error(
        "pipeline_failure stage=%s error_type=%s error=%s query=%r",
        stage,
        type(error).__name__,
        error,
        safe_query,
    )
