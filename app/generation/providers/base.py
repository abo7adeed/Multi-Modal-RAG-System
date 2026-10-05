from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class BaseMultimodalProvider(ABC):

    @abstractmethod
    def generate(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ) -> str:
        """
        Generate an answer using text and image context.

        `conversation` carries earlier turns so a follow-up question
        can be resolved; it is optional and may be None.
        """
        raise NotImplementedError

    def stream(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ) -> Iterator[str]:
        """
        Yield the answer incrementally, in text chunks.

        Default implementation: produce the full answer and emit it as
        a single chunk. Providers that support token streaming
        override this, so the API can stream for every provider while
        a non-streaming backend degrades to one chunk rather than
        failing.
        """
        yield self.generate(
            query=query,
            text_context=text_context,
            image_context=image_context,
            conversation=conversation,
        )
class ProviderError(Exception):
    """Base exception for generation provider failures."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider rate limit is exceeded."""


class ProviderTimeoutError(ProviderError):
    """Raised when the provider request times out."""