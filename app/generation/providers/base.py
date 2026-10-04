from abc import ABC, abstractmethod
from typing import Any


class BaseMultimodalProvider(ABC):

    @abstractmethod
    def generate(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
    ) -> str:
        """
        Generate an answer using text and image context.
        """
        raise NotImplementedError
class ProviderError(Exception):
    """Base exception for generation provider failures."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider rate limit is exceeded."""


class ProviderTimeoutError(ProviderError):
    """Raised when the provider request times out."""