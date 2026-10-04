"""
Generation providers and the factory that selects one.

Providers are chosen via Settings.generation_provider
("openrouter" | "ollama" | "opencode"), so switching backends is a
configuration change, not a code change.
"""
from app.config import settings

from .base import (
    BaseMultimodalProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from .ollama_provider import OllamaMultimodalProvider
from .openai_compatible import OpenAICompatibleProvider
from .opencode_provider import OpenCodeMultimodalProvider
from .openrouter_provider import OpenRouterMultimodalProvider


def build_provider() -> BaseMultimodalProvider:
    """
    Instantiate the configured generation provider.

    Raises ProviderError for an unknown provider name so the failure
    surfaces as a clean application error rather than a silent
    fallback to the wrong backend.
    """
    provider_name = settings.generation_provider

    if provider_name == "openrouter":
        return OpenRouterMultimodalProvider()

    if provider_name == "ollama":
        return OllamaMultimodalProvider()

    if provider_name == "opencode":
        return OpenCodeMultimodalProvider()

    raise ProviderError(
        f"Unknown generation provider: {provider_name}. "
        "Expected one of: openrouter, ollama, opencode."
    )


__all__ = [
    "BaseMultimodalProvider",
    "OllamaMultimodalProvider",
    "OpenAICompatibleProvider",
    "OpenCodeMultimodalProvider",
    "OpenRouterMultimodalProvider",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "build_provider",
]