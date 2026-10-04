from app.config import settings

from .openai_compatible import OpenAICompatibleProvider


class OpenRouterMultimodalProvider(OpenAICompatibleProvider):
    """
    OpenRouter provider (OpenAI-compatible API).

    All configuration comes from Settings; nothing is hardcoded here.
    """

    provider_name = "OpenRouter"

    default_base_url = "https://openrouter.ai/api/v1"

    supports_images = True

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        super().__init__(
            model=model or settings.openrouter_model,
            api_key=api_key or settings.openrouter_api_key,
            base_url=base_url,
            timeout=(
                timeout
                if timeout is not None
                else settings.openrouter_timeout
            ),
            max_retries=(
                max_retries
                if max_retries is not None
                else settings.openrouter_max_retries
            ),
        )