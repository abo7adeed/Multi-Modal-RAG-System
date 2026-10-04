from app.config import settings

from .openai_compatible import OpenAICompatibleProvider


class OpenCodeMultimodalProvider(OpenAICompatibleProvider):
    """
    OpenCode Zen provider (OpenAI-compatible gateway).

    Zen is a curated gateway for coding-focused models and is billed
    per request, so it avoids the free-tier rate limits of public
    aggregators. Its catalogue is text-only: image context is
    therefore dropped and a warning is logged instead of failing
    the request.
    """

    provider_name = "OpenCode"

    default_base_url = "https://opencode.ai/zen/v1"

    supports_images = False

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        super().__init__(
            model=model or settings.opencode_model,
            api_key=api_key or settings.opencode_api_key,
            base_url=(
                base_url or settings.opencode_base_url
            ),
            timeout=(
                timeout
                if timeout is not None
                else settings.opencode_timeout
            ),
            max_retries=(
                max_retries
                if max_retries is not None
                else settings.opencode_max_retries
            ),
        )