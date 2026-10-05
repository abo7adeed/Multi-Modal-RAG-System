"""
Shared base for OpenAI-compatible chat completion providers.

OpenRouter, OpenCode Zen and any other OpenAI-compatible gateway only
differ by base URL, credentials and model. Keeping prompt building and
SDK error translation here avoids duplicating that logic per provider.
"""
import base64
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .base import (
    BaseMultimodalProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a grounded multimodal RAG assistant.

Answer the user's question using ONLY
the provided document context.

If the context does not contain enough
information, say:

"I don't have enough information in the
provided documents."

Do not invent facts.

User question:
{query}

Text context:
{text_context_text}
"""

# Used instead of SYSTEM_PROMPT when the user attached an image. The
# document-grounded prompt above makes the model treat the retrieved
# context as the only admissible evidence, so a question about the
# picture itself ("what colour is it?") would be refused.
ATTACHMENT_SYSTEM_PROMPT = """
You are a grounded multimodal RAG assistant. The user attached an
image to their message and their question is about that image.

Rules you must follow:
1. The attached image is the primary source of truth. Look at it
   and answer the question directly from what you can actually see.
2. Never reply "I don't have enough information" merely because the
   question is not answered by the documents. Use that reply only
   when the image itself does not answer the question.
3. Use the retrieved text and any further images as supporting
   context: they can confirm a product name or add specifications.
   Say so when you rely on them.
4. Never invent facts you cannot see in the image or read in the
   provided context.

User question:
{query}

Retrieved text context (may or may not be relevant):
{text_context_text}
"""


class OpenAICompatibleProvider(BaseMultimodalProvider):
    """
    Base class for providers exposing an OpenAI-compatible
    /chat/completions endpoint.
    """

    #: Human-readable name used in logs and error messages.
    provider_name: str = "OpenAI-compatible"

    #: Endpoint base URL supplied by the concrete provider.
    default_base_url: str = ""

    #: Whether the model accepts image content parts.
    supports_images: bool = True

    #: Credential used when the model is not explicitly provided.
    default_api_key: str = ""

    #: Model used when none is explicitly provided.
    default_model: str = ""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        from app.config import settings

        self.model = model or self.default_model or settings.openrouter_model

        self._configured_key = api_key or self.default_api_key

        # The SDK raises its own error when the key is empty, which
        # would leak an SDK exception during construction. Pass a
        # placeholder instead and fail fast with a clean application
        # error on the first request (see generate()).
        self.client = OpenAI(
            api_key=self._configured_key or "missing-api-key",
            base_url=base_url or self.default_base_url,
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

    # ------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------

    def _build_text_context(
        self,
        text_context: list[dict[str, Any]],
    ) -> str:
        text_parts = []

        for item in text_context:
            page = item.get("page")
            content = item.get("content", "")

            text_parts.append(
                f"[Page {page}]\n{content}"
            )

        if not text_parts:
            return "No text context available."

        return "\n\n".join(text_parts)

    def _build_content(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ) -> list[dict[str, Any]]:
        # An attached image is the subject of the question, so it is
        # sent first and called out explicitly: models treat an
        # unlabelled leading image as just another context image.
        # The order is enforced here rather than assumed from the
        # caller, because the prompts below promise it. Sorting is
        # stable, so retrieved images keep their ranking.
        ordered_images = sorted(
            image_context,
            key=lambda item: (
                0 if item.get("kind") == "attachment" else 1
            ),
        )

        has_attachment = any(
            item.get("kind") == "attachment"
            for item in ordered_images
        )

        template = (
            ATTACHMENT_SYSTEM_PROMPT
            if has_attachment
            else SYSTEM_PROMPT
        )

        text = template.format(
            query=query,
            text_context_text=self._build_text_context(text_context),
        )

        if conversation:
            # Earlier turns let the model resolve a follow-up
            # ("what about its warranty?"). They are context only:
            # the grounding rules still bind the answer to the
            # retrieved documents below.
            text += (
                "\n\nEarlier in this conversation "
                "(for resolving references only):\n"
                f"{conversation}"
            )

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": text,
            }
        ]

        if not self.supports_images:
            if image_context:
                logger.info(
                    "provider=%s model=%s dropping %d image part(s): "
                    "model is text-only",
                    self.provider_name,
                    self.model,
                    len(image_context),
                )
            return content

        if has_attachment:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "The FIRST image below is the image the user "
                        "attached to their message. It is the subject "
                        "of the question: answer the question about it "
                        "directly. Any further images are pages "
                        "retrieved from the indexed documents."
                    ),
                }
            )

        for item in ordered_images:
            image_path = item.get("image_path")

            if not image_path:
                continue

            path = Path(image_path)

            if not path.exists():
                continue

            image_base64 = base64.b64encode(
                path.read_bytes()
            ).decode("utf-8")

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:image/jpeg;base64,"
                            f"{image_base64}"
                        )
                    },
                }
            )

        return content

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    def generate(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ) -> str:
        if not self._configured_key:
            raise ProviderError(
                f"{self.provider_name} API key is not configured. "
                "Set it in the environment."
            )

        content = self._build_content(
            query=query,
            text_context=text_context,
            image_context=image_context,
            conversation=conversation,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            )

        except RateLimitError as exc:
            raise ProviderRateLimitError(
                f"{self.provider_name} rate limit exceeded "
                f"for model: {self.model}"
            ) from exc

        except APITimeoutError as exc:
            raise ProviderTimeoutError(
                f"{self.provider_name} request timed out "
                f"for model: {self.model}"
            ) from exc

        except APIConnectionError as exc:
            raise ProviderError(
                f"Could not connect to {self.provider_name} "
                f"for model: {self.model}"
            ) from exc

        except APIStatusError as exc:
            raise ProviderError(
                f"{self.provider_name} returned an error "
                f"for model: {self.model}"
            ) from exc

        return response.choices[0].message.content or ""

    # ------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------

    def _translate_sdk_error(self, exc: Exception) -> ProviderError:
        """Map an SDK exception onto the provider error taxonomy."""
        if isinstance(exc, RateLimitError):
            return ProviderRateLimitError(
                f"{self.provider_name} rate limit exceeded "
                f"for model: {self.model}"
            )
        if isinstance(exc, APITimeoutError):
            return ProviderTimeoutError(
                f"{self.provider_name} request timed out "
                f"for model: {self.model}"
            )
        if isinstance(exc, APIConnectionError):
            return ProviderError(
                f"Could not connect to {self.provider_name} "
                f"for model: {self.model}"
            )
        if isinstance(exc, APIStatusError):
            return ProviderError(
                f"{self.provider_name} returned an error "
                f"for model: {self.model}"
            )
        return ProviderError(
            f"{self.provider_name} streaming failed "
            f"for model: {self.model}"
        )

    def stream(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ):
        """
        Yield the answer token-by-token.

        Errors can surface mid-stream, so they are raised as provider
        errors here; the pipeline converts them into an SSE error
        event rather than letting a truncated answer look complete.
        """
        if not self._configured_key:
            raise ProviderError(
                f"{self.provider_name} API key is not configured. "
                "Set it in the environment."
            )

        content = self._build_content(
            query=query,
            text_context=text_context,
            image_context=image_context,
            conversation=conversation,
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text

        except Exception as exc:  # noqa: BLE001 - re-raised below
            raise self._translate_sdk_error(exc) from exc