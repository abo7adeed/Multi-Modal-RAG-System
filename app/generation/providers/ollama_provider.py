import base64
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from ollama import Client, ResponseError

from app.config import settings

from .base import (
    BaseMultimodalProvider,
    ProviderError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


class OllamaMultimodalProvider(BaseMultimodalProvider):
    """
    Ollama provider for both local servers and Ollama Cloud.

    Local:  OLLAMA_BASE_URL=http://localhost:11434 (no auth)
    Cloud:  OLLAMA_BASE_URL=https://ollama.com with OLLAMA_API_KEY
    """

    provider_name = "Ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        self.model = model or settings.ollama_model

        self.base_url = base_url or settings.ollama_base_url

        self.api_key = api_key or settings.ollama_api_key

        self.timeout = (
            timeout
            if timeout is not None
            else settings.ollama_timeout
        )

        # Cloud endpoints require a bearer token; local servers do not.
        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else None
        )

        self.client = Client(
            host=self.base_url,
            timeout=self.timeout,
            headers=headers,
        )

    def _build_messages(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Build the chat messages for a request.

        Shared by generate() and stream() so both paths prompt the
        model identically. Returns the messages and whether an
        attachment was included.
        """
        text_parts = []

        for item in text_context:
            page = item.get("page")
            content = item.get("content", "")

            text_parts.append(
                f"[Page {page}]\n{content}"
            )

        text_context_text = (
            "\n\n".join(text_parts)
            if text_parts
            else "No text context available."
        )

        image_paths = []
        has_attachment = False

        # The prompt below promises that the attached image comes
        # first, so the order is enforced here rather than assumed
        # from the caller. Sorting is stable, so retrieved images keep
        # their ranking.
        ordered_images = sorted(
            image_context,
            key=lambda item: (
                0 if item.get("kind") == "attachment" else 1
            ),
        )

        for item in ordered_images:

            image_path = item.get("image_path")

            if not image_path:
                continue

            path = Path(image_path)

            # Images are sent base64-encoded: Ollama Cloud cannot read
            # paths on this machine, and encoding works for local too.
            if not path.exists():
                continue

            if item.get("kind") == "attachment":
                has_attachment = True

            image_paths.append(
                base64.b64encode(path.read_bytes()).decode("utf-8")
            )

        # Earlier turns let the model resolve a follow-up ("what
        # about its warranty?"). They are context only: the grounding
        # rules above still bind the answer to the retrieved
        # documents.
        conversation_section = (
            f"\n\nEarlier in this conversation "
            "(for resolving references only):\n"
            f"{conversation}\n"
            if conversation
            else ""
        )

        if has_attachment:
            # The document-grounded prompt makes the model treat the
            # retrieved context as the only admissible evidence, so
            # a question about the picture itself ("what colour is
            # it?") is refused. With an attachment the picture is
            # itself the primary evidence and the documents only
            # add detail.
            system_prompt = """
You are a grounded RAG assistant. The user attached an image to
their message and their question is about that image.

Rules you must follow:
1. The attached image is the primary source of truth. Look at it
   and answer the question directly from what you can actually
   see: the product, its colours, visible labels, its layout,
   anything else the question asks about.
2. Never reply "I don't have enough information" merely because
   the question is not answered by the documents. Use that reply
   only when the image itself does not answer the question.
3. Use the retrieved text and any further images as supporting
   context: they can confirm a product name or add
   specifications. Say so when you rely on them.
4. Never invent facts you cannot see in the image or read in the
   provided context.
"""

            user_prompt = f"""
User question:
{query}
{conversation_section}
The FIRST image below is the image the user attached. It is the
subject of the question - answer the question about it.

Retrieved text context (may or may not be relevant):
{text_context_text}

Any further images are pages retrieved from the indexed
documents. Give a concise, factual answer.
"""
        else:
            system_prompt = """
You are a grounded RAG assistant answering questions about a
specific set of documents.

Rules you must follow:
1. Answer ONLY from the provided context and images.
2. Never greet, chat, or answer general knowledge questions.
   You are not a general assistant.
3. If the context does not contain the answer, reply with exactly:
   "I don't have enough information in the provided documents."
4. Never invent facts, product names, or specifications.
"""

            user_prompt = f"""
User question:
{query}
{conversation_section}Text context:
{text_context_text}

Use the provided images as visual evidence when they are relevant
to the question. Give a concise, factual answer grounded in the
context above.
"""

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        if image_paths:
            messages[1]["images"] = image_paths
            logger.info(
                "provider=%s model=%s sending %d image(s) as base64 "
                "attachment=%s",
                self.provider_name,
                self.model,
                len(image_paths),
                has_attachment,
            )

        return messages, has_attachment

    def generate(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ) -> str:
        messages, _ = self._build_messages(
            query=query,
            text_context=text_context,
            image_context=image_context,
            conversation=conversation,
        )

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
            )

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out "
                f"for model: {self.model}"
            ) from exc

        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Could not connect to Ollama at "
                f"{self.base_url}. Is the server running?"
            ) from exc

        except ResponseError as exc:
            raise ProviderError(
                f"Ollama returned an error "
                f"for model: {self.model}"
            ) from exc

        return response["message"]["content"]

    # ------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------

    def stream(
        self,
        query: str,
        text_context: list[dict[str, Any]],
        image_context: list[dict[str, Any]],
        conversation: str | None = None,
    ) -> Iterator[str]:
        """
        Yield the answer incrementally.

        The SDK raises mid-iteration once the stream is open, so the
        whole iteration is wrapped: a failure after some text has
        already been emitted must surface as a provider error rather
        than ending the stream early and looking like a short answer.
        """
        messages, _ = self._build_messages(
            query=query,
            text_context=text_context,
            image_context=image_context,
            conversation=conversation,
        )

        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )

            for part in stream:
                text = part["message"]["content"]
                if text:
                    yield text

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out "
                f"for model: {self.model}"
            ) from exc

        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Could not connect to Ollama at "
                f"{self.base_url}. Is the server running?"
            ) from exc

        except ResponseError as exc:
            raise ProviderError(
                f"Ollama returned an error "
                f"for model: {self.model}"
            ) from exc