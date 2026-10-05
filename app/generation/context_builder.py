from pathlib import PureWindowsPath
from typing import Any

from .schemas import GenerationContext, Source


def document_name(source: str | None) -> str | None:
    """
    Extract the display filename from a stored document path.

    Stored paths are OS-native (backslash separated on Windows), so
    PureWindowsPath handles both separators.
    """
    if not source:
        return None
    name = PureWindowsPath(source).name
    return name or None


class ContextBuilder:
    """
    Converts multimodal retrieval results into a clean,
    structured context for the generation layer.
    """

    @staticmethod
    def _snippet(
        text: str,
        limit: int = 280,
    ) -> str:
        """Collapse whitespace and truncate for display."""
        if not text:
            return ""
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[:limit].rsplit(" ", 1)[0] + "…"

    def build(
        self,
        query: str,
        results: list[dict[str, Any]],
        attachments: list[dict[str, str]] | None = None,
        conversation: str | None = None,
    ) -> GenerationContext:
        text_context = []
        image_context = []
        sources = []

        # ----------------------------------------------------
        # Attached image
        #
        # The user's own image is the subject of the question, so
        # it leads the visual context. It is deliberately not
        # reported as a source: clients already display it, and it
        # is not part of the indexed corpus.
        # ----------------------------------------------------

        for attachment in attachments or []:
            image_context.append(
                {
                    "chunk_id": None,
                    "page": None,
                    "image_path": attachment["image_path"],
                    "source": attachment.get("filename") or "attachment",
                    "kind": "attachment",
                }
            )

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            content_type = metadata.get(
                "content_type"
            )

            page_number = metadata.get(
                "page_number"
            )

            # The chunk metadata carries the real document path;
            # the result-level "source" is only the RRF provenance
            # tag ("retrieved" / "related"). Prefer the document so
            # clients can attribute answers to a real file.
            # Page images the user's attachment merely resembles are
            # supporting context, not evidence: they must never be
            # reported as a source, or a photo question would appear to
            # cite an unrelated catalog page.
            is_visual_match = (
                result.get("source") == "visual_match"
            )

            source = document_name(
                metadata.get("source")
            ) or result.get(
                "source",
                "retrieved",
            )

            chunk_id = result.get(
                "chunk_id"
            )

            document = result.get(
                "document",
                "",
            )

            image_path = metadata.get(
                "image_path",
                "",
            )

            # ----------------------------------------------------
            # Text context
            # ----------------------------------------------------

            if content_type == "text":

                text_context.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "content": document,
                        "source": source,
                    }
                )

            # ----------------------------------------------------
            # Image context
            # ----------------------------------------------------

            elif content_type == "image":

                image_context.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "image_path": image_path,
                        "source": source,
                        "kind": (
                            "visual_match"
                            if is_visual_match
                            else "retrieved"
                        ),
                    }
                )

                if is_visual_match:
                    continue

            # ----------------------------------------------------
            # Unified source
            # ----------------------------------------------------

            sources.append(
                Source(
                    chunk_id=chunk_id,
                    page=page_number,
                    content_type=content_type,
                    source=source,
                    image_path=image_path or None,
                    # Text preview: lets the UI show why a source
                    # was retrieved and tell apart chunks that come
                    # from the same page.
                    snippet=(
                        self._snippet(document)
                        if content_type == "text"
                        else None
                    ),
                    score=result.get("score"),
                    kind="document",
                )
            )

        return GenerationContext(
            query=query,
            text_context=text_context,
            image_context=image_context,
            sources=sources,
            conversation=conversation,
        )