"""
Media service: maps stored image paths to URLs served by this API.

The server never exposes raw filesystem paths to clients; sources
carry an image_url that is resolved here and served by a locked-down
static route (see app/api/main.py).
"""
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class MediaService:
    def __init__(self, base_path: Path | str | None = None):
        self.base_path = Path(
            base_path or settings.processed_images_directory
        ).resolve()

    def resolve_image_url(self, image_path: str | None) -> str | None:
        """
        Convert a stored image path (absolute or relative) into a URL
        under the API media prefix.

        Returns None when the path is outside the served directory,
        does not exist, or is not an image.
        """
        if not image_path:
            return None

        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate

        try:
            resolved = candidate.resolve()
            resolved.relative_to(self.base_path)
        except (ValueError, OSError):
            logger.warning(
                "media_path_rejected path=%s", image_path
            )
            return None

        if not resolved.is_file():
            return None

        relative = resolved.relative_to(self.base_path).as_posix()
        return f"{settings.api_media_url_prefix}/{relative}"

    def resolve_safe_path(self, url_relative: str) -> Path | None:
        """
        Reverse mapping for the /media route: URL-relative path ->
        safe filesystem path, or None if outside the base directory.
        """
        relative = Path(url_relative)
        if relative.is_absolute() or ".." in relative.parts:
            return None

        candidate = (self.base_path / relative).resolve()

        try:
            candidate.relative_to(self.base_path)
        except ValueError:
            return None

        return candidate if candidate.is_file() else None
