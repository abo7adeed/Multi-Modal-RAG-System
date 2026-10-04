"""
Validation of images attached to a chat message.

Everything here treats the payload as hostile: the declared media type
and filename are client controlled, so the bytes themselves decide
what the file is. The service only produces a validated
ImageAttachment; it never touches the filesystem.
"""
import base64
import binascii
import io
import logging

from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.errors import InvalidRequestError
from app.generation.schemas import ImageAttachment

logger = logging.getLogger(__name__)


def _rejected(reason: str) -> InvalidRequestError:
    """
    Build a rejection whose reason is shown to the user.

    Every message here is written for end users ("not a valid image",
    "exceeds the maximum size"), so the generic
    InvalidRequestError message would only hide the useful part.
    """
    return InvalidRequestError(reason, user_message=reason)

# Pillow's own bomb guard fires far above this; ours rejects
# unreasonably large images long before they are decoded.
MAX_IMAGE_PIXELS = 20_000_000

# Pillow format name -> media type, so the attachment is labelled by
# what the bytes actually are.
FORMAT_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}


class AttachmentService:
    def decode(
        self,
        *,
        filename: str,
        media_type: str,
        data: str,
    ) -> ImageAttachment:
        """
        Turn a base64 payload into a validated ImageAttachment.

        Raises InvalidRequestError (HTTP 400) for anything that is not
        a reasonably sized, genuinely decodable image.
        """
        raw = self._decode_base64(data)
        self._check_size(raw)

        detected_type = self._detect_image(raw)
        filename = (filename or "").strip() or "attachment"

        logger.info(
            "image_attachment_validated filename=%s declared_type=%s "
            "detected_type=%s bytes=%d",
            filename,
            media_type,
            detected_type,
            len(raw),
        )

        return ImageAttachment(
            filename=filename,
            media_type=detected_type,
            data=raw,
        )

    # ------------------------------------------------------------
    # Validation steps
    # ------------------------------------------------------------

    @staticmethod
    def _decode_base64(data: str) -> bytes:
        # Some clients wrap base64 at 76 characters; whitespace is
        # never meaningful here, but other junk is a hard error.
        compact = "".join(data.split())

        try:
            raw = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _rejected(
                "The attached image is not valid base64 data."
            ) from exc

        if not raw:
            raise _rejected("The attached image is empty.")

        return raw

    @staticmethod
    def _check_size(raw: bytes) -> None:
        if len(raw) > settings.max_query_image_bytes:
            raise _rejected(
                "The attached image exceeds the maximum size of "
                f"{settings.api_max_query_image_size_mb} MB."
            )

    @staticmethod
    def _detect_image(raw: bytes) -> str:
        """
        Confirm the bytes are a real image and return their media type.

        Pillow is the judge: a renamed PDF, an HTML error page or a
        truncated upload all fail here instead of reaching the model.
        """
        try:
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size

                if width * height > MAX_IMAGE_PIXELS:
                    raise _rejected(
                        "The attached image has too many pixels "
                        f"({width}x{height})."
                    )

                image_format = image.format

                # verify() parses the file structure; it must run
                # while the file handle is still open.
                image.verify()

        except InvalidRequestError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.info(
                "image_attachment_rejected reason=not_an_image "
                "error_type=%s",
                type(exc).__name__,
            )
            raise _rejected(
                "The attached file is not a valid image."
            ) from exc

        if not width or not height:
            raise _rejected("The attached image is empty.")

        return FORMAT_MEDIA_TYPES.get(
            image_format, "image/jpeg"
        )
