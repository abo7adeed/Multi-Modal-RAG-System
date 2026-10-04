"""
Validation of images attached to a chat message.

The payload is client controlled, so the bytes - not the declared
media type - decide what the file is.
"""
import base64
import io

import pytest
from PIL import Image

from app.api.services.attachment_service import AttachmentService
from app.errors import InvalidRequestError


def image_bytes(fmt="PNG", size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# --------------------------------------------------------------
# Happy path
# --------------------------------------------------------------
def test_decodes_a_png():
    attachment = AttachmentService().decode(
        filename="photo.png",
        media_type="image/png",
        data=b64(image_bytes()),
    )

    assert attachment.data == image_bytes()
    assert attachment.filename == "photo.png"
    assert attachment.media_type == "image/png"


def test_media_type_comes_from_the_bytes_not_the_client():
    # The client claims JPEG; the payload is a PNG. Trust the bytes.
    attachment = AttachmentService().decode(
        filename="mislabelled.jpg",
        media_type="image/jpeg",
        data=b64(image_bytes(fmt="PNG")),
    )

    assert attachment.media_type == "image/png"


def test_accepts_jpeg_webp_and_gif():
    for fmt, expected in [
        ("JPEG", "image/jpeg"),
        ("WEBP", "image/webp"),
        ("GIF", "image/gif"),
    ]:
        attachment = AttachmentService().decode(
            filename=f"x.{fmt.lower()}",
            media_type="application/octet-stream",
            data=b64(image_bytes(fmt=fmt)),
        )

        assert attachment.media_type == expected


def test_tolerates_wrapped_base64():
    wrapped = "\n".join(
        b64(image_bytes())[i : i + 76]
        for i in range(0, len(b64(image_bytes())), 76)
    )

    attachment = AttachmentService().decode(
        filename="photo.png",
        media_type="image/png",
        data=wrapped,
    )

    assert attachment.data == image_bytes()


def test_blank_filename_gets_a_default():
    attachment = AttachmentService().decode(
        filename="   ",
        media_type="image/png",
        data=b64(image_bytes()),
    )

    assert attachment.filename == "attachment"


# --------------------------------------------------------------
# Rejections
# --------------------------------------------------------------
def test_rejects_invalid_base64():
    with pytest.raises(InvalidRequestError):
        AttachmentService().decode(
            filename="photo.png",
            media_type="image/png",
            data="not base64 !!!",
        )


def test_rejects_empty_payload():
    with pytest.raises(InvalidRequestError):
        AttachmentService().decode(
            filename="photo.png",
            media_type="image/png",
            data="",
        )


def test_rejects_a_pdf_disguised_as_an_image():
    with pytest.raises(InvalidRequestError):
        AttachmentService().decode(
            filename="invoice.png",
            media_type="image/png",
            data=b64(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n"),
        )


def test_rejects_html_error_page():
    with pytest.raises(InvalidRequestError):
        AttachmentService().decode(
            filename="cat.png",
            media_type="image/png",
            data=b64(b"<!DOCTYPE html><html><body>404</body></html>"),
        )


def test_rejects_truncated_image():
    truncated = image_bytes()[:20]

    with pytest.raises(InvalidRequestError):
        AttachmentService().decode(
            filename="broken.png",
            media_type="image/png",
            data=b64(truncated),
        )


def test_rejects_oversized_attachment(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_max_query_image_size_mb", 0)

    with pytest.raises(InvalidRequestError):
        AttachmentService().decode(
            filename="big.png",
            media_type="image/png",
            data=b64(image_bytes()),
        )


def test_rejects_excessive_pixel_count(monkeypatch):
    from app.api.services import attachment_service as module

    monkeypatch.setattr(module, "MAX_IMAGE_PIXELS", 4)

    with pytest.raises(InvalidRequestError):
        AttachmentService().decode(
            filename="huge.png",
            media_type="image/png",
            data=b64(image_bytes(size=(64, 64))),
        )
