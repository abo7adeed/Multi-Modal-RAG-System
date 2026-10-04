from pathlib import Path
from uuid import uuid4

from PIL import Image

from .base import BaseLoader
from .schemas import Document, ImageAsset, PageContent


class ImageLoader(BaseLoader):

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

    def load(self, file_path: Path) -> Document:
        # 1. Check that the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 2. Check that the file is an image format we support
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {file_path.suffix}"
            )

        # 3. Open the image and read basic information
        with Image.open(file_path) as image:
            width, height = image.size
            image_format = image.format

        # 4. Create the image asset
        image_asset = ImageAsset(
            image_id=str(uuid4()),
            path=str(file_path),
            metadata={
                "width": width,
                "height": height,
                "format": image_format,
            },
        )

        # 5. Put the image inside our normalized document structure.
        #    page_id is required by the schema and must follow the
        #    same convention as PDFLoader.
        document_id = str(uuid4())

        page = PageContent(
            page_id=f"{document_id}:page:1",
            page_number=1,
            text="",
            images=[image_asset],
            metadata={
                "source": str(file_path),
            },
        )

        # 6. Return normalized Document
        return Document(
            document_id=document_id,
            source=str(file_path),
            document_type="image",
            pages=[page],
            metadata={
                "filename": file_path.name,
                "width": width,
                "height": height,
                "format": image_format,
            },
        )