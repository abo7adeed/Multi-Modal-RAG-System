from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from .base import BaseLoader
from .schemas import Document, ImageAsset, PageContent


class PDFLoader(BaseLoader):

    def load(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: {file_path.suffix}"
            )

        reader = PdfReader(file_path)

        document_id = str(uuid4())

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            page_id = f"{document_id}:page:{page_number}"

            text = page.extract_text() or ""

            images = self._extract_images(
                page=page,
                document_id=document_id,
                page_number=page_number,
            )

            pages.append(
                PageContent(
                    page_id=page_id,
                    page_number=page_number,
                    text=text,
                    images=images,
                    metadata={
                        "source": str(file_path),
                    },
                )
            )

        return Document(
            document_id=document_id,
            source=str(file_path),
            document_type="pdf",
            pages=pages,
            metadata={
                "filename": file_path.name,
                "page_count": len(pages),
            },
        )

    def _extract_images(
        self,
        page,
        document_id: str,
        page_number: int,
    ) -> list[ImageAsset]:

        images = []

        for index, image in enumerate(page.images):

            image_id = str(uuid4())

            output_dir = (
                Path("data/processed/images")
                / document_id
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                output_dir
                / f"page_{page_number}_image_{index}"
                f"{Path(image.name).suffix}"
            )

            output_path.write_bytes(image.data)

            images.append(
                ImageAsset(
                    image_id=image_id,
                    path=str(output_path),
                    page_number=page_number,
                    metadata={
                        "filename": image.name,
                    },
                )
            )

        return images