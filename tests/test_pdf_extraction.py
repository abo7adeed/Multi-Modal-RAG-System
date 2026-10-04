from pathlib import Path

from app.ingestion.pdf_loader import PDFLoader


def main():
    pdf_path = Path("data/raw/documents/dell_catalog.pdf")

    document = PDFLoader().load(pdf_path)

    total_images = 0

    print("=" * 60)
    print("PDF MULTIMODAL EXTRACTION TEST")
    print("=" * 60)

    print(f"Document: {document.source}")
    print(f"Pages: {len(document.pages)}")

    for page in document.pages:
        image_count = len(page.images)
        total_images += image_count

        if image_count > 0:
            print(f"\n--- Page {page.page_number} ---")
            print(f"Images: {image_count}")

            for image in page.images:
                print(f"Image ID : {image.image_id}")
                print(f"Path     : {image.path}")
                print(f"Metadata : {image.metadata}")

    print("\n" + "=" * 60)
    print(f"TOTAL EXTRACTED IMAGES: {total_images}")
    print("=" * 60)


if __name__ == "__main__":
    main()