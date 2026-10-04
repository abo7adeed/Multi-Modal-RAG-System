from pathlib import Path

from app.ingestion.image_loader import ImageLoader


def main():
    image_path = Path("data/raw/images/test.jpg")

    loader = ImageLoader()

    document = loader.load(image_path)

    print("=" * 50)
    print("IMAGE INGESTION TEST")
    print("=" * 50)

    print(f"Document ID : {document.document_id}")
    print(f"Source      : {document.source}")
    print(f"Type        : {document.document_type}")

    print("\nDocument Metadata:")
    print(document.metadata)

    for page in document.pages:
        print(f"\n--- Page {page.page_number} ---")

        for image in page.images:
            print("Image ID :", image.image_id)
            print("Path     :", image.path)
            print("Metadata :", image.metadata)


if __name__ == "__main__":
    main()