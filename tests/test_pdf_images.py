from pathlib import Path

from pypdf import PdfReader


def main():
    pdf_path = Path("data/raw/documents/dell_catalog.pdf")

    reader = PdfReader(pdf_path)

    print("=" * 50)
    print("PDF IMAGE INSPECTION")
    print("=" * 50)

    for page_number, page in enumerate(reader.pages, start=1):

        images = list(page.images)

        print(f"\nPage {page_number}")
        print(f"Images found: {len(images)}")

        for image in images:
            print(f"  Name: {image.name}")
            print(f"  Size: {len(image.data)} bytes")


if __name__ == "__main__":
    main()