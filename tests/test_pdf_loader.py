from pathlib import Path

from app.ingestion.pdf_loader import PDFLoader


def main():
    pdf_path = Path("data/raw/documents/dell_catalog.pdf")

    loader = PDFLoader()

    document = loader.load(pdf_path)

    print("=" * 50)
    print("PDF INGESTION TEST")
    print("=" * 50)

    print(f"Document ID : {document.document_id}")
    print(f"Source      : {document.source}")
    print(f"Type        : {document.document_type}")
    print(f"Pages       : {len(document.pages)}")

    print("\nMetadata:")
    print(document.metadata)

    for page in document.pages:
        print(f"\n--- Page {page.page_number} ---")
        print(page.text)


if __name__ == "__main__":
    main()