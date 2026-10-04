from pathlib import Path

from app.ingestion.chunker import DocumentChunker
from app.ingestion.pdf_loader import PDFLoader


def main():
    pdf_path = Path("data/raw/documents/dell_catalog.pdf")

    document = PDFLoader().load(pdf_path)

    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.chunk(document)

    print("=" * 60)
    print("MULTIMODAL CHUNK TEST")
    print("=" * 60)

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:

        if chunk.page_number in [16, 17, 18]:

            print("\n------------------------------")
            print(f"Chunk ID       : {chunk.chunk_id}")
            print(f"Type           : {chunk.content_type}")
            print(f"Page           : {chunk.page_number}")
            print(f"Parent         : {chunk.parent_chunk_id}")
            print(f"Related chunks : {chunk.related_chunk_ids}")

            if chunk.text:
                print(f"Text           : {chunk.text[:100]}")

            if chunk.image_path:
                print(f"Image          : {chunk.image_path}")


if __name__ == "__main__":
    main()