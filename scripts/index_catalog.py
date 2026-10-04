from pathlib import Path

from app.ingestion.chunker import DocumentChunker
from app.ingestion.pdf_loader import PDFLoader
from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.vector_store.chroma_store import ChromaVectorStore
from app.retrieval.retriever import MultimodalRetriever


PDF_PATH = Path(
    "data/raw/documents/dell_catalog.pdf"
)

NORMALIZED_IMAGE_DIR = Path(
    "data/processed/normalized/dell_catalog"
)


def resolve_normalized_image(
    original_path: str,
) -> str:

    original = Path(original_path)

    normalized_path = (
        NORMALIZED_IMAGE_DIR
        / f"{original.stem}.jpg"
    )

    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Normalized image not found: "
            f"{normalized_path}"
        )

    return str(normalized_path)


def main():

    print("=" * 70)
    print("MULTIMODAL RAG INDEXING")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    print("\n[1/5] Loading PDF...")

    loader = PDFLoader()

    document = loader.load(
        PDF_PATH
    )

    print(
        f"Pages: {len(document.pages)}"
    )

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    print("\n[2/5] Creating chunks...")

    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.chunk(
        document
    )

    # Resolve normalized image paths
    for chunk in chunks:

        if (
            chunk.content_type == "image"
            and chunk.image_path
        ):
            chunk.image_path = (
                resolve_normalized_image(
                    chunk.image_path
                )
            )

    text_count = sum(
        chunk.content_type == "text"
        for chunk in chunks
    )

    image_count = sum(
        chunk.content_type == "image"
        for chunk in chunks
    )

    print(
        f"Total chunks : {len(chunks)}"
    )

    print(
        f"Text chunks  : {text_count}"
    )

    print(
        f"Image chunks : {image_count}"
    )

    # --------------------------------------------------
    # 3. Load CLIP
    # --------------------------------------------------

    print("\n[3/5] Loading CLIP...")

    embedder = CLIPEmbedder()

    # --------------------------------------------------
    # 4. Create vector store
    # --------------------------------------------------

    print("\n[4/5] Creating ChromaDB...")

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_store",
        collection_name="dell_catalog",
    )

    retriever = MultimodalRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # --------------------------------------------------
    # 5. Index
    # --------------------------------------------------

    print("\n[5/5] Indexing chunks...")

    retriever.index_chunks(
        chunks
    )

    print(
        f"\nVectors stored: "
        f"{vector_store.count()}"
    )

    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()