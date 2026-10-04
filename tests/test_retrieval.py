from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.vector_store.chroma_store import ChromaVectorStore
from app.retrieval.retriever import MultimodalRetriever


def main():

    print("=" * 70)
    print("MULTIMODAL RETRIEVAL TEST")
    print("=" * 70)

    embedder = CLIPEmbedder()

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_store",
        collection_name="dell_catalog",
    )

    retriever = MultimodalRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    print(
        f"\nIndexed vectors: {vector_store.count()}"
    )

    query = "Dell laptop computer"

    print(f"\nQuery: {query}")
    print("\nSearching...\n")

    results = retriever.search_multimodal(
        query=query,
        top_k=5,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print("-" * 70)

        print(f"Rank       : {rank}")
        print(
            f"Chunk ID    : "
            f"{result['chunk_id']}"
        )
        print(
            f"Source     : "
            f"{result['source']}"
        )

        distance = result["distance"]

        if distance is not None:
            print(
                f"Distance   : "
                f"{distance:.4f}"
            )
        else:
            print(
                "Distance   : related chunk"
            )

        metadata = result["metadata"]

        print(
            f"Type       : "
            f"{metadata.get('content_type')}"
        )

        print(
            f"Page       : "
            f"{metadata.get('page_number')}"
        )

        image_path = metadata.get(
            "image_path"
        )

        if image_path:
            print(
                f"Image      : "
                f"{image_path}"
            )

        document = result["document"] or ""

        print(
            f"Content    : "
            f"{document[:300]}"
        )

    print("-" * 70)

    print(
        f"\nTotal expanded results: "
        f"{len(results)}"
    )

    print(
        "\nMULTIMODAL RETRIEVAL TEST COMPLETE"
    )


if __name__ == "__main__":
    main()