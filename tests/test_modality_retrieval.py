from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.vector_store.chroma_store import ChromaVectorStore


def main():

    print("=" * 70)
    print("MODALITY-SPECIFIC RETRIEVAL TEST")
    print("=" * 70)

    embedder = CLIPEmbedder()

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_store",
        collection_name="dell_catalog",
    )

    queries = [
        "Dell laptop",
        "Dell workstation",
        "Intel Core i7 laptop",
        "Dell computer",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        query_embedding = embedder.embed_text(query)

        # Text-only retrieval
        text_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
            content_type="text",
        )

        # Image-only retrieval
        image_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
            content_type="image",
        )

        print("\nTEXT RESULTS")

        for rank, (
            metadata,
            distance,
        ) in enumerate(
            zip(
                text_results["metadatas"][0],
                text_results["distances"][0],
            ),
            start=1,
        ):
            print(
                f"{rank}. "
                f"page={metadata.get('page_number')} "
                f"distance={distance:.4f}"
            )

        print("\nIMAGE RESULTS")

        for rank, (
            metadata,
            distance,
        ) in enumerate(
            zip(
                image_results["metadatas"][0],
                image_results["distances"][0],
            ),
            start=1,
        ):
            print(
                f"{rank}. "
                f"page={metadata.get('page_number')} "
                f"distance={distance:.4f}"
            )

            print(
                f"   image={metadata.get('image_path')}"
            )

    print("\n" + "=" * 70)
    print("MODALITY RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()