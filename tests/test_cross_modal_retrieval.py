from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.vector_store.chroma_store import ChromaVectorStore


def main():

    print("=" * 70)
    print("CROSS-MODAL RETRIEVAL TEST")
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
        "Dell computer with wireless connectivity",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        query_embedding = embedder.embed_text(
            query
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):

            print("-" * 60)

            print(f"Rank     : {rank}")
            print(f"Distance : {distance:.4f}")
            print(
                f"Type     : "
                f"{metadata.get('content_type')}"
            )
            print(
                f"Page     : "
                f"{metadata.get('page_number')}"
            )

            image_path = metadata.get(
                "image_path"
            )

            if image_path:
                print(
                    f"Image    : "
                    f"{image_path}"
                )

            print(
                f"Content  : "
                f"{document[:150]}"
            )

    print("\n" + "=" * 70)
    print("CROSS-MODAL RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()