from collections import Counter

from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.vector_store.chroma_store import ChromaVectorStore


def main():

    print("=" * 70)
    print("RETRIEVAL MODALITY DISTRIBUTION")
    print("=" * 70)

    embedder = CLIPEmbedder()

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_store",
        collection_name="dell_catalog",
    )

    queries = [
        "Dell laptop",
        "Dell workstation",
        "Dell computer",
        "Intel Core i7 laptop",
        "Dell laptop product image",
    ]

    for query in queries:

        print("\n" + "-" * 70)
        print(f"QUERY: {query}")
        print("-" * 70)

        query_embedding = embedder.embed_text(query)

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=10,
        )

        types = [
            metadata.get("content_type")
            for metadata in results["metadatas"][0]
        ]

        counts = Counter(types)

        print(f"Text results  : {counts.get('text', 0)}")
        print(f"Image results : {counts.get('image', 0)}")

        for rank, metadata in enumerate(
            results["metadatas"][0],
            start=1,
        ):

            print(
                f"{rank:2}. "
                f"{metadata.get('content_type')} "
                f"| page={metadata.get('page_number')}"
            )

    print("\n" + "=" * 70)
    print("DISTRIBUTION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()