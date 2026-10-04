from app.pipeline import MultimodalRAGPipeline

from app.retrieval.retriever import MultimodalRetriever
from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.vector_store.chroma_store import ChromaVectorStore

from app.generation.context_builder import ContextBuilder
from app.generation.generator import MultimodalGenerator
from app.generation.providers import build_provider


def main():
    # Retrieval components
    embedder = CLIPEmbedder()

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_store",
        collection_name="dell_catalog",
    )

    retriever = MultimodalRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # Generation components
    context_builder = ContextBuilder()

    provider = build_provider()

    generator = MultimodalGenerator(
        provider=provider,
    )

    # RAG pipeline
    pipeline = MultimodalRAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
    )

    query = "What Dell products are shown in the catalog?"

    response = pipeline.run(query)

    print("\n=== ANSWER ===")
    print(response.answer)

    print("\n=== SOURCES ===")

    for source in response.sources:
        print(
            f"- {source.source} | "
            f"Page {source.page} | "
            f"{source.content_type} | "
            f"{source.chunk_id}"
        )


if __name__ == "__main__":
    main()