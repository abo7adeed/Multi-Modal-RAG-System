"""
Ad-hoc measurement: does conversation rewriting actually rescue
follow-up questions that the relevance gate would otherwise block?

Compares, on the real corpus:
  - the bare follow-up ("what about its warranty?")
  - the rewritten form ("<previous question> <follow-up>")

Run manually:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/verify_followup.py
"""
from app.config import settings
from app.generation.conversation import (
    ConversationMemory,
    ConversationTurn,
    rewrite_query,
)
from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.retriever import MultimodalRetriever
from app.retrieval.vector_store.chroma_store import ChromaVectorStore


# (previous question, follow-up that refers back to it)
CASES = [
    ("What is the Dell OptiPlex 3020?", "What about its warranty?"),
    ("What is the Dell OptiPlex 3020?", "And the price?"),
    ("What workstation products are in the catalog?", "Tell me more"),
    ("Which pages mention wireless connectivity?", "Does it support Wi-Fi 6?"),
    ("What is the Dell Precision tower?", "How much RAM does it have?"),
    ("What is the Dell OptiPlex 3020?", "Anything else?"),
]


def main() -> None:
    embedder = CLIPEmbedder()
    store = ChromaVectorStore(
        persist_directory=settings.vector_store_directory,
        collection_name="dell_catalog",
    )
    retriever = MultimodalRetriever(
        embedder=embedder, vector_store=store
    )

    rescued = 0

    for previous, follow_up in CASES:
        memory = ConversationMemory(
            [
                ConversationTurn(role="user", content=previous),
                ConversationTurn(role="assistant", content="An answer."),
            ]
        )
        rewritten = rewrite_query(follow_up, memory)

        bare_hits = retriever.search_multimodal(follow_up)
        rewritten_hits = retriever.search_multimodal(
            rewritten.search_query
        )

        blocked = not bare_hits
        now_answers = bool(rewritten_hits)

        if blocked and now_answers:
            rescued += 1

        print(f"\nprevious : {previous}")
        print(f"follow-up: {follow_up}")
        print(f"  rewritten -> {rewritten.search_query}")
        print(f"  bare:    {len(bare_hits)} results"
              f"{'  (BLOCKED)' if blocked else ''}")
        print(f"  rewrite: {len(rewritten_hits)} results")
        if now_answers:
            top = rewritten_hits[0]
            page = top.get("metadata", {}).get("page_number")
            print(f"    top hit: page {page}")

    print(
        f"\nrescued {rescued}/{len(CASES)} follow-ups that were "
        f"previously blocked"
    )


if __name__ == "__main__":
    main()