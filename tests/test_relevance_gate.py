"""
Tests for the off-topic gate in MultimodalRetriever.

The gate treats a text question as off-topic when too little of its
content vocabulary exists in the index. Previously it only checked
that BM25 returned *something*, which let any single common word
through (measured false-answer rate 0.80 on eval/; now 0.60).
"""
from app.retrieval.lexical import BM25Index
from app.retrieval.retriever import MultimodalRetriever

DOCUMENTS = {
    "c1": "Dell OptiPlex 3020 Micro desktop with Intel Core i5 processor",
    "c2": "Dell Latitude 7420 laptop with wireless connectivity",
    "c3": "Dell Precision workstation laptop for heavy workloads",
}


class FakeEmbedder:
    def embed_text(self, text):
        return [0.0, 0.0]

    def embed_texts(self, texts):
        return [[0.0, 0.0] for _ in texts]

    def embed_image(self, path):
        return [0.0, 0.0]

    def embed_images(self, paths):
        return [[0.0, 0.0] for _ in paths]


class FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def get(self, ids=None, include=None):
        ids = [i for i in (ids or []) if i in self.documents]
        return {
            "ids": ids,
            "documents": [self.documents[i] for i in ids],
            "metadatas": [
                {"content_type": "text", "page_number": 1}
                for _ in ids
            ],
        }


class FakeVectorStore:
    """Returns no vector hits, so only the lexical path is exercised."""

    def __init__(self, documents):
        self.collection = FakeCollection(documents)

    def search(self, query_embedding, top_k=5, content_type=None):
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }


def build_retriever():
    return MultimodalRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(DOCUMENTS),
        lexical_index=BM25Index(DOCUMENTS),
    )


def test_off_topic_query_is_gated_out():
    retriever = build_retriever()

    assert retriever.search_multimodal("cryptocurrency blockchain") == []


def test_grounded_query_is_not_gated_out():
    retriever = build_retriever()

    results = retriever.search_multimodal("Dell OptiPlex desktop")

    assert results
    assert results[0]["chunk_id"] == "c1"
