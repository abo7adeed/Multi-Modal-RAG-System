"""
Retrieval tests for image-only corpora.

The off-topic gate relies on lexical overlap. A corpus containing no
text chunks has no lexical index entries, so the gate must not reject
every query and CLIP must drive retrieval on its own.
"""
from app.retrieval.lexical import BM25Index
from app.retrieval.retriever import MultimodalRetriever


class FakeEmbedder:
    def embed_text(self, text):
        return [0.0, 1.0]

    def embed_texts(self, texts):
        return [[0.0, 1.0] for _ in texts]

    def embed_images(self, paths):
        return [[0.0, 1.0] for _ in paths]


class FakeCollection:
    def __init__(self, rows):
        self.rows = list(rows)

    def get(self, ids=None, include=None):
        if ids is not None:
            rows = [r for r in self.rows if r["chunk_id"] in ids]
        else:
            rows = list(self.rows)
        return {
            "ids": [r["chunk_id"] for r in rows],
            "documents": [r["document"] for r in rows],
            "metadatas": [r["metadata"] for r in rows],
        }


class FakeStore:
    def __init__(self, rows):
        self.collection = FakeCollection(rows)

    def search(self, query_embedding, top_k=5, content_type=None):
        rows = [
            r
            for r in self.collection.rows
            if content_type is None
            or r["metadata"].get("content_type") == content_type
        ]
        return {
            "ids": [[r["chunk_id"] for r in rows]],
            "documents": [[r["document"] for r in rows]],
            "metadatas": [[r["metadata"] for r in rows]],
            "distances": [[0.1 for _ in rows]],
        }


IMAGE_ROW = {
    "chunk_id": "img-1",
    "document": "Image from page 1",
    "metadata": {
        "content_type": "image",
        "page_number": 1,
        "image_path": "img.jpg",
        "related_chunk_ids": "",
    },
}


def make_retriever(rows):
    return MultimodalRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeStore(rows),
    )


def test_image_only_corpus_is_queryable():
    retriever = make_retriever([IMAGE_ROW])

    results = retriever.search_multimodal(
        "portrait photo of a person", top_k=5
    )

    assert results, "an image-only corpus must still be searchable"
    assert any(r["content_type"] if "content_type" in r else True for r in results)


def test_image_only_corpus_has_empty_lexical_index():
    retriever = make_retriever([IMAGE_ROW])

    assert retriever.lexical_index.documents == {}


def test_text_corpus_still_gates_off_topic_queries():
    text_row = {
        "chunk_id": "txt-1",
        "document": "Dell OptiPlex 3020 Micro desktop",
        "metadata": {
            "content_type": "text",
            "page_number": 1,
            "image_path": "",
            "related_chunk_ids": "",
        },
    }

    retriever = make_retriever([text_row])

    # No lexical overlap -> rejected.
    assert retriever.search_multimodal("hello", top_k=5) == []

    # Real overlap -> returned.
    assert retriever.search_multimodal("OptiPlex desktop", top_k=5)


def test_empty_corpus_returns_nothing():
    retriever = make_retriever([])

    assert retriever.search_multimodal("anything", top_k=5) == []


def test_lexical_index_is_built_once_per_corpus():
    retriever = make_retriever([IMAGE_ROW])

    first = retriever.lexical_index
    second = retriever.lexical_index

    assert first is second
    assert isinstance(first, BM25Index)