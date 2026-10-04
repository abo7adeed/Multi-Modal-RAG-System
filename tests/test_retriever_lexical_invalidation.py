"""
Regression tests for the lexical index lifecycle.

The BM25 index is cached in memory and built from Chroma. It must be
invalidated whenever new text is indexed, otherwise freshly uploaded
documents are invisible to text retrieval.
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
    """Minimal stand-in for a Chroma collection."""

    def __init__(self, documents=None):
        self.documents = dict(documents or {})

    def add(self, ids, documents, metadatas):
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            self.documents[chunk_id] = (text, metadata)

    def get(self, ids=None, include=None):
        ids = list(self.documents.keys())
        if ids is not None:
            ids = [i for i in ids if i in self.documents]
        texts = [self.documents[i][0] for i in ids]
        metadatas = [self.documents[i][1] for i in ids]
        return {
            "ids": ids,
            "documents": texts,
            "metadatas": metadatas,
        }


class FakeStore:
    def __init__(self, documents=None):
        self.collection = FakeCollection(documents)
        self.added = []

    def add(self, chunk_ids, embeddings, documents, metadatas):
        self.added.append(chunk_ids)
        self.collection.add(chunk_ids, documents, metadatas)

    def search(self, query_embedding, top_k=5, content_type=None):
        # Return every stored chunk; RRF/gating is not what is under
        # test here.
        rows = [
            (chunk_id, text, metadata)
            for chunk_id, (text, metadata) in (
                self.collection.documents.items()
            )
            if content_type is None
            or metadata.get("content_type") == content_type
        ]
        rows = rows[:top_k]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "distances": [[0.1 for _ in rows]],
        }


def make_retriever(store):
    return MultimodalRetriever(embedder=FakeEmbedder(), vector_store=store)


def test_newly_indexed_text_becomes_lexically_searchable():
    store = FakeStore(
        {
            "old": (
                "Dell OptiPlex desktop computer",
                {"content_type": "text", "page_number": 1},
            )
        }
    )
    retriever = make_retriever(store)

    # Warm the cache, as happens on the first real query.
    assert retriever.lexical_index.search("dell", top_k=5)

    # Ingest a new document.
    from app.ingestion.chunk_schemas import Chunk

    retriever.index_chunks([
        Chunk(
            chunk_id="new-1",
            document_id="doc-new",
            content_type="text",
            text="Product: AI Laptop. Price: $1200. RAM: 3GB.",
            page_number=1,
            metadata={
                "source": "data/uploads/test.pdf",
                "document_type": "pdf",
            },
        )
    ])

    # The freshly indexed chunk must be searchable right away.
    results = retriever.lexical_index.search("AI Laptop price", top_k=5)
    assert results, "newly indexed text must appear in the lexical index"
    assert results[0]["chunk_id"] == "new-1"


def test_invalidate_forces_rebuild():
    store = FakeStore(
        {
            "a": ("alpha beta", {"content_type": "text", "page_number": 1})
        }
    )
    retriever = make_retriever(store)

    first = retriever.lexical_index
    assert retriever.lexical_index is first  # cached

    retriever.invalidate_lexical_index()

    assert retriever.lexical_index is not first  # rebuilt


def test_index_chunks_with_no_chunks_leaves_index_intact():
    store = FakeStore(
        {
            "a": ("alpha beta", {"content_type": "text", "page_number": 1})
        }
    )
    retriever = make_retriever(store)
    first = retriever.lexical_index

    retriever.index_chunks([])

    assert retriever.lexical_index is first


def test_lexical_index_built_from_store_filters_image_chunks():
    store = FakeStore(
        {
            "t1": ("dell laptop", {"content_type": "text", "page_number": 1}),
            "i1": (
                "Image from page 2",
                {"content_type": "image", "page_number": 2},
            ),
        }
    )
    retriever = make_retriever(store)

    index = retriever.lexical_index

    assert isinstance(index, BM25Index)
    assert set(index.documents) == {"t1"}