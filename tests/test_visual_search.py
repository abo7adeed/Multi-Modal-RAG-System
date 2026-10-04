"""
Image-to-image retrieval.

When the user attaches their own picture, the index is searched with
that picture: CLIP's visual tower matches it against the page images
of the corpus, so the model can compare the upload with what the
documents actually show.
"""
from app.retrieval.retriever import MultimodalRetriever


class FakeEmbedder:
    def __init__(self):
        self.embedded = []

    def embed_image(self, image_path):
        self.embedded.append(image_path)
        return [0.0, 0.7, 0.7]


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def search(self, query_embedding, top_k=5, content_type=None):
        self.queries.append(
            {
                "embedding": query_embedding,
                "top_k": top_k,
                "content_type": content_type,
            }
        )
        rows = [
            row
            for row in self.rows
            if content_type is None
            or row["metadata"].get("content_type") == content_type
        ]
        return {
            "ids": [[row["chunk_id"] for row in rows]],
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[0.2 for _ in rows]],
        }


def image_row(chunk_id, page):
    return {
        "chunk_id": chunk_id,
        "document": f"Image from page {page}",
        "metadata": {
            "content_type": "image",
            "page_number": page,
            "image_path": f"page_{page}.jpg",
            "related_chunk_ids": "",
        },
    }


def text_row(chunk_id, page):
    return {
        "chunk_id": chunk_id,
        "document": "Dell OptiPlex 3020 Micro desktop.",
        "metadata": {
            "content_type": "text",
            "page_number": page,
            "image_path": "",
            "related_chunk_ids": "",
        },
    }


def test_search_by_image_queries_the_image_collection_only():
    store = FakeStore([image_row("img-1", 3)])
    embedder = FakeEmbedder()

    MultimodalRetriever(
        embedder=embedder, vector_store=store
    ).search_by_image("attachment.jpg", top_k=2)

    # Only page images can be visually compared with an upload.
    assert store.queries[0]["content_type"] == "image"
    assert store.queries[0]["top_k"] == 2
    assert embedder.embedded == ["attachment.jpg"]


def test_search_by_image_returns_page_images():
    store = FakeStore(
        [image_row("img-1", 3), image_row("img-2", 5), text_row("t-1", 1)]
    )

    results = MultimodalRetriever(
        embedder=FakeEmbedder(), vector_store=store
    ).search_by_image("attachment.jpg", top_k=2)

    assert [result["chunk_id"] for result in results] == [
        "img-1",
        "img-2",
    ]
    assert results[0]["metadata"]["page_number"] == 3
    assert results[0]["source"] == "retrieved"


def test_search_by_image_returns_nothing_for_a_text_only_corpus():
    store = FakeStore([text_row("t-1", 1)])

    results = MultimodalRetriever(
        embedder=FakeEmbedder(), vector_store=store
    ).search_by_image("attachment.jpg")

    assert results == []
