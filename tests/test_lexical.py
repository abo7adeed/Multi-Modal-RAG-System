"""
Tests for BM25 lexical retrieval: ranking precision, stopword and
off-topic handling, and Chroma-backed construction.
"""
from app.retrieval.lexical import BM25Index, tokenize

DOCUMENTS = {
    "c1": "Dell OptiPlex 3020 Micro desktop with Intel Core i5 processor",
    "c2": "Dell Latitude 7420 laptop with wireless connectivity",
    "c3": "Dell Precision workstation laptop for heavy workloads",
    "c4": "Accessories including keyboards mice and monitors",
}


def test_tokenize_drops_stopwords_and_lowercases():
    tokens = tokenize("What is the Dell OptiPlex 3020?")

    assert "dell" in tokens
    assert "optiplex" in tokens
    assert "the" not in tokens
    assert "what" not in tokens


def test_ranks_the_chunk_that_actually_contains_the_term():
    index = BM25Index(DOCUMENTS)

    results = index.search("OptiPlex 3020 desktop", top_k=3)

    assert results
    assert results[0]["chunk_id"] == "c1"


def test_workstation_query_ranks_workstation_chunk():
    index = BM25Index(DOCUMENTS)

    results = index.search("workstation laptop", top_k=3)

    assert results[0]["chunk_id"] == "c3"


def test_off_topic_query_returns_nothing():
    # This is the signal that rejects "hello" and similar queries:
    # no meaningful term overlap with the index.
    index = BM25Index(DOCUMENTS)

    assert index.search("hello", top_k=5) == []
    assert index.search("who are you?", top_k=5) == []
    assert index.search("weather in Paris", top_k=5) == []


def test_empty_index_returns_nothing():
    index = BM25Index({})

    assert index.search("dell", top_k=5) == []


def test_results_are_ranked_and_scored():
    index = BM25Index(DOCUMENTS)

    results = index.search("dell wireless laptop", top_k=3)

    assert [r["rank"] for r in results] == list(
        range(1, len(results) + 1)
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_keyword_relevance_is_high_for_grounded_query():
    index = BM25Index(DOCUMENTS)

    relevance = index.keyword_relevance("Dell OptiPlex desktop")

    assert relevance["coverage"] == 1.0
    assert relevance["matched"] == relevance["total"]
    assert relevance["top_score"] > 0


def test_keyword_relevance_penalises_absent_terms():
    # Every content term is unknown to the corpus: the strongest
    # possible signal that the question is off-topic.
    index = BM25Index(DOCUMENTS)

    relevance = index.keyword_relevance("cryptocurrency blockchain")

    assert relevance["coverage"] == 0.0
    assert relevance["top_score"] == 0.0


def test_keyword_relevance_ignores_stopword_only_query():
    index = BM25Index(DOCUMENTS)

    relevance = index.keyword_relevance("the and of")

    assert relevance["total"] == 0
    assert relevance["coverage"] == 0.0


def test_ignores_image_chunks_when_built_from_store():
    class FakeCollection:
        def get(self, include=None):
            return {
                "ids": ["t1", "i1"],
                "documents": ["Dell laptop", "Image from page 2"],
                "metadatas": [
                    {"content_type": "text"},
                    {"content_type": "image"},
                ],
            }

    class FakeStore:
        collection = FakeCollection()

    index = BM25Index.from_vector_store(FakeStore())

    assert list(index.documents.keys()) == ["t1"]