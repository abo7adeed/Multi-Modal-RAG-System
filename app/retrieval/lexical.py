"""
Lightweight lexical (BM25) retrieval over the indexed text chunks.

CLIP is a vision-language encoder: its text tower has very poor
lexical precision, so short queries ("hello", "workstation laptops")
land in nearly the same neighbourhood regardless of meaning. Measured
on this collection, "hello" matches *closer* than several genuinely
relevant queries.

BM25 supplies the missing lexical signal. It is fused with CLIP via
the existing RRF step, so retrieval keeps working for images (where
CLIP is the right tool) while text ranking becomes discriminative.
"""
import math
import re
from collections import Counter
from typing import Any

# Minimal English stoplist: only words that carry no retrieval signal.
STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how i in is it its
    of on or that the this to was were what when where which who why
    will with you your do does did can could should would me my we our
    """.split()
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


class BM25Index:
    """BM25 over chunk documents, held in memory."""

    def __init__(
        self,
        documents: dict[str, str],
        k1: float = K1,
        b: float = B,
    ):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_ids = list(documents.keys())
        self.doc_tokens = {
            doc_id: tokenize(text)
            for doc_id, text in documents.items()
        }
        self.doc_lengths = {
            doc_id: len(tokens)
            for doc_id, tokens in self.doc_tokens.items()
        }
        self.avg_length = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens.values():
            document_frequency.update(set(tokens))
        self.idf = {
            term: max(
                0.0,
                math.log(
                    1
                    + (len(self.doc_ids) - freq + 0.5)
                    / (freq + 0.5)
                ),
            )
            for term, freq in document_frequency.items()
        }

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    @classmethod
    def from_vector_store(cls, store: Any) -> "BM25Index":
        """
        Build the lexical index from documents already persisted in
        Chroma, so no re-indexing is required.
        """
        stored = store.collection.get(include=["documents", "metadatas"])

        documents: dict[str, str] = {}

        ids = stored.get("ids") or []
        texts = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []

        for chunk_id, text, metadata in zip(ids, texts, metadatas):
            if not text:
                continue
            # Only text chunks carry searchable prose.
            if (metadata or {}).get("content_type") != "text":
                continue
            documents[chunk_id] = text

        return cls(documents)

    # ------------------------------------------------------------
    # Search
    # ------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k chunks by BM25 score.

        An empty result means the query shares no meaningful term
        with the index, which is the signal used to reject off-topic
        questions.
        """
        terms = tokenize(query)

        if not terms or not self.doc_ids:
            return []

        scores: list[tuple[str, float]] = []

        for doc_id in self.doc_ids:
            tokens = self.doc_tokens[doc_id]
            if not tokens:
                continue

            counts = Counter(tokens)
            length = self.doc_lengths[doc_id]
            score = 0.0

            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1)
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator

            if score > 0:
                scores.append((doc_id, score))

        if not scores:
            return []

        scores.sort(key=lambda item: item[1], reverse=True)

        return [
            {
                "chunk_id": doc_id,
                "score": score,
                "rank": rank,
                "document": self.documents[doc_id],
            }            for rank, (doc_id, score) in enumerate(
                scores[:top_k], start=1
            )
        ]

    # ------------------------------------------------------------
    # Relevance signal
    # ------------------------------------------------------------

    def keyword_relevance(self, query: str) -> dict[str, Any]:
        """
        How well a query's own terms are grounded in the corpus.

        BM25 non-emptiness alone is a weak relevance signal: one
        common word is enough to make an unrelated question "match".
        This measures the share of the query's content terms that
        actually exist in the index, weighted by rarity, and pairs it
        with the best BM25 score. Off-topic questions that merely
        borrow a common word score low on both.
        """
        terms = list(dict.fromkeys(tokenize(query)))

        if not terms:
            return {
                "coverage": 0.0,
                "matched": 0,
                "total": 0,
                "top_score": 0.0,
            }

        # A term the corpus has never seen is as rare as it gets. It
        # is the strongest evidence the query is about something this
        # index knows nothing about, so it must not be ignored.
        rarest = max(self.idf.values()) if self.idf else 0.0

        weight_total = 0.0
        weight_matched = 0.0
        matched = 0

        for term in terms:
            weight = self.idf.get(term, rarest)
            weight_total += weight
            if term in self.idf:
                weight_matched += weight
                matched += 1

        top = self.search(query, top_k=1)

        return {
            "coverage": (
                weight_matched / weight_total
                if weight_total
                else 0.0
            ),
            "matched": matched,
            "total": len(terms),
            "top_score": top[0]["score"] if top else 0.0,
        }
