"""
Pure retrieval metrics for the evaluation harness.

Deliberately dependency-free (no CLIP, no Chroma, no network) so the
scoring logic can be unit-tested in milliseconds. The runner that
produces real rankings lives in scripts/evaluate_retrieval.py.
"""
from typing import Any, Iterable, Sequence


def recall_at_k(
    expected_pages: Sequence[int],
    retrieved_pages: Sequence[int | None],
    k: int,
) -> bool:
    """True when any expected page appears in the first k results."""
    if not expected_pages:
        return False
    expected = set(expected_pages)
    return any(page in expected for page in list(retrieved_pages)[:k])


def reciprocal_rank(
    expected_pages: Sequence[int],
    retrieved_pages: Sequence[int | None],
) -> float:
    """1 / rank of the first relevant result, else 0.0."""
    if not expected_pages:
        return 0.0
    expected = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return 1.0 / rank
    return 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize(
    rows: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """
    Aggregate per-query results into headline metrics.

    Each row needs: answerable (bool), expect_pages (list[int]),
    retrieved_pages (list[int | None]).
    """
    answerable = [row for row in rows if row["answerable"]]
    off_topic = [row for row in rows if not row["answerable"]]

    recall = _mean(
        1.0
        if recall_at_k(
            row["expect_pages"], row["retrieved_pages"], k
        )
        else 0.0
        for row in answerable
    )

    mrr = _mean(
        reciprocal_rank(row["expect_pages"], row["retrieved_pages"])
        for row in answerable
    )

    # An answerable query the retriever returned nothing for: the
    # off-topic gate refused something it should have answered.
    false_block_rate = _mean(
        1.0 if not row["retrieved_pages"] else 0.0
        for row in answerable
    )

    # An off-topic query the retriever answered anyway: the gate let
    # an ungrounded question through.
    false_answer_rate = _mean(
        1.0 if row["retrieved_pages"] else 0.0
        for row in off_topic
    )

    return {
        "k": k,
        "answerable_count": len(answerable),
        "off_topic_count": len(off_topic),
        "recall_at_k": recall,
        "mrr": mrr,
        "false_block_rate": false_block_rate,
        "false_answer_rate": false_answer_rate,
    }
