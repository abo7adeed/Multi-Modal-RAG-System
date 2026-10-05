"""Unit tests for the pure evaluation metrics (no models/network)."""
from eval.metrics import (
    recall_at_k,
    reciprocal_rank,
    summarize,
)


def test_recall_at_k_true_when_expected_page_in_window():
    assert recall_at_k([6], [1, 2, 6], k=3) is True


def test_recall_at_k_false_when_expected_page_outside_window():
    assert recall_at_k([6], [1, 2, 3, 6], k=3) is False


def test_recall_at_k_false_when_no_expectation():
    assert recall_at_k([], [1, 2], k=3) is False


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank([15], [1, 15, 3]) == 0.5


def test_reciprocal_rank_zero_when_absent():
    assert reciprocal_rank([99], [1, 2, 3]) == 0.0


def test_summarize_reports_recall_mrr_and_gate_rates():
    rows = [
        {
            "answerable": True,
            "expect_pages": [6],
            "retrieved_pages": [1, 6],
        },
        {
            # answerable, but the gate returned nothing
            "answerable": True,
            "expect_pages": [20],
            "retrieved_pages": [],
        },
        {
            # off-topic control the gate correctly blocked
            "answerable": False,
            "expect_pages": [],
            "retrieved_pages": [],
        },
        {
            # off-topic control that leaked through
            "answerable": False,
            "expect_pages": [],
            "retrieved_pages": [4],
        },
    ]

    metrics = summarize(rows, k=5)

    # One of two answerable queries hit its expected page.
    assert metrics["recall_at_k"] == 0.5
    # Hit at rank 2 => 0.5; blocked => 0.0.
    assert metrics["mrr"] == 0.25
    # One of two answerable queries was wrongly blocked.
    assert metrics["false_block_rate"] == 0.5
    # One of two off-topic controls leaked.
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["answerable_count"] == 2
    assert metrics["off_topic_count"] == 2
