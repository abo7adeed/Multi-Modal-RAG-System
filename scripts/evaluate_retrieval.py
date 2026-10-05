"""
Retrieval evaluation runner.

Scores the live retriever against the hand-labeled query set in
eval/queries.jsonl, reporting recall@k, MRR, the off-topic gate's
false-block rate, and its false-answer (leak) rate.

This loads CLIP and Chroma, so it is a manual/scheduled tool, not
part of the unit-test suite. The scoring logic it uses is pure and
lives in eval/metrics.py (covered by tests/test_eval_metrics.py).

Usage:
    .venv/Scripts/python.exe scripts/evaluate_retrieval.py
    .venv/Scripts/python.exe scripts/evaluate_retrieval.py --k 5 --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/...).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import summarize  # noqa: E402

DEFAULT_QUERIES = ROOT / "eval" / "queries.jsonl"


def load_queries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def retrieve_pages(results: list[dict], k: int) -> list[int | None]:
    """Ranked page numbers from a multimodal retrieval result list."""
    pages = []
    for result in results[:k]:
        metadata = result.get("metadata") or {}
        pages.append(metadata.get("page_number"))
    return pages


def evaluate(
    queries: list[dict],
    retriever,
    k: int,
) -> tuple[list[dict], dict]:
    rows = []
    for case in queries:
        results = retriever.search_multimodal(case["query"], top_k=k)
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "category": case.get("category"),
                "answerable": case["answerable"],
                "expect_pages": case.get("expect_pages", []),
                "retrieved_pages": retrieve_pages(results, k),
            }
        )
    return rows, summarize(rows, k)


def build_retriever(collection: str):
    """Construct the same retriever the API uses."""
    from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
    from app.retrieval.retriever import MultimodalRetriever
    from app.retrieval.vector_store.chroma_store import ChromaVectorStore

    return MultimodalRetriever(
        embedder=CLIPEmbedder(),
        vector_store=ChromaVectorStore(collection_name=collection),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--collection", default="dell_catalog")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query outcomes.",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} labeled queries from {args.queries}")

    retriever = build_retriever(args.collection)
    rows, metrics = evaluate(queries, retriever, args.k)

    if args.verbose:
        print("\n=== PER QUERY ===")
        for row in rows:
            status = (
                row["retrieved_pages"]
                or ["<blocked>"]
            )
            print(
                f"  {row['id']:<26} pages={status}"
            )

    print("\n=== RESULTS ===")
    print(f"  k                  : {metrics['k']}")
    print(f"  answerable         : {metrics['answerable_count']}")
    print(f"  off-topic controls : {metrics['off_topic_count']}")
    print(f"  recall@{metrics['k']:<12}: {metrics['recall_at_k']:.3f}")
    print(f"  MRR                : {metrics['mrr']:.3f}")
    print(f"  false-block rate   : {metrics['false_block_rate']:.3f}")
    print(f"  false-answer rate  : {metrics['false_answer_rate']:.3f}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {"metrics": metrics, "rows": rows},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
