# Retrieval evaluation baseline

Measured with `scripts/evaluate_retrieval.py --k 5` against the live
`dell_catalog` collection (268 chunks: 176 text, 92 image) and the 25
labeled queries in [`queries.jsonl`](queries.jsonl) (20 answerable, 5
off-topic controls).

Raw output: [`baseline.json`](baseline.json) (before),
[`after-gate.json`](after-gate.json) (after).

## Results

| metric              | before | after coverage gate |
| ------------------- | ------ | ------------------- |
| recall@5            | 0.900  | **0.900**           |
| MRR                 | 0.758  | **0.758**           |
| false-block rate    | 0.000  | **0.000**           |
| false-answer rate   | 0.800  | **0.600**           |

## What this revealed

Two things contradicted the earlier assumptions, and both are worth
stating plainly:

1. **Retrieval quality is good, not weak.** recall@5 of 0.90 with MRR
   0.758 means the CLIP + BM25 + RRF stack already finds the right
   page for the large majority of questions. The weak link was never
   ranking.
2. **The off-topic gate was the real defect.** It only checked that
   BM25 returned *something*, so a single common word made an
   unrelated question "match". Four of five off-topic controls were
   answered from the corpus (false-answer rate 0.80).

## The change

The gate now requires the rarity-weighted share of a query's content
terms that exist in the index to be at least
`RETRIEVAL_LEXICAL_MIN_COVERAGE` (default `0.30`). See
[`app/retrieval/lexical.py`](../app/retrieval/lexical.py)
(`keyword_relevance`) and
[`app/retrieval/retriever.py`](../app/retrieval/retriever.py).

The threshold is deliberately conservative. Labeled answerable queries
bottom out at coverage 0.381, and off-topic controls reach 0.685, so
the two distributions **overlap** — no threshold separates them
cleanly. `0.30` is the largest value that blocks the worst leaks while
leaving a safety margin above every answerable query.

## Known limitation

A clean separation is not achievable with BM25 alone here, because the
collection contains unrelated uploaded documents whose vocabulary
("python", "virtual", "environment") makes some off-topic questions
genuinely lexical matches. Closing that gap needs either a semantic
relevance judge (cross-encoder) or scoping retrieval to a single
document/collection — not a better threshold.

This number should be re-measured after any change to retrieval; the
harness exists precisely so that such changes are judged on data.
