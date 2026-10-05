"""
Ad-hoc verification: does the real Ollama provider actually stream
token-by-token, and do both paths agree?

Not part of the test suite: it needs a live provider and costs money.
Run manually with:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/verify_stream.py
"""
import time

from app.generation.providers import build_provider


def main() -> None:
    provider = build_provider()

    print(f"provider={type(provider).__name__} model={provider.model}")

    started = time.perf_counter()
    chunks: list[str] = []
    first_token_at: float | None = None

    for chunk in provider.stream(
        query="What colour is the sky? Answer in one short sentence.",
        text_context=[
            {
                "page": 1,
                "content": "A catalog page showing a blue Dell laptop.",
            }
        ],
        image_context=[],
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        chunks.append(chunk)
        # Show tokens arriving live rather than only at the end.
        print(f"  +{time.perf_counter() - started:6.2f}s {chunk!r}")

    total = time.perf_counter() - started
    answer = "".join(chunks)

    print()
    print(f"chunks={len(chunks)}")
    print(f"time to first token={first_token_at - started:.2f}s")
    print(f"total={total:.2f}s")
    print(f"answer={answer!r}")

    if len(chunks) <= 1:
        print(
            "\nWARNING: a single chunk means the backend is buffering - "
            "the client will not render progressively."
        )
    else:
        print("\nOK: tokens arrived incrementally.")

    buffered_started = time.perf_counter()
    buffered = provider.generate(
        query="What colour is the sky? Answer in one short sentence.",
        text_context=[
            {
                "page": 1,
                "content": "A catalog page showing a blue Dell laptop.",
            }
        ],
        image_context=[],
    )
    print(
        f"buffered took {time.perf_counter() - buffered_started:.2f}s, "
        f"{len(buffered)} chars"
    )
    print(f"streamed took {total:.2f}s, {len(answer)} chars")


if __name__ == "__main__":
    main()