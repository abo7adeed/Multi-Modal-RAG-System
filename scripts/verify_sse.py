"""
Ad-hoc verification of the SSE endpoint against a running server.

Prints each frame with the wall-clock time it arrived, so incremental
delivery is observable rather than assumed.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/verify_sse.py
"""
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8011"


def stream(query: str) -> None:
    started = time.perf_counter()
    print(f"\n=== {query!r} ===")

    event = None
    data_lines: list[str] = []
    tokens: list[str] = []
    sources = 0

    with httpx.stream(
        "POST",
        f"{BASE}/api/v1/rag/stream",
        json={"query": query},
        timeout=300.0,
    ) as response:
        print(f"status={response.status_code}")
        print(f"content-type={response.headers.get('content-type')}")

        if response.status_code != 200:
            print(response.read().decode())
            return

        for line in response.iter_lines():
            elapsed = time.perf_counter() - started

            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
            elif line == "":
                if event is None:
                    continue

                payload = json.loads("\n".join(data_lines)) if data_lines else {}

                if event == "token":
                    text = payload.get("text", "")
                    tokens.append(text)
                    print(f"  +{elapsed:6.2f}s token {text!r}")
                elif event == "sources":
                    sources = len(payload.get("sources", []))
                    print(f"  +{elapsed:6.2f}s sources ({sources})")
                    for source in payload.get("sources", []):
                        print(
                            f"        page={source.get('page')} "
                            f"chunk={source.get('chunk_id')} "
                            f"kind={source.get('kind')} "
                            f"score={source.get('score')}"
                        )
                else:
                    print(f"  +{elapsed:6.2f}s {event} {payload}")

                event = None
                data_lines = []

    print(f"\ntokens={len(tokens)} total={time.perf_counter() - started:.2f}s")
    print(f"answer={''.join(tokens)!r}")


def main() -> None:
    query = (
        sys.argv[1] if len(sys.argv) > 1
        else "What is the Dell OptiPlex 3020?"
    )
    stream(query)


if __name__ == "__main__":
    main()