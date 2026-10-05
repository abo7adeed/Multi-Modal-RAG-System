"""
Ad-hoc verification: does a follow-up question actually get answered
differently once history is supplied?

Sends the same follow-up twice against a running server - once with
no history (the reference behaviour) and once with a prior question -
and prints both answers side by side.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/verify_followup_live.py
"""
import json
import sys

import httpx

BASE = "http://127.0.0.1:8011"

FIRST = "What is the Dell OptiPlex 3020?"
FOLLOW_UP = "What about its warranty?"


def ask(query: str, history: list[dict] | None = None) -> tuple[str, int]:
    """Return (answer, source_count) for a streamed question."""
    answer: list[str] = []
    sources = 0
    event = None
    data: list[str] = []

    with httpx.stream(
        "POST",
        f"{BASE}/api/v1/rag/stream",
        json={"query": query, "history": history or []},
        timeout=300.0,
    ) as response:
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].strip())
            elif line == "":
                if event is None:
                    continue
                payload = json.loads("\n".join(data)) if data else {}
                if event == "token":
                    answer.append(payload.get("text", ""))
                elif event == "sources":
                    sources = len(payload.get("sources", []))
                    pages = [s.get("page") for s in payload.get("sources", [])]
                    print(f"    cited pages: {pages}")
                event = None
                data = []

    return "".join(answer), sources


def main() -> None:
    print(f"turn 1: {FIRST!r}")
    answer, sources = ask(FIRST)
    print(f"  sources={sources}\n  answer={answer!r}\n")

    print(f"follow-up WITHOUT history: {FOLLOW_UP!r}")
    bare, bare_sources = ask(FOLLOW_UP)
    print(f"  sources={bare_sources}\n  answer={bare!r}\n")

    print(f"follow-up WITH history: {FOLLOW_UP!r}")
    with_history, with_sources = ask(
        FOLLOW_UP,
        history=[
            {"role": "user", "content": FIRST},
            {"role": "assistant", "content": answer},
        ],
    )
    print(f"  sources={with_sources}\n  answer={with_history!r}\n")

    if bare == with_history:
        print("IDENTICAL: history made no difference for this pair.")
    else:
        print("DIFFERENT: history changed the answer.")


if __name__ == "__main__":
    main()