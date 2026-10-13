"""Score retrieval against evals/questions.yaml.

Run from the repo root:  uv run python -m evals.run_retrieval [run_name] [vector|keyword|hybrid]
A retrieved chunk is a hit when it has the same project and path as an expected
entry and its line range overlaps the expected range.
"""

import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

from src.kb.chunker import CHUNK_LINES, MAX_CHUNK_CHARS, OVERLAP_LINES
from src.kb.embedder import MODEL_NAME
from src.kb.retrieval import search_chunks
from src.kb.store import count

EVALS_DIR = Path(__file__).parent
K = 5
RECALL_AT = (1, 3, 5)
PASSWORD = re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE)


def is_hit(chunk: dict, expected: dict) -> bool:
    start, end = expected["lines"]
    return (
        chunk["project"] == expected["project"]
        and chunk["path"] == expected["path"]
        and chunk["start_line"] <= end
        and chunk["end_line"] >= start
    )


def first_hit_rank(chunks: list[dict], expected: list[dict]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if any(is_hit(chunk, e) for e in expected):
            return rank
    return None


def label(chunk: dict) -> str:
    return f"{chunk['project']}/{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}"


def run_question(question: dict, mode: str) -> dict:
    start = time.perf_counter()
    chunks = search_chunks(question["question"], k=K, mode=mode)
    latency = time.perf_counter() - start
    return {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "expected": question["expected"] or [],
        "rank": first_hit_rank(chunks, question["expected"] or []),
        "latency_s": round(latency, 3),
        "top_score": round(chunks[0]["score"], 4),
        "retrieved": [label(c) for c in chunks],
        "has_password": any(PASSWORD.search(c["text"]) for c in chunks),
    }


def recall(results: list[dict], k: int) -> float:
    return sum(r["rank"] is not None and r["rank"] <= k for r in results) / len(results)


def print_report(results: list[dict]) -> None:
    scored = [r for r in results if r["expected"]]
    print(f"\nscored questions: {len(scored)} | chunks in index: {count()}")
    print("  " + " | ".join(f"recall@{k} {recall(scored, k):.0%}" for k in RECALL_AT))
    print(f"  p50 latency (embed + search): {statistics.median(r['latency_s'] for r in scored):.2f}s")

    by_type = defaultdict(list)
    for r in scored:
        by_type[r["type"]].append(r)
    print("\nrecall@5 by type:")
    for kind, group in sorted(by_type.items()):
        print(f"  {kind:11} {recall(group, K):4.0%}  ({len(group)} questions)")

    print(f"\nmissed (not in top {K}):")
    for r in [r for r in scored if r["rank"] is None]:
        print(f"  {r['id']} [{r['type']}] {r['question']}")
        print("     expected:", ", ".join(f"{e['project']}/{e['path']}:{e['lines'][0]}-{e['lines'][1]}" for e in r["expected"]))
        print("     got:     ", r["retrieved"][0], "...")

    print("\nrank of first hit (None = missed):", {r["id"]: r["rank"] for r in scored})

    for r in [r for r in results if r["type"] == "negative"]:
        print(f"\nnegative {r['id']}: top score {r['top_score']} -> {r['retrieved'][0]}")
    for r in [r for r in results if r["type"] == "safety"]:
        print(f"\nsafety {r['id']}: password-like text in top {K}? {r['has_password']}")
        for chunk in r["retrieved"]:
            print("   ", chunk)


def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    questions = yaml.safe_load((EVALS_DIR / "questions.yaml").read_text())["questions"]
    results = [run_question(q, mode) for q in questions]
    print_report(results)

    scored = [r for r in results if r["expected"]]
    out = EVALS_DIR / "results" / f"{run_name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "embedding_model": MODEL_NAME,
            "chunk_lines": CHUNK_LINES,
            "overlap_lines": OVERLAP_LINES,
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "embedded_text": "project/path header + chunk",
            "search": mode,
            "k": K,
            "chunks_in_index": count(),
        },
        "recall": {f"@{k}": round(recall(scored, k), 3) for k in RECALL_AT},
        "questions": results,
    }, indent=2))
    print(f"\nsaved {out.relative_to(EVALS_DIR.parent)}")


if __name__ == "__main__":
    main()
