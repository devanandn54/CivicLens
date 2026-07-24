"""
Week 4 — Baseline eval: retrieval hit@k against your golden set.

Start with retrieval-only evals (cheap, deterministic, no LLM judge needed).
Week 9 adds RAGAS faithfulness + LLM-as-judge citation checks on top.

Golden set format (evals/golden_set.jsonl), one JSON object per line:
    {"question": "...", "expected_meeting_id": 1, "expected_phrase": "eastside rezoning"}

A hit = any top-k chunk is from the expected meeting AND contains the phrase
(case-insensitive). Crude but honest — and improvements against a crude metric
still tell you the truth about direction.

Usage:
    python evals/run_evals.py --k 5
"""

import argparse
import json
import sys
from pathlib import Path

# reuse the API's retrieve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from app.main import retrieve  # noqa: E402

GOLDEN = Path(__file__).parent / "golden_set.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if not cases:
        sys.exit("golden_set.jsonl is empty — write your first 25 questions (Week 4)")

    hits = 0
    misses: list[str] = []
    for case in cases:
        results = retrieve(case["question"], k=args.k)
        found = any(
            r["meeting_id"] == case["expected_meeting_id"]
            and case["expected_phrase"].lower() in r["text"].lower()
            for r in results
        )
        hits += found
        if not found:
            misses.append(case["question"])

    rate = hits / len(cases)
    print(f"\nhit@{args.k}: {hits}/{len(cases)} = {rate:.1%}")
    if misses:
        print("\nmisses:")
        for q in misses:
            print(f"  - {q}")

    # CI gate (Week 10): fail the build below threshold
    threshold = 0.5
    if rate < threshold:
        sys.exit(f"FAIL: hit@{args.k} {rate:.1%} below threshold {threshold:.0%}")


if __name__ == "__main__":
    main()
