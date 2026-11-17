"""Best-case retrieval score for golden Part B: search_knowledge WITH the breed filter (issue #9).

`eval/run_eval.py --part B` is the official baseline: it searches without any filter, because the golden
questions carry no breed. This script answers a second question: how good is retrieval when the agent
knows the breed correctly (from the photo or the chat) and passes it as `breed`? The breed is taken from the
expected chunk ID ("kb:<breed-slug>:<section>:<idx>"), so this is an upper bound that assumes a perfect
breed. It does not touch eval/run_eval.py and does not use the section filter.

Output: eval/results_B_week8_breed.json (same shape as results_B.json, plus the breed used per item).

Usage:
    python scripts/eval_search_breed_filter.py
    python scripts/eval_search_breed_filter.py --output eval/results_B_week8_breed.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from mcp_server.tools.knowledge import search_knowledge  # noqa: E402
from shared.breeds import load_breeds_map, slugify  # noqa: E402


def breed_of(chunk_id: str, by_slug: dict[str, str]) -> str:
    slug = chunk_id.split(":")[1]
    if slug not in by_slug:
        sys.exit(f"Chunk ID {chunk_id!r}: no breed in breeds_map.json has the slug {slug!r}")
    return by_slug[slug]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default=config.EVAL_DIR / "results_B_week8_breed.json")
    args = parser.parse_args()

    by_slug = {slugify(display): display for display in load_breeds_map()}
    items = [json.loads(line) for line in args.golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [item for item in items if item["part"] == "B"]
    if not items:
        sys.exit(f"No Part B items in {args.golden}")

    per_item, hit1_count, hit3_count = [], 0, 0
    for item in items:
        expected = item["expected"]["source_chunk_ids"]
        breeds = sorted({breed_of(chunk_id, by_slug) for chunk_id in expected})
        breed = breeds[0] if len(breeds) == 1 else None  # several breeds expected: no filter for that item
        out = search_knowledge(query=item["input"]["query"], breed=breed)
        retrieved = [chunk.chunk_id for chunk in out.chunks]
        hit1, hit3 = any(c in retrieved[:1] for c in expected), any(c in retrieved[:3] for c in expected)
        hit1_count += hit1
        hit3_count += hit3
        per_item.append({"id": item["id"], "query": item["input"]["query"], "breed_filter": breed,
                         "expected_chunk_ids": expected, "retrieved_chunk_ids": retrieved,
                         "hit@1": hit1, "hit@3": hit3, "pass": hit3})

    n = len(items)
    result = {
        "part": "B",
        "mode": "breed filter taken from the expected chunk ID (best case: the breed is known and correct)",
        "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),  # same hash as eval/run_eval.py
        "aggregates": {"num_items": n, "hit@1": round(hit1_count / n, 4), "hit@3": round(hit3_count / n, 4),
                       "items_without_filter": sum(1 for i in per_item if i["breed_filter"] is None)},
        "per_item": per_item,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Part B with breed filter ({n} items): hit@1 {result['aggregates']['hit@1']:.1%}, "
          f"hit@3 {result['aggregates']['hit@3']:.1%}. Written to {args.output}")


if __name__ == "__main__":
    main()
