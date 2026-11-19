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
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from shared.breeds import load_breeds_map, slugify  # noqa: E402


def _load_run_eval():
    """eval/run_eval.py as a module (eval/ is not a package). It is reused, not copied, and not modified."""
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden_sha256(path: Path) -> str:
    """SHA-256 of the file with CRLF turned into LF, so a Windows checkout (autocrlf) and a Linux/macOS one agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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

    run_eval = _load_run_eval()
    by_slug = {slugify(display): display for display in load_breeds_map()}
    items = run_eval.load_golden(args.golden, "B")
    if not items:
        sys.exit(f"No Part B items in {args.golden}")

    breed_for_query = {}
    for item in items:
        breeds = sorted({breed_of(chunk_id, by_slug) for chunk_id in item["expected"]["source_chunk_ids"]})
        breed_for_query[item["input"]["query"]] = breeds[0] if len(breeds) == 1 else None  # several: no filter

    # run_eval.eval_part_b calls its own search_knowledge(query=...); the only change here is the breed argument.
    official_search = run_eval.search_knowledge
    run_eval.search_knowledge = lambda query: official_search(query=query, breed=breed_for_query[query])
    try:
        evaluated = run_eval.eval_part_b(items)
    finally:
        run_eval.search_knowledge = official_search
    per_item = [{**row, "breed_filter": breed_for_query[row["query"]]} for row in evaluated["per_item"]]

    aggregates = {**evaluated["aggregates"],
                  "items_without_filter": sum(1 for row in per_item if row["breed_filter"] is None)}
    result = {
        "part": "B",
        "mode": "breed filter taken from the expected chunk ID (best case: the breed is known and correct)",
        "golden_sha256": golden_sha256(args.golden),  # line endings normalised; eval/run_eval.py hashes raw bytes
        "aggregates": aggregates,
        "per_item": per_item,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Part B with breed filter ({aggregates['num_items']} items): hit@1 {result['aggregates']['hit@1']:.1%}, "
          f"hit@3 {result['aggregates']['hit@3']:.1%}. Written to {args.output}")


if __name__ == "__main__":
    main()
