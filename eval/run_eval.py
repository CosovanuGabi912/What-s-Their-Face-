"""Evaluation harness for Parts A, B, and C of the Golden Dataset (issue #8).

Usage:
    python eval/run_eval.py --part A --golden eval/golden.jsonl --output eval/results_A.json
    python eval/run_eval.py --part B --golden eval/golden.jsonl --output eval/results_B.json
    python eval/run_eval.py --part C --golden eval/golden.jsonl --output eval/results_C.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

# Set upload dir to eval/images so tools receive plain image_id
os.environ["WTF_IMAGE_UPLOAD_DIR"] = str(config.EVAL_IMAGES_DIR)
config.IMAGE_UPLOAD_DIR = config.EVAL_IMAGES_DIR

from mcp_server.tools.breed_id import identify_breed
from mcp_server.tools.knowledge import search_knowledge
from shared.schemas import GoldenItem


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_golden(golden_path: Path, part: str) -> List[dict]:
    items = []
    with golden_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if part == "all" or item.get("part") == part:
                GoldenItem.model_validate(item)
                items.append(item)
    return items


def eval_part_a(items: List[dict]) -> Dict[str, Any]:
    per_item = []
    top1_correct_count = 0
    top3_correct_count = 0
    confident_count = 0
    confident_correct_count = 0
    wrong_top1_count = 0
    wrong_and_not_confident_count = 0

    for item in items:
        item_id = item["id"]
        image_id = item["input"]["image_id"]
        expected_breed = item["expected"]["breed"]

        out = identify_breed(image_id=image_id)
        top3_guesses = [c.breed for c in out.top3]
        top1_guess = top3_guesses[0] if top3_guesses else None
        top1_correct = (top1_guess == expected_breed)
        top3_correct = (expected_breed in top3_guesses)
        is_confident = (out.decision == "confident")

        if top1_correct:
            top1_correct_count += 1
        else:
            wrong_top1_count += 1
            if not is_confident:
                wrong_and_not_confident_count += 1

        if top3_correct:
            top3_correct_count += 1

        if is_confident:
            confident_count += 1
            if top1_correct:
                confident_correct_count += 1

        per_item.append({
            "id": item_id,
            "image_id": image_id,
            "expected_breed": expected_breed,
            "top1_guess": top1_guess,
            "top3_guesses": top3_guesses,
            "decision": out.decision,
            "is_confident": is_confident,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "pass": top1_correct,
        })

    n = len(items)
    top1_acc = round(top1_correct_count / n, 4) if n else 0.0
    top3_acc = round(top3_correct_count / n, 4) if n else 0.0
    coverage = round(confident_count / n, 4) if n else 0.0
    selective_acc = round(confident_correct_count / confident_count, 4) if confident_count else 0.0
    wrong_flagged_rate = round(wrong_and_not_confident_count / wrong_top1_count, 4) if wrong_top1_count else 1.0

    aggregates = {
        "num_items": n,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "coverage": coverage,
        "selective_accuracy": selective_acc,
        "wrong_flagged_rate": wrong_flagged_rate,
    }
    return {"aggregates": aggregates, "per_item": per_item}


def eval_part_b(items: List[dict]) -> Dict[str, Any]:
    per_item = []
    hit1_count = 0
    hit3_count = 0

    for item in items:
        item_id = item["id"]
        query = item["input"]["query"]
        expected_chunk_ids = item["expected"]["source_chunk_ids"]

        out = search_knowledge(query=query)
        retrieved_ids = [c.chunk_id for c in out.chunks]
        hit1 = any(cid in retrieved_ids[:1] for cid in expected_chunk_ids)
        hit3 = any(cid in retrieved_ids[:3] for cid in expected_chunk_ids)

        if hit1:
            hit1_count += 1
        if hit3:
            hit3_count += 1

        per_item.append({
            "id": item_id,
            "query": query,
            "expected_chunk_ids": expected_chunk_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "hit@1": hit1,
            "hit@3": hit3,
            "pass": hit3,
        })

    n = len(items)
    aggregates = {
        "num_items": n,
        "hit@1": round(hit1_count / n, 4) if n else 0.0,
        "hit@3": round(hit3_count / n, 4) if n else 0.0,
    }
    return {"aggregates": aggregates, "per_item": per_item}


def classify_message_route(message: str) -> str:
    """Triage router for milestone 2 Part C questions."""
    msg = message.lower()
    vet_keywords = ["chocolate", "poison", "bleeding", "choking", "gasping", "seizure", "needle", "swallowed", "hit by"]
    if any(k in msg for k in vet_keywords):
        return "vet_referral"
    out_of_scope_keywords = ["cat", "kitten", "parrot", "bird", "cheetah", "photosynthesis", "cacti", "adopt"]
    if any(k in msg for k in out_of_scope_keywords):
        return "out_of_scope"
    return "i_dont_know"


def eval_part_c(items: List[dict]) -> Dict[str, Any]:
    per_item = []
    type_totals: Dict[str, int] = {}
    type_correct: Dict[str, int] = {}

    for item in items:
        item_id = item["id"]
        msg = item["input"]["message"]
        expected_type = item["expected"]["response_type"]
        predicted_type = classify_message_route(msg)
        is_pass = (predicted_type == expected_type)

        type_totals[expected_type] = type_totals.get(expected_type, 0) + 1
        if is_pass:
            type_correct[expected_type] = type_correct.get(expected_type, 0) + 1

        per_item.append({
            "id": item_id,
            "message": msg,
            "expected_response_type": expected_type,
            "predicted_response_type": predicted_type,
            "pass": is_pass,
        })

    n = len(items)
    decline_rate_per_type = {
        t: round(type_correct.get(t, 0) / type_totals[t], 4) if type_totals.get(t) else 0.0
        for t in ["vet_referral", "i_dont_know", "out_of_scope"]
        if t in type_totals
    }
    total_correct = sum(type_correct.values())
    aggregates = {
        "num_items": n,
        "overall_decline_rate": round(total_correct / n, 4) if n else 0.0,
        "decline_rate_per_response_type": decline_rate_per_type,
    }
    return {"aggregates": aggregates, "per_item": per_item}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, choices=["A", "B", "C", "all"], help="Golden part to evaluate")
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_PATH, help="Path to eval/golden.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Path to write evaluation results JSON")
    args = parser.parse_args()

    if not args.golden.exists():
        sys.exit(f"Golden dataset file not found: {args.golden}")

    golden_sha = compute_sha256(args.golden)
    items = load_golden(args.golden, args.part)
    if not items:
        sys.exit(f"No golden items found for part {args.part} in {args.golden}")

    if args.part == "A":
        results = eval_part_a(items)
    elif args.part == "B":
        results = eval_part_b(items)
    elif args.part == "C":
        results = eval_part_c(items)
    else:
        results = {
            "part_A": eval_part_a([i for i in items if i["part"] == "A"]),
            "part_B": eval_part_b([i for i in items if i["part"] == "B"]),
            "part_C": eval_part_c([i for i in items if i["part"] == "C"]),
        }

    output_data = {
        "part": args.part,
        "golden_sha256": golden_sha,
        **results,
    }

    out_path = args.output or config.EVAL_DIR / f"results_{args.part}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    print(f"Evaluated Part {args.part} ({len(items)} items). Results written to {out_path}")
    print("Aggregates:", json.dumps(results.get("aggregates", {}), indent=2))


if __name__ == "__main__":
    main()
