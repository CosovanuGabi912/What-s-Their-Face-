"""Evaluation harness for Parts A, B, C, E, and F of the Golden Dataset (issues #8, #16, #17).

Usage:
    python eval/run_eval.py --part A --golden eval/golden.jsonl --output eval/results_A.json
    python eval/run_eval.py --part B --golden eval/golden.jsonl --output eval/results_B.json
    python eval/run_eval.py --part C --golden eval/golden.jsonl --output eval/results_C.json
    python eval/run_eval.py --part E --golden eval/golden.jsonl --output eval/results_E.json
    python eval/run_eval.py --part F --golden eval/golden.jsonl --output eval/results_F.json
"""

from __future__ import annotations

import argparse
import asyncio
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
from mcp_server.tools.names import suggest_names
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


async def agent_routes(messages: List[str], model: str) -> List[Dict[str, Any]]:
    """Run each message through the real agent, each in a fresh conversation (one MCP server for all)."""
    from agent.loop import AgentLoop

    routes = []
    async with AgentLoop(model_id=model, session_id="eval_part_c") as server:
        for message in messages:
            agent = AgentLoop(model_id=model, session=server.session, llm_client=server.llm,
                              session_id="eval_part_c")
            agent.tools, agent._tool_names = server.tools, server._tool_names
            reply = await agent.run_turn(message)
            routes.append({"predicted_response_type": agent.last_reply_kind, "reply": reply,
                           "tools": [log.tool for log in agent.trace]})
    return routes


def eval_part_c(items: List[dict], mode: str = "agent", model: str = config.AGENT_MODEL_ID) -> Dict[str, Any]:
    per_item = []
    type_totals: Dict[str, int] = {}
    type_correct: Dict[str, int] = {}

    messages = [item["input"]["message"] for item in items]
    if mode == "agent":
        routes = asyncio.run(agent_routes(messages, model))
    else:
        routes = [{"predicted_response_type": classify_message_route(m)} for m in messages]

    for item, route in zip(items, routes):
        item_id = item["id"]
        msg = item["input"]["message"]
        expected_type = item["expected"]["response_type"]
        predicted_type = route["predicted_response_type"]
        is_pass = (predicted_type == expected_type)

        type_totals[expected_type] = type_totals.get(expected_type, 0) + 1
        if is_pass:
            type_correct[expected_type] = type_correct.get(expected_type, 0) + 1

        per_item.append({
            "id": item_id,
            "message": msg,
            "expected_response_type": expected_type,
            "predicted_response_type": predicted_type,
            **{key: value for key, value in route.items() if key != "predicted_response_type"},
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
        "mode": mode,
        **({"model": model, "stub_tools": sorted(config.STUB_TOOLS)} if mode == "agent" else {}),
        "overall_decline_rate": round(total_correct / n, 4) if n else 0.0,
        "decline_rate_per_response_type": decline_rate_per_type,
    }
    return {"aggregates": aggregates, "per_item": per_item}


def eval_part_e(items: List[dict]) -> Dict[str, Any]:
    per_item = []
    top1_hit_count = 0
    top5_hit_count = 0

    for item in items:
        item_id = item["id"]
        description = item["input"]["description"]
        expected_real_name = item["expected"]["real_name"]

        out = suggest_names(species="dog", colour=None, description=description)
        suggested_names = [n.name for n in out.names]
        top5_names = suggested_names[:5]

        top1_hit = (bool(suggested_names) and suggested_names[0].casefold() == expected_real_name.casefold())
        top5_hit = any(n.casefold() == expected_real_name.casefold() for n in top5_names)

        if top1_hit:
            top1_hit_count += 1
        if top5_hit:
            top5_hit_count += 1

        per_item.append({
            "id": item_id,
            "description": description,
            "expected_real_name": expected_real_name,
            "suggested_names_top5": top5_names,
            "top1_hit": top1_hit,
            "top5_hit": top5_hit,
            "pass": top5_hit,
        })

    n = len(items)
    real_name_in_top_5 = round(top5_hit_count / n, 4) if n else 0.0
    top1_hit_rate = round(top1_hit_count / n, 4) if n else 0.0

    aggregates = {
        "num_items": n,
        "real_name_in_top_5": real_name_in_top_5,
        "top5_hit_rate": real_name_in_top_5,
        "top1_hit_rate": top1_hit_rate,
    }
    return {"aggregates": aggregates, "per_item": per_item}


def is_subsequence(sub: list[str], full: list[str]) -> bool:
    """Returns True if sub elements appear in full in the same order."""
    it = iter(full)
    return all(x in it for x in sub)


def simulate_journey_turns(turns: list[dict]) -> tuple[list[str], str]:
    """Simulate tool calls and outcome for a scripted MVP journey."""
    tools = []
    for turn in turns:
        msg = turn.get("content", "").lower()
        image_id = turn.get("image_id")

        if image_id or ("breed" in msg and any(k in msg for k in ["what breed", "identify", "which breed is this"])):
            tools.append("identify_breed")
            tools.append("search_knowledge")
        elif any(k in msg for k in ["recommend", "suit", "best for me", "which breeds"]):
            tools.append("recommend_breeds")
        elif any(k in msg for k in ["unusual", "unique", "rarer"]):
            tools.append("suggest_names")
        elif any(k in msg for k in ["name", "names", "ideas", "call"]):
            tools.append("suggest_names")
        elif any(k in msg for k in ["exercise", "grooming", "temperament", "active", "children", "strangers", "trainable", "shed"]):
            tools.append("search_knowledge")

    outcome = "names_suggested" if "suggest_names" in tools else "unknown"
    return tools, outcome


async def run_agent_journeys(items: List[dict], model: str) -> List[tuple[list[str], str]]:
    from agent.loop import AgentLoop

    results = []
    async with AgentLoop(model_id=model, session_id="eval_part_f") as server:
        for item in items:
            turns = item["input"]["scripted_turns"]
            agent = AgentLoop(model_id=model, session=server.session, llm_client=server.llm,
                              session_id=f"eval_part_f_{item['id']}")
            agent.tools, agent._tool_names = server.tools, server._tool_names

            for turn in turns:
                content = turn.get("content", "")
                image_id = turn.get("image_id")
                await agent.run_turn(content, image_id=image_id)

            executed_tools = [log.tool for log in agent.trace]
            outcome = "names_suggested" if "suggest_names" in executed_tools else "unknown"
            results.append((executed_tools, outcome))
    return results


def eval_part_f(items: List[dict], mode: str = "agent", model: str = config.AGENT_MODEL_ID) -> Dict[str, Any]:
    per_item = []
    completed_count = 0
    exact_match_count = 0

    actual_mode = mode
    if mode == "agent":
        try:
            import urllib.request
            urllib.request.urlopen(f"{config.LM_STUDIO_BASE_URL}/models", timeout=1.0)
            journey_results = asyncio.run(run_agent_journeys(items, model))
        except Exception:
            actual_mode = "simulated"
            journey_results = [simulate_journey_turns(it["input"]["scripted_turns"]) for it in items]
    else:
        actual_mode = "simulated"
        journey_results = [simulate_journey_turns(it["input"]["scripted_turns"]) for it in items]

    for item, (actual_tools, actual_outcome) in zip(items, journey_results):
        item_id = item["id"]
        expected_tools = item["expected"]["tool_sequence"]
        expected_outcome = item["expected"]["final_outcome"]

        exact_match = (actual_tools == expected_tools)
        is_completed = is_subsequence(expected_tools, actual_tools) and (actual_outcome == expected_outcome)

        if exact_match:
            exact_match_count += 1
        if is_completed:
            completed_count += 1

        per_item.append({
            "id": item_id,
            "expected_tool_sequence": expected_tools,
            "actual_tool_sequence": actual_tools,
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "tool_sequence_match": exact_match,
            "journey_completed": is_completed,
            "pass": is_completed,
        })

    n = len(items)
    journey_completion_rate = round(completed_count / n, 4) if n else 0.0
    tool_sequence_match_rate = round(exact_match_count / n, 4) if n else 0.0

    aggregates = {
        "num_items": n,
        "mode": actual_mode,
        "journey_completion_rate": journey_completion_rate,
        "tool_sequence_match_rate": tool_sequence_match_rate,
    }
    return {"aggregates": aggregates, "per_item": per_item}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, choices=["A", "B", "C", "E", "F", "all"], help="Golden part to evaluate")
    parser.add_argument("--golden", type=Path, default=config.GOLDEN_PATH, help="Path to eval/golden.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Path to write evaluation results JSON")
    parser.add_argument("--mode", choices=["agent", "router", "simulated"], default="agent",
                        help="Part C / F: run the real agent (LM Studio) or the router/simulated baseline")
    parser.add_argument("--model", default=config.AGENT_MODEL_ID, help="Part C / F agent model id in LM Studio")
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
        results = eval_part_c(items, args.mode, args.model)
    elif args.part == "E":
        results = eval_part_e(items)
    elif args.part == "F":
        results = eval_part_f(items, args.mode, args.model)
    else:
        results = {
            "part_A": eval_part_a([i for i in items if i["part"] == "A"]),
            "part_B": eval_part_b([i for i in items if i["part"] == "B"]),
            "part_C": eval_part_c([i for i in items if i["part"] == "C"], args.mode, args.model),
            "part_E": eval_part_e([i for i in items if i["part"] == "E"]),
            "part_F": eval_part_f([i for i in items if i["part"] == "F"], args.mode, args.model),
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
