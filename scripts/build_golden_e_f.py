"""Generate Golden Part E (50 name items) and Part F (10 journey items) for issue #17.

Outputs:
    eval/holdout_listing_ids.txt (50 held-out PetFinder listing IDs)
    eval/images/E_001.jpg ... E_050.jpg
    eval/images/F_001.jpg ... F_010.jpg
    Appends E and F items to eval/golden.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from shared.schemas import GoldenItem, LISTING_ID_PATTERN


def build_part_e() -> tuple[list[dict], list[str]]:
    con = sqlite3.connect(config.CHROMA_LISTINGS_DIR / "chroma.sqlite3")
    cur = con.cursor()

    query = """
    SELECT m1.string_value AS listing_id, m2.string_value AS name, m3.string_value AS breed,
           m4.string_value AS colour, m5.string_value AS size, m6.string_value AS doc
    FROM embedding_metadata m1
    JOIN embedding_metadata m2 ON m1.id = m2.id AND m2.key = 'name'
    JOIN embedding_metadata m3 ON m1.id = m3.id AND m3.key = 'breed'
    JOIN embedding_metadata m4 ON m1.id = m4.id AND m4.key = 'colour'
    JOIN embedding_metadata m5 ON m1.id = m5.id AND m5.key = 'size'
    JOIN embedding_metadata m6 ON m1.id = m6.id AND m6.key = 'chroma:document'
    JOIN embedding_metadata m7 ON m1.id = m7.id AND m7.key = 'species'
    WHERE m1.key = 'listing_id' AND m7.string_value = 'dog'
    ORDER BY m1.id ASC
    """
    rows = cur.execute(query).fetchall()

    bad_names = {
        "dog", "puppy", "puppies", "unknown", "no name", "baby", "boy", "girl", "cutie", "female", "male",
        "doggy", "doggie", "sister", "brother", "mother", "father", "mama", "papa"
    }

    # Load val files and part A files to avoid image overlap
    val_path = config.DATA_DIR / "splits" / "stanford_dogs_val.json"
    val_files = set()
    if val_path.exists():
        data = json.loads(val_path.read_text(encoding="utf-8"))
        val_files = {Path(f).name for f in data.get("val_files", [])}

    part_a_files = set()
    if config.GOLDEN_PATH.exists():
        with config.GOLDEN_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("part") == "A":
                    for src in item.get("source_ids", []):
                        part_a_files.add(Path(src).name)

    # Collect available photos from Stanford Dogs test split
    test_dir = config.DATA_DIR / "raw" / "stanford_dogs" / "test"
    all_photos = sorted([
        p for p in test_dir.rglob("*.jpg")
        if p.name not in val_files and p.name not in part_a_files
    ], key=lambda p: p.name)

    selected_listings = []
    seen_names = set()
    seen_ids = set()

    for lid, name, breed, colour, size, doc in rows:
        name_clean = name.strip().capitalize()
        if not (2 <= len(name_clean) <= 12):
            continue
        if not re.match(r"^[A-Za-z]+$", name_clean):
            continue
        if name_clean.lower() in bad_names:
            continue
        if name_clean in seen_names or lid in seen_ids:
            continue
        if not re.match(LISTING_ID_PATTERN, lid):
            continue

        # Extract free-text description part from doc
        # doc is like: Species: dog. Breed: X. Colour: Y. Size: Z. Age: N. <free text>
        parts = doc.split("Age: ", 1)
        if len(parts) == 2:
            sub = parts[1]
            # split at next period
            after_age = sub.split(".", 1)
            raw_desc = after_age[1].strip() if len(after_age) > 1 else sub.strip()
        else:
            raw_desc = doc

        # Remove name mentions from description
        desc_anonymized = re.sub(rf"\b{re.escape(name_clean)}\b", "this dog", raw_desc, flags=re.IGNORECASE)
        # Clean up repeated spaces
        desc_anonymized = re.sub(r"\s+", " ", desc_anonymized).strip()

        # Build concise user description with physical traits and personality
        # Make sure real name is strictly absent
        if re.search(rf"\b{re.escape(name_clean)}\b", desc_anonymized, flags=re.IGNORECASE):
            continue

        # Limit description to 25 words
        words = desc_anonymized.split()[:20]
        if len(words) < 5:
            continue
        short_desc = f"{size} {colour} dog, " + " ".join(words)
        short_desc = re.sub(rf"\b{re.escape(name_clean)}\b", "pup", short_desc, flags=re.IGNORECASE)

        seen_names.add(name_clean)
        seen_ids.add(lid)
        selected_listings.append((lid, name_clean, short_desc))
        if len(selected_listings) == 50:
            break

    assert len(selected_listings) == 50, f"Expected 50 listings, got {len(selected_listings)}"

    part_e_items = []
    holdout_ids = []

    for idx, (lid, name, desc) in enumerate(selected_listings, start=1):
        item_id = f"E_{idx:03d}"
        image_name = f"{item_id}.jpg"

        # Copy photo to eval/images/
        src_photo = all_photos[idx - 1]
        dst_photo = config.EVAL_IMAGES_DIR / image_name
        shutil.copyfile(src_photo, dst_photo)

        item = {
            "id": item_id,
            "part": "E",
            "input": {
                "image_id": image_name,
                "description": desc,
            },
            "expected": {
                "real_name": name,
            },
            "source_ids": [lid],
        }
        part_e_items.append(item)
        holdout_ids.append(lid)

    return part_e_items, holdout_ids


def build_part_f(photo_offset: int = 50) -> list[dict]:
    # 10 complete journey scripts following MVP flow
    # At least 2 include the rejection loop ("something more unusual")
    journeys_spec = [
        {
            "id": "F_001",
            "photo_breed_dir": "n02106662-German_shepherd",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_001.jpg"},
                {"role": "user", "content": "I live in a flat, work 6 hours away, and have no kids. Which breeds suit me?"},
                {"role": "user", "content": "I'll take the Cavalier King Charles Spaniel. Name ideas?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_002",
            "photo_breed_dir": "n02110185-Siberian_husky",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_002.jpg"},
                {"role": "user", "content": "I have an apartment with a terrace and mild allergies. Which breeds do you recommend?"},
                {"role": "user", "content": "I'll take the Bichon Frise. Name ideas for a white fluffy dog?"},
                {"role": "user", "content": "Something more unusual, please! Any rarer name ideas?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_003",
            "photo_breed_dir": "n02099601-golden_retriever",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_003.jpg"},
                {"role": "user", "content": "I live in a house with a garden and have 2 kids. Which breeds suit our family?"},
                {"role": "user", "content": "I'll take the Labrador. Name ideas for a yellow lab?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_004",
            "photo_breed_dir": "n02088364-beagle",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_004.jpg"},
                {"role": "user", "content": "I live in a small flat and walk 45 minutes daily. Which breeds suit me best?"},
                {"role": "user", "content": "I'll take the French Bulldog. Name ideas for a small fawn pup?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_005",
            "photo_breed_dir": "n02110958-pug",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_005.jpg"},
                {"role": "user", "content": "I have an apartment and work 8 hours a day. Which breeds do you recommend?"},
                {"role": "user", "content": "I'll take the Cavalier. Name ideas for a tricolor puppy?"},
                {"role": "user", "content": "Something more unusual, please! Any unique suggestions?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_006",
            "photo_breed_dir": "n02085620-Chihuahua",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_006.jpg"},
                {"role": "user", "content": "I live in a studio flat, first-time dog owner. Which breeds suit me?"},
                {"role": "user", "content": "I'll take the Miniature Poodle. Name ideas for a smart little pup?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_007",
            "photo_breed_dir": "n02106550-Rottweiler",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_007.jpg"},
                {"role": "user", "content": "We have a large fenced yard and active outdoor lifestyle. Which breeds do you recommend?"},
                {"role": "user", "content": "I'll take the German Shepherd. Name ideas for a brave puppy?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_008",
            "photo_breed_dir": "n02113023-Pembroke",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_008.jpg"},
                {"role": "user", "content": "I have an apartment and mild dog allergies. Which breeds suit me?"},
                {"role": "user", "content": "I'll take the Bichon Frise. Name ideas for a cheerful dog?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_009",
            "photo_breed_dir": "n02108089-boxer",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_009.jpg"},
                {"role": "user", "content": "House with a yard, family with children. Which breeds do you recommend?"},
                {"role": "user", "content": "I'll take the Golden Retriever. Name ideas for a friendly pup?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
        {
            "id": "F_010",
            "photo_breed_dir": "n02107142-Doberman",
            "turns": [
                {"role": "user", "content": "What breed is this dog?", "image_id": "F_010.jpg"},
                {"role": "user", "content": "Suburban home with experience in dog training. Which breeds suit me?"},
                {"role": "user", "content": "I'll take the German Shepherd. Name ideas for an alert loyal dog?"},
            ],
            "expected": {
                "tool_sequence": ["identify_breed", "search_knowledge", "recommend_breeds", "suggest_names"],
                "final_outcome": "names_suggested",
            },
        },
    ]

    test_dir = config.DATA_DIR / "raw" / "stanford_dogs" / "test"
    part_f_items = []

    for spec in journeys_spec:
        item_id = spec["id"]
        image_name = f"{item_id}.jpg"

        # Find matching photo from the breed directory
        b_dir = test_dir / spec["photo_breed_dir"]
        photos = sorted(list(b_dir.glob("*.jpg")))
        src_photo = photos[0] if photos else (test_dir / "n02085620-Chihuahua").glob("*.jpg")[0]
        dst_photo = config.EVAL_IMAGES_DIR / image_name
        shutil.copyfile(src_photo, dst_photo)

        item = {
            "id": item_id,
            "part": "F",
            "input": {
                "scripted_turns": spec["turns"],
            },
            "expected": spec["expected"],
            "source_ids": [],
        }
        part_f_items.append(item)

    return part_f_items


def main():
    print("Building Golden Part E (50 items)...")
    part_e_items, holdout_ids = build_part_e()

    # Write eval/holdout_listing_ids.txt
    holdout_file = config.EVAL_DIR / "holdout_listing_ids.txt"
    with holdout_file.open("w", encoding="utf-8") as f:
        for hid in holdout_ids:
            f.write(f"{hid}\n")
    print(f"Wrote {len(holdout_ids)} holdout IDs to {holdout_file}")

    print("Building Golden Part F (10 journey items)...")
    part_f_items = build_part_f()

    # Validate with GoldenItem schema
    for it in part_e_items + part_f_items:
        GoldenItem.model_validate(it)
    print("All Part E & F items validated against GoldenItem schema.")

    # Read existing golden items (A, B, C)
    existing_items = []
    if config.GOLDEN_PATH.exists():
        with config.GOLDEN_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Keep existing parts that are not E or F
                    if item.get("part") not in ("E", "F"):
                        existing_items.append(item)

    all_items = existing_items + part_e_items + part_f_items
    print(f"Total golden items: {len(all_items)} (existing: {len(existing_items)}, E: {len(part_e_items)}, F: {len(part_f_items)})")

    with config.GOLDEN_PATH.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Updated {config.GOLDEN_PATH} successfully.")


if __name__ == "__main__":
    main()
