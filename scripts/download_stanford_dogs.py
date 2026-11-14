"""Download Stanford Dogs (120 breeds) with the official train/test split.

The official server (vision.stanford.edu) was unreachable in November 2026, so we use a
Hugging Face mirror that keeps the official split (12,000 train / 8,580 test, 100 train
images per breed). The files are pinned to one revision and checked with SHA-256.

Output layout, read by scripts/train_classifier.py:
    data/raw/stanford_dogs/train/n02085620-Chihuahua/n02085620_10074.jpg
    data/raw/stanford_dogs/test/n02085620-Chihuahua/n02085620_10131.jpg

Usage:
    python scripts/download_stanford_dogs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "stanford_dogs"

MIRROR = "maurice-fp/stanford-dogs"
REVISION = "ee102b9a35f00829e65bdad7cda8adcfa185b400"
SPLITS = {
    # split: (file in the mirror, sha256, expected number of images)
    "train": (
        "data/train-00000-of-00001.parquet",
        "07a5729ae8cf8c21c1f18d8f93b05fc0c7d7e010300082ad028b1ea8b42ed3c1",
        12000,
    ),
    "test": (
        "data/test-00000-of-00001.parquet",
        "a3f55298e59d6e87012e4183390373cd39266494f960ac24f9165344343298ee",
        8580,
    ),
}
NUM_CLASSES = 120
TRAIN_IMAGES_PER_CLASS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(remote: str, expected_sha: str, dest: Path) -> None:
    if dest.exists() and sha256(dest) == expected_sha:
        print(f"Already downloaded: {dest.name}")
        return
    url = f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/{remote}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    if sha256(dest) != expected_sha:
        dest.unlink()
        sys.exit(f"SHA-256 mismatch for {dest.name}; the file was deleted.")


def label_names(parquet: pq.ParquetFile) -> list[str]:
    """Class names stored by Hugging Face in the parquet metadata, e.g. 'n02085620-Chihuahua'."""
    metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
    return metadata["info"]["features"]["label"]["names"]


def extract(split: str, parquet_path: Path, out_dir: Path, expected: int) -> None:
    split_dir = out_dir / split
    if split_dir.exists() and sum(1 for _ in split_dir.rglob("*.jpg")) == expected:
        print(f"Already extracted: {split} ({expected} images)")
        return

    parquet = pq.ParquetFile(parquet_path)
    names = label_names(parquet)
    if len(names) != NUM_CLASSES:
        sys.exit(f"Expected {NUM_CLASSES} classes, found {len(names)}")

    counts: Counter[str] = Counter()
    for batch in parquet.iter_batches(batch_size=256, columns=["image", "label"]):
        for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            class_name = names[label]
            data = image["bytes"]
            # One original file (n02105855_2933.jpg) is a PNG with a .jpg name, as in ImageNet.
            if not data.startswith((b"\xff\xd8", b"\x89PNG")):
                sys.exit(f"Image in class {class_name} is neither JPEG nor PNG")
            file_name = Path(image["path"]).name if image.get("path") else f"{split}_{sum(counts.values()):05d}.jpg"
            target = split_dir / class_name / file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            counts[class_name] += 1

    total = sum(counts.values())
    if total != expected or len(counts) != NUM_CLASSES:
        sys.exit(f"{split}: expected {expected} images in {NUM_CLASSES} classes, got {total} in {len(counts)}")
    if split == "train" and set(counts.values()) != {TRAIN_IMAGES_PER_CLASS}:
        sys.exit("train: the official split has exactly 100 images per breed")
    print(f"Extracted {split}: {total} images, {len(counts)} breeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for split, (remote, expected_sha, expected) in SPLITS.items():
        parquet_path = args.out_dir / "_parquet" / Path(remote).name
        download(remote, expected_sha, parquet_path)
        extract(split, parquet_path, args.out_dir, expected)
    print(f"Stanford Dogs ready in {args.out_dir}")


if __name__ == "__main__":
    main()
