"""Download the trained breed classifier (issue #6) from the GitHub Release.

The checkpoint (about 112 MB) is too large for git, so it is published as an asset of the
GitHub Release RELEASE_TAG on this repository. The file is checked with SHA-256 and only then
moved into place, so a failed or wrong download never replaces a working checkpoint. An older
copy (for example one from before the #3 class-name rename) has a different hash and is replaced.

Usage:
    python scripts/download_classifier.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

REPOSITORY = "CosovanuGabi912/What-s-Their-Face-"
RELEASE_TAG = "breed-classifier-v1"
FILE_NAME = "convnext_tiny_stanford120.pt"
SHA256 = "f41aada229c43e85baba7583efdd8badf4a6669f5eef013b236f11b6e61f6688"
URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{FILE_NAME}"
DEST = config.BREED_CLASSIFIER_DIR / FILE_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


def download(dest: Path = DEST) -> bool:
    """Return True when dest holds the published checkpoint."""
    if dest.exists() and sha256(dest) == SHA256:
        print(f"Already downloaded: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    print(f"Downloading {URL}")
    try:
        fetch(URL, partial)
    except OSError as error:
        partial.unlink(missing_ok=True)
        print(f"Download failed: {error}")
        return False
    if sha256(partial) != SHA256:
        partial.unlink()
        print("SHA-256 mismatch: the downloaded file was deleted and nothing was replaced.")
        return False
    partial.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


if __name__ == "__main__":
    sys.exit(0 if download() else 1)
