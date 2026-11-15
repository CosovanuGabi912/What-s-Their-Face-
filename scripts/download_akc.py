"""Download the AKC breed table (277 breeds) to data/akc_breeds.csv.

Source: https://github.com/tmfilho/akcdata (MIT licence), pinned to one commit and checked with
SHA-256. The file is small and already committed; this script documents where it came from and
restores it.

Usage:
    python scripts/download_akc.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AKC_TABLE_PATH  # noqa: E402

COMMIT = "84a69877c2e1337f0052e80e3b79522c3fd66dde"
URL = f"https://raw.githubusercontent.com/tmfilho/akcdata/{COMMIT}/data/akc-data-latest.csv"
SHA256 = "bece823fe30d9fd5207757385ffc4d84fb172cc70cd372b2e4828497b0241154"


def main() -> None:
    if AKC_TABLE_PATH.exists() and hashlib.sha256(AKC_TABLE_PATH.read_bytes()).hexdigest() == SHA256:
        print(f"Already present: {AKC_TABLE_PATH}")
        return
    with urllib.request.urlopen(URL) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        sys.exit("SHA-256 mismatch; nothing written.")
    AKC_TABLE_PATH.write_bytes(data)
    print(f"Wrote {AKC_TABLE_PATH}")


if __name__ == "__main__":
    main()
