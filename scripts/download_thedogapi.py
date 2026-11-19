"""Download TheDogAPI breed data (free tier) to data/raw/thedogapi/breeds.json (issue #5).

Needs THEDOGAPI_KEY in .env (or the environment); the key is sent as a header and never printed.
TheDogAPI has no entry for the wild canids (Dingo, Dhole, African Wild Dog): that is expected.

After downloading, run `python scripts/build_breeds_map.py` to record which TheDogAPI name belongs to
which display name in breeds_map.json (it prints the names it could not match).

Usage:
    python scripts/download_thedogapi.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402  (also loads .env)

OUT_PATH = DATA_DIR / "raw" / "thedogapi" / "breeds.json"
URL = "https://api.thedogapi.com/v1/breeds"
USER_AGENT = "WhatsTheirFace-course-project/1.0 (student project; breed knowledge base)"


def main() -> None:
    key = os.getenv("THEDOGAPI_KEY")
    if not key:
        sys.exit("THEDOGAPI_KEY is not set. Put it in .env (see .env.example); it is never printed or committed.")
    breeds: list[dict] = []
    page = 0
    while True:  # the API paginates; a short page means the end
        response = requests.get(URL, headers={"x-api-key": key, "User-Agent": USER_AGENT},
                                params={"limit": 100, "page": page}, timeout=30)
        if response.status_code in (401, 403):
            sys.exit(f"TheDogAPI rejected the key (HTTP {response.status_code}).")
        response.raise_for_status()
        batch = response.json()
        breeds.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(breeds, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"TheDogAPI: {len(breeds)} breeds saved to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
