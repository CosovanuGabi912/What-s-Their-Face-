"""Download the PetFinder.my Adoption Prediction tables from Kaggle to data/raw/petfinder/ (issue #5).

Only the small CSV files are fetched (train.csv, breed/color/state labels), never the ~2 GB of photos.
Needs a Kaggle token (~/.kaggle/access_token or KAGGLE_API_TOKEN) and the competition rules accepted on
kaggle.com/c/petfinder-adoption-prediction. The token is read by the Kaggle CLI and never printed here.

Output (flat folder):
    data/raw/petfinder/train.csv  breed_labels.csv  color_labels.csv  state_labels.csv

Usage:
    python scripts/download_petfinder.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402

COMPETITION = "petfinder-adoption-prediction"
RAW_DIR = DATA_DIR / "raw" / "petfinder"
FILES = {  # file in the competition -> file name saved here
    "train/train.csv": "train.csv",
    "breed_labels.csv": "breed_labels.csv",
    "color_labels.csv": "color_labels.csv",
    "state_labels.csv": "state_labels.csv",
}


def kaggle_command() -> list[str]:
    # The kaggle executable sits next to the interpreter in a venv; fall back to `python -m kaggle`.
    exe = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    return [str(exe)] if exe.exists() else (["kaggle"] if shutil.which("kaggle") else [sys.executable, "-m", "kaggle"])


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        target = RAW_DIR / local
        if target.exists() and target.stat().st_size > 0:
            print(f"Already present: {target.relative_to(ROOT)}")
            continue
        command = kaggle_command() + ["competitions", "download", "-c", COMPETITION, "-f", remote,
                                      "-p", str(RAW_DIR), "--force"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"Kaggle download of {remote} failed (exit {result.returncode}). Check that the token is set "
                     f"and that the competition rules are accepted.\n{result.stdout[-400:]}{result.stderr[-400:]}")
        # The CLI keeps the competition's sub-folder (train/train.csv.zip) and may or may not zip the file.
        downloaded = RAW_DIR / remote
        archive = downloaded.with_name(downloaded.name + ".zip")
        if archive.exists():
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(archive.parent)
            archive.unlink()
        if not downloaded.exists():
            sys.exit(f"Kaggle did not produce {downloaded}")
        if downloaded != target:
            downloaded.replace(target)
            if downloaded.parent != RAW_DIR and not any(downloaded.parent.iterdir()):
                downloaded.parent.rmdir()
        print(f"Downloaded {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
