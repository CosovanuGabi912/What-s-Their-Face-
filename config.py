"""Central configuration (issue #3).

Every path is absolute and resolved from the repository root, so scripts, tests, the MCP
server and the Chainlit app behave the same from any working directory. Values that differ per
laptop (LM Studio URL, model IDs, a few folders) can be overridden in `.env` (see `.env.example`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:  # python-dotenv is in requirements.txt; config still imports without it
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def _path(env_var: str, default: Path) -> Path:
    """Path from an env var (relative values are taken from the repo root) or the default."""
    value = os.getenv(env_var)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


# --- Paths ------------------------------------------------------------------------------
DATA_DIR             = ROOT / "data"
CHROMA_KNOWLEDGE_DIR = _path("WTF_CHROMA_KNOWLEDGE_DIR", DATA_DIR / "chroma_knowledge")
CHROMA_LISTINGS_DIR  = _path("WTF_CHROMA_LISTINGS_DIR", DATA_DIR / "chroma_listings")
# Overridable so eval/run_eval.py can point tools at eval/images/ and pass image_id="A_001.jpg"
IMAGE_UPLOAD_DIR     = _path("WTF_IMAGE_UPLOAD_DIR", DATA_DIR / "uploads")
AKC_TABLE_PATH       = DATA_DIR / "akc_breeds.csv"
BREEDS_MAP_PATH      = DATA_DIR / "breeds_map.json"
MODELS_DIR           = ROOT / "models"
BREED_CLASSIFIER_DIR = MODELS_DIR / "breed_classifier"   # weights + calibration from #6 / #10
EVAL_DIR             = ROOT / "eval"
GOLDEN_PATH          = EVAL_DIR / "golden.jsonl"
EVAL_IMAGES_DIR      = EVAL_DIR / "images"
LOG_DIR              = _path("WTF_LOG_DIR", ROOT / "logs")

# --- LM Studio / models -----------------------------------------------------------------
LM_STUDIO_BASE_URL   = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY    = os.getenv("LM_STUDIO_API_KEY", "lm-studio")   # LM Studio ignores it; the client needs one
AGENT_MODEL_ID       = os.getenv("AGENT_MODEL", "qwen3-4b")
EMBED_MODEL_ID       = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
VISION_MODEL_ID      = os.getenv("VISION_MODEL", "qwen2.5-vl-3b")
EMBED_DOC_PREFIX     = "search_document: "
EMBED_QUERY_PREFIX   = "search_query: "

# --- Fixed vocabularies shared by tools, ingestion and golden items -------------------------
KNOWLEDGE_SECTIONS   = ["Overview", "Temperament", "Exercise", "Grooming", "Health",
                        "Training", "Feeding", "Living Environment", "History", "Other"]  # Part A may revise before #5 merges
PET_COLOURS          = ["black", "brown", "golden", "yellow", "cream", "gray", "white"]  # = PetFinder color_labels.csv


# --- Breed-ID confidence (agent decision 2) ---------------------------------------------
# Tuned on the validation split by scripts/calibrate_classifier.py (#10), which writes
# models/breed_classifier/threshold.json. These fallbacks apply until that file exists.
#   top-1 >= CONFIDENCE_HIGH                  -> "confident": name the breed
#   CONFIDENCE_LOW <= top-1 < CONFIDENCE_HIGH -> "top2": show the two most likely breeds
#   top-1 < CONFIDENCE_LOW                    -> "retry": ask for a clearer photo
def _threshold(key: str, default: float) -> float:
    path = BREED_CLASSIFIER_DIR / "threshold.json"
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))[key])
    except (OSError, ValueError, KeyError, TypeError):
        return default


CONFIDENCE_HIGH = _threshold("confidence_threshold", 0.82)  # 0.82 = #10 result (95% precision on val)
CONFIDENCE_LOW  = _threshold("top2_threshold", 0.50)        # provisional until #10 tunes it

# --- Stub tools ------------------------------------------------------------------------------
# Tools listed here are served from mcp_server/tools/stubs.py instead of the real implementation,
# e.g. WTF_STUB_TOOLS=identify_breed,describe_pet on a laptop without the classifier weights,
# or WTF_STUB_TOOLS=all for UI work. Empty (default) = real implementations everywhere.
STUB_TOOLS = {name.strip() for name in os.getenv("WTF_STUB_TOOLS", "").split(",") if name.strip()}

# --- Agent limits (enforced in code, not in the prompt) -----------------------------------
MAX_TOOLS_PER_TURN        = 5
MAX_SEARCHES_PER_QUESTION = 2
SEARCH_TOP_K              = 3    # search_knowledge returns at most 3 chunks
NAME_NEIGHBOURS_K         = 20   # suggest_names looks at the 20 most similar listings
