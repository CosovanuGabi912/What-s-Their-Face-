"""The image_id convention (issue #3).

The UI saves each uploaded photo once as IMAGE_UPLOAD_DIR/<image_id> and passes only the
`image_id` (a plain file name such as "3f2a9c...e1.jpg") through the agent and MCP, never bytes,
base64 or paths: a small SLM copies a short ID into a tool call reliably, a long base64 string not.
Golden items reference eval photos the same way ({"image_id": "A_001.jpg"}) with
WTF_IMAGE_UPLOAD_DIR=eval/images.
"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

import config

try:  # HEIC/HEIF (iPhone photos) needs the optional pillow-heif plugin
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # pragma: no cover - HEIC uploads are then rejected with a clear error
    pass

JPEG_QUALITY = 90

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
# A plain file name: no folders, no "..", one of the extensions above (any case).
# Spelled without (?i:...) so it is also valid in JSON Schema (ECMA-262): it is advertised to
# MCP clients / LM Studio, which may reject Python-only regex syntax.
IMAGE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.([Jj][Pp][Ee]?[Gg]|[Pp][Nn][Gg]|[Ww][Ee][Bb][Pp])$"
_IMAGE_ID_RE = re.compile(IMAGE_ID_PATTERN)


def is_valid_image_id(image_id: str) -> bool:
    return bool(image_id) and ".." not in image_id and bool(_IMAGE_ID_RE.match(image_id))


def new_image_id(original_name: str = "upload.jpg") -> str:
    """Fresh ID for an upload, keeping the original extension: 'dog.PNG' -> '<uuid hex>.png'."""
    suffix = Path(original_name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"
    return uuid.uuid4().hex + suffix


def image_path(image_id: str, upload_dir: Path | None = None) -> Path:
    """Absolute path of an uploaded image. ValueError for a bad ID, FileNotFoundError if missing."""
    if not is_valid_image_id(image_id):
        raise ValueError(f"Invalid image_id: {image_id!r} (expected a file name like 'abc123.jpg')")
    path = (upload_dir or config.IMAGE_UPLOAD_DIR) / image_id
    if not path.is_file():
        raise FileNotFoundError(f"No uploaded image named {image_id!r}")
    return path


def save_upload(data: bytes, original_name: str = "upload.jpg", upload_dir: Path | None = None) -> str:
    """Save uploaded bytes under a new image_id and return the ID (used by the Chainlit UI)."""
    image_id = new_image_id(original_name)
    folder = upload_dir or config.IMAGE_UPLOAD_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / image_id).write_bytes(data)
    return image_id


def save_as_jpeg(data: bytes, upload_dir: Path | None = None) -> str:
    """Re-encode any uploaded photo (JPEG, PNG, WebP, HEIC) to an RGB JPEG and return its new image_id.

    Tools only ever see ".jpg". The EXIF rotation of phone photos is applied before it is dropped, and
    transparency is flattened onto white. ValueError if the bytes are not an image Pillow can read.
    """
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                rgba = image.convert("RGBA")
                image = Image.new("RGB", rgba.size, "white")
                image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                image = image.convert("RGB")
            image_id = uuid.uuid4().hex + ".jpg"
            folder = upload_dir or config.IMAGE_UPLOAD_DIR
            folder.mkdir(parents=True, exist_ok=True)
            image.save(folder / image_id, "JPEG", quality=JPEG_QUALITY)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Not a readable image (JPEG, PNG, WebP or HEIC expected): {exc}") from exc
    return image_id
