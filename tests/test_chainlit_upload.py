"""Issue #13: uploads from the Chainlit UI become plain .jpg image_ids on disk."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.app import save_upload
from shared.images import image_path, is_valid_image_id


def _bytes(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **kwargs)
    return buffer.getvalue()


def _element(**attrs):
    """Stand-in for cl.Image: save_upload only reads .content and .path."""
    return SimpleNamespace(content=attrs.get("content"), path=attrs.get("path"), name="dog")


def _assert_saved_jpeg(image_id: str, folder) -> Image.Image:
    assert image_id.endswith(".jpg") and "/" not in image_id and "\\" not in image_id
    assert is_valid_image_id(image_id)
    assert (folder / image_id).is_file()
    assert image_path(image_id, folder) == folder / image_id
    with Image.open(folder / image_id) as saved:
        assert saved.format == "JPEG" and saved.mode == "RGB"
        return saved.copy()


def test_png_content_is_saved_as_jpg(tmp_path):
    image_id = save_upload(_element(content=_bytes(Image.new("RGB", (8, 6), "red"), "PNG")), tmp_path)
    assert _assert_saved_jpeg(image_id, tmp_path).size == (8, 6)


def test_upload_from_path_is_saved_as_jpg(tmp_path):
    source = tmp_path / "in" / "photo.webp"
    source.parent.mkdir()
    source.write_bytes(_bytes(Image.new("RGB", (5, 5), "blue"), "WEBP"))
    image_id = save_upload(_element(path=str(source)), tmp_path / "out")
    _assert_saved_jpeg(image_id, tmp_path / "out")


def test_transparent_png_is_flattened_on_white(tmp_path):
    rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 0))   # fully transparent
    image_id = save_upload(_element(content=_bytes(rgba, "PNG")), tmp_path)
    pixel = _assert_saved_jpeg(image_id, tmp_path).getpixel((1, 1))
    assert all(channel > 240 for channel in pixel)


def test_exif_rotation_is_applied(tmp_path):
    photo = Image.new("RGB", (10, 4), "green")
    exif = Image.Exif()
    exif[0x0112] = 6   # "rotate 90 degrees clockwise to display"
    image_id = save_upload(_element(content=_bytes(photo, "JPEG", exif=exif)), tmp_path)
    assert _assert_saved_jpeg(image_id, tmp_path).size == (4, 10)


def test_each_upload_gets_a_new_id(tmp_path):
    data = _bytes(Image.new("RGB", (4, 4)), "PNG")
    assert save_upload(_element(content=data), tmp_path) != save_upload(_element(content=data), tmp_path)


def test_default_upload_dir_is_config(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "IMAGE_UPLOAD_DIR", tmp_path)
    image_id = save_upload(_element(content=_bytes(Image.new("RGB", (4, 4)), "PNG")))
    assert (tmp_path / image_id).is_file()


@pytest.mark.parametrize("element", [_element(content=b"not an image"), _element(content=b""), _element()])
def test_non_image_is_rejected_and_nothing_saved(tmp_path, element):
    with pytest.raises(ValueError):
        save_upload(element, tmp_path)
    assert not any(tmp_path.iterdir())
