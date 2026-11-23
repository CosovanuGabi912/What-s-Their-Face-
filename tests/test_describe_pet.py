"""describe_pet (issue #15), with a mocked LM Studio: no vision model is needed."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

pytest.importorskip("torch")
from PIL import Image  # noqa: E402

import config  # noqa: E402
from mcp_server import server  # noqa: E402
from mcp_server.tools import breed_id  # noqa: E402
from shared.schemas import DescribePetOutput  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOOD_REPLY = json.dumps({"species": "dog", "main_colour": "white", "size": "large",
                         "description": "A fluffy white and gray dog with upright ears."})
EXIF_ORIENTATION = 0x0112


class FakeVision:
    """Stands in for the OpenAI client and remembers the request."""

    def __init__(self, reply=GOOD_REPLY, error: Exception | None = None):
        self.reply, self.error, self.request = reply, error, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.request = request
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])

    def sent_photo(self) -> Image.Image:
        url = self.request["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


@pytest.fixture(autouse=True)
def uploads_in_fixtures(monkeypatch):
    monkeypatch.setattr(breed_id, "IMAGE_UPLOAD_DIR", FIXTURES)


def use(monkeypatch, **kwargs) -> FakeVision:
    fake = FakeVision(**kwargs)
    monkeypatch.setattr(breed_id, "_vision_client", lambda: fake)
    return fake


def test_valid_reply_returns_all_four_fields(monkeypatch):
    fake = use(monkeypatch)
    result = breed_id.describe_pet("test_husky.jpg")
    assert isinstance(result, DescribePetOutput)
    assert result.model_dump() == json.loads(GOOD_REPLY)
    assert fake.request["model"] == config.VISION_MODEL_ID
    assert fake.request["temperature"] == 0
    content = fake.request["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[1]["text"] == breed_id.VISION_PROMPT


def test_prompt_lists_every_allowed_colour():
    for colour in config.PET_COLOURS:
        assert f'"{colour}"' in breed_id.VISION_PROMPT


@pytest.mark.parametrize("reply", [
    f"```json\n{GOOD_REPLY}\n```",
    f"Here is the JSON:\n{GOOD_REPLY}\nHope this helps!",
    f"<think>Looks like a husky {{maybe}}.</think>{GOOD_REPLY}",
])
def test_json_is_found_inside_extra_text(monkeypatch, reply):
    use(monkeypatch, reply=reply)
    assert breed_id.describe_pet("test_husky.jpg").model_dump() == json.loads(GOOD_REPLY)


@pytest.mark.parametrize("reply", [
    "", None, "A nice dog.", "{not json}", "[1, 2, 3]",
    '{"species": "hamster", "main_colour": "white", "size": "small", "description": "A hamster."}',
    '{"species": "dog", "main_colour": "white", "size": "huge", "description": "A huge dog."}',
    '{"species": "dog", "main_colour": "white", "size": "large", "description": ""}',
    '{"species": "dog", "main_colour": "white", "size": "large"}',
])
def test_unusable_reply_returns_the_fallback(monkeypatch, reply):
    use(monkeypatch, reply=reply)
    assert breed_id.describe_pet("test_husky.jpg") == breed_id.FALLBACK_DESCRIPTION


@pytest.mark.parametrize("error", [
    openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:1234/v1/chat/completions")),
    openai.APITimeoutError(request=httpx.Request("POST", "http://localhost:1234/v1/chat/completions")),
    OSError("connection reset"),
])
def test_lm_studio_failure_returns_the_fallback(monkeypatch, error):
    use(monkeypatch, error=error)
    assert breed_id.describe_pet("test_husky.jpg") == breed_id.FALLBACK_DESCRIPTION


def test_empty_choices_return_the_fallback(monkeypatch):
    fake = use(monkeypatch)
    fake.chat.completions.create = lambda **_: SimpleNamespace(choices=[])
    assert breed_id.describe_pet("test_husky.jpg") == breed_id.FALLBACK_DESCRIPTION


def test_unreadable_photo_returns_the_fallback(monkeypatch, tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"this is not an image")
    monkeypatch.setattr(breed_id, "IMAGE_UPLOAD_DIR", tmp_path)
    fake = use(monkeypatch)
    assert breed_id.describe_pet("broken.jpg") == breed_id.FALLBACK_DESCRIPTION
    assert fake.request is None


@pytest.mark.parametrize("said, stored", [
    ("Grey", "gray"), ("gold", "golden"), (" WHITE ", "white"),
    ("purple", None), ("brown and white", None), (None, None), (7, None),
])
def test_colour_is_normalised_or_none(said, stored):
    reply = json.dumps({"species": "dog", "main_colour": said, "size": "small", "description": "A small dog."})
    assert breed_id.parse_description(reply).main_colour == stored


def test_long_description_is_cut_to_one_short_sentence():
    long_sentence = " ".join(f"word{i}" for i in range(40))
    reply = json.dumps({"species": "cat", "main_colour": "black", "size": "small", "description": long_sentence})
    assert len(breed_id.parse_description(reply).description.split()) == 20

    two = json.dumps({"species": "cat", "main_colour": "black", "size": "small",
                      "description": "A small black cat. It sits on a sofa."})
    assert breed_id.parse_description(two).description == "A small black cat."

    saint = json.dumps({"species": "dog", "main_colour": "brown", "size": "large",
                        "description": "A large St. Bernard resting on grass."})
    assert breed_id.parse_description(saint).description == "A large St. Bernard resting on grass."


def test_photo_is_resized_and_sent_upright(monkeypatch, tmp_path):
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6       # stored sideways; viewers turn it 90 degrees clockwise
    Image.new("RGB", (2000, 1000), "red").save(tmp_path / "phone.jpg", exif=exif)
    monkeypatch.setattr(breed_id, "IMAGE_UPLOAD_DIR", tmp_path)
    fake = use(monkeypatch)
    breed_id.describe_pet("phone.jpg")
    assert fake.sent_photo().size == (breed_id.VISION_MAX_SIDE // 2, breed_id.VISION_MAX_SIDE)


@pytest.mark.parametrize("image_id", ["../secret.jpg", "sub/dir.jpg", ""])
def test_bad_image_id_raises_before_calling_the_model(monkeypatch, image_id):
    fake = use(monkeypatch)
    with pytest.raises(ValueError):
        breed_id.describe_pet(image_id)
    assert fake.request is None


def test_missing_photo_raises(monkeypatch):
    fake = use(monkeypatch)
    with pytest.raises(FileNotFoundError):
        breed_id.describe_pet("does_not_exist.jpg")
    assert fake.request is None


def test_server_uses_the_real_tool_and_passes_the_contract(monkeypatch):
    monkeypatch.setattr(config, "STUB_TOOLS", set())
    use(monkeypatch)
    result = server.describe_pet("test_husky.jpg")
    assert isinstance(result, DescribePetOutput) and result.main_colour == "white"
