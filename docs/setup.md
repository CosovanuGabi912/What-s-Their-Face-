# Setup: vision model for `describe_pet` (issue #15)

`describe_pet` sends the photo to **Qwen2.5-VL-3B** in LM Studio and reads back the species, main colour, size and one short sentence. Without the model the tool still works: it returns a neutral answer (`main_colour: null`) and the agent asks the user for the colour.

## 1. Download the model

In LM Studio, search for `lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF` and download the **Q4_K_M** version. LM Studio also needs the vision file from the same repository, `mmproj-model-f16.gguf`.

- Download size: 1.9 GB for the model plus 1.3 GB for the vision file, about 3.3 GB in total.

## 2. Load it under the name the project uses

The code calls the model `qwen2.5-vl-3b`, so it must be loaded with that identifier. First find the model key:

```bash
lms ls
```

Then load it, replacing `<model-key>` with the key from the list:

```bash
lms load <model-key> --identifier qwen2.5-vl-3b
```

To use another identifier, set `VISION_MODEL` in `.env`.

## 3. Start the server and check it

Start the LM Studio server (default `http://localhost:1234`), then check that the identifier is listed:

```bash
curl http://localhost:1234/v1/models
```

If the identifier is wrong or the server is off, `describe_pet` does not crash: it logs a warning to stderr and returns the neutral answer. So if every photo comes back as "Unable to describe pet from photo.", check this step first.

## Tests

The tests use a mocked LM Studio, so they need neither the model nor the server:

```bash
python -m pytest tests/test_describe_pet.py
```
