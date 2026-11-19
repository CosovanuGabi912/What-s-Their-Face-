# What's Their Face

A local pet assistant built around small language models (SLMs). From a photo of a dog it identifies the breed, answers questions about it with cited sources, recommends breeds that fit the user's lifestyle, and suggests names based on similar real pets.

## Documentation

- [Milestone 2 – design and plan](docs/milestone2.md): problem and data, golden dataset, RAG design, agent and MCP tools, model choices, week-by-week plan.
- [RFC](docs/What's_their_face_RFC.docx): the original project proposal.

## Setup

Python 3.11 (see `.python-version`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: py -3.11 -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional: LM Studio URL, model IDs, WTF_STUB_TOOLS
python -m pytest                # contract + MCP stdio smoke tests
python scripts/validate_breeds_map.py
mcp dev mcp_server/server.py    # MCP Inspector (needs Node.js)
```

## How to run: get the data (issue #5)

The knowledge and listings databases (ChromaDB) are not in git. This repository is **public**, and the TheDogAPI
data and the PetFinder listings may not be redistributed, so neither they nor the databases built from them can be
on GitHub. Part A shares them privately in the team. The Wikipedia articles (CC BY-SA, see
`data/raw/wikipedia/README.md`) and `data/ingestion_manifest.json` (checksums, chunk counts and the embedding
model, no data) are in git.

**Everyone must use the same embedding model file: nomic-ai `nomic-embed-text-v1.5`, quantisation `Q4_K_M`**
(LM Studio id `text-embedding-nomic-embed-text-v1.5`). Another quantisation or model gives slightly different
vectors, and search then quietly gets worse: no error, just worse answers. `validate_ingestion.py` checks it.

### Fast path (recommended)

1. Get `data_<commit>.zip`: the download link is shared privately in the team chat, so ask Part A (Gabi) for it. The link
   is never put in the repository. Unzip the file into `data/`.
   It contains `chroma_knowledge/`, `chroma_listings/` and `raw/thedogapi/breeds.json`.
   On Windows, check that `data/chroma_knowledge/` sits directly inside `data/`. "Extract All" can add an extra folder.
2. In LM Studio load `nomic-embed-text-v1.5` **Q4_K_M** and start the server (default `http://localhost:1234`).
3. `cp .env.example .env` (Windows: `copy .env.example .env`).
4. `python scripts/validate_ingestion.py`. It must end with `OK`; any `MISMATCH` line says what differs from the manifest.

### Rebuild path (only if you cannot get the zip)

1. Get `breeds.json` the same way (ask Part A, Gabi, in the team chat) and save it as `data/raw/thedogapi/breeds.json`.
2. Accept the rules of the [PetFinder competition](https://www.kaggle.com/c/petfinder-adoption-prediction) on Kaggle and
   save a Kaggle token (`~/.kaggle/access_token`), then `python scripts/download_petfinder.py`.
3. Load the Q4_K_M model in LM Studio and start the server, `cp .env.example .env`.
4. `python scripts/ingest.py --all` (stops with an error if a raw file is missing or its checksum differs from the
   manifest; `--skip-thedogapi` ingests Wikipedia only, and the result will not match the manifest), then
   `python scripts/validate_ingestion.py`. Running `ingest.py --all` twice is safe: the second run adds nothing.

Part A only: after changing raw data on purpose, `python scripts/ingest.py --all --update-manifest` rewrites the manifest;
commit it and share a new `data_<commit>.zip`.

## Shared contracts (issue #3)

- `config.py`: every path (absolute, from the repo root), model IDs, confidence thresholds, agent limits.
- `shared/schemas.py`: inputs and outputs of the 6 MCP tools, `GoldenItem`, `ToolCallLog`. The server validates every tool result against them.
- `data/breeds_map.json`: one canonical `display` name per breed. Use `shared.breeds.to_display()` to turn any spelling (Stanford label, AKC, PetFinder, ...) into it.
- `shared/images.py`: photos are saved once as `data/uploads/<image_id>`; only the `image_id` travels through the agent and MCP.
- `mcp_server/server.py`: the `pet-advisor` MCP server. Tools without weights or data on your laptop can be served from `mcp_server/tools/stubs.py` with `WTF_STUB_TOOLS=identify_breed,describe_pet` (or `all`).
- `shared/mcp_client.py`: start the server with `server_params()`. The MCP stdio client does not pass the parent's environment to the server, so without it `WTF_*`, `LM_STUDIO_*`, model and ML runtime variables (`HF_*`, `TORCH_*`, `PYTORCH_*`, `CUDA_VISIBLE_DEVICES`) set in code or in the terminal never reach the tools (values in `.env` still work). `tests/test_server_hygiene.py` enforces the stdout rule below and JSON-Schema-safe regex patterns.
- Tool code must never print to stdout: over stdio, stdout carries the MCP protocol. Use `logging` (stderr).

## Agent SLM (issue #7)

- In LM Studio load **Qwen3-4B** (id `qwen3-4b`, or set `AGENT_MODEL` in `.env`) with **Context Length 8192**
  (load settings, before loading the model). The default of 4096 is too small: the system prompt and the tool
  schemas alone are about 2.2k tokens, and one turn with two searches adds about 2k more, plus Qwen3's thinking.
- `agent/loop.py` keeps only the user messages and the final replies between turns. Tool calls and results stay
  in the turn that made them; what later turns need (breed, questionnaire, rejected names) is in the conversation state.
