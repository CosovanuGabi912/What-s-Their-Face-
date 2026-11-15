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

## Shared contracts (issue #3)

- `config.py`: every path (absolute, from the repo root), model IDs, confidence thresholds, agent limits.
- `shared/schemas.py`: inputs and outputs of the 6 MCP tools, `GoldenItem`, `ToolCallLog`. The server validates every tool result against them.
- `data/breeds_map.json`: one canonical `display` name per breed. Use `shared.breeds.to_display()` to turn any spelling (Stanford label, AKC, PetFinder, ...) into it.
- `shared/images.py`: photos are saved once as `data/uploads/<image_id>`; only the `image_id` travels through the agent and MCP.
- `mcp_server/server.py`: the `pet-advisor` MCP server. Tools without weights or data on your laptop can be served from `mcp_server/tools/stubs.py` with `WTF_STUB_TOOLS=identify_breed,describe_pet` (or `all`).
