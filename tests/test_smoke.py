"""Smoke test (issue #3): an async MCP client spawns mcp_server/server.py over stdio, exactly as
the agent loop and the Chainlit app will, lists the tools and resource, and calls every tool.

The server is started from an unrelated working directory to prove paths come from config.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from shared.schemas import BREEDS_LIST_URI, TOOL_SCHEMAS
from tests.test_contracts import SAMPLE_CALLS

SERVER = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"


def _params(cwd: Path) -> StdioServerParameters:
    # Stubs: this tests the protocol and the contract, not the model weights or Chroma
    return StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(cwd),
                                 env={"WTF_STUB_TOOLS": "all"})


@pytest.mark.asyncio
async def test_stdio_handshake_lists_and_calls_every_tool(tmp_path):
    async with stdio_client(_params(tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.serverInfo.name == "pet-advisor"

            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert set(tools) == set(TOOL_SCHEMAS)
            for tool in tools.values():
                assert tool.description and "Example:" in tool.description
                assert tool.inputSchema["type"] == "object"
                assert tool.outputSchema is not None

            for name, args in SAMPLE_CALLS.items():
                result = await session.call_tool(name, args)
                assert not result.isError, result.content
                TOOL_SCHEMAS[name][1].model_validate(result.structuredContent)
                # the text content carries the same JSON, for clients that ignore structuredContent
                assert json.loads(result.content[0].text) == result.structuredContent


@pytest.mark.asyncio
async def test_stdio_resource_and_errors(tmp_path):
    async with stdio_client(_params(tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            resources = [str(r.uri) for r in (await session.list_resources()).resources]
            assert BREEDS_LIST_URI in resources
            content = await session.read_resource(BREEDS_LIST_URI)
            names = json.loads(content.contents[0].text)
            assert {"Dingo", "German Shepherd Dog"} <= set(names)

            dingo = await session.call_tool("get_breed_traits", {"breed": "Dingo"})
            assert not dingo.isError and dingo.structuredContent["status"] == "no_akc_data"

            # Bad arguments come back as a tool error the agent can show the SLM, not a crash
            bad = await session.call_tool("search_knowledge", {"query": "x", "section": "Diet"})
            assert bad.isError and "Unknown section" in bad.content[0].text
            bad = await session.call_tool("identify_breed", {"image_id": "../etc/passwd"})
            assert bad.isError


def test_chainlit_is_installed():
    """Part E's UI will run the client above inside Chainlit's asyncio loop."""
    assert importlib.util.find_spec("chainlit") is not None
