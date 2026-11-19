"""Guards for the MCP server (issue #3 follow-up).

- Over stdio, stdout carries the MCP protocol: tool code must never print (use logging -> stderr).
- Tool schemas are sent to LM Studio / MCP clients as JSON Schema, whose regexes are ECMA-262:
  Python-only syntax such as inline flags (?i:...) can make a client reject the tool.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from shared.images import IMAGE_ID_PATTERN, is_valid_image_id
from shared.schemas import TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
SERVER_CODE_DIRS = ["mcp_server", "shared"]
STDOUT_WRITE = re.compile(r"(?<![\w.])print\s*\(|sys\.stdout\.write")
# Python-only regex syntax: inline flags, named groups, lookbehind, possessive/atomic groups, \A \Z
NON_ECMA_REGEX = re.compile(r"\(\?[aiLmsux-]|\(\?P|\(\?<[=!]|\(\?>|\\A|\\Z")


def _code_outside_main(path: Path) -> str:
    """Source without the `if __name__ == "__main__":` block (printing there is fine)."""
    return re.split(r'^if __name__ == ["\']__main__["\']:', path.read_text(encoding="utf-8"), flags=re.M)[0]


@pytest.mark.parametrize("path", sorted(p for d in SERVER_CODE_DIRS for p in (ROOT / d).rglob("*.py")),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_server_code_never_writes_to_stdout(path):
    hits = [line.strip() for line in _code_outside_main(path).splitlines()
            if STDOUT_WRITE.search(line) and not line.lstrip().startswith("#")]
    assert not hits, f"{path.relative_to(ROOT)} writes to stdout (breaks MCP stdio); use logging: {hits}"


def _patterns(schema) -> list[str]:
    if isinstance(schema, dict):
        found = [schema["pattern"]] if isinstance(schema.get("pattern"), str) else []
        return found + [p for value in schema.values() for p in _patterns(value)]
    if isinstance(schema, list):
        return [p for value in schema for p in _patterns(value)]
    return []


def _advertised_schemas() -> list[dict]:
    from mcp_server.server import mcp
    tools = asyncio.run(mcp.list_tools())
    return [t.inputSchema for t in tools] + [t.outputSchema for t in tools if t.outputSchema]


def _contract_schemas() -> list[dict]:
    return [model.model_json_schema() for pair in TOOL_SCHEMAS.values() for model in pair]


@pytest.mark.parametrize("source", [_advertised_schemas, _contract_schemas], ids=["mcp_tools", "TOOL_SCHEMAS"])
def test_schema_patterns_are_valid_json_schema_regex(source):
    patterns = [p for schema in source() for p in _patterns(schema)]
    bad = [p for p in patterns if NON_ECMA_REGEX.search(p)]
    assert not bad, f"Python-only regex syntax in JSON Schema patterns: {bad}"


def test_image_id_pattern_is_advertised_to_mcp_clients():
    from mcp_server.server import mcp
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("identify_breed", "describe_pet"):
        assert tools[name].inputSchema["properties"]["image_id"]["pattern"] == IMAGE_ID_PATTERN


@pytest.mark.parametrize("image_id, ok", [
    ("abc123.jpg", True), ("A_001.JPG", True), ("x.JpEg", True), ("d.png", True), ("d.WEBP", True),
    ("d.gif", False), ("../a.jpg", False), ("a/b.jpg", False), (".a.jpg", False), ("a.jpgx", False),
    ("a.jpeeg", False), ("", False), ("a..b.jpg", False),
])
def test_image_id_rules(image_id, ok):
    assert is_valid_image_id(image_id) is ok
