"""Test doubles for the agent loop (shared by test_agent_loop.py and test_agent_limits.py).

FakeLLM plays the SLM: it returns scripted chat-completion messages and records every request.
FakeSession plays the MCP server: it answers from mcp_server/tools/stubs.py, so results match the
real contract without a subprocess.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

from mcp.types import CallToolResult, TextContent, Tool

from agent.loop import AgentLoop
from mcp_server.tools import stubs, traits
from shared.schemas import TOOL_SCHEMAS


# --- Fakes -------------------------------------------------------------------------------------
def tool_call(name: str, args: dict | str, call_id: str = "c1") -> NS:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return NS(id=call_id, type="function", function=NS(name=name, arguments=arguments))


def reply(content: str | None = None, calls: list[NS] | None = None) -> NS:
    """Shaped like openai's ChatCompletion: response.choices[0].message.{content, tool_calls}."""
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


class FakeLLM:
    """Returns the scripted replies in order, then `default()` forever."""

    def __init__(self, *script: NS, default=None):
        self.script = list(script)
        self.default = default or (lambda: reply("Default answer."))
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0) if self.script else self.default()


class FakeSession:
    """MCP ClientSession stand-in backed by the stub tools (get_breed_traits has no stub:
    like the server, it falls back to the real one, which only reads the AKC CSV in git)."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return NS(tools=[Tool(name=name, description=f"{name} tool. Example: {name}()",
                              inputSchema=model.model_json_schema())
                         for name, (model, _) in TOOL_SCHEMAS.items()])

    async def call_tool(self, name: str, args: dict) -> CallToolResult:
        self.calls.append((name, args))
        try:
            tool = getattr(stubs, name, None) or getattr(traits, name)
            data = tool(**args).model_dump(mode="json")
        except Exception as exc:  # what FastMCP does with a failing tool or bad arguments
            return CallToolResult(content=[TextContent(type="text", text=f"Error executing tool {name}: {exc}")],
                                  isError=True)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))], structuredContent=data)


def make_agent(llm: FakeLLM) -> tuple[AgentLoop, FakeSession]:
    session = FakeSession()
    return AgentLoop(llm_client=llm, session=session), session


def tool_messages(llm: FakeLLM) -> list[dict]:
    """The tool results the SLM saw in its last request."""
    return [m for m in llm.requests[-1]["messages"] if m["role"] == "tool"]
