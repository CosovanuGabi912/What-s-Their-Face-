"""How the agent, the Chainlit app and eval/run_eval.py start the MCP server (issue #3 follow-up).

The MCP stdio client does NOT give the server process the parent's environment: only a few system
variables (PATH, APPDATA, TEMP, ... see mcp.client.stdio.get_default_environment) plus what is passed
in StdioServerParameters(env=...). A variable set in code or in the terminal, for example
run_eval.py setting WTF_IMAGE_UPLOAD_DIR=eval/images, would never reach the tools.

server_params() forwards every project variable explicitly. Values in .env are read by config.py
inside the server process either way.

    from mcp.client.stdio import stdio_client
    from shared.mcp_client import server_params

    async with stdio_client(server_params()) as (read, write): ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp import StdioServerParameters

import config

SERVER_SCRIPT = config.ROOT / "mcp_server" / "server.py"
FORWARDED_PREFIXES = ("WTF_", "LM_STUDIO_")
FORWARDED_NAMES = ("AGENT_MODEL", "EMBEDDING_MODEL", "VISION_MODEL")


def project_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The project's variables from the current environment, plus `overrides` (which win)."""
    env = {name: value for name, value in os.environ.items()
           if name.startswith(FORWARDED_PREFIXES) or name in FORWARDED_NAMES}
    env.update(overrides or {})
    return env


def server_params(overrides: dict[str, str] | None = None, cwd: Path | None = None) -> StdioServerParameters:
    """Parameters that start mcp_server/server.py with this Python and the project's variables."""
    return StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)],
                                 env=project_env(overrides), cwd=str(cwd) if cwd else None)
