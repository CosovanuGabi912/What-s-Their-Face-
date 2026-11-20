# Its presence makes pytest add the repository root to sys.path,
# so tests can import project packages (mcp_server, shared, ...) with a plain `pytest`.

import pytest

import config


@pytest.fixture(autouse=True)
def _tool_call_logs_in_tmp(tmp_path, monkeypatch):
    """The agent loop appends tool calls to config.LOG_DIR; keep test runs out of the repo's logs/."""
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
