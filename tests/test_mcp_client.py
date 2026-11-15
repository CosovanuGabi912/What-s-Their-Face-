"""shared/mcp_client.py: project variables must reach the MCP server process (issue #3 follow-up)."""

from __future__ import annotations

import sys

from shared.mcp_client import SERVER_SCRIPT, project_env, server_params


def test_project_variables_are_forwarded(monkeypatch):
    monkeypatch.setenv("WTF_IMAGE_UPLOAD_DIR", "eval/images")  # what eval/run_eval.py does
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://localhost:4321/v1")
    monkeypatch.setenv("AGENT_MODEL", "phi-4-mini")
    monkeypatch.setenv("SOME_SECRET_TOKEN", "do-not-forward")
    env = project_env()
    assert env["WTF_IMAGE_UPLOAD_DIR"] == "eval/images"
    assert env["LM_STUDIO_BASE_URL"] == "http://localhost:4321/v1"
    assert env["AGENT_MODEL"] == "phi-4-mini"
    assert "SOME_SECRET_TOKEN" not in env


def test_overrides_win(monkeypatch):
    monkeypatch.setenv("WTF_STUB_TOOLS", "")
    assert project_env({"WTF_STUB_TOOLS": "all"})["WTF_STUB_TOOLS"] == "all"


def test_server_params_start_the_project_server(tmp_path):
    params = server_params({"WTF_STUB_TOOLS": "all"}, cwd=tmp_path)
    assert params.command == sys.executable
    assert params.args == [str(SERVER_SCRIPT)]
    assert params.env["WTF_STUB_TOOLS"] == "all"
    assert params.cwd == str(tmp_path)
