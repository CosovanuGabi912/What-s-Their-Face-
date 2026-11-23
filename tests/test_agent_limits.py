"""Limits enforced in code by the agent loop (issues #7 and #16): tool calls per turn, searches per
question, invalid tool calls. The SLM is mocked (tests/agent_fakes.py)."""

from __future__ import annotations

import pytest

import config
from agent.loop import APOLOGY_REPLY
from tests.agent_fakes import FakeLLM, make_agent, reply, tool_call, tool_messages


# --- Limits (enforced in code) -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_call_limit_per_turn():
    # an SLM that never stops asking for a tool
    llm = FakeLLM(default=lambda: reply(calls=[tool_call("get_breed_traits", {"breed": "Beagle"})]))
    agent, session = make_agent(llm)

    answer = await agent.run_turn("Tell me about beagles")

    assert isinstance(answer, str) and answer
    assert len(session.calls) == config.MAX_TOOLS_PER_TURN
    # once the limit is reached the SLM is asked to answer without tools
    choices = [r["tool_choice"] for r in llm.requests]
    assert choices[:config.MAX_TOOLS_PER_TURN] == ["auto"] * config.MAX_TOOLS_PER_TURN
    assert set(choices[config.MAX_TOOLS_PER_TURN:]) == {"none"}


@pytest.mark.asyncio
async def test_tool_call_limit_within_one_message():
    calls = [tool_call("get_breed_traits", {"breed": "Beagle"}, call_id=f"c{i}") for i in range(7)]
    llm = FakeLLM(reply(calls=calls), reply("Beagles are friendly."))
    agent, session = make_agent(llm)

    assert await agent.run_turn("Beagle traits?") == "Beagles are friendly."
    assert len(session.calls) == config.MAX_TOOLS_PER_TURN
    # every tool call still gets an answer, or the next request would be malformed
    results = tool_messages(llm)
    assert [m["tool_call_id"] for m in results] == [f"c{i}" for i in range(7)]
    assert all("Limit reached" in m["content"] for m in results[config.MAX_TOOLS_PER_TURN:])


@pytest.mark.asyncio
async def test_search_limit_per_question():
    llm = FakeLLM(
        *[reply(calls=[tool_call("search_knowledge", {"query": f"try {i}"}, call_id=f"c{i}")]) for i in range(3)],
        reply("I don't know."),
    )
    agent, session = make_agent(llm)

    await agent.run_turn("Something obscure about huskies?")

    assert [name for name, _ in session.calls] == ["search_knowledge"] * config.MAX_SEARCHES_PER_QUESTION
    assert "searches per question" in tool_messages(llm)[-1]["content"]


@pytest.mark.asyncio
async def test_limits_reset_every_turn():
    script = [reply(calls=[tool_call("search_knowledge", {"query": "q"})])] * 2 + [reply("Answer.")]
    llm = FakeLLM(*script, *script)
    agent, session = make_agent(llm)
    await agent.run_turn("Question 1")
    await agent.run_turn("Question 2")
    assert len(session.calls) == 4


# --- Invalid tool calls ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalid_json_is_sent_back_once_and_the_slm_can_recover():
    llm = FakeLLM(
        reply(calls=[tool_call("identify_breed", '{"image_id": "test.jpg"')]),   # missing }
        reply(calls=[tool_call("identify_breed", {"image_id": "test.jpg"})]),
        reply("It's a German Shepherd Dog."),
    )
    agent, session = make_agent(llm)

    assert await agent.run_turn("Breed?", image_id="test.jpg") == "It's a German Shepherd Dog."
    assert len(session.calls) == 1   # the broken call never reached MCP
    assert "not a valid JSON object" in tool_messages(llm)[0]["content"]


@pytest.mark.asyncio
async def test_second_invalid_tool_call_ends_the_turn_with_an_apology(caplog):
    llm = FakeLLM(
        reply(calls=[tool_call("identify_breed", "{oops")]),
        reply(calls=[tool_call("suggest_names", {"species": "hamster", "colour": None, "description": "x"})]),  # schema error from MCP
    )
    agent, _ = make_agent(llm)

    assert await agent.run_turn("Breed?", image_id="test.jpg") == APOLOGY_REPLY
    assert len(llm.requests) == 2
    assert "invalid tool calls" in caplog.text   # logged


@pytest.mark.asyncio
async def test_unknown_tool_counts_as_invalid():
    llm = FakeLLM(reply(calls=[tool_call("delete_everything", {})]), reply("Done."))
    agent, session = make_agent(llm)
    assert await agent.run_turn("Hi") == "Done."
    assert session.calls == []
    assert "unknown tool" in tool_messages(llm)[0]["content"]
