"""Agent loop (issue #7) with a mocked LM Studio and a mocked MCP session.

FakeLLM plays the SLM: it returns scripted chat-completion messages and records every request.
FakeSession plays the MCP server: it answers from mcp_server/tools/stubs.py, so results match the
real contract without a subprocess. One test at the end uses the real server over stdio.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from mcp.types import CallToolResult, TextContent, Tool

import config
from agent import loop as loop_module
from agent.guards import VET_REPLY, vet_check
from agent.loop import APOLOGY_REPLY, DONT_KNOW_REPLY, AgentLoop, strip_think, to_openai_tool
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


# --- Turn cycle --------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_turn_cycle_returns_answer_and_updates_state():
    llm = FakeLLM(
        reply(calls=[tool_call("identify_breed", {"image_id": "test.jpg"})]),
        reply(calls=[tool_call("search_knowledge", {"query": "what is it like", "section": "Overview"})]),
        reply("<think>The tools say German Shepherd.</think>A German Shepherd Dog [kb:german-shepherd-dog:overview:0]."),
    )
    agent, session = make_agent(llm)

    answer = await agent.run_turn("What breed is this?", image_id="test.jpg")

    assert answer == "A German Shepherd Dog [kb:german-shepherd-dog:overview:0]."
    assert [name for name, _ in session.calls] == ["identify_breed", "search_knowledge"]
    # code adds the breed filter from the state (agent decision 3)
    assert session.calls[1][1]["breed"] == "German Shepherd Dog"
    assert agent.state.current_breed == "German Shepherd Dog"
    assert agent.state.breed_confidence == pytest.approx(0.91)
    # the image id reaches the SLM in the user message, and the state reaches it in the system prompt
    assert "[image_id: test.jpg]" in llm.requests[0]["messages"][-1]["content"]
    assert "current_breed: German Shepherd Dog" in llm.requests[-1]["messages"][0]["content"]
    assert [t.tool for t in agent.trace] == ["identify_breed", "search_knowledge"]


@pytest.mark.asyncio
async def test_tools_come_from_the_mcp_session_in_openai_format():
    llm = FakeLLM(reply("Hello!"))
    agent, _ = make_agent(llm)
    await agent.run_turn("Hi")
    tools = llm.requests[0]["tools"]
    assert {t["function"]["name"] for t in tools} == set(TOOL_SCHEMAS)
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in tools)
    assert llm.requests[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_history_is_kept_across_turns():
    llm = FakeLLM(reply("First answer."), reply("Second answer."))
    agent, _ = make_agent(llm)
    await agent.run_turn("First question")
    await agent.run_turn("Second question")
    contents = [m["content"] for m in llm.requests[-1]["messages"][1:]]
    assert contents == ["First question", "First answer.", "Second question"]



@pytest.mark.asyncio
async def test_tool_results_are_dropped_from_history_when_a_turn_ends():
    llm = FakeLLM(
        reply(calls=[tool_call("search_knowledge", {"query": "exercise", "breed": "Beagle"})]),
        reply("Beagles need an hour a day [kb:beagle:exercise:0]."),
        reply("Second answer."),
    )
    agent, _ = make_agent(llm)
    await agent.run_turn("How much exercise does a beagle need?")
    # inside the turn the SLM sees the tool call and its result
    assert [m["role"] for m in llm.requests[1]["messages"][1:]] == ["user", "assistant", "tool"]

    await agent.run_turn("And grooming?")
    # the next turn only gets the user messages and the final replies
    history = llm.requests[-1]["messages"][1:]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[1]["content"] == "Beagles need an hour a day [kb:beagle:exercise:0]."
    assert agent.messages[-1] == {"role": "assistant", "content": "Second answer."}

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
        reply(calls=[tool_call("identify_breed", {"image_id": "../secret.txt"})]),   # schema error from MCP
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


# --- Guards ------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_vet_check_runs_before_the_slm():
    llm = FakeLLM()
    agent, session = make_agent(llm)
    assert await agent.run_turn("Help, my dog ate chocolate!") == VET_REPLY
    assert llm.requests == [] and session.calls == []



@pytest.mark.parametrize("message", [
    "My dog is bleeding from the paw",
    "Our puppy had a seizure this morning",
    "my dog swallowed a sock",
    "He ate chocolate, what do I do?",          # patterns fire without "my"/"our"
])
def test_vet_check_fires_on_own_pet_emergencies(message):
    assert vet_check(message)


@pytest.mark.parametrize("message", [
    "What bleeding disorder is genetically inherited in Dobermans?",   # golden Part B
    "Are Boxers prone to seizures?",
    "Which breeds are prone to choking on food?",
])
def test_vet_check_ignores_breed_questions_with_vet_words(message):
    assert not vet_check(message)

@pytest.mark.asyncio
async def test_failed_citation_check_retries_once(monkeypatch):
    results = iter([False, True])
    monkeypatch.setattr(loop_module, "citation_check", lambda answer, ids: next(results))
    llm = FakeLLM(reply("Bad citation [kb:beagle:health:9]."), reply("Fixed answer."))
    agent, _ = make_agent(llm)

    assert await agent.run_turn("Are beagles healthy?") == "Fixed answer."
    assert llm.requests[-1]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_failed_citation_check_twice_says_i_dont_know(monkeypatch):
    monkeypatch.setattr(loop_module, "citation_check", lambda answer, ids: False)
    llm = FakeLLM(reply("Bad [kb:beagle:health:9]."), reply("Still bad [kb:beagle:health:9]."))
    agent, _ = make_agent(llm)
    assert await agent.run_turn("Are beagles healthy?") == DONT_KNOW_REPLY


# --- State -------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unsure_breed_is_not_used_as_a_filter():
    llm = FakeLLM(reply(calls=[tool_call("identify_breed", {"image_id": "unsure.jpg"})]), reply("Husky or Eskimo Dog?"))
    agent, _ = make_agent(llm)
    await agent.run_turn("Breed?", image_id="unsure.jpg")
    assert agent.state.current_breed is None
    assert agent.state.breed_confidence == pytest.approx(0.64)


@pytest.mark.asyncio
async def test_questionnaire_and_rejected_names_are_kept():
    answers = {"home": "flat", "hours_alone": 8, "daily_walks_min": 30, "kids_at_home": False,
               "has_allergies": True, "experience": "none"}
    llm = FakeLLM(
        reply(calls=[tool_call("recommend_breeds", answers)]),
        reply("Here are three breeds."),
        reply(calls=[tool_call("suggest_names", {"species": "dog", "colour": "brown",
                                                 "description": "A small brown dog.", "exclude_names": ["Max"]})]),
        reply("How about Bella?"),
        reply(calls=[tool_call("suggest_names", {"species": "dog", "colour": "brown",
                                                 "description": "A small brown dog.", "exclude_names": ["Bella"]})]),
        reply("How about Luna?"),
    )
    agent, session = make_agent(llm)
    await agent.run_turn("Which dog suits me?")
    await agent.run_turn("Name ideas? Not Max.")
    await agent.run_turn("Not Bella either.")

    assert agent.state.questionnaire == answers
    assert agent.state.rejected_names == ["Max", "Bella"]
    assert session.calls[-1][1]["exclude_names"] == ["Bella", "Max"]   # code adds earlier rejections


# --- Helpers -----------------------------------------------------------------------------------
def test_strip_think():
    assert strip_think("<think>\nreasoning\n</think>\n\nAnswer.") == "Answer."
    assert strip_think("reasoning without opening tag</think>Answer.") == "Answer."
    assert strip_think(None) == ""


def test_to_openai_tool():
    tool = Tool(name="get_breed_traits", description="Traits.",
                inputSchema={"type": "object", "properties": {"breed": {"type": "string"}}})
    assert to_openai_tool(tool) == {"type": "function", "function": {
        "name": "get_breed_traits", "description": "Traits.", "parameters": tool.inputSchema}}


# --- Real MCP server over stdio (stub tools), mocked SLM ---------------------------------------
@pytest.mark.asyncio
async def test_turn_against_the_real_mcp_server():
    llm = FakeLLM(reply(calls=[tool_call("identify_breed", {"image_id": "test.jpg"})]), reply("A German Shepherd Dog."))
    async with AgentLoop(server_env={"WTF_STUB_TOOLS": "all"}, llm_client=llm) as agent:
        assert await agent.run_turn("What breed is this?", image_id="test.jpg") == "A German Shepherd Dog."
        assert agent.state.current_breed == "German Shepherd Dog"
        assert {t["function"]["name"] for t in llm.requests[0]["tools"]} == set(TOOL_SCHEMAS)


