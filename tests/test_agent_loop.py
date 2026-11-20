"""Agent loop (issue #7) with a mocked LM Studio and a mocked MCP session.

FakeLLM plays the SLM: it returns scripted chat-completion messages and records every request.
FakeSession plays the MCP server: it answers from mcp_server/tools/stubs.py, so results match the
real contract without a subprocess. One test at the end uses the real server over stdio.
"""

from __future__ import annotations

import pytest
from mcp.types import Tool

from agent import loop as loop_module
from agent.guards import OUT_OF_SCOPE_REPLY, VET_REPLY, vet_check
from agent.loop import (DONT_KNOW_REPLY, RETRY_PHOTO_REPLY, AgentLoop, reply_kind, strip_think,
                        to_openai_tool)
from agent.state import extract_answers
from shared.schemas import ToolCallLog
from shared.schemas import TOOL_SCHEMAS
from tests.agent_fakes import FakeLLM, make_agent, reply, tool_call, tool_messages


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
    llm = FakeLLM(reply(calls=[tool_call("identify_breed", {"image_id": "unsure.jpg"})]))
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




# --- Low-confidence flow (issue #16): code replies, driven by IdentifyBreedOutput.decision ----------
@pytest.mark.asyncio
@pytest.mark.parametrize("image_id,expected,kind", [
    ("unsure.jpg", "I'm not sure: it could be a Siberian Husky or a Eskimo Dog. Could you send a clearer photo?",
     "low_confidence"),
    ("blurry.jpg", RETRY_PHOTO_REPLY, "low_confidence"),
    ("dingo.jpg", "This looks like a Dingo, a wild animal", "out_of_scope"),
])
async def test_unsure_or_wild_identification_gets_a_fixed_reply(image_id, expected, kind):
    llm = FakeLLM(reply(calls=[tool_call("identify_breed", {"image_id": image_id})]),
                  default=lambda: pytest.fail("the SLM must not be called after an unsure identification"))
    agent, session = make_agent(llm)

    answer = await agent.run_turn("What breed is this?", image_id=image_id)

    assert answer.startswith(expected)
    assert agent.last_reply_kind == kind
    assert len(llm.requests) == 1 and [name for name, _ in session.calls] == ["identify_breed"]
    assert agent.state.current_breed is None


@pytest.mark.asyncio
async def test_confident_identification_lets_the_slm_answer():
    llm = FakeLLM(reply(calls=[tool_call("identify_breed", {"image_id": "dog.jpg"})]), reply("A German Shepherd Dog."))
    agent, _ = make_agent(llm)
    assert await agent.run_turn("Breed?", image_id="dog.jpg") == "A German Shepherd Dog."
    assert agent.last_reply_kind == "answer"


@pytest.mark.asyncio
async def test_no_threshold_constants_in_agent():
    import agent.loop as loop_source
    from pathlib import Path
    for path in Path(loop_source.__file__).parent.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "CONFIDENCE_HIGH" not in text and "CONFIDENCE_LOW" not in text, path


# --- Out-of-scope guard ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_out_of_scope_runs_before_the_slm():
    llm = FakeLLM()
    agent, session = make_agent(llm)
    assert await agent.run_turn("Where can I adopt a stray kitten?") == OUT_OF_SCOPE_REPLY
    assert agent.last_reply_kind == "out_of_scope"
    assert llm.requests == [] and session.calls == []


# --- Safety-claim guard ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_uncited_safety_claim_is_retried_then_declined():
    llm = FakeLLM(reply("Grapes are safe for dogs."), reply("Yes, grapes are harmless."))
    agent, _ = make_agent(llm)
    assert await agent.run_turn("Are grapes OK as a treat for a Beagle?") == DONT_KNOW_REPLY
    retry_prompt = llm.requests[-1]["messages"][-1]["content"]
    assert "safe without citing" in retry_prompt and llm.requests[-1]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_bad_citation_retry_names_the_ids():
    llm = FakeLLM(reply("Fact [kb:beagle:health:9]."), reply("I don't know."))
    agent, _ = make_agent(llm)
    assert await agent.run_turn("Are beagles healthy?") == "I don't know."
    assert "['kb:beagle:health:9']" in llm.requests[-1]["messages"][-1]["content"]
    assert agent.last_reply_kind == "i_dont_know"


@pytest.mark.asyncio
async def test_cited_safety_claim_passes():
    llm = FakeLLM(
        reply(calls=[tool_call("search_knowledge", {"query": "is it safe", "breed": "Beagle", "section": "Exercise"})]),
        reply("Short walks are safe for Beagles [kb:beagle:exercise:0]."),
    )
    agent, _ = make_agent(llm)
    assert await agent.run_turn("Are walks safe for a beagle?") == "Short walks are safe for Beagles [kb:beagle:exercise:0]."


# --- Questionnaire state (agent decision 5) --------------------------------------------------------
@pytest.mark.asyncio
async def test_answers_from_the_users_words_reach_the_prompt_and_the_tool():
    llm = FakeLLM(
        reply("How many hours would the dog be alone?"),
        reply(calls=[tool_call("recommend_breeds", {"hours_alone": 8, "daily_walks_min": 30, "kids_at_home": False,
                                                    "has_allergies": False, "experience": "none"})]),
        reply("Here are three breeds."),
    )
    agent, session = make_agent(llm)
    await agent.run_turn("I live in a flat, which dog would suit me?")
    system = llm.requests[0]["messages"][0]["content"]
    assert "questionnaire: home=flat" in system
    assert "questionnaire still to ask: hours_alone, daily_walks_min, kids_at_home, has_allergies, experience" in system

    await agent.run_turn("8 hours alone, 30 minutes of walks, no kids or allergies, first dog.")
    assert session.calls[-1] == ("recommend_breeds", {"hours_alone": 8, "daily_walks_min": 30, "kids_at_home": False,
                                                      "has_allergies": False, "experience": "none", "home": "flat"})
    assert agent.state.missing_answers() == []


@pytest.mark.parametrize("message,expected", [
    ("I live in a flat, which dog would suit me?", {"home": "flat"}),
    ("We live in a house with a garden", {"home": "house_with_garden"}),
    ("I work 9 hours a day", {"hours_alone": 9}),
    ("I am out for 8 hours and can walk 30 minutes", {"hours_alone": 8, "daily_walks_min": 30}),
    ("I can do an hour of walks", {"daily_walks_min": 60}),
    ("No kids, but I am allergic to dogs", {"kids_at_home": False, "has_allergies": True}),
    ("We have two kids", {"kids_at_home": True}),
    ("It would be my first dog", {"experience": "none"}),
    ("I have no allergies", {"has_allergies": False}),
    ("Is a Husky OK in a flat?", {}),                      # about a breed, not the user
    ("Does it need two hours of walks?", {}),
])
def test_extract_answers(message, expected):
    assert extract_answers(message) == expected


# --- Tool-call log ---------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_every_tool_call_is_appended_to_the_session_log(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    llm = FakeLLM(
        reply(calls=[tool_call("get_breed_traits", {"breed": "Beagle"})]), reply("Beagles have energy 4/5."),
        reply(calls=[tool_call("suggest_names", {"species": "hamster", "colour": None, "description": "x"})]),  # MCP error: still logged
        reply(calls=[tool_call("get_breed_traits", {"breed": "Pug"})]), reply("Pugs are calm."),
    )
    agent = AgentLoop(llm_client=llm, session=make_agent(llm)[1], session_id="test123")
    await agent.run_turn("Beagle energy?")
    await agent.run_turn("And a pug?")

    lines = (tmp_path / "tool_calls_test123.jsonl").read_text(encoding="utf-8").splitlines()
    logs = [ToolCallLog.model_validate_json(line) for line in lines]
    assert [(log.turn, log.tool) for log in logs] == [(1, "get_breed_traits"), (2, "suggest_names"),
                                                      (2, "get_breed_traits")]
    assert logs[0].result_ids == ["Beagle"] and logs[1].result_ids == []
    assert logs == agent.trace


@pytest.mark.parametrize("text,kind", [
    (VET_REPLY, "vet_referral"), (OUT_OF_SCOPE_REPLY, "out_of_scope"), (DONT_KNOW_REPLY, "i_dont_know"),
    ("I don't know: the profiles do not list blood pressure.", "i_dont_know"),
    ("Sorry, that's outside what I can help with: I only cover dog breeds.", "out_of_scope"),
    ("Please see your vet.", "vet_referral"),
    ("Beagles need an hour a day [kb:beagle:exercise:0].", "answer"),
])
def test_reply_kind(text, kind):
    assert reply_kind(text) == kind


@pytest.mark.parametrize("text", [
    "I don't have that specific information in my breed profiles. Please ask your veterinarian.",
    "The profiles don't include breeder statistics, so I can't provide that number.",
])
def test_declines_that_mention_a_vet_are_i_dont_know(text):
    assert reply_kind(text) == "i_dont_know"


@pytest.mark.asyncio
async def test_photo_tools_refuse_an_image_id_the_user_never_sent():
    llm = FakeLLM(reply(calls=[tool_call("identify_breed", {"image_id": "made_up.jpg"})]), reply("I don't know."))
    agent, session = make_agent(llm)
    assert await agent.run_turn("Which vet clinic is open now?") == "I don't know."
    assert session.calls == []
    assert tool_messages(llm)[0]["content"].startswith("Refused: identify_breed needs a photo")


@pytest.mark.asyncio
async def test_photo_from_an_earlier_turn_can_still_be_used():
    llm = FakeLLM(reply("Nice photo!"), reply(calls=[tool_call("describe_pet", {"image_id": "dog.jpg"})]),
                  reply("A large brown dog."))
    agent, session = make_agent(llm)
    await agent.run_turn("Look", image_id="dog.jpg")
    await agent.run_turn("Describe it")
    assert session.calls == [("describe_pet", {"image_id": "dog.jpg"})]
