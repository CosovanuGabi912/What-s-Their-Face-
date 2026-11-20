"""Agent loop (issues #7 and #16): one SLM (Qwen3-4B in LM Studio) calling the pet-advisor MCP tools.

One turn:
    vet_check, out_of_scope_check (code, before the SLM) -> fixed reply, no SLM call
    -> questionnaire answers from the message -> state
    -> SLM: answer or call tools?
    -> each tool call: limits checked in code -> run through MCP -> logged -> state updated -> result to SLM
       (identify_breed with decision "top2"/"retry" or a wild canid ends the turn with a fixed reply)
    -> ... until the SLM answers -> citation_check + safety_claim_check (one retry) -> reply

Every reply has a kind (self.last_reply_kind): "vet_referral", "out_of_scope", "i_dont_know",
"low_confidence", "apology" or "answer". eval/run_eval.py --part C scores it.
Every executed tool call is appended to config.LOG_DIR / f"tool_calls_{session_id}.jsonl".

The SLM decides what to do; code decides whether it is allowed (docs/milestone2.md).

Usage. Open and close the loop in the same asyncio task: the MCP stdio client must be closed by
the task that opened it.

    async with AgentLoop() as agent:
        reply = await agent.run_turn("What breed is this?", image_id="abc123.jpg")
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool
from openai import AsyncOpenAI

import config
from agent.guards import (OUT_OF_SCOPE_REPLY, VET_REPLY, citation_check, out_of_scope_check, safety_claim_check,
                          unbacked_safety_claims, unretrieved_ids, vet_check)
from agent.state import QUESTIONNAIRE_FIELDS, ConversationState, extract_answers
from shared.mcp_client import project_env, server_params
from shared.schemas import ToolCallLog

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PATH = config.ROOT / "agent" / "prompts" / "system.md"
MAX_INVALID_TOOL_CALLS = 2                                  # 1st: error sent back; 2nd: apologise
MAX_LLM_CALLS_PER_TURN = config.MAX_TOOLS_PER_TURN + 3      # safety net if the SLM ignores tool_choice

APOLOGY_REPLY = "Sorry, something went wrong on my side. Could you rephrase that or try again?"
DONT_KNOW_REPLY = "Sorry, I don't know. I couldn't find a reliable source for that."
TOOL_LIMIT_MESSAGE = (f"Limit reached: at most {config.MAX_TOOLS_PER_TURN} tool calls per question. "
                      "Do not call more tools. Answer with what you have, or say you don't know.")
SEARCH_LIMIT_MESSAGE = (f"Limit reached: at most {config.MAX_SEARCHES_PER_QUESTION} searches per question. "
                        "Do not search again. Answer from the chunks you have, or say you don't know.")
NO_PHOTO_MESSAGE = ("Refused: {name} needs a photo the user sent, and no photo with that image_id was sent. "
                    "Do not call photo tools without one. Answer the question without it, or ask for a photo.")

# Low-confidence replies after identify_breed (agent decision 2). The thresholds behind `decision` live in
# models/breed_classifier/threshold.json (#10) and are applied by the tool, never here.
RETRY_PHOTO_REPLY = "I can't identify the breed confidently. Could you send a clearer photo?"
TOP2_REPLY = "I'm not sure: it could be a {first} or a {second}. Could you send a clearer photo?"
OUT_OF_SCOPE_WILD = ("This looks like a {breed}, a wild animal, not a pet dog breed, so I can't help with it. "
                     "Send me a photo of a dog and I'll tell you its breed.")

# Kinds of reply (self.last_reply_kind); the SLM's own replies are classified by reply_kind().
_DONT_KNOW_RE = re.compile(
    r"\b(i\s+(don't|do not|dont)\s+know|i couldn't find|i could not find|no reliable source|"
    r"i\s+(don't|do not)\s+have\s+(that|this|any|enough|access|the|specific|information|data|details)|"
    r"(can't|cannot|can not|unable to)\s+(provide|answer|give you|tell you|find)|"
    r"(doesn't|does not|don't|do not)\s+(include|cover|contain|mention|list|say))\b")
_OUT_OF_SCOPE_RE = re.compile(r"\b(outside what i can help|out of scope|outside my scope|i only (cover|help))\b")
_VET_RE = re.compile(r"\b(vet|veterinarian|veterinary)\b")


def reply_kind(text: str) -> str:
    """What kind of reply the user got: the fixed replies by identity, the SLM's own by its wording."""
    fixed = {VET_REPLY: "vet_referral", OUT_OF_SCOPE_REPLY: "out_of_scope", DONT_KNOW_REPLY: "i_dont_know",
             APOLOGY_REPLY: "apology", RETRY_PHOTO_REPLY: "low_confidence"}
    if text in fixed:
        return fixed[text]
    lowered = text.casefold().replace("\u2019", "'")
    if _OUT_OF_SCOPE_RE.search(lowered):
        return "out_of_scope"
    if _DONT_KNOW_RE.search(lowered):   # before vet: "I don't have that data, ask your vet" is a decline
        return "i_dont_know"
    if _VET_RE.search(lowered):
        return "vet_referral"
    return "answer"


def guard_errors(answer: str, retrieved_ids: set[str]) -> list[str]:
    """Why the final answer fails the citation and safety guards (empty list = it passes)."""
    errors = []
    if not citation_check(answer, retrieved_ids):
        bad = unretrieved_ids(answer, retrieved_ids)
        errors.append(f"These chunk IDs were not retrieved: {bad}. Cite only chunk ids that search_knowledge "
                      "returned for this question.")
    if not safety_claim_check(answer, retrieved_ids):
        claims = unbacked_safety_claims(answer, retrieved_ids)
        errors.append(f"These sentences say something is safe without citing a retrieved chunk: {claims}. "
                      "Never say something is safe unless a retrieved chunk says so, and cite it.")
    return errors

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str | None) -> str:
    """Remove Qwen3's <think>...</think> reasoning so only the answer reaches the user."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:  # opening tag cut off by the chat template
        text = text.split("</think>")[-1]
    return text.strip()


def to_openai_tool(tool: Tool) -> dict[str, Any]:
    """MCP tool definition -> OpenAI function-calling format (what LM Studio expects)."""
    return {"type": "function",
            "function": {"name": tool.name, "description": tool.description or "",
                         "parameters": tool.inputSchema}}


@dataclass
class TurnBudget:
    """Per-turn counters behind the limits that are enforced in code."""
    tool_calls: int = 0
    searches: int = 0
    invalid_calls: int = 0
    retrieved_ids: set[str] = field(default_factory=set)   # chunk ids search_knowledge returned
    forced_reply: tuple[str, str] | None = None             # (text, kind) set by code, ends the turn

    @property
    def tools_exhausted(self) -> bool:
        return self.tool_calls >= config.MAX_TOOLS_PER_TURN


class AgentLoop:
    def __init__(self, mcp_server_command: list[str] | None = None,
                 lm_studio_url: str = config.LM_STUDIO_BASE_URL,
                 model_id: str = config.AGENT_MODEL_ID, *,
                 server_env: dict[str, str] | None = None,
                 llm_client: Any = None, session: Any = None, session_id: str | None = None) -> None:
        """
        mcp_server_command: e.g. ["python", "-m", "mcp_server.server"]; None starts
            mcp_server/server.py with this Python (shared.mcp_client.server_params).
        server_env: extra variables for the server process, e.g. {"WTF_STUB_TOOLS": "all"}.
        llm_client / session: test doubles for the OpenAI client and the MCP ClientSession (or an already
            open session to share, e.g. one server for a whole eval run).
        session_id: names the tool-call log, logs/tool_calls_<session_id>.jsonl (default: a new uuid).
        """
        self.model_id = model_id
        self.llm = llm_client or AsyncOpenAI(base_url=lm_studio_url, api_key=config.LM_STUDIO_API_KEY)
        if mcp_server_command:
            # project_env: the stdio client does not pass our environment to the server by itself
            self._server = StdioServerParameters(command=mcp_server_command[0], args=list(mcp_server_command[1:]),
                                                 env=project_env(server_env), cwd=str(config.ROOT))
        else:
            self._server = server_params(server_env)
        self.session = session
        self._stack: AsyncExitStack | None = None
        self.tools: list[dict[str, Any]] | None = None
        self._tool_names: set[str] = set()

        self.state = ConversationState()
        # Chat history without the system prompt. Between turns it holds only user messages and final
        # replies: tool calls and results are dropped when a turn ends (_reply), so retrieved chunks do not
        # fill the SLM's context. What later turns need (breed, questionnaire, rejected names) is in self.state.
        self.messages: list[dict[str, Any]] = []
        self._turn_start = 0                       # index of this turn's user message in self.messages
        self.image_ids: set[str] = set()           # photos the user actually sent in this conversation
        self.trace: list[ToolCallLog] = []         # every tool call of the conversation
        self.turn = 0
        self.session_id = session_id or uuid.uuid4().hex
        self.last_reply_kind: str | None = None
        self._prompt_template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    # --- MCP connection ---------------------------------------------------------------------
    async def connect(self) -> None:
        """Start the MCP server (unless a session was given) and load its tool list. Idempotent."""
        if self.session is None:
            stack = AsyncExitStack()
            try:
                read, write = await stack.enter_async_context(stdio_client(self._server))
                self.session = await stack.enter_async_context(ClientSession(read, write))
                await self.session.initialize()
            except BaseException:
                await stack.aclose()
                self.session = None
                raise
            self._stack = stack
        if self.tools is None:
            # Advertised schemas are the source of truth, not shared.schemas.TOOL_SCHEMAS (see #7 comment)
            listed = (await self.session.list_tools()).tools
            self.tools = [to_openai_tool(tool) for tool in listed]
            self._tool_names = {tool.name for tool in listed}

    async def aclose(self) -> None:
        """Stop the MCP server if this loop started it."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack, self.session, self.tools = None, None, None

    async def __aenter__(self) -> "AgentLoop":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- One turn ---------------------------------------------------------------------------
    async def run_turn(self, user_message: str, image_id: str | None = None) -> str:
        """Single conversation turn. Returns the agent's reply."""
        self.turn += 1
        content = user_message if image_id is None else f"{user_message}\n[image_id: {image_id}]"
        if image_id is not None:
            self.image_ids.add(image_id)
        self._turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})

        # Code first; the SLM never sees these messages, so it cannot overrule the guards.
        if vet_check(user_message):
            logger.info("turn %d: vet_check fired", self.turn)
            return self._reply(VET_REPLY, "vet_referral")
        if out_of_scope_check(user_message):
            logger.info("turn %d: out_of_scope_check fired", self.turn)
            return self._reply(OUT_OF_SCOPE_REPLY, "out_of_scope")

        answers = extract_answers(user_message)       # "I live in a flat" -> the home question is skipped
        if answers:
            self.state.questionnaire.update(answers)

        await self.connect()
        budget = TurnBudget()
        for _ in range(MAX_LLM_CALLS_PER_TURN):
            message = await self._complete(allow_tools=not budget.tools_exhausted)
            if not message.tool_calls:
                return await self._finish(strip_think(message.content), budget)

            self.messages.append(_assistant_tool_message(message))
            for call in message.tool_calls:   # every tool_call needs a tool reply, even a refused one
                result = await self._run_tool_call(call, budget)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if budget.forced_reply:        # e.g. unsure breed: code answers, the rest of the turn is dropped
                    return self._reply(*budget.forced_reply)

            if budget.invalid_calls >= MAX_INVALID_TOOL_CALLS:
                logger.error("turn %d: %d invalid tool calls, giving up", self.turn, budget.invalid_calls)
                return self._reply(APOLOGY_REPLY, "apology")

        logger.warning("turn %d: no answer after %d SLM calls", self.turn, MAX_LLM_CALLS_PER_TURN)
        return self._reply(DONT_KNOW_REPLY, "i_dont_know")

    async def _complete(self, allow_tools: bool, extra: list[dict[str, Any]] | None = None) -> Any:
        """One SLM call with the system prompt (state filled in), the history and the tools."""
        system = self._prompt_template.replace("{state}", self.state.to_prompt())  # not .format(): braces
        response = await self.llm.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system}, *self.messages, *(extra or [])],
            tools=self.tools,
            tool_choice="auto" if allow_tools else "none",
            temperature=0,
        )
        return response.choices[0].message

    async def _finish(self, answer: str, budget: TurnBudget) -> str:
        """Citation and safety guards on the final answer: one retry with the errors, else "I don't know"."""
        if not answer:
            logger.warning("turn %d: empty answer from the SLM", self.turn)
            return self._reply(DONT_KNOW_REPLY, "i_dont_know")
        errors = guard_errors(answer, budget.retrieved_ids)
        if errors:
            logger.warning("turn %d: answer guards failed, retrying once: %s", self.turn, errors)
            retry = await self._complete(allow_tools=False, extra=[
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "\n".join(errors) + "\nRewrite your answer, or say you don't know."}])
            answer = strip_think(retry.content)
            if not answer or guard_errors(answer, budget.retrieved_ids):
                logger.warning("turn %d: answer guards failed again, replying I don't know", self.turn)
                return self._reply(DONT_KNOW_REPLY, "i_dont_know")
        return self._reply(answer)

    def _reply(self, text: str, kind: str | None = None) -> str:
        """End the turn: drop its tool calls and results from the history, keep the user message and the reply."""
        del self.messages[self._turn_start + 1:]
        self.messages.append({"role": "assistant", "content": text})
        self.last_reply_kind = kind or reply_kind(text)
        return text

    # --- Tool calls -------------------------------------------------------------------------
    async def _run_tool_call(self, call: Any, budget: TurnBudget) -> str:
        """Check limits and arguments, run the tool through MCP, update the state.
        Returns the text for the tool message (the result, or an error the SLM can act on)."""
        name = call.function.name
        if budget.tools_exhausted:
            return TOOL_LIMIT_MESSAGE
        budget.tool_calls += 1   # refused requests count too, so a model stuck on one tool is stopped
        if name == "search_knowledge" and budget.searches >= config.MAX_SEARCHES_PER_QUESTION:
            return SEARCH_LIMIT_MESSAGE

        if name not in self._tool_names:
            budget.invalid_calls += 1
            return f"Error: unknown tool {name!r}. Available tools: {sorted(self._tool_names)}."
        try:
            args = json.loads(call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            budget.invalid_calls += 1
            logger.warning("turn %d: invalid JSON for %s: %r", self.turn, name, call.function.arguments)
            return f"Error: the arguments for {name} are not a valid JSON object ({exc}). Call it again with valid JSON."

        if "image_id" in args and args["image_id"] not in self.image_ids:   # the SLM must not invent photos
            logger.warning("turn %d: %s called with image_id %r the user never sent", self.turn, name,
                           args["image_id"])
            return NO_PHOTO_MESSAGE.format(name=name)

        args = self._fill_from_state(name, args)
        if name == "search_knowledge":
            budget.searches += 1

        started = time.perf_counter()
        result = await self.session.call_tool(name, args)
        duration_ms = int((time.perf_counter() - started) * 1000)
        text = _result_text(result)
        if result.isError:  # e.g. pydantic rejected an argument: send the error back to the SLM
            budget.invalid_calls += 1
            logger.warning("turn %d: %s returned an error: %s", self.turn, name, text)
            self._log_tool_call(name, args, [], duration_ms)
            return f"Error from {name}: {text}\nFix the arguments and try again."

        data = result.structuredContent or {}
        self._update_state(name, args, data)
        ids = _result_ids(name, data)
        if name == "search_knowledge":
            budget.retrieved_ids.update(ids)
        if name == "identify_breed":
            budget.forced_reply = low_confidence_reply(data)
        self._log_tool_call(name, args, ids, duration_ms)
        return text

    @property
    def log_path(self) -> Path:
        return Path(config.LOG_DIR) / f"tool_calls_{self.session_id}.jsonl"

    def _log_tool_call(self, name: str, args: dict[str, Any], ids: list[str], duration_ms: int) -> None:
        """Keep the call in self.trace and append it to logs/tool_calls_<session_id>.jsonl."""
        entry = ToolCallLog(turn=self.turn, tool=name, arguments=args, result_ids=ids, duration_ms=duration_ms)
        self.trace.append(entry)
        logger.info("tool call: %s", entry.model_dump_json())
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(entry.model_dump_json() + "\n")
        except OSError:   # a full disk or read-only folder must not break the conversation
            logger.exception("could not write the tool-call log %s", self.log_path)

    def _fill_from_state(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Arguments that code adds from the state, so the SLM cannot forget or miscopy them."""
        args = dict(args)
        if name == "search_knowledge" and not args.get("breed"):
            breed = self.state.chosen_breed or self.state.current_breed   # agent decision 3
            if breed:
                args["breed"] = breed
        if name == "suggest_names" and self.state.rejected_names:
            args["exclude_names"] = _merge(args.get("exclude_names") or [], self.state.rejected_names)
        if name == "recommend_breeds":   # answers the user already gave, in case the SLM left them out
            for key in QUESTIONNAIRE_FIELDS:
                if key not in args and key in self.state.questionnaire:
                    args[key] = self.state.questionnaire[key]
        return args

    def _update_state(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> None:
        """Copy what the tools found into the state (only from successful results)."""
        if name == "identify_breed":
            top = data["top3"][0]
            confident = data["decision"] == "confident" and data["kind"] == "breed"
            self.state.current_breed = top["breed"] if confident else None   # unsure -> no breed filter
            self.state.breed_confidence = top["confidence"]
        elif name == "describe_pet":
            self.state.pet_description = data["description"]
        elif name == "recommend_breeds":
            self.state.questionnaire = dict(args)        # the arguments are the questionnaire answers
        elif name == "suggest_names" and args.get("exclude_names"):
            self.state.rejected_names = _merge(self.state.rejected_names, args["exclude_names"])
        # chosen_breed comes from the user, not from a tool: set it from the UI / a later issue.


def low_confidence_reply(data: dict[str, Any]) -> tuple[str, str] | None:
    """Fixed reply after identify_breed, or None when the SLM may go on (a confident dog breed)."""
    top3 = data["top3"]
    if data["kind"] == "wild_canid" and data["decision"] == "confident":
        return OUT_OF_SCOPE_WILD.format(breed=top3[0]["breed"]), "out_of_scope"
    if data["decision"] == "retry":
        return RETRY_PHOTO_REPLY, "low_confidence"
    if data["decision"] == "top2":
        if len(top3) < 2:
            return RETRY_PHOTO_REPLY, "low_confidence"
        return TOP2_REPLY.format(first=top3[0]["breed"], second=top3[1]["breed"]), "low_confidence"
    return None


def _assistant_tool_message(message: Any) -> dict[str, Any]:
    """The SLM's tool-call message in the dict form the next request sends back."""
    return {"role": "assistant",
            "content": strip_think(message.content) or None,
            "tool_calls": [{"id": call.id, "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments}}
                           for call in message.tool_calls]}


def _result_text(result: CallToolResult) -> str:
    """Compact JSON of the structured result, or the text content (errors have no structured part)."""
    if result.structuredContent is not None and not result.isError:
        return json.dumps(result.structuredContent, ensure_ascii=False)
    return "\n".join(getattr(item, "text", "") for item in result.content).strip()


def _result_ids(name: str, data: dict[str, Any]) -> list[str]:
    """IDs a result rests on, for ToolCallLog and the citation check."""
    if name == "search_knowledge":
        return [chunk["chunk_id"] for chunk in data.get("chunks", [])]
    if name in ("identify_breed", "recommend_breeds"):
        return [item["breed"] for item in data.get("top3", [])]
    if name == "get_breed_traits":
        return [data["traits"]["breed"]] if data.get("traits") else list(data.get("closest_matches", []))
    if name == "suggest_names":
        return [pid for item in data.get("names", []) for pid in item["example_listing_ids"]]
    return []


def _merge(first: list[str], second: list[str]) -> list[str]:
    """Union of two name lists, order kept, case-insensitive."""
    seen: set[str] = set()
    merged = []
    for item in [*first, *second]:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            merged.append(item)
    return merged
