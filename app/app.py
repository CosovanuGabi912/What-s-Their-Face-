"""Chainlit UI shell (issue #13): chat with text and photo upload, wired to the AgentLoop.

Run from the repository root:  chainlit run app/app.py

Photos are re-encoded to RGB JPEG and saved once as IMAGE_UPLOAD_DIR/<image_id>. Only that plain file
name goes to the AgentLoop and MCP, never bytes, base64 or a path (shared/images.py).

The AgentLoop must be opened and closed by the same asyncio task (its MCP stdio client is bound to it),
but Chainlit calls the handlers below from different tasks. So one small task per chat session owns the
loop (`_own_agent`); the handlers only call `run_turn` on it.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `chainlit run app/app.py` does not put the repository root on the path
    sys.path.insert(0, str(ROOT))

import chainlit as cl  # noqa: E402

from agent.loop import AgentLoop  # noqa: E402
from shared.images import save_as_jpeg  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_PHOTO_QUESTION = "What breed is this dog?"   # used when a photo is sent without any text
ERROR_REPLY = ("Sorry, I could not get an answer. Check that LM Studio is running with the agent model "
               "loaded, then try again.")
BAD_IMAGE_REPLY = "I could not read that file as a photo. Please send a JPEG, PNG, WebP or HEIC image."


def save_upload(element: cl.Image, upload_dir: Path | None = None) -> str:
    """Save an uploaded photo as an RGB JPEG in IMAGE_UPLOAD_DIR and return its image_id (file name only).

    Raises ValueError if the file is not an image.
    """
    data = element.content
    if data is None and element.path:
        data = Path(element.path).read_bytes()
    if not data:
        raise ValueError("The uploaded element has no content")
    return save_as_jpeg(data, upload_dir)


def _is_image(element: object) -> bool:
    return isinstance(element, cl.Image) or str(getattr(element, "mime", "") or "").startswith("image/")


async def _own_agent(agent: AgentLoop, ready: asyncio.Event, stop: asyncio.Event) -> None:
    """Open the agent's MCP connection and close it in this same task when the chat ends."""
    try:
        async with agent:
            ready.set()
            await stop.wait()
    except Exception:
        logger.exception("agent loop failed to start or stop")
    finally:
        ready.set()   # never leave on_chat_start waiting


@cl.on_chat_start
async def on_chat_start() -> None:
    agent = AgentLoop()
    ready, stop = asyncio.Event(), asyncio.Event()
    owner = asyncio.create_task(_own_agent(agent, ready, stop))
    await ready.wait()
    if owner.done():   # could not start (e.g. the MCP server crashed)
        await cl.Message(content=ERROR_REPLY).send()
        return
    cl.user_session.set("agent_loop", agent)
    cl.user_session.set("agent_owner", (owner, stop))


@cl.on_chat_end
async def on_chat_end() -> None:
    owned = cl.user_session.get("agent_owner")
    if owned:
        owner, stop = owned
        stop.set()
        await owner


@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent: AgentLoop | None = cl.user_session.get("agent_loop")
    if agent is None:
        await cl.Message(content=ERROR_REPLY).send()
        return

    image_id = None
    for element in message.elements or []:
        if _is_image(element):
            try:
                image_id = save_upload(element)
            except ValueError:
                logger.warning("unreadable upload %r", getattr(element, "name", None), exc_info=True)
                await cl.Message(content=BAD_IMAGE_REPLY).send()
                return
            break   # one photo per message

    text = message.content.strip() or (DEFAULT_PHOTO_QUESTION if image_id else "")
    if not text:
        return
    try:
        async with cl.Step(name="Thinking", type="run"):   # shows a loading state until the turn is done
            reply = await agent.run_turn(text, image_id=image_id)
    except Exception:
        logger.exception("run_turn failed")
        reply = ERROR_REPLY
    await cl.Message(content=reply).send()
