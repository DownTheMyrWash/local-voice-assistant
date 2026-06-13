"""LLM layer that understands actions.

The LLM is given the registry's tool specs. If it returns a tool call,
we run the action and TTS the result. If it returns plain text, we
TTS the text. The result is an async iterator of PCM chunks either way.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

import httpx

from shared.config import Config
from shared.logging_utils import get_logger, LatencyTimer

from .actions import get_registry
from .actions.base import ActionContext

log = get_logger("llm")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class ActionAwareLLM:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._registry = get_registry()
        self._ctx = ActionContext()  # could carry client_addr in the future

    async def stream_reply(self, user_text: str) -> AsyncIterator[tuple[str, str]]:
        """Yield (kind, payload) tuples.

        Kinds:
            "token" : payload is a text fragment from the LLM (streamed)
            "action": payload is a dict describing the chosen action
            "action_result": payload is the action's text result (streamed)
        """
        tool_specs = self._registry.tool_specs()
        # Tell the model which tools exist + how to call them
        system = (
            (self._cfg.llm_system_prompt or "")
            + "\n\nAvailable actions:\n"
            + self._registry.descriptions()
            + "\n\nIf an action fits, reply with EXACTLY:\n"
              "<tool_call>{\"name\": \"action_name\", \"arguments\": {}}</tool_call>\n"
              "Otherwise reply with normal conversational text."
        )
        body = {
            "model": self._cfg.ollama_model or "llama3.1:8b",
            "prompt": user_text,
            "system": system,
            "stream": True,
        }
        url = (self._cfg.ollama_host or "http://localhost:11434") + "/api/generate"
        log.info("LLM: prompt=%r", user_text[:120])

        full = ""
        with LatencyTimer("llm", log):
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            full += token
                            yield ("token", token)
                        if chunk.get("done"):
                            break

        # Look for a tool call in the full response
        m = TOOL_CALL_RE.search(full)
        if m:
            try:
                payload = json.loads(m.group(1))
                yield ("action", payload)
                return
            except json.JSONDecodeError as e:
                log.warning("Bad tool_call JSON: %s", e)

        # No action -> the streamed text IS the reply. Caller will TTS it.
        log.info("LLM reply: %r", full.strip()[:200])

    async def run_action(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Execute a tool call and stream its text result."""
        name = payload.get("name", "")
        args = payload.get("arguments") or {}
        action = self._registry.get(name)
        if action is None:
            yield f"I don't know how to do {name}."
            return
        try:
            async for chunk in action.run(args, self._ctx):
                yield chunk
        except Exception as e:
            log.exception("Action %s failed", name)
            yield f"That action failed: {e}"
