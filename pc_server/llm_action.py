"""LLM layer that understands actions.

The LLM is given the registry's tool specs. If it returns a tool call,
we run the action and TTS the result. If it returns plain text, we
TTS the text. The result is an async iterator of PCM chunks either way.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator
import os

import httpx

from shared.config import Config, load_config
from shared.logging_utils import get_logger, LatencyTimer

from .actions import get_registry
from .actions.base import ActionContext
log = get_logger("llm")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

cfg = load_config() 
HISTORY_FILE = cfg.chat_history_file or "chat_history.json"

MAX_HISTORY_MESSAGES = 40  # Keeps the last n/2 back-and-forths

class ActionAwareLLM:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._registry = get_registry()
        self._ctx = ActionContext()  
        self._history = self._load_history()

    def _load_history(self) -> list[dict[str, str]]:
        """Loads chat history from disk."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.error("Failed to load chat history: %s", e)
        return []

    def _save_history(self) -> None:
        """Prunes and saves chat history to disk."""
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]
            
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            log.error("Failed to save chat history: %s", e)

    async def stream_reply(self, user_text: str) -> AsyncIterator[tuple[str, str]]:
        tool_specs = self._registry.tool_specs()
        
        # Explicit system instructions clarifying clear_history is an available structural action
        system = (
            (self._cfg.llm_system_prompt or "")
            + "\n\nAvailable actions:\n"
            + self._registry.descriptions()
            + "\n- name: clear_history\n  description: Clear or reset the chat conversation history, forgetting everything discussed so far. Arguments: None"
            + "\n\nIf an action fits, reply with EXACTLY:\n"
              "<tool_call>{\"name\": \"action_name\", \"arguments\": {}}</tool_call>\n"
              "or reply with raw JSON:\n"
              "{\"name\": \"action_name\", \"arguments\": {}}\n"
              "Otherwise reply with normal conversational text."
        )
        
        messages = [{"role": "system", "content": system}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})

        body = {
            "model": self._cfg.ollama_model or "llama3.1:8b",
            "messages": messages,
            "stream": True,
        }
        
        url = (self._cfg.ollama_host or "http://localhost:11434") + "/api/chat"
        log.info("LLM: prompt=%r", user_text[:120])

        full = ""
        is_tool_suspect = False
        buffered_tokens: list[str] = []

        with LatencyTimer("llm", log):
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        
                        message_data = chunk.get("message", {})
                        token = message_data.get("content", "")
                        
                        if token:
                            full += token
                            
                            # Check the beginning of the stream to see if it looks like JSON or a tool tag
                            if len(full) <= 12:
                                stripped = full.strip()
                                if stripped.startswith("{") or stripped.startswith("<"):
                                    is_tool_suspect = True
                            
                            # If it looks like a tool call, we don't stream tokens to TTS yet
                            if is_tool_suspect:
                                buffered_tokens.append(token)
                            else:
                                # Safe plain text, stream it immediately
                                yield ("token", token)

                        if chunk.get("done"):
                            break

        # 1. Process regular tags: <tool_call>...</tool_call>
        m = TOOL_CALL_RE.search(full)
        if m:
            try:
                payload = json.loads(m.group(1))
                yield ("action", payload)
                return
            except json.JSONDecodeError as e:
                log.warning("Bad tool_call JSON inside tags: %s", e)

        # 2. Process fallback raw JSON objects
        cleaned_full = full.strip()
        if cleaned_full.startswith("{") and cleaned_full.endswith("}"):
            try:
                payload = json.loads(cleaned_full)
                if "name" in payload:
                    yield ("action", payload)
                    return
            except json.JSONDecodeError:
                pass

        # 3. If it wasn't a tool call after all, flush any tokens we held back to the TTS engine
        if is_tool_suspect and buffered_tokens:
            for token in buffered_tokens:
                yield ("token", token)

        # Update and save conversational history
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": full.strip()})
        self._save_history()

        log.info("LLM reply: %r", full.strip()[:200])

    async def run_action(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Execute a tool call and stream its text result."""
        name = payload.get("name", "")
        args = payload.get("arguments") or {}
        
        # Super simple internal clear history handler
        if name == "clear_history":
            self._history = []
            if os.path.exists(HISTORY_FILE):
                try:
                    os.remove(HISTORY_FILE)
                    yield "I have cleared our conversation history and forgotten past sessions."
                except Exception as e:
                    log.error("Failed to delete history file: %s", e)
                    yield "I encountered an error trying to clear the history file."
            else:
                yield "Your conversation history is already empty."
            return

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