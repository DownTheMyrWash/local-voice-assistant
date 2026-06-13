"""Ollama streaming client.

Talks to a local Ollama daemon over HTTP. The default base URL is
`http://localhost:11434`. To use a remote Ollama, set OLLAMA_HOST.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from shared.config import Config
from shared.logging_utils import get_logger, LatencyTimer

log = get_logger("llm")


SYSTEM_PROMPT_DEFAULT = (
    "You are a concise voice assistant. Keep answers under 2 sentences "
    "unless asked to elaborate. Speak naturally."
)


class OllamaClient:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._system_prompt = cfg.llm_system_prompt or SYSTEM_PROMPT_DEFAULT

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield tokens as they arrive from Ollama."""
        body = {
            "model": self._cfg.ollama_model or "llama3.1:8b",
            "prompt": prompt,
            "system": self._system_prompt,
            "stream": True,
        }
        url = (self._cfg.ollama_host or "http://localhost:11434") + "/api/generate"
        log.info("LLM: prompt=%r", prompt[:120])
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
                            yield token
                        if chunk.get("done"):
                            break
