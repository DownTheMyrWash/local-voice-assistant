"""Pi-side streaming STT client.

Opens a persistent WebSocket connection to the PC's /ws/stt endpoint.
During an utterance, the caller feeds PCM frames via feed(); when the
utterance ends, finalize() flushes Vosk and returns the full transcript.

The connection is kept alive across utterances so there's no reconnect
latency between wake words.
"""
from __future__ import annotations

import asyncio
import json

import aiohttp

from shared.config import Config
from shared.logging_utils import get_logger

log = get_logger("stt_client")


class StreamingSTTClient:
    def __init__(self, cfg: Config) -> None:
        self._url = f"ws://{cfg.pc_host}:{cfg.pc_port}/ws/stt"
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._recv_task: asyncio.Task | None = None

        # Set by _recv_loop when Vosk sends back a "final" result
        self._final_event: asyncio.Event = asyncio.Event()
        self._final_text: str = ""

    async def connect(self) -> None:
        """Open the WebSocket connection to the PC STT server."""
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self._url, max_msg_size=2**20)
        self._recv_task = asyncio.create_task(self._recv_loop())
        log.info("STT stream connected to %s", self._url)

    async def _recv_loop(self) -> None:
        """Background task: listens for Vosk transcript messages."""
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                kind = data.get("type")
                text = data.get("text", "")

                if kind == "partial":
                    # Useful for logging / future UI feedback
                    log.debug("STT partial: %r", text)

                elif kind == "final":
                    self._final_text = text
                    self._final_event.set()

            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                log.warning("STT WebSocket closed unexpectedly")
                break

    async def feed(self, pcm_bytes: bytes) -> None:
        """Send a raw PCM chunk to Vosk. Call this for every mic frame."""
        if self._ws is not None and not self._ws.closed:
            await self._ws.send_bytes(pcm_bytes)

    async def finalize(self, timeout: float = 5.0) -> str:
        """Tell Vosk to flush, wait for the final transcript, and return it.

        Safe to call even if no audio was fed (returns empty string).
        """
        if self._ws is None or self._ws.closed:
            return ""

        self._final_event.clear()
        self._final_text = ""

        await self._ws.send_str("FINALIZE")

        try:
            await asyncio.wait_for(self._final_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("STT finalize timed out after %.1fs", timeout)

        return self._final_text

    async def close(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
