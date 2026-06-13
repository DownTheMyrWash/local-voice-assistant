"""Pi-side streaming STT client with auto-reconnect."""
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
        self._connect_task: asyncio.Task | None = None

        self._final_event: asyncio.Event = asyncio.Event()
        self._final_text: str = ""
        self._connected: asyncio.Event = asyncio.Event()
        self._stop: asyncio.Event = asyncio.Event()

    async def connect(self) -> None:
        """Start the background reconnect loop. Returns immediately."""
        self._connect_task = asyncio.create_task(self._connect_loop())

    async def _connect_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._session = aiohttp.ClientSession()
                async with self._session.ws_connect(self._url, max_msg_size=2**20) as ws:
                    self._ws = ws
                    self._connected.set()
                    backoff = 1.0
                    log.info("STT stream connected to %s", self._url)
                    await self._recv_loop()
            except Exception as e:
                log.warning("STT connection failed (%s), retrying in %.1fs...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                self._connected.clear()
                self._ws = None
                if self._session and not self._session.closed:
                    await self._session.close()
                self._session = None

    async def _recv_loop(self) -> None:
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
                    log.debug("STT partial: %r", text)
                elif kind == "final":
                    self._final_text = text
                    self._final_event.set()
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                log.warning("STT WebSocket closed")
                break
        self._connected.clear()

    async def feed(self, pcm_bytes: bytes) -> None:
        """Send a PCM chunk to Vosk. No-ops silently if not connected."""
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.send_bytes(pcm_bytes)
            except Exception:
                pass

    async def finalize(self, timeout: float = 5.0) -> str:
        """Flush Vosk and return the final transcript.

        If the STT connection is down, waits briefly for reconnect before
        giving up and returning an empty string.
        """
        # Give reconnect a moment if we just dropped
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            log.warning("STT not connected at finalize time — returning empty.")
            return ""

        if self._ws is None or self._ws.closed:
            return ""

        self._final_event.clear()
        self._final_text = ""

        try:
            await self._ws.send_str("FINALIZE")
        except Exception as e:
            log.warning("STT finalize send failed: %s", e)
            return ""

        try:
            await asyncio.wait_for(self._final_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("STT finalize timed out after %.1fs", timeout)

        return self._final_text

    async def close(self) -> None:
        self._stop.set()
        self._connected.set()  # unblock any waiting finalize
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
