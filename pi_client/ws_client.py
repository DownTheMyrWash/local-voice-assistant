"""WebSocket client with auto-reconnect and binary audio framing."""
from __future__ import annotations

import asyncio
import json

import aiohttp

from shared.config import Config
from shared.logging_utils import get_logger

log = get_logger("ws_client")


class VoiceClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.url = f"ws://{cfg.pc_host}:{cfg.pc_port}/ws/voice"
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._sender_queue: asyncio.Queue = asyncio.Queue()
        self._receiver_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._on_binary = None  # async callable(bytes)
        self._on_text = None    # async callable(dict)
        self._connected = asyncio.Event()
        self._stop = asyncio.Event()

    def set_handlers(self, on_binary, on_text) -> None:
        self._on_binary = on_binary
        self._on_text = on_text

    async def connect(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                log.info("DEBUG: Attempting connection to URL: %s", self.url)
                self._session = aiohttp.ClientSession()
                
                async with self._session.ws_connect(self.url, max_msg_size=2**20) as ws:
                    self._ws = ws
                    self._connected.set()
                    backoff = 1.0  # Reset backoff on success
                    log.info("Connected to %s", self.url)
                    
                    # Start background send/receive handlers for THIS connection
                    self._sender_task = asyncio.create_task(self._send_loop())
                    self._receiver_task = asyncio.create_task(self._recv_loop())
                    
                    # CRITICAL: Wait here until the receiver task finishes 
                    # (meaning the socket disconnected or closed)
                    await self._receiver_task
                    
            except Exception as e:
                log.warning("Connection failed (%s), retrying in %.1fs...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                # Clean up this specific session instance before trying again
                self._connected.clear()
                if self._sender_task and not self._sender_task.done():
                    self._sender_task.cancel()
                if self._receiver_task and not self._receiver_task.done():
                    self._receiver_task.cancel()
                if self._session and not self._session.closed:
                    await self._session.close()

    async def stop(self) -> None:
        self._stop.set()
        self._connected.set()  # unblock the connect() loop
        await self._cleanup()

    async def _cleanup(self) -> None:
        for t in (self._receiver_task, self._sender_task):
            if t and not t.done():
                t.cancel()
        self._receiver_task = None
        self._sender_task = None
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
        self._session = None
        self._ws = None
        self._connected.clear()

    # ---------- queues ----------

    async def send_text(self, payload: dict) -> None:
        await self._sender_queue.put(("text", json.dumps(payload)))

    async def send_binary(self, data: bytes) -> None:
        await self._sender_queue.put(("binary", data))

    async def _send_loop(self) -> None:
        assert self._ws is not None
        while True:
            kind, data = await self._sender_queue.get()
            try:
                if kind == "text":
                    await self._ws.send_str(data)
                else:
                    await self._ws.send_bytes(data)
            except Exception as e:
                log.warning("Send failed: %s", e)
                # Drop everything still in the queue to avoid stalling
                while not self._sender_queue.empty():
                    try:
                        self._sender_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self._connected.clear()
                return

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if self._on_binary:
                    await self._on_binary(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    obj = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if self._on_text:
                    await self._on_text(obj)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.warning("WS error: %s", msg.data)
                break
        self._connected.clear()
