"""Voice Session Pipeline: Coordinates LLM, Actions, and TTS via aiohttp WebSockets.

STT is no longer performed here — the Pi streams audio to /ws/stt in real time
and sends the finished transcript as MSG_STT_RESULT before MSG_UTTERANCE_END.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web

from shared.config import Config
from shared.logging_utils import get_logger
from shared.protocol import (
    MSG_STT_FINAL,
    MSG_STT_RESULT,   # new: Pi sends us the pre-transcribed text
    MSG_TTS_END,
    MSG_UTTERANCE_END,
    MSG_WAKE,
    encode_json,
)

from .llm_action import ActionAwareLLM
from .tts import TTS

log = get_logger("pipeline")


class VoiceSession:
    """Manages the full lifecycle of a single WebSocket voice client session."""

    def __init__(self, cfg: Config, ws: web.WebSocketResponse) -> None:
        self.cfg = cfg
        self.ws = ws

        self.llm = ActionAwareLLM(cfg)
        self.tts = TTS(cfg)

        self._pending_transcript: str = ""
        self._processing_task: asyncio.Task | None = None

    async def run(self) -> None:
        async for msg in self.ws:
            if msg.type == web.WSMsgType.TEXT:
                await self._handle_text(msg.data)
            elif msg.type in (
                web.WSMsgType.CLOSE,
                web.WSMsgType.CLOSING,
                web.WSMsgType.ERROR,
            ):
                break

    async def _handle_text(self, text_data: str) -> None:
        try:
            data = json.loads(text_data)
            if isinstance(data, str):
                data = json.loads(data)
        except json.JSONDecodeError:
            log.warning("Invalid JSON: %r", text_data)
            return

        if not isinstance(data, dict):
            return

        msg_type = data.get("type")

        if msg_type == MSG_WAKE:
            log.info("Wake received — ready for utterance.")
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                log.info("Interrupted active generation task.")
            self._pending_transcript = ""

        elif msg_type == MSG_STT_RESULT:
            # Pi has finished streaming to Vosk and got the final transcript back.
            # Store it so MSG_UTTERANCE_END can kick off the LLM immediately.
            transcript = data.get("text", "").strip()
            if transcript:
                self._pending_transcript = transcript
                log.info("Received pre-transcribed text: %r", transcript)
                # Echo to client UI
                await self.ws.send_str(
                    encode_json({"type": MSG_STT_FINAL, "text": transcript})
                )

        elif msg_type == MSG_UTTERANCE_END:
            if not self._pending_transcript:
                log.info("Utterance ended but no transcript — dropping turn.")
                await self.ws.send_str(encode_json({"type": MSG_TTS_END}))
                return

            transcript = self._pending_transcript
            self._pending_transcript = ""
            self._processing_task = asyncio.create_task(
                self._process_and_respond(transcript)
            )

    async def _process_and_respond(self, user_text: str) -> None:
        """LLM → TTS, skipping STT entirely."""
        try:
            text_to_synthesize = ""
            async for kind, payload in self.llm.stream_reply(user_text):
                if kind == "action":
                    log.info("Action triggered: %s", payload.get("name"))
                    chunks = []
                    async for chunk in self.llm.run_action(payload):
                        chunks.append(chunk)
                    text_to_synthesize = "".join(chunks)
                    log.info("Action result: %s", text_to_synthesize)
                    break
                elif kind == "token":
                    text_to_synthesize += payload

            if text_to_synthesize.strip():
                await self.tts_stream_to_client(text_to_synthesize)

            await self.ws.send_str(encode_json({"type": MSG_TTS_END}))
            log.info("Pipeline turn complete.")

        except asyncio.CancelledError:
            log.info("Pipeline task cancelled.")
        except Exception as e:
            log.exception("Pipeline error: %s", e)

    async def tts_stream_to_client(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, self.tts._generate_wav, text)

        if not wav_bytes or len(wav_bytes) < 44:
            return

        pcm_payload = wav_bytes[44:]
        chunk_size = 2048
        for i in range(0, len(pcm_payload), chunk_size):
            await self.ws.send_bytes(pcm_payload[i : i + chunk_size])
