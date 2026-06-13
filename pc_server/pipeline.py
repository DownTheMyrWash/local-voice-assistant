"""Voice Session Pipeline."""
from __future__ import annotations

import asyncio
import json
import re

from aiohttp import web

from shared.config import Config
from shared.logging_utils import get_logger
from shared.protocol import (
    MSG_STT_FINAL,
    MSG_STT_RESULT,
    MSG_TTS_END,
    MSG_UTTERANCE_END,
    MSG_WAKE,
    encode_json,
)

from .llm_action import ActionAwareLLM
from .tts import TTS

log = get_logger("pipeline")

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
_TOOL_CALL_START = re.compile(r'<tool_call>')
_MIN_CHUNK_CHARS = 10


def _split_sentences(text: str) -> tuple[list[str], str]:
    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        return [], text
    if not re.search(r'[.!?]$', parts[-1].strip()):
        return parts[:-1], parts[-1]
    return parts, ""


class VoiceSession:
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
            transcript = data.get("text", "").strip()
            if transcript:
                self._pending_transcript = transcript
                log.info("Received transcript: %r", transcript)
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
        try:
            await self._stream_llm_to_tts(user_text)
            await self.ws.send_str(encode_json({"type": MSG_TTS_END}))
            log.info("Pipeline turn complete.")
        except asyncio.CancelledError:
            log.info("Pipeline task cancelled.")
        except Exception as e:
            log.exception("Pipeline error: %s", e)

    async def _stream_llm_to_tts(self, user_text: str) -> None:
        """Stream LLM tokens into sentences, synthesize each as it completes.

        Tool call detection: if we see <tool_call> appear in the buffer at any
        point we stop enqueuing text sentences immediately and let the action
        path handle everything. Any text already enqueued before the tool call
        started is allowed to finish playing.
        """
        buffer = ""
        loop = asyncio.get_running_loop()
        tool_call_detected = False

        tts_queue: asyncio.Queue[asyncio.Future[bytes] | None] = asyncio.Queue()

        async def synthesis_worker() -> None:
            while True:
                fut = await tts_queue.get()
                if fut is None:
                    break
                try:
                    wav_bytes = await fut
                    await self._stream_wav(wav_bytes)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.warning("TTS chunk failed: %s", e)

        def _enqueue(sentence: str) -> None:
            sentence = sentence.strip()
            if not sentence:
                return
            log.info("Queuing TTS sentence: %r", sentence)
            fut = loop.run_in_executor(None, self.tts._generate_wav, sentence)
            tts_queue.put_nowait(fut)

        worker = asyncio.create_task(synthesis_worker())

        try:
            async for kind, payload in self.llm.stream_reply(user_text):
                if kind == "action":
                    tool_call_detected = True

                    # Drain the worker before playing action result so order is preserved
                    await tts_queue.put(None)
                    await worker

                    log.info("Action triggered: %s", payload.get("name"))
                    chunks: list[str] = []
                    async for chunk in self.llm.run_action(payload):
                        chunks.append(chunk)
                    action_text = "".join(chunks)
                    if action_text.strip():
                        log.info("Action result: %r", action_text)
                        await self._synthesize_and_stream(action_text)
                    return

                elif kind == "token":
                    buffer += payload

                    # If a tool call tag appears in the buffer, stop sending
                    # anything to TTS — the LLM is going to emit a tool call
                    if _TOOL_CALL_START.search(buffer):
                        tool_call_detected = True
                        continue

                    if tool_call_detected:
                        continue

                    sentences, buffer = _split_sentences(buffer)
                    for sentence in sentences:
                        if len(sentence.strip()) >= _MIN_CHUNK_CHARS:
                            _enqueue(sentence)
                        else:
                            buffer = sentence.strip() + " " + buffer.lstrip()

            # LLM finished without a tool call — flush remainder
            if not tool_call_detected and buffer.strip():
                _enqueue(buffer)

        finally:
            if not tool_call_detected:
                await tts_queue.put(None)
                await worker
            # If tool_call_detected, worker was already shut down in the action branch

    async def _synthesize_and_stream(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, self.tts._generate_wav, text)
        await self._stream_wav(wav_bytes)

    async def _stream_wav(self, wav_bytes: bytes) -> None:
        if not wav_bytes or len(wav_bytes) < 44:
            return
        pcm = wav_bytes[44:]
        chunk_size = 2048
        for i in range(0, len(pcm), chunk_size):
            await self.ws.send_bytes(pcm[i : i + chunk_size])
