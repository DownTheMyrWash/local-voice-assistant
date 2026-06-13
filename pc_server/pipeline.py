"""Voice Session Pipeline: Coordinates STT, LLM, Actions, and TTS via aiohttp WebSockets."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web

from shared.config import Config
from shared.logging_utils import get_logger
from shared.protocol import (
    MSG_STT_FINAL,
    MSG_TTS_END,
    MSG_UTTERANCE_END,
    MSG_WAKE,
    encode_json,
)

from .llm_action import ActionAwareLLM
from .stt import Transcriber
from .tts import TTS

log = get_logger("pipeline")


class VoiceSession:
    """Manages the full lifecycle of a single WebSocket voice client session."""

    def __init__(self, cfg: Config, ws: web.WebSocketResponse) -> None:
        self.cfg = cfg
        self.ws = ws

        # Instantiate server modules
        self.transcriber = Transcriber(cfg)
        self.llm = ActionAwareLLM(cfg)
        self.tts = TTS(cfg)

        # Session state tracking
        self.audio_buffer = bytearray()
        self.is_recording = False
        self._processing_task: asyncio.Task | None = None

    async def run(self) -> None:
        """Main framework listening loop matching the aiohttp entrypoint."""
        async for msg in self.ws:
            if msg.type == web.WSMsgType.TEXT:
                await self._handle_text(msg.data)
            elif msg.type == web.WSMsgType.BINARY:
                await self._handle_binary(msg.data)
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.ERROR):
                break

    async def _handle_text(self, text_data: str) -> None:
        """Processes structured pipeline JSON protocol events from the client."""
        try:
            data = json.loads(text_data)
            # Fix double serialization edge-cases if the client sent a stringified string
            if isinstance(data, str):
                data = json.loads(data)
        except json.JSONDecodeError:
            log.warning("Received invalid JSON on pipeline socket: %r", text_data)
            return

        if not isinstance(data, dict):
            log.warning("Parsed JSON data is type %s, expected dict. Data: %r", type(data), data)
            return

        msg_type = data.get("type")

        if msg_type == MSG_WAKE:
            log.info("Wake word detected! Readying client utterance audio buffer.")
            # Clear or interrupt any active speech tasks if the user cuts off the assistant
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                log.info("Active generation task interrupted by incoming wake event.")

            self.audio_buffer.clear()
            self.is_recording = True

        elif msg_type == MSG_UTTERANCE_END:
            if not self.is_recording:
                log.warning("Received utterance end signal, but recording state was inactive.")
                return

            log.info("Client utterance ended. Submitting buffer for inference.")
            self.is_recording = False
            
            # Extract current audio data snapshot and clear tracking buffer
            captured_audio = bytes(self.audio_buffer)
            self.audio_buffer.clear()

            # Schedule pipeline processing out of line to prevent blocking the socket listener
            self._processing_task = asyncio.create_task(
                self._process_and_respond(captured_audio)
            )

    async def _handle_binary(self, binary_data: bytes) -> None:
        """Ingests raw PCM audio streaming frames from the client connection."""
        if self.is_recording:
            self.audio_buffer.extend(binary_data)

    async def _process_and_respond(self, audio_data: bytes) -> None:
        """Executes full inference: STT -> Action-Aware LLM -> Windows SAPI5 TTS Playback."""
        try:
            # 1. Speech-To-Text Transcription via faster-whisper
            user_text = await self.transcriber.transcribe(audio_data)
            if not user_text.strip():
                log.info("STT returned empty string. Dropping execution pipeline turn.")
                # Force-signal the client that this turn is over so it unlocks its wake word engine
                await self.ws.send_str(encode_json({"type": MSG_TTS_END}))
                return

            # Echo text back to client interface
            await self.ws.send_str(encode_json({"type": MSG_STT_FINAL, "text": user_text}))

            # 2. LLM Processing & Action Dispatch Loop
            text_to_synthesize = ""
            async for kind, payload in self.llm.stream_reply(user_text):
                if kind == "action":
                    log.info("LLM triggered action tool execution: %s", payload.get("name"))
                    
                    action_result_chunks = []
                    async for chunk in self.llm.run_action(payload):
                        action_result_chunks.append(chunk)
                    
                    text_to_synthesize = "".join(action_result_chunks)
                    log.info("RAW TOOL CALL: %s | TOOL RESULT: %s", payload, text_to_synthesize)
                    break
                elif kind == "token":
                    text_to_synthesize += payload

            # 3. Text-To-Speech Generation using your local SAPI5 code
            if text_to_synthesize.strip():
                # Note: SAPI5 reads out full responses sequentially from a WAV file.
                # Since your tts.py reads chunks inside an internal WAV generator loop,
                # we call it directly and pass our active server socket object down to send chunks.
                await self.tts_stream_to_client(text_to_synthesize)

            # 4. signal client playback can close safely
            await self.ws.send_str(encode_json({"type": MSG_TTS_END}))
            log.info("Successfully finished execution pipeline turn.")

        except asyncio.CancelledError:
            log.info("Voice assistant generation processing task was cancelled.")
        except Exception as e:
            log.exception("Pipeline error during processing cycle: %s", e)

    async def tts_stream_to_client(self, text: str) -> None:
        """Generates SAPI5 speech and streams binary payload chunks straight over the socket."""
        loop = asyncio.get_running_loop()
        # Call your pyttsx3 generator wrapper method via executor
        wav_bytes = await loop.run_in_executor(None, self.tts._generate_wav, text)
        
        if not wav_bytes or len(wav_bytes) < 44:
            return

        # Skip over the 44-byte standard RIFF/WAV header to isolate the raw PCM frames
        pcm_payload = wav_bytes[44:]
        chunk_size = 2048

        # Stream chunks down the WebSocket immediately
        for i in range(0, len(pcm_payload), chunk_size):
            chunk = pcm_payload[i : i + chunk_size]
            await self.ws.send_bytes(chunk)