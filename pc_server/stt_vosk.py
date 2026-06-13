"""Vosk streaming STT server.

Runs a WebSocket endpoint at /ws/stt alongside the existing voice pipeline.
The Pi streams raw PCM frames in real time; Vosk transcribes as they arrive.
When the Pi sends a plain-text "FINALIZE" message, we flush and return the
best final transcript accumulated so far, then reset for the next utterance.

Vosk model is loaded once at startup and shared across all connections.

Setup:
    pip install vosk
    # Download a model from https://alphacephei.com/vosk/models
    # Recommended: vosk-model-en-us-0.22  (good accuracy, ~1.8 GB)
    # Lightweight:  vosk-model-small-en-us-0.15 (fast, ~40 MB)
    # Set env var:  VOSK_MODEL_PATH=/path/to/vosk-model-en-us-0.22
"""
from __future__ import annotations

import json
import os

from aiohttp import web

from shared.logging_utils import get_logger

log = get_logger("stt_vosk")

# Loaded once at startup via init_vosk()
_recognizer_model = None


def init_vosk() -> None:
    """Load the Vosk model. Call once before starting the server."""
    global _recognizer_model
    from vosk import Model  # type: ignore

    model_path = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Vosk model not found at '{model_path}'. "
            "Download one from https://alphacephei.com/vosk/models and set "
            "VOSK_MODEL_PATH to point to the extracted folder."
        )
    log.info("Loading Vosk model from %s ...", model_path)
    _recognizer_model = Model(model_path)
    log.info("Vosk model loaded.")


async def ws_stt(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler: stream PCM in, get transcript out.

    Protocol (client → server):
        binary frames  : raw PCM s16le, 16000 Hz, mono
        text "FINALIZE": flush and return final transcript, then reset

    Protocol (server → client):
        text JSON {"type": "partial", "text": "..."}   — intermediate results
        text JSON {"type": "final",   "text": "..."}   — response to FINALIZE
    """
    from vosk import KaldiRecognizer  # type: ignore

    if _recognizer_model is None:
        raise RuntimeError("Vosk model not initialised — call init_vosk() first.")

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    peer = request.remote or "unknown"
    log.info("STT client connected: %s", peer)

    # 16 kHz is what Vosk expects; Pi must send at this rate
    rec = KaldiRecognizer(_recognizer_model, 16000)
    rec.SetWords(True)

    # Accumulate final phrases across the utterance so FINALIZE returns
    # the complete sentence even if Vosk emitted several final chunks.
    accumulated_finals: list[str] = []

    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            pcm = msg.data
            if rec.AcceptWaveform(pcm):
                # Vosk is confident this phrase is complete
                result = json.loads(rec.Result())
                phrase = result.get("text", "").strip()
                if phrase:
                    accumulated_finals.append(phrase)
                    log.debug("Vosk final phrase: %r", phrase)
                    await ws.send_str(json.dumps({"type": "partial", "text": phrase}))

            else:
                # In-progress partial — useful for UI feedback but we don't
                # feed it to the LLM
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text:
                    await ws.send_str(json.dumps({"type": "partial", "text": partial_text}))

        elif msg.type == web.WSMsgType.TEXT:
            if msg.data.strip().upper() == "FINALIZE":
                # Flush whatever Vosk is still holding
                result = json.loads(rec.FinalResult())
                tail = result.get("text", "").strip()
                if tail:
                    accumulated_finals.append(tail)

                full_transcript = " ".join(accumulated_finals).strip()
                log.info("STT finalized: %r", full_transcript)
                await ws.send_str(json.dumps({"type": "final", "text": full_transcript}))

                # Reset for the next utterance on this same connection
                rec = KaldiRecognizer(_recognizer_model, 16000)
                rec.SetWords(True)
                accumulated_finals.clear()

        elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
            break

    log.info("STT client disconnected: %s", peer)
    return ws