"""Pi-side orchestrator."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config
from shared.logging_utils import get_logger
from shared.protocol import (
    MSG_INTERRUPT,
    MSG_TTS_END,
    MSG_UTTERANCE_END,
    MSG_WAKE,
    MSG_STT_FINAL,
    encode_json,
)
from shared.audio_utils import SAMPLE_RATE

from pi_client.audio_io import MicCapture, Speaker
from pi_client.wake_word import WakeWordDetector
from pi_client.vad import VoiceActivityDetector
from pi_client.ws_client import VoiceClient

log = get_logger("pi-main")


async def main() -> None:
    cfg = load_config()
    mic = MicCapture(cfg)
    speaker = Speaker(cfg)
    
    # Pass initialized mic reference into the wake word detector to use the shared pipeline
    wake = WakeWordDetector(cfg, mic=mic)
    
    vad = VoiceActivityDetector(cfg, aggressiveness=3)
    client = VoiceClient(cfg)

    tts_done = asyncio.Event()

    async def on_binary(pcm: bytes) -> None:
        speaker.play(pcm)

    async def on_text(msg: dict) -> None:
        kind = msg.get("type")
        if kind == MSG_STT_FINAL:
            text = msg.get("text", "")
            log.info("STT Final: %s", text)
        elif kind == MSG_TTS_END:
            log.info("Server finished sending TTS audio chunks")
            tts_done.set()

    client.set_handlers(on_binary, on_text)

    # Start network client background loop
    connect_task = asyncio.create_task(client.connect())

    # Start local audio infrastructure layers
    mic.start()
    speaker.start()
    wake.start()

    try:
        while True:
            log.info("Idle. Say the wake word to begin.")
            await wake.wait_for_wake()
            
            # Wipes out backlogged trailing audio (the spoken word 'Jarvis') 
            # so the VAD starts analyzing a completely clean audio stream
            mic.clear()

            log.info("Wake word detected! Listening for utterance...")
            wake.suppress()

            # Tell server we are beginning streaming pipeline input
            await client.send_text(encode_json({"type": MSG_WAKE}))
            tts_done.clear()

            try:
                # Blocks here, reading mic frames and streaming until user stops talking
                await _run_utterance(cfg, client, mic, vad)
                
                # Close out client recording handle
                await client.send_text(encode_json({"type": MSG_UTTERANCE_END}))
                log.info("Finished utterance. Waiting for assistant reply...")

                # Wait until server sends MSG_TTS_END
                try:
                    await asyncio.wait_for(tts_done.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    log.warning("TTS done timeout — resuming anyway")

            finally:
                wake.resume()

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        await client.stop()
        connect_task.cancel()
        mic.stop()
        speaker.stop()
        wake.stop()


async def _run_utterance(cfg, client, mic, vad):
    """Stream mic audio until VAD detects sustained silence, or hard timeout hits."""
    silence_frames_needed = int(0.8 / (cfg.frame_duration_ms / 1000))
    trailing_silence = 0
    speaking = False

    start_time = time.monotonic()
    max_duration = 10.0

    async for pcm in mic.frames():
        if time.monotonic() - start_time > max_duration:
            log.info("Max utterance duration reached (10s). Forcing stop.")
            return

        await client.send_binary(pcm)
        decisions = vad.feed(pcm)
        for is_speech in decisions:
            if is_speech:
                speaking = True
                trailing_silence = 0
            elif speaking:
                trailing_silence += 1

        if speaking and trailing_silence >= silence_frames_needed:
            log.info("VAD: End of speech detected natively via silence metrics.")
            return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass