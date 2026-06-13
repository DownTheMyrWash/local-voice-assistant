"""Pi-side orchestrator — with streaming STT via Vosk."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
import wave  # Add this import

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config
from shared.logging_utils import get_logger
from shared.protocol import (
    MSG_TTS_END,
    MSG_UTTERANCE_END,
    MSG_WAKE,
    MSG_STT_FINAL,
    MSG_STT_RESULT,   # new: we send the transcript ourselves now
    encode_json,
)

from pi_client.audio_io import MicCapture, Speaker
from pi_client.wake_word import WakeWordDetector
from pi_client.vad import VoiceActivityDetector
from pi_client.ws_client import VoiceClient
from pi_client.stt_client import StreamingSTTClient

log = get_logger("pi-main")

def load_wake_sound(filepath: str) -> bytes:
    """Loads a WAV file into memory as raw PCM bytes."""
    try:
        with wave.open(filepath, 'rb') as wf:
            return wf.readframes(wf.getnframes())
    except Exception as e:
        log.error("Could not load wake sound %s: %s", filepath, e)
        return b""
    
async def main() -> None:
    
    cfg = load_config()
    mic = MicCapture(cfg)
    speaker = Speaker(cfg)
    wake = WakeWordDetector(cfg, mic=mic)
    vad = VoiceActivityDetector(cfg, aggressiveness=3)

    wake_sound_pcm = load_wake_sound(str(Path(__file__).parent / "wake_beep.wav"))

    # Pipeline WebSocket (LLM + TTS)
    client = VoiceClient(cfg)

    # Streaming STT WebSocket (Vosk — runs on the PC, separate endpoint)
    stt = StreamingSTTClient(cfg)

    tts_done = asyncio.Event()

    async def on_binary(pcm: bytes) -> None:
        speaker.play(pcm)

    async def on_text(msg: dict) -> None:
        kind = msg.get("type")
        if kind == MSG_STT_FINAL:
            log.info("STT Final (echo): %s", msg.get("text", ""))
        elif kind == MSG_TTS_END:
            log.info("Server finished TTS")
            tts_done.set()

    client.set_handlers(on_binary, on_text)

    # Start both WebSocket connections
    connect_task = asyncio.create_task(client.connect())
    await stt.connect()

    mic.start()
    speaker.start()
    wake.start()

    try:
        while True:
            log.info("Idle. Waiting for wake word...")
            await wake.wait_for_wake()

            log.info("Wake word detected! Listening...")
            
            # 1. Play the beep without stopping the mic hardware
            if wake_sound_pcm:
                speaker.play(wake_sound_pcm)
                
                # 2. Burn/Ignore mic frames for the exact duration of the beep
                # This prevents the beep from being sent to Vosk or triggering VAD
                beep_duration = 0.5  # Adjust this to match your WAV file duration
                start_beep_time = time.monotonic()
                
                log.info("Dropping mic frames while beep plays...")
                async for pcm in mic.frames():
                    if time.monotonic() - start_beep_time > beep_duration:
                        break

            # 3. Flush whatever is left in the buffer right before processing speech
            mic.clear()
            wake.suppress()

            # Notify pipeline server a new turn is starting
            await client.send_text(encode_json({"type": MSG_WAKE}))
            tts_done.clear()

            try:
                # Stream audio to BOTH the pipeline server and Vosk
                transcript = await _run_utterance(cfg, client, stt, mic, vad)

                if transcript.strip():
                    log.info("Transcript ready: %r", transcript)
                    await client.send_text(
                        encode_json({"type": MSG_STT_RESULT, "text": transcript})
                    )
                else:
                    log.info("Empty transcript — skipping turn.")

                await client.send_text(encode_json({"type": MSG_UTTERANCE_END}))

                if transcript.strip():
                    log.info("Waiting for assistant reply...")
                    try:
                        await asyncio.wait_for(tts_done.wait(), timeout=60.0)
                    except asyncio.TimeoutError:
                        log.warning("TTS done timeout — resuming anyway")

            finally:
                # --- FIX STARTS HERE ---
                log.info("Turn complete. Cleaning up buffers...")
                
                # 1. Give physical audio hardware a moment to fully quiet down
                await asyncio.sleep(0.4) 
                
                # 2. Flush the microphone buffer so old assistant audio is thrown out
                mic.clear() 
                
                # 3. Only now is it safe to let the wake word engine look at the mic
                wake.resume()

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        await client.stop()
        await stt.close()
        connect_task.cancel()
        mic.stop()
        speaker.stop()
        wake.stop()


async def _run_utterance(
    cfg, client: VoiceClient, stt: StreamingSTTClient, mic: MicCapture, vad: VoiceActivityDetector
) -> str:
    """Stream mic frames to Vosk and the pipeline server simultaneously.

    Returns the final transcript once VAD detects end of speech.
    Audio is still sent to the pipeline server as binary so it can be
    used for anything else (e.g. barge-in detection), but STT is handled
    entirely by Vosk now.
    """
    silence_frames_needed = int(0.8 / (cfg.frame_duration_ms / 1000))
    trailing_silence = 0
    speaking = False

    start_time = time.monotonic()
    max_duration = 10.0

    async for pcm in mic.frames():
        if time.monotonic() - start_time > max_duration:
            log.info("Max utterance duration reached (10s).")
            break

        # Send to pipeline server (binary framing, kept for protocol compatibility)
        await client.send_binary(pcm)

        # Stream to Vosk in real time
        await stt.feed(pcm)

        # VAD to detect end of speech
        decisions = vad.feed(pcm)
        for is_speech in decisions:
            if is_speech:
                speaking = True
                trailing_silence = 0
            elif speaking:
                trailing_silence += 1

        if speaking and trailing_silence >= silence_frames_needed:
            log.info("VAD: end of speech detected.")
            break

    # Vosk has been receiving audio this whole time — just flush and return
    transcript = await stt.finalize()
    return transcript


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
