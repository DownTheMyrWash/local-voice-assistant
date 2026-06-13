"""Text-to-speech using Edge-TTS (Online) and Kokoro-ONNX (Offline) with Auto Fallback."""
from __future__ import annotations

import os
import io
import wave
import asyncio
import edge_tts
from pydub import AudioSegment
from kokoro_onnx import Kokoro as KokoroOnnx

from shared.logging_utils import get_logger

log = get_logger("tts")


class TTS:
    def __init__(self, cfg=None) -> None:
        self._cfg = cfg
        # Read the mode from environment, defaulting to 'auto'
        self.mode = os.getenv("TTS_MODE", "auto").lower()
        
        # Voice configuration
        self.online_voice = "en-US-BrianNeural"
        self.offline_voice = "am_liam"
        
        # Lazy load Kokoro only if needed to save startup memory/time
        self.kokoro = None

    def _init_kokoro(self) -> None:
        """Initializes the offline Kokoro engine using explicit absolute project paths."""
        if self.kokoro is None:
            log.info("Loading offline Kokoro model weights...")
            
            # Since tts.py lives inside pc_server/, we go up one level to find the root folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(current_dir)
            
            # Explicitly point to the assistant-v2 root folder files
            onnx_path = os.path.normpath(os.path.join(root_dir, "kokoro-v0_19.onnx"))
            voices_path = os.path.normpath(os.path.join(root_dir, "voices.bin"))
            
            log.info("Looking for ONNX at: %s", onnx_path)
            log.info("Looking for voices at: %s", voices_path)
            
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(
                    f"Could not find model weights at '{onnx_path}'. "
                    f"Please verify the file exists in your project root."
                )
            self.kokoro = KokoroOnnx(onnx_path, voices_path)

    async def _edge_synthesize(self, text: str) -> bytes:
        """Fetch MP3 audio from Edge TTS."""
        communicate = edge_tts.Communicate(text, self.online_voice)
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk["data"]
        return mp3_data

    def _generate_offline_wav(self, text: str) -> bytes:
        """Generate high-quality local PCM bytes using Kokoro."""
        self._init_kokoro()
        # speed=1.1 gives it a slightly more natural conversational pacing
        samples, sample_rate = self.kokoro.create(text, voice=self.offline_voice, speed=1.1)
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((samples * 32767).astype("<i2").tobytes())
        
        return wav_io.getvalue()

    def _generate_wav(self, text: str) -> bytes:
        """Orchestrates generation based on online/offline/auto rules."""
        
        # 1. Strict Offline Mode
        if self.mode == "offline":
            return self._generate_offline_wav(text)

        # 2. Try Online (For 'online' or 'auto' modes)
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._edge_synthesize(text), loop)
                # Apply a 4-second timeout to online requests so 'auto' doesn't hang
                mp3_bytes = future.result(timeout=4.0)
            else:
                mp3_bytes = loop.run_until_complete(self._edge_synthesize(text))

            if mp3_bytes:
                audio_seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
                wav_io = io.BytesIO()
                audio_seg.export(wav_io, format="wav")
                return wav_io.getvalue()

        except Exception as e:
            if self.mode == "online":
                log.error("Online TTS failed: %s. No fallback configured.", e)
                return b""
            log.warning("Online TTS failed or timed out. Falling back to offline Kokoro. Error: %s", e)

        # 3. Fallback to Offline Mode (For 'auto' mode when online fails)
        return self._generate_offline_wav(text)

    async def stream(self, text_input):
        """Accept a plain string or async generator, synthesize, stream audio chunks."""
        if hasattr(text_input, "__anext__"):
            fragments = []
            async for token in text_input:
                fragments.append(token)
            full_text = "".join(fragments)
        else:
            full_text = str(text_input)

        if not full_text.strip():
            return

        log.info("TTS synthesizing (Mode: %s): %r", self.mode, full_text[:100])

        audio_bytes = self._generate_wav(full_text)

        buffer = io.BytesIO(audio_bytes)
        chunk_size = 2048 
        while True:
            data = buffer.read(chunk_size)
            if not data:
                break
            yield data