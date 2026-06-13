"""openWakeWord-based wake word detection."""
from __future__ import annotations

import asyncio
import threading
import time
import queue
import numpy as np
from openwakeword.model import Model

from shared.config import Config
from shared.logging_utils import get_logger
from pi_client.audio_io import MicCapture

log = get_logger("wake_word")

SENSITIVITY = 0.5


class WakeWordDetector:
    def __init__(self, cfg: Config, mic: MicCapture) -> None:
        self.cfg = cfg
        self.mic = mic
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake_event = asyncio.Event()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._suppressed = False
        self._resume_after: float = 0.0
        self._model = None
        
        # Subscribe to the shared mic capture pipeline stream
        self._audio_queue = self.mic.subscribe()
        self._audio_buffer = bytearray()

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            self._model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            log.info("Wake word engine initialized using shared audio pipeline subscription")

            self._thread = threading.Thread(target=self._detect_loop, daemon=True)
            self._thread.start()
            log.info("openWakeWord engine started")

        except Exception as e:
            log.warning("Could not launch openWakeWord (%s); using REPL fallback", e)
            self._thread = threading.Thread(target=self._repl_loop, daemon=True)
            self._thread.start()

    def suppress(self) -> None:
        """Pause wake word scoring while pipeline is active."""
        self._suppressed = True

    def resume(self, cooldown: float = 1.5) -> None:
        # Drain stale audio
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Exception:
                break
        self._audio_buffer.clear()

        # Reinitialize the model to wipe its internal context buffer
        try:
            self._model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            log.info("Wake word model reinitialized for next turn.")
        except Exception as e:
            log.warning("Could not reinitialize wake word model: %s", e)

        self._resume_after = time.monotonic() + cooldown
        self._suppressed = False

    def _detect_loop(self) -> None:
        self._suppressed = False
        
        # openWakeWord works best when fed slices representing 1280 samples at 16000Hz
        target_chunk_samples = 1280
        native_rate = self.cfg.sample_rate
        
        # Calculate how many bytes we need to accumulate from the mic based on native rate
        # (samples * native_rate / 16000) * 2 bytes per sample
        sample_factor = native_rate // 16000
        required_bytes = target_chunk_samples * sample_factor * 2

        while not self._stop.is_set():
            try:
                # Safely pull raw chunks from the async queue via a future threadsafe call
                future = asyncio.run_coroutine_threadsafe(self._audio_queue.get(), self._loop)
                chunk = future.result(timeout=0.2)
                
                if self._suppressed or time.monotonic() < self._resume_after:
                    continue

                self._audio_buffer.extend(chunk)

                # Process chunks once we accumulate enough data to match openWakeWord's context window
                while len(self._audio_buffer) >= required_bytes:
                    raw_block = self._audio_buffer[:required_bytes]
                    del self._audio_buffer[:required_bytes]

                    audio = np.frombuffer(raw_block, dtype=np.int16)
                    
                    # Downsample using NumPy slicing if the hardware is running higher than 16kHz
                    if sample_factor > 1:
                        audio = audio[::sample_factor]

                    prediction = self._model.predict(audio)

                    for name, score in prediction.items():
                        if score >= SENSITIVITY:
                            log.info("Wake word detected: %s (score=%.2f)", name, score)
                            if self._loop is not None:
                                self._loop.call_soon_threadsafe(self._wake_event.set)
                            break

            except queue.Empty:
                continue
            except Exception as e:
                if not self._stop.is_set():
                    log.error("Wake word detect error: %s", e)
                break

    def stop(self) -> None:
        self._stop.set()

    def _repl_loop(self) -> None:
        print("\n>>> REPL MODE ACTIVE: Press ENTER to trigger the assistant <<<\n", flush=True)
        while True:
            try:
                input()
                print("\n[REPL] Enter key detected! Triggering wake event...", flush=True)
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._wake_event.set)
            except EOFError:
                return

    async def wait_for_wake(self) -> None:
        await self._wake_event.wait()
        self._wake_event.clear()