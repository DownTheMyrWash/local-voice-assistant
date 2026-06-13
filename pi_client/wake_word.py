"""openWakeWord-based wake word detection."""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
import numpy as np
from openwakeword.model import Model

from shared.config import Config
from shared.logging_utils import get_logger
from pi_client.audio_io import MicCapture

log = get_logger("wake_word")

SENSITIVITY = 0.2


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
        self._model_lock = threading.Lock()

        # Subscribe to the audio stream
        self._audio_queue = self.mic.subscribe()
        
        # Guarded by self._model_lock to ensure thread-safety across reset/detection boundaries
        self._audio_buffer = bytearray()

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            self._model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            log.info("Wake word engine initialized")
            self._thread = threading.Thread(target=self._detect_loop, daemon=True)
            self._thread.start()
            log.info("openWakeWord engine started")
        except Exception as e:
            log.warning("Could not launch openWakeWord (%s); using REPL fallback", e)
            self._thread = threading.Thread(target=self._repl_loop, daemon=True)
            self._thread.start()

    def suppress(self) -> None:
        self._suppressed = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._wake_event.clear)

    def resume(self, cooldown: float = 1.5) -> None:
        """Thread-safe state reset executed from the main async loop."""
        self._suppressed = True  # Stop processing incoming chunks immediately

        # 1. Clear out the private thread-safe background processing buffer
        with self._model_lock:
            self._audio_buffer.clear()
            try:
                self._model = Model(
                    wakeword_models=["hey_jarvis"],
                    inference_framework="onnx",
                )
                log.info("Wake word model reinitialized for next turn.")
            except Exception as e:
                log.warning("Could not reinitialize wake word model: %s", e)

        # 2. Schedule a clean drainage of the async queue on the main loop thread
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._drain_async_queue)

        # 3. Set the cooldown timing window
        self._resume_after = time.monotonic() + cooldown
        self._suppressed = False

    def _drain_async_queue(self) -> None:
        """Helper to drop queue items on the thread where the loop resides."""
        try:
            while not self._audio_queue.empty():
                self._audio_queue.get_nowait()
        except Exception:
            pass

    def _detect_loop(self) -> None:
        self._suppressed = False

        target_chunk_samples = 1280
        native_rate = self.cfg.sample_rate
        sample_factor = native_rate // 16000
        required_bytes = target_chunk_samples * sample_factor * 2

        while not self._stop.is_set():
            try:
                # Use threadsafe execution bridge to fetch fresh microphone frames
                future = asyncio.run_coroutine_threadsafe(
                    self._audio_queue.get(), self._loop
                )
                chunk = future.result(timeout=0.2)
            except concurrent.futures.TimeoutError:
                continue
            except Exception as e:
                if not self._stop.is_set():
                    log.warning("Audio queue read failed (continuing): %s", e)
                continue

            # --- CRITICAL CORRECTION HERE ---
            # If suppressed or cooling down, discard the audio completely 
            # and do not pass it down to the window matching blocks.
            if self._suppressed or time.monotonic() < self._resume_after:
                with self._model_lock:
                    self._audio_buffer.clear()
                continue

            with self._model_lock:
                self._audio_buffer.extend(chunk)
                
                # Process window blocks once requirements are satisfied
                while len(self._audio_buffer) >= required_bytes:
                    raw_block = self._audio_buffer[:required_bytes]
                    del self._audio_buffer[:required_bytes]

                    # --- RE-CHECK SUPPRESSION RE-ENTRANCY ---
                    if self._suppressed or time.monotonic() < self._resume_after:
                        continue

                    audio = np.frombuffer(raw_block, dtype=np.int16)
                    if sample_factor > 1:
                        audio = audio[::sample_factor]

                    if self._model is None:
                        continue

                    try:
                        prediction = self._model.predict(audio)
                    except Exception as e:
                        log.warning("Wake word predict error (skipping frame): %s", e)
                        continue

                    for name, score in prediction.items():
                        if score >= SENSITIVITY:
                            # DOUBLE CHECK: Ensure we didn't get suppressed while openwakeword was calculating
                            if not self._suppressed and time.monotonic() >= self._resume_after:
                                log.info("Wake word detected: %s (score=%.2f)", name, score)
                                if self._loop is not None:
                                    self._loop.call_soon_threadsafe(self._wake_event.set)
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