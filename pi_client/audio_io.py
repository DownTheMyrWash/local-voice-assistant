"""Audio capture and playback using PyAudio."""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import AsyncIterator

import pyaudio
import numpy as np

from shared.config import Config
from shared.logging_utils import get_logger

log = get_logger("audio_io")


class MicCapture:
    """Captures mic audio using PyAudio and exposes it as an async iterator."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=0)
        # Separate subscriber queues for anything listening to the mic (like the wake word engine)
        self._subscribers: list[asyncio.Queue[bytes]] = []
        self._p = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()

    def _get_webcam_device_index(self) -> int | None:
        # Use system default (None) so PipeWire shares input across scripts cleanly
        return None

    def subscribe(self) -> asyncio.Queue[bytes]:
        """Allows external modules to tap into the microphone audio stream."""
        sub_q = asyncio.Queue(maxsize=0)
        self._subscribers.append(sub_q)
        return sub_q

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        blocksize = int(self.cfg.sample_rate * self.cfg.frame_duration_ms / 1000)

        def _callback(in_data, frame_count, time_info, status):
            if status and self._loop:
                self._loop.call_soon_threadsafe(log.debug, "mic status: %s", status)
            if in_data and self._loop:
                # Dispatch audio chunk to the primary queue
                self._loop.call_soon_threadsafe(self._q.put_nowait, in_data)
                # Dispatch the exact same chunk to any active subscribers
                for sub_q in self._subscribers:
                    self._loop.call_soon_threadsafe(sub_q.put_nowait, in_data)
            return (None, pyaudio.paContinue)

        device_index = self._get_webcam_device_index()
        
        target_rate = self.cfg.sample_rate
        if device_index is not None:
            try:
                dev_info = self._p.get_device_info_by_index(device_index)
                target_rate = int(dev_info.get("defaultSampleRate", target_rate))
            except Exception:
                pass

        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=target_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=blocksize,
            stream_callback=_callback,
        )
        log.info("Mic capture started on system default device (%d Hz, 1 ch)", target_rate)

    async def frames(self) -> AsyncIterator[bytes]:
        """Async iterator yielding raw PCM frames as they arrive."""
        while not self._stop.is_set():
            try:
                # Use a small timeout so the loop stays responsive to cancellations
                frame = await asyncio.wait_for(self._q.get(), timeout=0.1)
                yield frame
                self._q.task_done()
            except asyncio.TimeoutError:
                continue

    def clear(self) -> None:
        """Purges any backlogged audio frames from the primary capture queue."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except Exception:
                break
        log.debug("Primary microphone frame queue cleared.")

    def stop(self) -> None:
        self._stop.set()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._p.terminate()


class Speaker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._p = pyaudio.PyAudio()
        self._tts_rate = cfg.tts_sample_rate or 22050
        self._stream: pyaudio.Stream | None = None

    def _get_bt_device_index(self) -> int | None:
        """Find the Bluetooth/PipeWire output device."""
        for i in range(self._p.get_device_count()):
            info = self._p.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if info.get("maxOutputChannels", 0) > 0 and (
                "bluez" in name or "bluetooth" in name or "pulse" in name
            ):
                return i
        return None

    def start(self) -> None:
        device_index = self._get_bt_device_index()
        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._tts_rate,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=1024,
        )
        log.info("Speaker ready on device %s (%d Hz)", device_index, self._tts_rate)

    def stop(self) -> None:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._p.terminate()

    def play(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes or self._stream is None:
            return
        try:
            self._stream.write(pcm_bytes)
        except Exception as e:
            log.error("Speaker playback failed: %s", e)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return True

    def clear(self) -> None:
        pass