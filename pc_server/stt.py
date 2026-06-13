"""STT wrapper around `faster-whisper`.

Loaded lazily on first request to keep server startup fast and idle memory low.
"""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Iterable

import numpy as np

from shared.config import Config
from shared.audio_utils import pcm_to_float32, SAMPLE_RATE
from shared.logging_utils import get_logger, LatencyTimer

log = get_logger("stt")


class Transcriber:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._model = None  # lazy

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # heavy import

        device = "auto"
        compute_type = self._cfg.whisper_compute_type or "int8"
        log.info(
            "Loading Whisper model=%s compute=%s device=%s",
            self._cfg.whisper_model, compute_type, device,
        )
        self._model = WhisperModel(
            self._cfg.whisper_model or "base.en",
            device=device,
            compute_type=compute_type,
        )
        log.info("Whisper loaded")

    async def transcribe(self, pcm_bytes: bytes) -> str:
        """Transcribe a complete utterance. Returns the final text."""
        self._load()
        if not pcm_bytes:
            return ""
        audio = pcm_to_float32(pcm_bytes)
        loop = asyncio.get_running_loop()
        with LatencyTimer("stt", log):
            segments, _ = await loop.run_in_executor(
                None, partial(self._transcribe_sync, audio)
            )
        text = "".join(seg.text for seg in segments).strip()
        log.info("STT: %r", text)
        return text

    def _transcribe_sync(self, audio: np.ndarray) -> Iterable:
        # vad_filter skips silence segments -> faster, more accurate
        return self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
