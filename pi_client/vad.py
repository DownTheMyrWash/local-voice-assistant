"""WebRTC VAD wrapper.

Detects whether a frame of audio contains speech. Used to auto-end
the utterance when the user stops talking (no need to press a button).
"""
from __future__ import annotations

from typing import Iterable

import webrtcvad

from shared.config import Config


class VoiceActivityDetector:
    """Buffers audio in 30ms frames and reports whether recent activity
    includes speech."""

    def __init__(self, cfg: Config, aggressiveness: int = 3) -> None:
        # webrtcvad only accepts 10/20/30 ms frames at 8/16/32/48 kHz
        if cfg.frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if cfg.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("sample_rate must be 8000, 16000, 32000, or 48000")
        self.cfg = cfg
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_bytes = int(cfg.sample_rate * cfg.frame_duration_ms / 1000) * 2  # int16 mono
        self._buffer = bytearray()

    def feed(self, pcm16: bytes) -> list[bool]:
        """Consume PCM bytes. Returns one decision per complete frame."""
        self._buffer.extend(pcm16)
        decisions: list[bool] = []
        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            decisions.append(self._vad.is_speech(frame, self.cfg.sample_rate))
        return decisions

    def reset(self) -> None:
        self._buffer.clear()
