"""Audio helpers shared by both sides.

The on-the-wire format is always:
    16-bit signed little-endian PCM, mono, 16kHz
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000  # Whisper native


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw int16 PCM bytes to a float32 numpy array in [-1, 1]."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert a float32 numpy array in [-1, 1] to int16 PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling. Good enough for VAD and basic STT handoff.

    For higher-quality resampling, swap in `scipy.signal.resample_poly`.
    Kept dependency-free for the Pi.
    """
    if src_rate == dst_rate:
        return samples
    duration = len(samples) / src_rate
    target_len = int(duration * dst_rate)
    if target_len <= 1:
        return samples[:1]
    src_x = np.linspace(0, duration, num=len(samples), endpoint=False)
    dst_x = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(samples.dtype)
