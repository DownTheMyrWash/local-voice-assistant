"""WebRTC VAD wrapper with enhanced noise and silence filtering."""
from __future__ import annotations

from collections import deque
import numpy as np
import webrtcvad

from shared.config import Config


class VoiceActivityDetector:
    """Buffers audio in 30ms frames, applies an RMS energy threshold, 
    and uses a sliding window to smooth out erratic background noise triggers.
    """

    def __init__(self, cfg: Config, aggressiveness: int = 3, rms_threshold: float = 300.0, history_frames: int = 10) -> None:
        if cfg.frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if cfg.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("sample_rate must be 8000, 16000, 32000, or 48000")
        
        self.cfg = cfg
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_bytes = int(cfg.sample_rate * cfg.frame_duration_ms / 1000) * 2  # int16 mono
        self._buffer = bytearray()

        # --- NEW TUNING PARAMETERS FOR NOISE REJECTION ---
        # 1. Anything below this raw volume level is ignored as background noise
        # Typical range for 16-bit PCM: 100 (very sensitive) to 800 (very strict)
        self.rms_threshold = rms_threshold 

        # 2. Sliding window to smooth out sudden clicks, pops, or noise spikes
        self.history_frames = history_frames
        self._speech_history = deque(maxlen=history_frames)

    def feed(self, pcm16: bytes) -> list[bool]:
        """Consume PCM bytes. Returns one smoothed decision per complete frame."""
        self._buffer.extend(pcm16)
        decisions: list[bool] = []
        
        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            
            # --- LAYER 1: RMS Energy Threshold ---
            audio_np = np.frombuffer(frame, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
            
            if rms < self.rms_threshold:
                # Too quiet to be human speech; bypass WebRTC VAD entirely
                is_speech = False
            else:
                # --- LAYER 2: WebRTC VAD Check ---
                is_speech = self._vad.is_speech(frame, self.cfg.sample_rate)
            
            # --- LAYER 3: Sliding Window Smoothing ---
            self._speech_history.append(is_speech)
            
            # Calculate the ratio of "speech" frames in our recent history
            speech_ratio = sum(self._speech_history) / len(self._speech_history)
            
            # If the history is mostly speech, consider the current state as active speech.
            # For 10 frames (300ms total), requiring > 30% means at least 4 frames must be True.
            # This instantly drops random 1-frame background clicks/pops.
            smoothed_decision = speech_ratio > 0.3
            
            decisions.append(smoothed_decision)
            
        return decisions

    def reset(self) -> None:
        """Clear buffers and history for a clean slate on a new utterance."""
        self._buffer.clear()
        self._speech_history.clear()