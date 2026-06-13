"""Shared utilities used by both the Pi client and the PC server."""
from .protocol import (
    MSG_WAKE,
    MSG_UTTERANCE_END,
    MSG_STT_FINAL,
    MSG_LLM_TOKEN,
    MSG_TTS_END,
    MSG_INTERRUPT,
    MSG_ERROR,
    encode_json,
    decode_json,
)
from .audio_utils import (
    SAMPLE_RATE,
    pcm_to_float32,
    float32_to_pcm16,
    resample,
)

__all__ = [
    "MSG_WAKE",
    "MSG_UTTERANCE_END",
    "MSG_STT_FINAL",
    "MSG_LLM_TOKEN",
    "MSG_TTS_END",
    "MSG_INTERRUPT",
    "MSG_ERROR",
    "encode_json",
    "decode_json",
    "SAMPLE_RATE",
    "pcm_to_float32",
    "float32_to_pcm16",
    "resample",
]
