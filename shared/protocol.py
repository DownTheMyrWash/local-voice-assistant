"""WebSocket message protocol shared by the Pi client and PC server.

Control messages are JSON over the WebSocket text channel. Audio is sent
as binary frames (raw 16-bit, 16kHz, mono PCM).

Wire format:
    Pi -> PC:
        text  : {"type": "wake"}                          start of utterance
        binary: <PCM bytes>                               streaming audio
        text  : {"type": "utterance_end"}                 end of utterance
        text  : {"type": "interrupt", "phrase": "stop"}   user spoke stop word

    PC -> Pi:
        text  : {"type": "stt_final", "text": "..."}       transcription
        text  : {"type": "llm_token", "text": "..."}       streamed token
        binary: <PCM bytes>                               streamed TTS audio
        text  : {"type": "tts_end"}                        playback complete
        text  : {"type": "error", "message": "..."}        error
"""
from __future__ import annotations

import json
from typing import Any

# Message type constants
MSG_WAKE = "wake"
MSG_UTTERANCE_END = "utterance_end"
MSG_STT_FINAL = "stt_final"
MSG_LLM_TOKEN = "llm_token"
MSG_TTS_END = "tts_end"
MSG_INTERRUPT = "interrupt"
MSG_ERROR = "error"


def encode_json(payload: dict[str, Any]) -> str:
    """Serialize a control message. Drops None values for compactness."""
    return json.dumps({k: v for k, v in payload.items() if v is not None})


def decode_json(text: str) -> dict[str, Any]:
    """Parse a control message. Raises on malformed input."""
    obj = json.loads(text)
    if not isinstance(obj, dict) or "type" not in obj:
        raise ValueError(f"Invalid message: missing 'type' field in {text!r}")
    return obj
