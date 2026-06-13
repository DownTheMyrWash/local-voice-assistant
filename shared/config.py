"""Load .env once and expose typed config to both sides."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of where the script is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _get_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _get_list(key: str, default: str) -> list[str]:
    return [s.strip() for s in os.getenv(key, default).split(",") if s.strip()]


@dataclass(frozen=True)
class Config:
    pc_host: str
    pc_port: int
    sample_rate: int
    channels: int
    frame_duration_ms: int
    wake_phrase: str
    stop_phrases: list[str]
    log_level: str
    latency_log: str

    # PC-only fields (None on the Pi)
    whisper_model: str | None
    whisper_compute_type: str | None
    whisper_device: str | None
    ollama_host: str | None
    ollama_model: str | None
    llm_system_prompt: str | None
    tts_model: str | None
    tts_device: str | None
    tts_sample_rate: int | None
    chat_history_file: str | None


def load_config() -> Config:
    return Config(
        pc_host=os.getenv("PC_HOST", "127.0.0.1"),
        pc_port=_get_int("PC_PORT", 8765),
        sample_rate=_get_int("SAMPLE_RATE", 16000),
        channels=_get_int("CHANNELS", 1),
        frame_duration_ms=_get_int("FRAME_DURATION_MS", 30),
        wake_phrase=os.getenv("WAKE_PHRASE", "hey assistant"),
        stop_phrases=_get_list("STOP_PHRASES", "stop,cancel,never mind"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        latency_log=os.getenv("LATENCY_LOG", "latency.log"),
        whisper_model=os.getenv("WHISPER_MODEL"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE"),
        whisper_device=os.getenv("WHISPER_DEVICE"),
        ollama_host=os.getenv("OLLAMA_HOST"),
        ollama_model=os.getenv("OLLAMA_MODEL"),
        llm_system_prompt=os.getenv("LLM_SYSTEM_PROMPT"),
        tts_model=os.getenv("TTS_MODEL"),
        tts_device=os.getenv("TTS_DEVICE"),
        tts_sample_rate=_get_int("TTS_SAMPLE_RATE", 22050) or None,
        chat_history_file=os.getenv("CHAT_HISTORY_FILE"),
    )
