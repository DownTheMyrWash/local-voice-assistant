"""Smoke tests for the WebSocket protocol and the action registry.

Runs without any heavy ML models — only Python imports.

    python -m scripts.verify_protocol
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import protocol
from shared.config import load_config
from pc_server.actions import get_registry
from pc_server.actions.builtins import register_default_actions


def test_protocol_roundtrip() -> None:
    msgs = [
        {"type": protocol.MSG_WAKE},
        {"type": protocol.MSG_UTTERANCE_END},
        {"type": protocol.MSG_INTERRUPT, "phrase": "stop"},
        {"type": protocol.MSG_STT_FINAL, "text": "hello"},
        {"type": protocol.MSG_LLM_TOKEN, "text": "hi"},
        {"type": protocol.MSG_TTS_END},
        {"type": protocol.MSG_ERROR, "message": "boom"},
    ]
    for m in msgs:
        encoded = protocol.encode_json(m)
        decoded = protocol.decode_json(encoded)
        assert decoded["type"] == m["type"], decoded
        # Optional fields preserved
        for k, v in m.items():
            if k == "type":
                continue
            assert decoded.get(k) == v, (k, decoded)
    print("PASS protocol roundtrip")


def test_action_registry() -> None:
    reg = get_registry()
    for name in ("get_time", "open_youtube", "tell_joke"):
        a = reg.get(name)
        assert a is not None, f"missing action: {name}"
        assert a.name == name
        assert a.description
    specs = reg.tool_specs()
    assert len(specs) == len(reg._actions)
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["name"]
    print(f"PASS action registry ({len(specs)} actions)")


def test_audio_utils() -> None:
    import numpy as np
    from shared.audio_utils import pcm_to_float32, float32_to_pcm16, resample

    f = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm = float32_to_pcm16(f)
    f2 = pcm_to_float32(pcm)
    assert np.allclose(f, f2, atol=1e-3), (f, f2)

    big = np.sin(np.linspace(0, 6.28, 16000)).astype(np.float32)
    r = resample(big, 16000, 8000)
    assert len(r) == 8000
    r2 = resample(big, 16000, 16000)
    assert np.allclose(big, r2)
    print("PASS audio_utils")


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg.sample_rate == 16000
    assert cfg.pc_port > 0
    print(f"PASS config (PC at {cfg.pc_host}:{cfg.pc_port}, wake='{cfg.wake_phrase}')")


if __name__ == "__main__":
    register_default_actions()
    test_protocol_roundtrip()
    test_action_registry()
    test_audio_utils()
    test_config_loads()
    print("\nAll smoke tests passed.")
