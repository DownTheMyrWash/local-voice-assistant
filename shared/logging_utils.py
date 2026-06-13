"""Lightweight logging + per-stage latency tracking."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from .config import load_config

_cfg = load_config()
LOG_FILE = Path(_cfg.latency_log)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = getattr(logging, _cfg.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


class LatencyTimer:
    """Context manager: `with LatencyTimer("stt"): ...` logs the elapsed ms."""

    def __init__(self, stage: str, logger: logging.Logger | None = None) -> None:
        self.stage = stage
        self.logger = logger or get_logger("latency")
        self._t0 = 0.0
        self._elapsed_ms = 0.0

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms

    def __enter__(self) -> "LatencyTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        line = f"{self.stage} elapsed={self._elapsed_ms:.1f}ms"
        self.logger.info(line)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
