"""aiohttp server entrypoint."""
from __future__ import annotations

from aiohttp import web, WSMsgType

from shared.config import load_config
from shared.logging_utils import get_logger
from shared.protocol import MSG_ERROR, encode_json

from .pipeline import VoiceSession
from .stt_vosk import init_vosk, ws_stt

log = get_logger("server")


async def health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def ws_voice(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=2**20)
    await ws.prepare(request)
    peer = request.remote or "unknown"
    log.info("Client connected: %s", peer)

    session = VoiceSession(load_config(), ws)
    try:
        await session.run()
    except Exception as e:
        log.exception("Session error: %s", e)
        try:
            await ws.send_str(encode_json({"type": MSG_ERROR, "message": str(e)}))
        except Exception:
            pass
    finally:
        await ws.close()
        log.info("Client disconnected: %s", peer)
    return ws


def make_app() -> web.Application:
    # Load Vosk model once at startup before the server begins accepting requests
    init_vosk()

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/ws/voice", ws_voice)
    app.router.add_get("/ws/stt", ws_stt)   # <-- new streaming STT endpoint
    return app


def main() -> None:
    cfg = load_config()
    app = make_app()
    log.info("Starting server on 0.0.0.0:%d", cfg.pc_port)
    web.run_app(app, host="0.0.0.0", port=cfg.pc_port, print=None)


if __name__ == "__main__":
    main()
