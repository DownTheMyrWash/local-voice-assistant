"""CLI entrypoint: `python -m pc_server.main` or `python -m pc_server.main --mock`."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# =====================================================================
# FORCE ENVIRONMENT PATH ISOLATION FOR WINDOWS ERROR 127
# =====================================================================
if sys.platform == "win32":
    # 1. Force Python to check site-packages first for binary DLLs
    v_env_site = Path(__file__).resolve().parent.parent / "pcenv" / "Lib" / "site-packages"
    if v_env_site.exists():
        os.add_dll_directory(str(v_env_site))
        
    # 2. Strip global NVIDIA/Toolkit paths from this specific running session's PATH.
    # This prevents Windows from pulling broken global DLLs into PyTorch's memoryspace.
    clean_paths = []
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        # Drop paths containing NVIDIA or CUDA Toolkit links
        if "nvidia" not in path_dir.lower() and "cuda" not in path_dir.lower():
            clean_paths.append(path_dir)
            
    os.environ["PATH"] = os.pathsep.join(clean_paths)
# =====================================================================

# Allow `python pc_server/main.py` style invocation as well as `python -m pc_server.main`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="Voice assistant PC server")
    ap.add_argument("--mock", action="store_true",
                    help="Use mock TTS (no model load). Useful for protocol testing.")
    ap.add_argument("--host", default=None, help="Override PC_HOST")
    ap.add_argument("--port", type=int, default=None, help="Override PC_PORT")
    args = ap.parse_args()

    if args.mock:
        os.environ["TTS_MODEL"] = "__mock__"
    if args.host:
        os.environ["PC_HOST"] = args.host
    if args.port:
        os.environ["PC_PORT"] = str(args.port)

    from pc_server.server import main as server_main
    server_main()


if __name__ == "__main__":
    main()