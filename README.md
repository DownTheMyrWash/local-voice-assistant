# Voice Assistant v2

Two-part local voice assistant. **Raspberry Pi 4** captures audio and plays responses. **Windows PC** runs the heavy ML (STT, LLM, TTS). Communication over WebSocket on your LAN.

```
                        LAN WebSocket (PCM audio + control JSON)
                  ┌────────────────────────────────────────────────┐
                  │                                                │
[USB Mic] → [Precise] → [sounddevice]   ═══════►   [aiohttp server]
 "hey assistant"                                │
                                                ├─→ faster-whisper (STT)
                                                ├─→ Ollama (LLM, streaming)
                                                └─→ Coqui TTS (streaming)
                  ◄════════  TTS audio chunks + control  ◄─────────┘
[3.5mm Jack / BT speaker] ← [sounddevice playback]
```

## Quick start

### PC (Windows)

```powershell
# 1. Install Ollama and pull a model
ollama pull llama3.1:8b

# 2. Set up Python env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-pc.txt

# 3. Configure
cp .env.example .env
# edit .env: set PC_HOST to your PC's LAN IP (the one the Pi can reach)

# 4. Run
python -m pc-server.main
```

### Raspberry Pi 4

```bash
sudo apt install python3-pip portaudio19-dev libasound2-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi.txt

# Configure
cp .env.example .env
# edit .env: set PC_HOST to your PC's LAN IP

# Run
python -m pi-client.main
```

## Wake word

Default: **"hey assistant"** or just **"assistant"**. The model is at `models/hey-assistant.pb`. To retrain for different words, see [docs/wake_word_training.md](docs/wake_word_training.md).

## Stop word

Saying **"stop"**, **"cancel"**, or **"never mind"** mid-response cancels the current TTS playback immediately.

## Actions

A few starter actions ship in `pc-server/actions/`:

- `get_time` — "what time is it"
- `open_youtube` — "open youtube"
- `tell_joke` — "tell me a joke"
- `stop` — interrupts current playback

Add your own by dropping a new file in that directory. See [docs/actions.md](docs/actions.md).

## Layout

```
assistant-v2/
├── pi-client/       # Runs on the Pi
├── pc-server/       # Runs on the PC
│   └── actions/     # Drop-in action plugins
├── shared/          # Imported by both sides
├── scripts/         # Setup helpers (BT pairing, etc.)
├── docs/            # Training, actions, troubleshooting
└── requirements-*.txt
```

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).
