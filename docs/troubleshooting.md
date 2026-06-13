# Troubleshooting

## PC server won't start

- **Port in use**: change `PC_PORT` in `.env`
- **Ollama not running**: `ollama serve` (usually auto-starts)
- **Ollama model missing**: `ollama pull model_name`
- **CUDA errors**: set `WHISPER_DEVICE=cpu` and `TTS_DEVICE=cpu` in `.env`

## Pi can't connect to PC

1. **Same network?** Pi and PC must be on the same LAN (or Pi on Wi-Fi, PC on Ethernet — same subnet).
2. **PC firewall**: Windows Defender blocks new inbound connections by default. Allow port 8765 (or whatever you set) or temporarily disable the firewall for testing.
3. **Find PC IP**: on Windows run `ipconfig` — look for IPv4 Address.
4. **Test the connection**: from the Pi, `curl http://<PC_IP>:8765/health` should return `{"status": "ok"}`.

## Wake word never fires

- **No model**: `pi-client/wake_word.py` falls back to "press Enter to wake" mode if the `.pb` file is missing. Train one (see [wake_word_training.md](wake_word_training.md)) or use the REPL fallback.
- **Mic muted**: test with `arecord -d 5 test.wav && aplay test.wav`.
- **Sensitivity too low**: raise `SENSITIVITY` in `pi-client/wake_word.py`.

## No sound out of Pi speaker

- **3.5mm jack**: ensure `amixer cset numid=3 1` (select headphone jack, not HDMI).
- **Volume**: `amixer set Master 80%` (or use `alsamixer`).
- **Bluetooth**: Pi 4 BT audio is glitchy. Use the 3.5mm jack for reliable playback, or follow [scripts/bt_speaker_setup.sh](../scripts/bt_speaker_setup.sh) for a documented BT path.

## High latency

- **Use a smaller Whisper model**: `WHISPER_MODEL=tiny.en` in `.env`.
- **Use a smaller LLM**: `ollama pull llama3.1:8b-q4` or `phi3:mini`.
- **Use a faster TTS**: try `tts_models/en/ljspeech/fast_pitch`.
- **Check the latency log**: `tail -f latency.log` — each line shows which stage is slow.

## Latency log reference

```
stt elapsed=412.3ms            # Whisper transcription
llm elapsed=890.1ms           # LLM first token to last token
tts[...] elapsed=210.0ms      # TTS per sentence
total elapsed=1650.4ms        # wake -> TTS end (target: < 1500ms)
```

## Pi runs out of memory

- Use `tiny.en` for Whisper (won't help much — STT runs on PC).
- If you're using on-device STT as a fallback, prefer `tiny` or `base`.
- Make sure the swap file is enabled: `sudo dphys-swapfile setup` then reboot.

## PC is too slow

- **GPU**: a single NVIDIA GPU with 6GB+ VRAM makes a huge difference.
- **CPU-only fallback**: works but expect 3-5s latency. Use the smallest models.

## Stop word doesn't interrupt

- The "stop" word is handled **client-side** on the Pi. Make sure `STOP_PHRASES` in `.env` includes "stop" (default does).
- It also requires a working wake-word engine — if you're in REPL mode, press Enter to interrupt.
- See the (future) `StopWordDetector` in `pi-client/` for the more advanced version that listens for the word mid-playback.
