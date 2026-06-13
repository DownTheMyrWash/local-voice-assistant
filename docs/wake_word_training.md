# Wake word training

The default wake word is **"hey assistant"** (or just **"assistant"**). The model file lives at `models/hey-assistant.pb`. Mycroft Precise lets you train your own model for any word/phrase.

## Train a custom model

```bash
# 1. Install Precise
pip install precise-runner

# 2. Record your wake-word samples (~30-50 recordings of you saying the phrase)
precise-collect train hey-assistant.net hey assistant 50

# 3. Record "not-wake-word" samples (other speech that should NOT trigger)
precise-collect train hey-assistant.net "anything else you might say" 100

# 4. Train
precise-train hey-assistant.net hey-assistant.pb

# 5. Drop the .pb into models/ and update MODEL_PATH in pi-client/wake_word.py
```

Tips:
- Vary your distance, volume, and speaking speed across recordings.
- Use the exact mic you'll deploy with.
- Aim for 50 wake + 200+ not-wake samples for a solid model.

## Sensitivity tuning

`SENSITIVITY` in [pi-client/wake_word.py](../pi-client/wake_word.py) — higher = more sensitive (more false positives). 0.5 is a reasonable starting point.

## Alternative: openWakeWord

If you want a more modern, ONNX-based engine with simpler training, swap the contents of [pi-client/wake_word.py](../pi-client/wake_word.py) for the `openwakeword` library. The async interface in `WakeWordDetector.wait_for_wake()` is stable — the rest of the code won't need to change.
