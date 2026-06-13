"""Utility to list available audio input devices using PyAudio."""

import pyaudio

p = pyaudio.PyAudio()
print("\n--- AVAILABLE INPUT DEVICES ---")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"Index {i}: Name='{dev['name']}' | MaxInput={dev['maxInputChannels']} | DefaultRate={int(dev['defaultSampleRate'])}")
print("-------------------------------\n")
p.terminate()