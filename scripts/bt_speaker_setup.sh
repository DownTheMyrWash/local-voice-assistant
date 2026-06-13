#!/usr/bin/env bash
# Pair a Bluetooth speaker with the Raspberry Pi 4 for audio output.
# Note: Pi 4 BT audio has known latency/glitches vs the 3.5mm jack.

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

echo "Installing Bluetooth + PulseAudio packages..."
apt update
apt install -y pulseaudio pulseaudio-module-bluetooth bluez bluez-tools

echo "Enabling Bluetooth service..."
systemctl enable bluetooth
systemctl start bluetooth

echo
echo "1. Put your speaker in pairing mode (hold the BT button until the LED flashes)."
echo "2. When ready, press Enter to scan..."
read -r

# Discover MAC address
echo "Scanning for 10 seconds..."
MAC=$(bluetoothctl scan on & sleep 10; bluetoothctl devices | tail -1 | awk '{print $2}')
killall bluetoothctl 2>/dev/null || true

if [ -z "${MAC:-}" ]; then
    echo "No device found. Re-run and try again." >&2
    exit 1
fi

echo "Pairing with $MAC..."
bluetoothctl pair "$MAC"
bluetoothctl trust "$MAC"
bluetoothctl connect "$MAC"

echo
echo "Done. To switch the Pi's audio output to the BT speaker, use:"
echo "  pactl set-default-sink bluez_sink.${MAC//:/_}.a2dp_sink"
echo
echo "To switch back to 3.5mm jack:"
echo "  pactl set-default-sink alsa_output.platform-3f980000.usb..."
