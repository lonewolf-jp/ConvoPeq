#!/usr/bin/env python3
"""Create a Dirac impulse WAV file (1 sample = 1.0, rest = 0.0)"""
import wave, struct, os

sr = 48000
n_samples = 8192  # ~170ms

path = "C:/TEMP/dirac_test.wav"
with wave.open(path, 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)  # 16-bit
    w.setframerate(sr)
    data = bytearray()
    for i in range(n_samples):
        val = 32767 if i == 0 else 0
        data += struct.pack('<h', val)
    w.writeframes(bytes(data))

sz = os.path.getsize(path)
print("Created:", path)
print("  Size:", sz, "bytes")
print("  Duration: %.3fs" % (n_samples / sr))
print("  Format: 48kHz, 16-bit, mono")
