#!/usr/bin/env python3
"""work52 診断用テスト信号生成ツール

低音入力に対する Convolver 出力を診断するためのテストWAVファイルを生成する。

生成ファイル:
  C:/TEMP/conv_diag_input.wav  (48kHz, 2ch, 32bit float, 約30秒)

信号内容:
  - 0-5秒:   40Hz 正弦波 (-6dBFS)
  - 5-10秒:  60Hz 正弦波 (-6dBFS)
  - 10-15秒: 80Hz 正弦波 (-6dBFS)
  - 15-20秒: 40Hz + 60Hz + 80Hz 混合 (-9dBFS each)
  - 20-25秒: 40Hz 連打 (200ms ON / 200ms OFF, -6dBFS)
  - 25-30秒: 40Hz 正弦波 (-3dBFS, 強め)
"""

import struct, math, os, sys
from pathlib import Path

SAMPLE_RATE = 48000
NUM_CHANNELS = 2
BITS_PER_SAMPLE = 32  # float
DURATION_SEC = 30
OUTPUT_PATH = "C:/TEMP/conv_diag_input.wav"

def generate_signal() -> list:
    """テスト信号を生成。list of [L, R] samples を返す。"""
    total_samples = SAMPLE_RATE * DURATION_SEC
    signal = []

    for i in range(total_samples):
        t = i / SAMPLE_RATE
        val = 0.0

        if t < 5.0:          # 40Hz sine
            val = 0.5 * math.sin(2 * math.pi * 40 * t)
        elif t < 10.0:       # 60Hz sine
            val = 0.5 * math.sin(2 * math.pi * 60 * t)
        elif t < 15.0:       # 80Hz sine
            val = 0.5 * math.sin(2 * math.pi * 80 * t)
        elif t < 20.0:       # 40+60+80Hz 混合
            val = 0.35 * (math.sin(2 * math.pi * 40 * t)
                         + math.sin(2 * math.pi * 60 * t)
                         + math.sin(2 * math.pi * 80 * t))
        elif t < 25.0:       # 40Hz 連打
            beat = t % 0.4  # 200ms ON, 200ms OFF
            if beat < 0.2:
                val = 0.5 * math.sin(2 * math.pi * 40 * t)
            # else: silence during OFF
        else:                # 40Hz stronger
            val = 0.7 * math.sin(2 * math.pi * 40 * t)

        signal.append([val, val])  # L/R same

    return signal


def write_wav(filename: str, signal: list):
    """32bit float WAV ファイルを書き出す。"""
    num_samples = len(signal)
    data_size = num_samples * NUM_CHANNELS * 4  # 4 bytes per float
    file_size = 36 + data_size

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))           # chunk size
        f.write(struct.pack("<H", 3))            # format = IEEE float
        f.write(struct.pack("<H", NUM_CHANNELS))
        f.write(struct.pack("<I", SAMPLE_RATE))
        f.write(struct.pack("<I", SAMPLE_RATE * NUM_CHANNELS * 4))  # byte rate
        f.write(struct.pack("<H", NUM_CHANNELS * 4))  # block align
        f.write(struct.pack("<H", BITS_PER_SAMPLE))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for sample in signal:
            for ch in range(NUM_CHANNELS):
                f.write(struct.pack("<f", sample[ch]))

    print(f"[OK] WAV file created: {filename}")
    print(f"     Duration: {num_samples / SAMPLE_RATE:.1f}s, {SAMPLE_RATE}Hz, {NUM_CHANNELS}ch, {BITS_PER_SAMPLE}bit float")


if __name__ == "__main__":
    signal = generate_signal()
    write_wav(OUTPUT_PATH, signal)
    print("Done.")
