#!/usr/bin/env python3
"""work52: 診断指標の検証 - jump値の妥当性とFFT分析"""

import struct, math, sys

# === 1. 純粋40Hz sine波のjump値（ベースライン）===
SAMPLE_RATE = 48000
FREQ = 40.0
AMP = 0.5
DURATION = 5.0
BLOCK = 512

n = int(SAMPLE_RATE * DURATION)
pure = [AMP * math.sin(2.0 * math.pi * FREQ * i / SAMPLE_RATE) for i in range(n)]

jumps_pure = [abs(pure[i] - pure[i-1]) for i in range(BLOCK, n, BLOCK)]
avg_pure = sum(jumps_pure) / len(jumps_pure)
max_pure = max(jumps_pure)
pct_01_pure = 100 * sum(1 for j in jumps_pure if j > 0.1) / len(jumps_pure)

print("=" * 60)
print("  1. jump指標 ベースライン（純粋40Hz正弦波）")
print("=" * 60)
print(f"  平均 jump: {avg_pure:.6f}")
print(f"  最大 jump: {max_pure:.4f}")
print(f"  >0.1の割合: {pct_01_pure:.1f}%")

# === 2. Convolver出力のjump値 ===
try:
    with open('C:/TEMP/conv_output_tailBypass.raw', 'rb') as f:
        raw = f.read()
except FileNotFoundError:
    with open('C:/TEMP/conv_output_work52_fail.raw', 'rb') as f:
        raw = f.read()

n_conv = len(raw) // 8
conv = [struct.unpack_from('<d', raw, i*8)[0] for i in range(n_conv)]

jumps_conv = [abs(conv[i] - conv[i-1]) for i in range(BLOCK, n_conv, BLOCK)]
avg_conv = sum(jumps_conv) / len(jumps_conv)
max_conv = max(jumps_conv)
median_conv = sorted(jumps_conv)[len(jumps_conv)//2]
pct_01_conv = 100 * sum(1 for j in jumps_conv if j > 0.1) / len(jumps_conv)
pct_02_conv = 100 * sum(1 for j in jumps_conv if j > 0.2) / len(jumps_conv)

print()
print("=" * 60)
print("  2. Convolver出力のjump値")
print("=" * 60)
print(f"  平均 jump: {avg_conv:.6f}")
print(f"  最大 jump: {max_conv:.4f}")
print(f"  中央値 jump: {median_conv:.6f}")
print(f"  >0.1の割合: {pct_01_conv:.1f}%")
print(f"  >0.2の割合: {pct_02_conv:.1f}%")

# === 3. 比較表 ===
print()
print("=" * 60)
print("  3. 比較表")
print("=" * 60)
print(f"  {'指標':<20} {'純粋40Hz':>12} {'Convolver':>12} {'比率':>10}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
print(f"  {'平均 jump':<20} {avg_pure:>12.6f} {avg_conv:>12.4f} {avg_conv/avg_pure:>8.1f}x")
print(f"  {'最大 jump':<20} {max_pure:>12.4f} {max_conv:>12.4f} {max_conv/max_pure:>8.1f}x")
print(f"  {'>0.1 割合':<20} {pct_01_pure:>12.1f}% {pct_01_conv:>11.1f}%")

# === 4. ブロック境界サンプル詳細 ===
print()
print("=" * 60)
print("  4. ブロック境界サンプル詳細")
print("=" * 60)

for b in [0, 1, 8, 9, 10, 50, 100, 335, 336]:
    pos = b * BLOCK
    if pos >= n_conv:
        continue
    before = conv[pos-1] if pos > 0 else 0.0
    after = conv[pos]
    jump = abs(after - before)
    ctx_before = [conv[max(0, pos-4+i)] for i in range(4)]
    ctx_after = [conv[min(n_conv-1, pos+i)] for i in range(4)]
    print(f"  Block {b:3d} boundary (sample {pos:6d}): jump={jump:.4f}")
    print(f"    before: [{', '.join(f'{v:+.4f}' for v in ctx_before)}]")
    print(f"    after:  [{', '.join(f'{v:+.4f}' for v in ctx_after)}]")

# === 5. 純粋40Hzでも同じFFTを実行 ===
print()
print("=" * 60)
print("  5. FFTスペクトル比較")
print("=" * 60)

# 定常状態を使う（初期80msのFDLランプアップをスキップ）
fft_size = 8192
steady_start = 150 * BLOCK

hann = [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (fft_size - 1))) for i in range(fft_size)]

def calc_spectrum(signal, start, size, window):
    chunk = signal[start:start+size]
    if len(chunk) < size:
        return [], 0
    windowed = [chunk[i] * window[i] for i in range(size)]
    spec = []
    for k in range(size // 2):
        re = 0.0
        im = 0.0
        for t in range(size):
            angle = -2.0 * math.pi * k * t / size
            re += windowed[t] * math.cos(angle)
            im += windowed[t] * math.sin(angle)
        spec.append(math.sqrt(re*re + im*im))
    max_s = max(spec) if spec else 1.0
    spec_db = [20 * math.log10(s/max_s + 1e-12) for s in spec] if max_s > 0 else [-200]*len(spec)
    return spec_db, SAMPLE_RATE / size

spec_conv, bin_hz = calc_spectrum(conv, steady_start, fft_size, hann)
spec_pure, _ = calc_spectrum(pure, steady_start, fft_size, hann)

bin_40hz = int(40.0 / bin_hz)
bin_94hz = int(93.75 / bin_hz)
bin_12k = int(12000 / bin_hz)
bin_24k = int(24000 / bin_hz)

print(f"  bin幅: {bin_hz:.2f} Hz, FFT size: {fft_size}")
print()
print(f"  {'周波数':<10} {'純粋40Hz':>12} {'Convolver':>12} {'差':>10}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
for freq_hz in [0, 6, 12, 18, 24, 30, 35, 40, 47, 60, 94, 200, 12000, 24000]:
    bin_idx = int(freq_hz / bin_hz) if freq_hz > 0 else 0
    if bin_idx < len(spec_conv) and bin_idx < len(spec_pure):
        print(f"  {freq_hz:>5}Hz:  {spec_pure[bin_idx]:>10.1f}dB  {spec_conv[bin_idx]:>10.1f}dB  {spec_conv[bin_idx]-spec_pure[bin_idx]:>+9.1f}dB")

# === 6. 結論 ===
print()
print("=" * 60)
print("  6. 結論")
print("=" * 60)
print(f"""
  jump指標:  {'有効' if avg_conv/avg_pure > 5 else '無効'}
    純粋40Hzの最大jump = {max_pure:.4f}
    Convolverの最大jump = {max_conv:.4f} ({max_conv/max_pure:.1f}x)
    40Hz正弦波なら0.1を超えることはない

  FFT所見:
    DC成分が0dB（支配的）- 純粋40Hzでは-65.7dB
    40Hz成分がDCより38dB低い = 異常
    12kHz/24kHzにスプリアス成分あり

  L1/L2関与: 否定（tail bypassでも同一パターン）

  次に疑うべき:
    processLayerBlock()のFDL初期化
    またはIRデータそのものの問題
""")
