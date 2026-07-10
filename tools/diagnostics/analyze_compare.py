#!/usr/bin/env python3
"""work52: 全キャプチャ比較分析"""

import struct, math

captures = {
    'conv_bypass': 'C:/TEMP/conv_output_bypass.raw',
    'tail_bypass': 'C:/TEMP/conv_output_tailBypass.raw',
    'tail_enabled': 'C:/TEMP/conv_output_work52_fail.raw',
}

for name, fpath in captures.items():
    with open(fpath, 'rb') as f:
        raw = f.read()
    n = len(raw) // 8
    data = [struct.unpack_from('<d', raw, i*8)[0] for i in range(n)]

    block_size = 512
    jumps = [abs(data[i] - data[i-1]) for i in range(block_size, n, block_size)]
    first20dc = [sum(data[b*block_size:(b+1)*block_size])/block_size for b in range(min(30, n//block_size))]

    # FFT (steady state, skip first 100 blocks)
    fft_start = 100 * block_size
    fft_n = min(8192, n - fft_start)
    hann = [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (fft_n - 1))) for i in range(fft_n)]
    chunk = data[fft_start:fft_start+fft_n]
    windowed = [chunk[i] * hann[i] for i in range(fft_n)]
    spec = []
    for k in range(fft_n // 2):
        re, im = 0.0, 0.0
        for t in range(fft_n):
            a = -2.0 * math.pi * k * t / fft_n
            re += windowed[t] * math.cos(a)
            im += windowed[t] * math.sin(a)
        spec.append(math.sqrt(re*re + im*im))
    max_s = max(spec)
    spec_db = [20 * math.log10(s/max_s + 1e-12) for s in spec]
    bin_hz = 48000.0 / fft_n

    print(f"=== {name} ===")
    print(f"  n={n}, avg_jump={sum(jumps)/len(jumps):.6f}, max_jump={max(jumps):.4f}, >0.1: {100*sum(1 for j in jumps if j>0.1)/len(jumps):.1f}%")
    print(f"  DC={sum(data)/n:+.6e}")
    print(f"  Block DC (first 20): {[f'{v:+.4f}' for v in first20dc]}")
    print(f"  FFT (fft_n={fft_n}, bin={bin_hz:.2f}Hz):")
    for freq in [0, 6, 12, 18, 24, 30, 35, 40, 47, 60, 94, 200, 12000, 24000]:
        bi = int(freq / bin_hz) if freq > 0 else 0
        if bi < len(spec_db):
            print(f"    {freq:5d}Hz: {spec_db[bi]:+.1f}dB")
    print()
