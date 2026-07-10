#!/usr/bin/env python3
"""Generate test FIR filter IRs as WAV files (LPF 200Hz, HPF 20Hz, 129 taps)"""
import wave, struct, math, os

SR = 48000

def windowed_sinc_lpf(cutoff_hz, num_taps, window='hamming'):
    """Generate low-pass FIR filter coefficients using windowed sinc method."""
    nyq = SR / 2.0
    fc = cutoff_hz / nyq  # normalized cutoff (0..1)
    half = (num_taps - 1) // 2
    taps = []
    for n in range(num_taps):
        i = n - half
        if i == 0:
            val = 2.0 * fc
        else:
            val = math.sin(2.0 * math.pi * fc * i) / (math.pi * i)
        # Hamming window
        w = 0.54 - 0.46 * math.cos(2.0 * math.pi * n / (num_taps - 1))
        taps.append(val * w)
    # Normalize to unity gain at DC
    gain = sum(taps)
    return [t / gain for t in taps]

def windowed_sinc_hpf(cutoff_hz, num_taps, window='hamming'):
    """Generate high-pass FIR filter via spectral inversion of LPF."""
    lpf = windowed_sinc_lpf(cutoff_hz, num_taps, window)
    half = (num_taps - 1) // 2
    hpf = []
    for n in range(num_taps):
        # δ[n - half] - lpf[n]
        impulse = 1.0 if n == half else 0.0
        hpf.append(impulse - lpf[n])
    return hpf

def write_wav(path, taps, label):
    """Write float64 filter taps as 32-bit float WAV (stereo, same L/R)."""
    n = len(taps)
    with wave.open(path, 'w') as w:
        w.setnchannels(2)
        w.setsampwidth(4)  # 32-bit
        w.setframerate(SR)
        data = bytearray()
        for v in taps:
            packed = struct.pack('<f', float(v))
            data += packed  # L
            data += packed  # R
        w.writeframes(bytes(data))
    sz = os.path.getsize(path)
    print("Created: %s" % path)
    print("  Taps: %d, Duration: %.3fs, Size: %d bytes" % (n, n/SR, sz))
    print("  Peak: %.4f, RMS: %.6f" % (max(abs(v) for v in taps), math.sqrt(sum(v*v for v in taps)/n)))
    # Frequency response at key points (simple DFT approximation)
    fft_n = 4096
    padded = taps + [0.0] * (fft_n - n)
    window = [0.5 * (1.0 - math.cos(2.0*math.pi*i/(fft_n-1))) for i in range(fft_n)]
    windowed = [padded[i] * window[i] for i in range(fft_n)]
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
    bin_hz = SR / fft_n
    print("  Freq response (dB):")
    for freq in [0, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000]:
        bi = int(freq / bin_hz) if freq > 0 else 0
        if bi < len(spec_db):
            print("    %5d Hz: %+.1f dB" % (freq, spec_db[bi]))
    print()

# Generate test IRs
print("=" * 60)
print("  Test FIR IR Generation")
print("=" * 60)
print()

# LPF 200Hz, 129 taps
lpf_taps = windowed_sinc_lpf(200.0, 129)
write_wav("C:/TEMP/fir_lpf_200hz.wav", lpf_taps, "LPF 200Hz (129 taps)")

# HPF 20Hz, 129 taps
hpf_taps = windowed_sinc_hpf(20.0, 129)
write_wav("C:/TEMP/fir_hpf_20hz.wav", hpf_taps, "HPF 20Hz (129 taps)")
