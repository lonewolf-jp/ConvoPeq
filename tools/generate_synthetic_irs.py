#!/usr/bin/env python3
"""
Synthetic IR Generator — Auto Gain Staging Benchmark Tool
=========================================================
Generates synthetic impulse responses for Week 2 benchmarks.

Usage: python3 generate_synthetic_irs.py [output_dir]

Output:
  sampledata/synthetic/
    dirac_k2.wav, dirac_k4.wav, dirac_k8.wav, dirac_k16.wav, dirac_k32.wav
    minimum_phase/*.wav
    linear_phase/*.wav
    mixed_phase/*.wav
    reverb_plate.wav, reverb_spring.wav
"""

import os
import sys
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import wave
import struct
import argparse
from pathlib import Path

# ─── Constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 48000
MAX_LENGTH = 65536  # kMaxAnalysisWindow
BITS_PER_SAMPLE = 32  # float
NUM_CHANNELS = 2

def write_wav(filepath, data, sr=SAMPLE_RATE):
    """Write multi-channel float64 numpy array to WAV file (32-bit float)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [-1, 1] range
    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data = data / max_abs * 0.95  # Headroom

    n_channels = data.shape[1] if data.ndim > 1 else 1
    n_frames = data.shape[0]

    with wave.open(str(filepath), 'w') as w:
        w.setnchannels(n_channels)
        w.setsampwidth(4)  # 32-bit
        w.setframerate(sr)

        # Interleave channels
        if n_channels == 1:
            samples = data
        else:
            samples = data.flatten()

        # Convert to 32-bit float PCM
        packed = struct.pack(f'<{len(samples)}f', *samples)
        w.writeframes(packed)

    print(f"  ✓ {filepath.name}: {n_frames}samples, {n_channels}ch, {sr}Hz")
    return filepath

# ─── 1. Dirac × k ──────────────────────────────────────────────────────────
def generate_dirac(gain_linear, length=1024):
    """Generate a Dirac impulse scaled by gain_linear."""
    ir = np.zeros((length, NUM_CHANNELS))
    ir[0] = gain_linear
    return ir

def generate_all_dirac(output_dir):
    """Generate Dirac IRs with gains: 2, 4, 8, 16, 32 (≈ 6, 12, 18, 24, 30 dB)."""
    print("\n[1/4] Generating Dirac × k IRs...")
    gains = [2.0, 4.0, 8.0, 16.0, 32.0]
    for k in gains:
        ir = generate_dirac(k)
        write_wav(output_dir / f"dirac_k{k:.0f}.wav", ir)
    print(f"  → {len(gains)} files created")

# ─── 2. Minimum Phase ──────────────────────────────────────────────────────
def generate_minimum_phase(length=4096, fc=2000.0, gain_db=12.0):
    """
    Generate a minimum-phase IR using cepstral method.
    """
    # Create full magnitude response (0 to Nyquist)
    N = length
    half = N // 2 + 1
    freq = np.linspace(0, SAMPLE_RATE / 2, half)
    H_mag = np.exp(-((freq - fc) / (fc * 0.5))**2) * 10**(gain_db / 20.0)
    H_mag = np.maximum(H_mag, 0.001)

    # Log magnitude
    log_mag = np.log(H_mag)

    # Create symmetric log magnitude for IFFT
    log_mag_full = np.zeros(N)
    log_mag_full[0:half] = log_mag
    log_mag_full[half:] = log_mag[-2:0:-1] if half > 1 else log_mag[0:0]
    # Handle odd N edge
    if N % 2 == 0 and half < N:
        log_mag_full[half] = log_mag[-1]

    # IFFT to get cepstrum
    cepstrum = fft.ifft(log_mag_full).real

    # Minimum phase filter in quefrency domain
    m = np.zeros(N)
    m[0] = 1.0
    m[1:N//2] = 2.0
    m[N//2] = 1.0  # Half-way point for even N
    # Upper half (> Nyquist) is 0

    cepstrum_min = cepstrum * m

    # FFT back to get log spectrum
    log_spec_min = fft.fft(cepstrum_min).real

    # Exponential to get complex spectrum, then IFFT
    H_min = np.exp(log_spec_min + 1j * 0)  # Phase is embedded in cepstrum

    # Actually, minimum phase is derived from the Hilbert transform of log magnitude
    # More direct approach: H_min = exp(real_cepstrum + j * hilbert(real_cepstrum))
    # Using the cepstral method correctly:
    # The minimum phase spectrum is: log|H| + j * φ_min where φ_min = -hilbert(log|H|)
    from scipy.signal import hilbert
    analytic = hilbert(log_mag_full)
    H_min_phase = np.exp(analytic)  # exp(log|H| + j*φ_min)

    # IFFT to get minimum phase IR
    ir_mono = fft.ifft(H_min_phase).real[:length]

    # Normalize peak to 0dB
    ir_mono = ir_mono / np.max(np.abs(ir_mono)) * 10**(gain_db / 20.0)

    # Stereo (same for both channels)
    ir = np.zeros((length, NUM_CHANNELS))
    ir[:, 0] = ir_mono
    ir[:, 1] = ir_mono * 0.95  # Slight difference

    return ir

def generate_all_minimum_phase(output_dir):
    """Generate minimum phase IRs with various characteristics."""
    print("\n[2/4] Generating Minimum Phase IRs...")
    configs = [
        ("minphase_bandpass_2k_12dB.wav", 2000, 12.0),
        ("minphase_bandpass_5k_18dB.wav", 5000, 18.0),
        ("minphase_lowshelf_200_12dB.wav", 200, 12.0),
        ("minphase_highshelf_8k_12dB.wav", 8000, 12.0),
        ("minphase_fullband_24dB.wav", 1000, 24.0),
    ]
    for name, fc, gain_db in configs:
        ir = generate_minimum_phase(fc=fc, gain_db=gain_db)
        write_wav(output_dir / name, ir)
    print(f"  → {len(configs)} files created")

# ─── 3. Linear Phase ───────────────────────────────────────────────────────
def generate_linear_phase(length=4096, fc=2000.0, gain_db=12.0):
    """
    Generate a linear-phase FIR from a target magnitude response.
    Uses windowed FIR design with symmetric coefficients.
    """
    # Design FIR filter with linear phase
    # Use frequency sampling method
    num_taps = length if length % 2 == 0 else length - 1  # Even for linear phase

    # Target magnitude
    freq = np.linspace(0, 1, num_taps // 2 + 1)  # Normalized 0 to Nyquist
    H_target = np.exp(-((freq * SAMPLE_RATE / 2 - fc) / (fc * 0.5))**2) * 10**(gain_db / 20.0)
    H_target[0] = 1.0  # DC
    H_target += 0.001  # Floor

    # Linear phase: group delay = (num_taps - 1) / 2
    group_delay = (num_taps - 1) / 2
    phase = -2 * np.pi * group_delay * freq

    H_complex = H_target * np.exp(1j * phase)

    # IFFT to get filter coefficients
    H_full = np.zeros(num_taps, dtype=complex)
    H_full[0] = H_complex[0].real
    H_full[1:len(H_complex)] = H_complex[1:]
    # Upper half is conjugate mirror
    H_full[len(H_complex):] = H_complex[-2:0:-1].conj() if len(H_complex) > 2 else np.array([])

    ir_mono = fft.ifft(H_full).real[:length]

    # Apply Tukey window to smooth edges
    from scipy.signal.windows import tukey
    window = tukey(length, alpha=0.2)
    ir_mono = ir_mono * window

    # Normalize
    ir_mono = ir_mono / np.max(np.abs(ir_mono)) * 10**(gain_db / 20.0)

    ir = np.zeros((length, NUM_CHANNELS))
    ir[:, 0] = ir_mono
    ir[:, 1] = ir_mono
    return ir

def generate_all_linear_phase(output_dir):
    """Generate linear phase IRs with various characteristics."""
    print("\n[3/4] Generating Linear Phase IRs...")
    configs = [
        ("linphase_bandpass_2k_12dB.wav", 2000, 12.0),
        ("linphase_bandpass_5k_18dB.wav", 5000, 18.0),
        ("linphase_brickwall_3k_12dB.wav", 3000, 12.0),
        ("linphase_bandpass_500_6dB.wav", 500, 6.0),
        ("linphase_fullband_24dB.wav", 1000, 24.0),
    ]
    for name, fc, gain_db in configs:
        ir = generate_linear_phase(fc=fc, gain_db=gain_db)
        write_wav(output_dir / name, ir)
    print(f"  → {len(configs)} files created")

# ─── 4. Mixed Phase ────────────────────────────────────────────────────────
def generate_mixed_phase(length=4096, fc=2000.0, gain_db=12.0):
    """
    Generate a mixed-phase IR: minimum phase early part +
    linear phase late reverberation tail.
    """
    # Early part: minimum phase
    early_len = length // 4
    ir_early = generate_minimum_phase(length=early_len, fc=fc, gain_db=gain_db)
    ir_early = ir_early[:, 0]  # Mono

    # Late part: exponentially decaying noise (reverb tail)
    late_len = length - early_len
    noise = np.random.randn(late_len)

    # Exponential decay
    t = np.arange(late_len) / SAMPLE_RATE
    decay = np.exp(-t * 3.0)  # RT60 ≈ 1s
    ir_late = noise * decay

    # Apply bandpass to late part
    low_freq = max(10, fc * 0.3)
    high_freq = min(SAMPLE_RATE * 0.49, fc * 3.0)
    sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=SAMPLE_RATE, output='sos')
    ir_late = signal.sosfilt(sos, ir_late)

    # Scale late part relative to early part
    early_energy = np.sum(ir_early**2)
    late_energy = np.sum(ir_late**2)
    if late_energy > 0:
        ir_late = ir_late * np.sqrt(early_energy / late_energy) * 0.5

    # Concatenate
    ir_mono = np.concatenate([ir_early[:early_len], ir_late])

    # Pad or truncate to exact length
    if len(ir_mono) < length:
        ir_mono = np.pad(ir_mono, (0, length - len(ir_mono)))
    else:
        ir_mono = ir_mono[:length]

    # Normalize
    ir_mono = ir_mono / np.max(np.abs(ir_mono)) * 10**(gain_db / 20.0)

    ir = np.zeros((length, NUM_CHANNELS))
    ir[:, 0] = ir_mono
    ir[:, 1] = ir_mono * 0.98  # Very slight decorrelation
    return ir

def generate_all_mixed_phase(output_dir):
    """Generate mixed phase IRs."""
    print("\n[4/4] Generating Mixed Phase IRs...")
    configs = [
        ("mixedphase_room_2k_12dB.wav", 2000, 12.0),
        ("mixedphase_hall_500_18dB.wav", 500, 18.0),
        ("mixedphase_chamber_4k_6dB.wav", 4000, 6.0),
        ("mixedphase_plate_8k_12dB.wav", 8000, 12.0),
        ("mixedphase_stadium_200_24dB.wav", 200, 24.0),
    ]
    for name, fc, gain_db in configs:
        ir = generate_mixed_phase(fc=fc, gain_db=gain_db)
        write_wav(output_dir / name, ir)
    print(f"  → {len(configs)} files created")

# ─── 5. Plate / Spring Reverb ──────────────────────────────────────────────
def generate_plate_reverb(length=24000):
    """Generate synthetic plate reverb IR using FDN-like structure."""
    t = np.arange(length) / SAMPLE_RATE

    # Early reflections
    early = np.zeros(length)
    early_delays = [0.001, 0.003, 0.007, 0.012, 0.018, 0.025]
    early_gains = [0.8, 0.6, 0.5, 0.3, 0.2, 0.1]
    for delay, gain in zip(early_delays, early_gains):
        idx = int(delay * SAMPLE_RATE)
        if idx < length:
            early[idx] = gain

    # Late diffuse tail
    noise = np.random.randn(length)
    decay = np.exp(-t * 2.0)
    late = noise * decay

    # Bandpass
    sos = signal.butter(2, [200, min(SAMPLE_RATE*0.49, 8000)], btype='band', fs=SAMPLE_RATE, output='sos')
    late = signal.sosfilt(sos, late)

    ir_mono = early + late * 0.3
    ir_mono = ir_mono / np.max(np.abs(ir_mono)) * 0.95

    ir = np.zeros((length, NUM_CHANNELS))
    ir[:, 0] = ir_mono
    ir[:, 1] = ir_mono
    return ir

def generate_spring_reverb(length=24000):
    """Generate synthetic spring reverb IR."""
    t = np.arange(length) / SAMPLE_RATE

    # Spring reverb: chirp-like, metallic
    # Modulated delay lines
    d = 0.002  # Base delay
    chirp = np.sin(2 * np.pi * 200 * t + 5000 * t**2)
    noise = np.random.randn(length)

    # Heavy modulation
    mod = np.sin(2 * np.pi * 3 * t) * 0.0005
    delays = d + mod

    ir_mono = chirp * 0.3 + noise * 0.7
    decay = np.exp(-t * 5.0)
    ir_mono = ir_mono * decay

    # Bandpass (spring reverb is mid-heavy)
    sos = signal.butter(2, [500, min(SAMPLE_RATE*0.49, 5000)], btype='band', fs=SAMPLE_RATE, output='sos')
    ir_mono = signal.sosfilt(sos, ir_mono)

    ir_mono = ir_mono / np.max(np.abs(ir_mono)) * 0.95

    ir = np.zeros((length, NUM_CHANNELS))
    ir[:, 0] = ir_mono
    ir[:, 1] = ir_mono * 0.99
    return ir

def generate_standard_reverbs(output_dir):
    """Generate plate and spring reverb IRs."""
    print("\n[5] Generating Standard Reverb IRs...")
    ir_plate = generate_plate_reverb()
    write_wav(output_dir / "reverb_plate.wav", ir_plate)

    ir_spring = generate_spring_reverb()
    write_wav(output_dir / "reverb_spring.wav", ir_spring)
    print(f"  → 2 files created")

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic IR files for Auto Gain Staging benchmarks")
    parser.add_argument("output_dir", nargs="?",
                        default="sampledata/synthetic",
                        help="Output directory (default: sampledata/synthetic)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"  Synthetic IR Generator")
    print(f"  Output: {output_dir}")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Max Length: {MAX_LENGTH} samples ({MAX_LENGTH/SAMPLE_RATE:.1f}s)")
    print(f"=" * 60)

    generate_all_dirac(output_dir)
    generate_all_minimum_phase(output_dir / "minimum_phase")
    generate_all_linear_phase(output_dir / "linear_phase")
    generate_all_mixed_phase(output_dir / "mixed_phase")
    generate_standard_reverbs(output_dir)

    # Summary
    n_files = sum(1 for _ in output_dir.rglob("*.wav"))

    print(f"\n{'=' * 60}")
    print(f"  Done: {n_files} files generated")
    print(f"  Location: {output_dir.resolve()}")
    print(f"{'=' * 60}")

    # Print analysis info for each file
    print(f"\n  File summary:")
    for f in sorted(output_dir.rglob("*.wav")):
        with wave.open(str(f), 'r') as w:
            duration = w.getnframes() / w.getframerate()
            print(f"  {f.relative_to(output_dir.parent)}: {w.getnframes()}samples, {duration:.2f}s")

if __name__ == "__main__":
    main()
