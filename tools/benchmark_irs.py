#!/usr/bin/env python3
"""
Auto Gain Staging — upperBound 過大評価分布測定 & EmpiricalSafetyMarginPolicy 係数較正
====================================================================================

Week 2 benchmark tool. Uses real and synthetic IRs to measure:
1. boundExcessDb 分布（平均・95% tile・最大）
2. EmpiricalSafetyMarginPolicy の過小評価量分布
3. 推奨係数の再較正

Usage:
  python benchmark_irs.py [--irs_dir sampledata/real_irs/wavs] [--synthetic_dir sampledata/synthetic]
"""

import os
import sys
import argparse
import json
import math
import wave
import struct
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Constants (matching C++ implementation) ────────────────────────────────
SAMPLE_RATE = 48000  # Target SR for analysis
NUM_BANDS = 20
K_TWENTY_OVER_LOG10 = 20.0 / math.log(10.0)

# EQ band default frequencies (from EQProcessor.h)
DEFAULT_FREQS = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600,
                 2500, 4000, 6300, 10000, 11000, 12500, 14000, 16500, 18000, 19500]

# ─── IR Analysis ────────────────────────────────────────────────────────────
def read_wav(filepath):
    """Read WAV file as numpy array (float64, normalized to [-1, 1])."""
    with wave.open(str(filepath), 'rb') as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        data = w.readframes(n_frames)

    # Convert to numpy
    if sampwidth == 1:  # 8-bit unsigned
        dtype = np.uint8
        raw = np.frombuffer(data, dtype=dtype).reshape(-1, n_channels).astype(np.float64)
        raw = (raw - 128.0) / 128.0
    elif sampwidth == 2:  # 16-bit signed
        raw = np.frombuffer(data, dtype=np.int16).reshape(-1, n_channels).astype(np.float64)
        raw = raw / 32768.0
    elif sampwidth == 3:  # 24-bit signed (packed)
        raw_bytes = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3 * n_channels)
        raw = np.zeros((n_frames, n_channels), dtype=np.float64)
        for ch in range(n_channels):
            ch_bytes = raw_bytes[:, ch*3:(ch+1)*3]
            raw[:, ch] = ch_bytes[:, 0].astype(np.int32) | (ch_bytes[:, 1].astype(np.int32) << 8) | (ch_bytes[:, 2].astype(np.int32) << 16)
            raw[:, ch] = np.where(raw[:, ch] >= 0x800000, raw[:, ch] - 0x1000000, raw[:, ch])
            raw[:, ch] = raw[:, ch] / 8388608.0
    elif sampwidth == 4:  # 32-bit float
        raw = np.frombuffer(data, dtype=np.float32).reshape(-1, n_channels).astype(np.float64)
        raw = np.clip(raw, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    return raw, framerate

def compute_fft_peak(ir_mono, sr=SAMPLE_RATE):
    """Compute peak magnitude response using FFT (simplified IRAnalyzer)."""
    length = len(ir_mono)
    if length < 2:
        return 0.0, 0.0

    # Apply Hann window
    window = np.hanning(length)
    ir_windowed = ir_mono * window

    # FFT
    fft_size = 1 << (length - 1).bit_length()
    spectrum = np.fft.rfft(ir_windowed, n=fft_size)
    magnitude = np.abs(spectrum)

    # Coherent gain correction (Hann window mean)
    window_mean = np.mean(window)
    if window_mean > 1e-18:
        magnitude /= window_mean

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

    # Find peak
    peak_idx = np.argmax(magnitude[:len(freqs)])
    peak_mag = magnitude[peak_idx]
    peak_freq = freqs[peak_idx]

    # Simple parabolic interpolation
    if 0 < peak_idx < len(magnitude) - 1:
        ym1 = magnitude[peak_idx - 1]
        y0 = magnitude[peak_idx]
        yp1 = magnitude[peak_idx + 1]
        if y0 > ym1 and y0 > yp1 and y0 > 1e-18:
            denom = math.log(ym1) - 2.0 * math.log(y0) + math.log(yp1)
            if abs(denom) > 1e-18:
                delta = 0.5 * (math.log(ym1) - math.log(yp1)) / denom
                peak_mag = y0 * math.exp(-delta * (math.log(y0) - math.log(ym1)))

    peak_gain_db = 20.0 * math.log10(peak_mag) if peak_mag > 1e-18 else -200.0
    return peak_gain_db, peak_freq

def compute_upper_bound(ir_mono, sr=SAMPLE_RATE):
    """
    Simplified upperBound calculation: Π(1+|Hi-1|) in dB.
    Uses FFT bins as proxies for band magnitudes.
    """
    length = len(ir_mono)
    if length < 2:
        return 0.0

    # FFT
    fft_size = 1 << (length - 1).bit_length()
    spectrum = np.fft.rfft(ir_mono, n=fft_size)
    magnitude = np.abs(spectrum)

    # Group into 20 bands (simplified: use frequency bins as Hi)
    # For each "band", compute |Hi-1| and accumulate log1p
    log_bound = 0.0

    # Use frequency bins as independent bands
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    freqs = freqs[1:]  # Skip DC
    magnitudes = magnitude[1:]

    # Pick significant peaks as "bands"
    threshold = np.max(magnitudes) * 0.1
    significant = magnitudes > threshold

    for mag in magnitudes[significant]:
        delta = abs(mag - 1.0)
        if math.isfinite(delta) and delta > 1e-6:
            log_bound += math.log1p(delta)

    bound_db = K_TWENTY_OVER_LOG10 * log_bound
    return bound_db

def analyze_ir(filepath):
    """Full IR analysis: peak gain, upperBound, boundExcessDb."""
    try:
        ir, sr = read_wav(filepath)
    except Exception as e:
        return None, str(e)

    # Mix to mono
    if ir.ndim > 1 and ir.shape[1] > 1:
        ir_mono = np.mean(ir[:, :2], axis=1)  # Use first 2 channels
    else:
        ir_mono = ir.flatten()

    # Limit to kMaxAnalysisWindow
    max_samples = 65536
    if len(ir_mono) > max_samples:
        ir_mono = ir_mono[:max_samples]

    # Compute metrics
    peak_db, peak_freq = compute_fft_peak(ir_mono, sr)
    bound_db = compute_upper_bound(ir_mono, sr)
    rms_db = 20.0 * math.log10(np.sqrt(np.mean(ir_mono**2)) + 1e-18)
    peak_linear = np.max(np.abs(ir_mono))
    peak_peak_db = 20.0 * math.log10(peak_linear + 1e-18)

    return {
        'peak_gain_db': float(peak_db),
        'bound_db': float(bound_db),
        'bound_excess_db': float(max(0.0, bound_db - peak_db)),
        'rms_db': float(rms_db),
        'peak_peak_db': float(peak_peak_db),
        'peak_freq_hz': float(peak_freq),
        'length': len(ir_mono),
        'sample_rate': sr,
    }, None

# ─── Batch Processing ───────────────────────────────────────────────────────
def scan_wav_files(directory):
    """Recursively find all WAV files."""
    results = []
    for f in sorted(Path(directory).rglob("*.wav")):
        # Skip very short files (likely metadata)
        try:
            with wave.open(str(f), 'r') as w:
                if w.getnframes() >= 512:
                    results.append(f)
        except:
            pass
    return results

def run_benchmark(irs_dir, synthetic_dir, output_file):
    """Run the full benchmark across all IRs."""
    all_irs = []

    # Real IRs
    if irs_dir and Path(irs_dir).exists():
        all_irs.extend(scan_wav_files(irs_dir))
        print(f"  Real IRs found: {len(all_irs)}")

    # Synthetic IRs
    if synthetic_dir and Path(synthetic_dir).exists():
        synthetic_irs = scan_wav_files(synthetic_dir)
        all_irs.extend(synthetic_irs)
        print(f"  Synthetic IRs found: {len(synthetic_irs)}")

    print(f"  Total IRs: {len(all_irs)}")

    results = []
    errors = 0

    for i, filepath in enumerate(all_irs):
        rel_path = filepath.relative_to(filepath.anchor) if filepath.is_absolute() else filepath.name
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(all_irs)}...")

        result, error = analyze_ir(filepath)
        if result:
            result['file'] = str(rel_path)
            result['name'] = filepath.stem[:60]
            results.append(result)
        else:
            errors += 1
            if errors <= 5:
                print(f"  ⚠ {filepath.name}: {error}")

    # Compute statistics
    bound_excess_values = [r['bound_excess_db'] for r in results]
    peak_gain_values = [r['peak_gain_db'] for r in results]

    stats = {
        'total_files': len(results),
        'errors': errors,
        'bound_excess_db': {
            'mean': float(np.mean(bound_excess_values)) if bound_excess_values else 0,
            'median': float(np.median(bound_excess_values)) if bound_excess_values else 0,
            'std': float(np.std(bound_excess_values)) if bound_excess_values else 0,
            'p95': float(np.percentile(bound_excess_values, 95)) if bound_excess_values else 0,
            'p99': float(np.percentile(bound_excess_values, 99)) if bound_excess_values else 0,
            'max': float(np.max(bound_excess_values)) if bound_excess_values else 0,
            'min': float(np.min(bound_excess_values)) if bound_excess_values else 0,
        },
        'peak_gain_db': {
            'mean': float(np.mean(peak_gain_values)) if peak_gain_values else 0,
            'max': float(np.max(peak_gain_values)) if peak_gain_values else 0,
        },
    }

    # EmpiricalSafetyMarginPolicy calibration
    # Current formula: margin = min(2.5, max(0, 0.8 + (maxQ-0.707)*0.12 + gain*0.04))
    # For calibration, we estimate required margin as bound_excess_db
    # since bound_excess_db represents the overestimation of upperBound vs measured

    stats['calibration'] = {
        'current_coefficients': {
            'base': 0.8,
            'q_coeff': 0.12,
            'gain_coeff': 0.04,
            'max_margin': 2.5,
        },
        'recommended_base': float(np.percentile(bound_excess_values, 50)) if bound_excess_values else 0.8,
        'worst_case_95pct': stats['bound_excess_db']['p95'],
        'worst_case_99pct': stats['bound_excess_db']['p99'],
    }

    # Output
    output = {
        'statistics': stats,
        'all_results': sorted(results, key=lambda r: r['bound_excess_db'], reverse=True),
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {output_file}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total IRs analyzed: {results}")
    print(f"  Errors: {errors}")
    print(f"\n  boundExcessDb Distribution:")
    print(f"    Mean:   {stats['bound_excess_db']['mean']:.2f} dB")
    print(f"    Median: {stats['bound_excess_db']['median']:.2f} dB")
    print(f"    P95:    {stats['bound_excess_db']['p95']:.2f} dB")
    print(f"    P99:    {stats['bound_excess_db']['p99']:.2f} dB")
    print(f"    Max:    {stats['bound_excess_db']['max']:.2f} dB")
    print(f"\n  EmpiricalSafetyMarginPolicy Calibration:")
    print(f"    Current coefficients: base={0.8}, q_coeff={0.12}, gain_coeff={0.04}, max={2.5}")
    print(f"    Recommended base (median): {stats['calibration']['recommended_base']:.2f} dB")
    print(f"    Needed for 95% coverage:  {stats['calibration']['worst_case_95pct']:.2f} dB")
    print(f"    Needed for 99% coverage:  {stats['calibration']['worst_case_99pct']:.2f} dB")

    # Top 10 worst cases
    print(f"\n  Top 10 worst boundExcessDb cases:")
    for r in output['all_results'][:10]:
        print(f"    {r['bound_excess_db']:.1f}dB | {r['peak_gain_db']:.1f}dB | {r['name']}")

    return output

def main():
    parser = argparse.ArgumentParser(
        description="Auto Gain Staging IR Benchmark Tool")
    parser.add_argument("--irs-dir", default="sampledata/real_irs/wavs",
                        help="Directory with real IR WAV files")
    parser.add_argument("--synthetic-dir", default="sampledata/synthetic",
                        help="Directory with synthetic IR WAV files")
    parser.add_argument("--output", default="doc/work77/benchmark-results.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  Auto Gain Staging Benchmark")
    print(f"{'=' * 60}")

    run_benchmark(args.irs_dir, args.synthetic_dir, args.output)

if __name__ == "__main__":
    main()
