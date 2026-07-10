#!/usr/bin/env python3
"""
ConvoPeq Convolver出力診断 自動解析ツール (work52)

使用方法:
    $env:PYTHONUTF8="1"; python analyze_conv_output.py [--raw C:/TEMP/conv_output_l.raw] [--sr 48000]
    $env:PYTHONUTF8="1"; python analyze_conv_output.py --raw C:/TEMP/conv_output_l.raw --header-only

出力ファイル: C:/TEMP/conv_output_l.raw を読み取り、解析レポートを表示する。
ヘッダ（CaptureHeader, 64byte）が存在する場合は自動検出して表示する。

評価基準:
  PASS: DC offset < 1e-6, peak < 0.99, 高調波歪み < -60dB
  WARN: 上記のいずれかが許容範囲を超えている
  FAIL: DC offset > 1e-3, peak >= 1.0, または明らかな異常共振
"""

import argparse
import struct
import sys
import os
import math
from pathlib import Path
import struct
import sys

# ── CaptureHeader ──
CAPTURE_HEADER_MAGIC = 0xCAD0DE52
CAPTURE_HEADER_SIZE = 64

def read_capture_header(filename: str) -> dict:
    """RAWファイル先頭からCaptureHeaderを読み取る。存在しなければ空dict。"""
    path = Path(filename)
    if not path.exists() or path.stat().st_size < CAPTURE_HEADER_SIZE:
        return {}
    with open(str(path), "rb") as f:
        magic = struct.unpack_from("<I", f.read(4))[0]
        if magic != CAPTURE_HEADER_MAGIC:
            return {}
        f.seek(0)
        raw = f.read(CAPTURE_HEADER_SIZE)
    # Manual parse - struct has 4-byte padding after numSamples for int64_t alignment
    # Layout: magic(4) + ver(4) + flags(4) + sr(4) + ns(4) = 20 bytes
    #         padding(4) = 24 bytes
    #         build_gen(8) + ts(8) = 40 bytes
    #         tf(8) + ta(8) + tbf(8) = 64 bytes
    magic2, ver, flags, sr, ns = struct.unpack_from("<IIIii", raw, 0)
    build_gen, ts = struct.unpack_from("<qq", raw, 24)
    tf, ta, tbf = struct.unpack_from("<ddd", raw, 40)
    return {
        "magic": hex(magic2), "version": ver,
        "captureInput": bool(flags & 1),
        "tailBypass": bool(flags & 2),
        "directHeadOff": bool(flags & 4),
        "injectTone": bool(flags & 8),
        "sampleRate": sr, "numSamples": ns,
        "buildGeneration": build_gen,
        "timestamp": ts,
        "toneFreq": tf, "toneAmp": ta, "toneBeatFreq": tbf,
    }

def print_capture_header(hdr: dict) -> None:
    """CaptureHeaderを見やすく表示。"""
    if not hdr:
        print("  [ヘッダなし: 旧フォーマット]")
        return
    print(f"  Magic: {hdr['magic']}, Version: {hdr['version']}")
    print(f"  CaptureInput={hdr['captureInput']}, TailBypass={hdr['tailBypass']}, "
          f"DirectHeadOff={hdr['directHeadOff']}, InjectTone={hdr['injectTone']}")
    print(f"  SampleRate={hdr['sampleRate']}, NumSamples={hdr['numSamples']}")
    print(f"  BuildGeneration={hdr['buildGeneration']}, Timestamp={hdr['timestamp']}")
    print(f"  Tone: {hdr['toneFreq']}Hz, {hdr['toneAmp']}amp, {hdr['toneBeatFreq']}Hz beat")
    if hdr['buildGeneration'] == 0:
        print("  [buildGeneration=0: 未設定]")

# ── 評価閾値 ──
THRESH_DC_OFFSET_PASS = 1e-6      # PASS: DC offset < 1e-6
THRESH_DC_OFFSET_WARN = 1e-4      # WARN: DC offset < 1e-4
THRESH_DC_OFFSET_FAIL = 1e-3      # FAIL: DC offset >= 1e-3
THRESH_PEAK_PASS = 0.99           # PASS: peak < 0.99
THRESH_PEAK_WARN = 0.999          # WARN: peak < 0.999
THRESH_PEAK_FAIL = 1.0            # FAIL: peak >= 1.0 (clipping!)
THRESH_RMS_ABNORMAL = 1e-12       # RMSがこれ以下なら無音とみなす

# ── 評価関数 ──

def analyze(filename: str, sample_rate: int = 48000) -> dict:
    """RAW doubleファイルを解析し、評価結果を返す。"""
    path = Path(filename)
    if not path.exists():
        return {"error": f"File not found: {filename}"}

    file_size = path.stat().st_size

    # CaptureHeader検出
    hdr = read_capture_header(str(path))
    header_bytes = CAPTURE_HEADER_SIZE if hdr else 0
    data_bytes = file_size - header_bytes
    num_samples = data_bytes // 8  # sizeof(double) = 8
    if num_samples <= 0:
        return {"error": f"Empty data section ({data_bytes} bytes, header={header_bytes})"}

    # ファイル読み込み（ヘッダ以降）
    data = []
    with open(str(path), "rb") as f:
        if header_bytes > 0:
            f.seek(header_bytes)
        raw = f.read()
    for i in range(num_samples):
        val = struct.unpack_from("<d", raw, i * 8)[0]
        data.append(val)

    # ── 基本統計量 ──
    n = len(data)
    _min = min(data)
    _max = max(data)
    _mean = sum(data) / n
    _abs_max = max(abs(_min), abs(_max))
    _rms = math.sqrt(sum(v * v for v in data) / n)

    # ── DCオフセット ──
    if _mean < 0:
        dc_offset = -_mean
    else:
        dc_offset = _mean

    # ── クリップ判定 ──
    clipped_samples = sum(1 for v in data if abs(v) >= 1.0)
    near_clip_samples = sum(1 for v in data if abs(v) >= 0.99)

    # ── ゼロクロッシングレート（異常信号検出） ──
    zcr = 0
    for i in range(1, n):
        if (data[i-1] >= 0 and data[i] < 0) or (data[i-1] < 0 and data[i] >= 0):
            zcr += 1
    zcr_rate = zcr / n if n > 0 else 0

    # ── ブロック間不連続性（ブロック境界でのジャンプ） ──
    # デフォルトブロックサイズ512と仮定
    block_size = 512
    boundary_jumps = []
    for i in range(block_size, n, block_size):
        if i > 0 and i < n:
            jump = abs(data[i] - data[i-1])
            boundary_jumps.append(jump)
    max_boundary_jump = max(boundary_jumps) if boundary_jumps else 0.0
    avg_boundary_jump = sum(boundary_jumps) / len(boundary_jumps) if boundary_jumps else 0.0

    # ── 簡易周波数分析（FFT） ──
    # 最初の4096サンプルについてFFT
    fft_size = min(4096, n)
    fft_data = data[:fft_size]

    # Hanning窓
    window = [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (fft_size - 1))) for i in range(fft_size)]
    windowed = [fft_data[i] * window[i] for i in range(fft_size)]

    # FFT（単純なDFT、低速だが依存関係不要）
    spectrum = []
    for k in range(fft_size // 2):
        re = 0.0
        im = 0.0
        for t in range(fft_size):
            angle = -2.0 * math.pi * k * t / fft_size
            re += windowed[t] * math.cos(angle)
            im += windowed[t] * math.sin(angle)
        spectrum.append(math.sqrt(re*re + im*im))

    # 正規化
    max_spec = max(spectrum) if spectrum else 1.0
    if max_spec > 0:
        spectrum = [s / max_spec for s in spectrum]

    # 基本周波数成分（40-200Hz）のエネルギー比率
    bin_hz = sample_rate / fft_size
    bass_bins = int(200 / bin_hz)
    high_bins = int(2000 / bin_hz)
    bass_energy = sum(spectrum[:bass_bins]) if bass_bins < len(spectrum) else sum(spectrum)
    high_energy = sum(spectrum[bass_bins:high_bins]) if high_bins < len(spectrum) else 0.0
    total_energy = sum(spectrum)

    bass_ratio = bass_energy / total_energy if total_energy > 0 else 0
    high_ratio = high_energy / total_energy if total_energy > 0 else 0

    # ── 高調波歪み率（THD）の簡易推定 ──
    # 基本周波数成分以外のエネルギー比率
    # 40-200Hzを基本波帯域と仮定
    noise_floor_bins = int(10000 / bin_hz)  # 10kHz以上のノイズフロア
    noise_floor = sum(spectrum[noise_floor_bins:]) / max(1, len(spectrum) - noise_floor_bins) if noise_floor_bins < len(spectrum) else 0

    # ── 総合評価 ──
    issues = []
    verdict = "PASS"

    # DC offset
    if dc_offset > THRESH_DC_OFFSET_FAIL:
        issues.append(f"CRITICAL: DC offset = {dc_offset:.6e} (FAIL)")
        verdict = "FAIL"
    elif dc_offset > THRESH_DC_OFFSET_WARN:
        issues.append(f"WARNING: DC offset = {dc_offset:.6e}")
        if verdict == "PASS":
            verdict = "WARN"
    elif dc_offset > THRESH_DC_OFFSET_PASS:
        issues.append(f"INFO: DC offset = {dc_offset:.6e} (minor)")

    # Peak
    if _abs_max >= THRESH_PEAK_FAIL:
        issues.append(f"CRITICAL: Peak = {_abs_max:.4f} (CLIPPING! FAIL)")
        verdict = "FAIL"
    elif _abs_max >= THRESH_PEAK_WARN:
        issues.append(f"WARNING: Peak = {_abs_max:.4f} (near clipping)")
        if verdict == "PASS":
            verdict = "WARN"
    elif _abs_max >= THRESH_PEAK_PASS:
        issues.append(f"INFO: Peak = {_abs_max:.4f}")
    else:
        issues.append(f"OK: Peak = {_abs_max:.4f}")

    # Clipped samples
    if clipped_samples > 0:
        issues.append(f"CRITICAL: {clipped_samples} samples >= 1.0 (HARD CLIP!)")
        verdict = "FAIL"

    # Boundary jumps (Partition境界グリッチ検出)
    if avg_boundary_jump > 0.01 and n > block_size:
        issues.append(f"WARNING: Avg block-boundary jump = {avg_boundary_jump:.6f} (partition glitch?)")
        if verdict == "PASS":
            verdict = "WARN"
    if max_boundary_jump > 0.1:
        issues.append(f"WARNING: Max block-boundary jump = {max_boundary_jump:.4f}")
        verdict = "FAIL"

    # Noise floor (高域ノイズ)
    if noise_floor > 0.001 and _abs_max > 0.01:
        issues.append(f"WARNING: High noise floor at 10kHz+ = {noise_floor:.6f} (relative)")
        if verdict == "PASS":
            verdict = "WARN"

    # RMS異常
    if _rms < THRESH_RMS_ABNORMAL:
        issues.append("INFO: Signal is near-silent (RMS below threshold)")
        verdict = "INFO (silent)"

    # ── 結果の構築 ──
    result = {
        "filename": str(path),
        "verdict": verdict,
        "num_samples": n,
        "duration_sec": n / sample_rate,
        "sample_rate": sample_rate,
        "stats": {
            "min": _min,
            "max": _max,
            "mean": _mean,
            "abs_max": _abs_max,
            "rms": _rms,
        },
        "capture_header": hdr,
        "dc_offset": dc_offset,
        "clipped_samples": clipped_samples,
        "near_clip_samples": near_clip_samples,
        "zcr_rate": zcr_rate,
        "boundary_jump": {
            "avg": avg_boundary_jump,
            "max": max_boundary_jump,
        },
        "spectral": {
            "bass_ratio": bass_ratio,
            "high_ratio": high_ratio,
            "noise_floor_10k": noise_floor,
        },
        "issues": issues,
    }

    return result


def print_report(result: dict):
    """結果を整形して表示する。"""
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return result.get("verdict", "ERROR")

    print("=" * 60)
    print("  ConvoPeq Convolver出力診断 解析レポート")
    print("=" * 60)

    # CaptureHeader表示
    hdr = result.get("capture_header", {})
    if hdr:
        print("  [キャプチャ状態]")
        print_capture_header(hdr)
        print()

    # 総合評価
    v = result["verdict"]
    if v == "PASS":
        print(f"  総合評価: ✅ PASS - 異常なし")
    elif v == "WARN":
        print(f"  総合評価: ⚠️ WARN - 軽微な異常")
    elif v == "FAIL":
        print(f"  総合評価: ❌ FAIL - 異常あり")
    else:
        print(f"  総合評価: {v}")

    print(f"  ファイル: {result['filename']}")
    print(f"  サンプル数: {result['num_samples']} ({result['duration_sec']:.2f}秒 @ {result['sample_rate']}Hz)")
    print()

    # 基本統計
    s = result["stats"]
    print(f"  [基本統計]")
    print(f"    Min: {s['min']:.8f}")
    print(f"    Max: {s['max']:.8f}")
    print(f"    Mean (DC): {s['mean']:.8e}")
    print(f"    Peak: {s['abs_max']:.4f} ({20*math.log10(max(s['abs_max'],1e-15)):.1f} dBFS)" if s['abs_max'] > 0 else "    Peak: -inf dBFS")
    print(f"    RMS: {s['rms']:.8f} ({20*math.log10(max(s['rms'],1e-15)):.1f} dBFS)" if s['rms'] > 0 else "    RMS: -inf dBFS")
    print()

    # DCオフセット
    dc = result['dc_offset']
    if dc < THRESH_DC_OFFSET_PASS:
        print(f"  [DCオフセット] ✅ {dc:.2e} (正常範囲)")
    elif dc < THRESH_DC_OFFSET_WARN:
        print(f"  [DCオフセット] ⚠️ {dc:.2e} (軽微)")
    else:
        print(f"  [DCオフセット] ❌ {dc:.2e} (異常)")

    # クリップ
    if result['clipped_samples'] > 0:
        print(f"  [クリップ] ❌ {result['clipped_samples']} samples >= 1.0 (ハードクリップ発生!)")
    elif result['near_clip_samples'] > 0:
        print(f"  [クリップ] ⚠️ {result['near_clip_samples']} samples >= 0.99 (ニアクリップ)")
    else:
        print(f"  [クリップ] ✅ クリップなし")

    # ブロック境界
    bj = result['boundary_jump']
    print(f"  [ブロック境界] avg jump={bj['avg']:.8f}, max jump={bj['max']:.6f}")

    # スペクトル
    sp = result['spectral']
    print(f"  [スペクトル]")
    print(f"    低域比率(40-200Hz): {sp['bass_ratio']*100:.1f}%")
    print(f"    中域比率(200-2000Hz): {sp['high_ratio']*100:.1f}%")
    print(f"    10kHz+ ノイズフロア: {sp['noise_floor_10k']:.6f}")
    print()

    # Issue一覧
    if result['issues']:
        print(f"  [検出項目]")
        for issue in result['issues']:
            print(f"    • {issue}")
    else:
        print(f"  [検出項目] なし")

    print("=" * 60)

    # 総評
    print()
    if v == "PASS":
        print("  ▶ Convolver出力に異常は検出されませんでした。")
        print("    問題は Convolver より後段（outputMakeup → SoftClip → DC Blocker → NS 経路）")
        print("    にある可能性が高いです。")
    elif v == "FAIL":
        print("  ▶ Convolver出力に異常を検出しました。")
        if result['clipped_samples'] > 0:
            print("    → クリップが発生しています。IRのスケールまたはinputHeadroom設定を見直してください。")
        if result['dc_offset'] > 1e-3:
            print("    → DCオフセットが大きいです。IRにDC成分が含まれている可能性があります。")
        if any("boundary" in i for i in result['issues']):
            print("    → ブロック境界で不連続が検出されました。パーティション畳み込みの可能性があります。")
    elif v == "WARN":
        print("  ▶ Convolver出力に軽微な異常を検出しました。")
        print("    実機での聴感テストと併せて判断してください。")

    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvoPeq Convolver出力診断ツール")
    parser.add_argument("--raw", default=r"C:\TEMP\conv_output_l.raw",
                        help="RAW doubleファイルのパス")
    parser.add_argument("--sr", type=int, default=48000,
                        help="サンプルレート (default: 48000)")
    parser.add_argument("--header-only", action="store_true",
                        help="ヘッダ情報のみ表示")
    args = parser.parse_args()

    if args.header_only:
        hdr = read_capture_header(args.raw)
        if hdr:
            print("=" * 60)
            print("  CaptureHeader")
            print("=" * 60)
            print_capture_header(hdr)
        else:
            print("[INFO] No CaptureHeader found (old format or invalid file)")
        sys.exit(0)

    result = analyze(args.raw, args.sr)
    verdict = print_report(result)
    sys.exit(0 if verdict == "PASS" else (1 if verdict == "WARN" else 2))
